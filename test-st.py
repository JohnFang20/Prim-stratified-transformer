import collections
import os
import time
import random
import numpy as np
import logging
import argparse
import shutil

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter

from util import dataset, config
from util.abc import ABC_Dataset
from util.s3dis import S3DIS
from util.scannet_v2 import Scannetv2
from util.common_util import AverageMeter, intersectionAndUnionGPU, find_free_port, poly_learning_rate, smooth_loss
from util.data_util import collate_fn, collate_fn_limit
from util.loss_util import compute_embedding_loss, mean_shift_gpu, compute_iou
from util import transform
from util.logger import get_logger

from functools import partial
from util.lr import MultiStepWithWarmup, PolyLR, PolyLRwithWarmup
import torch_points_kernels as tp

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Primitive Segmentation')
    parser.add_argument('--config', type=str, default='config/abc/abc.yaml', help='config file')
    parser.add_argument('opts', help='see config/abc/abc.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg

# def get_logger():
#     logger_name = "main-logger"
#     logger = logging.getLogger(logger_name)
#     logger.setLevel(logging.INFO)
#     handler = logging.StreamHandler()
#     fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
#     handler.setFormatter(logging.Formatter(fmt))
#     logger.addHandler(handler)
#     return logger


def main_process():
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def main():
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    # import torch.backends.mkldnn
    # ackends.mkldnn.enabled = False
    # os.environ["LRU_CACHE_CAPACITY"] = "1"
    # cudnn.deterministic = True

    # if args.manual_seed is not None:
    #     random.seed(args.manual_seed)
    #     np.random.seed(args.manual_seed)
    #     torch.manual_seed(args.manual_seed)
    #     torch.cuda.manual_seed(args.manual_seed)
    #     torch.cuda.manual_seed_all(args.manual_seed)
    #     cudnn.benchmark = False
    #     cudnn.deterministic = True

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False

    if args.multiprocessing_distributed:
        port = find_free_port()
        args.dist_url = f"tcp://127.0.0.1:{port}"
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args, best_iou
    args, best_iou = argss, 0
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    
    # get model
    if args.arch == 'stratified_transformer':
        
        from model.stratified_transformer import Stratified

        args.patch_size = args.grid_size * args.patch_size
        args.window_size = [args.patch_size * args.window_size * (2**i) for i in range(args.num_layers)]
        args.grid_sizes = [args.patch_size * (2**i) for i in range(args.num_layers)]
        args.quant_sizes = [args.quant_size * (2**i) for i in range(args.num_layers)]

        model = Stratified(args.downsample_scale, args.depths, args.channels, args.num_heads, args.window_size, \
            args.up_k, args.grid_sizes, args.quant_sizes, rel_query=args.rel_query, \
            rel_key=args.rel_key, rel_value=args.rel_value, drop_path_rate=args.drop_path_rate, concat_xyz=args.concat_xyz, num_classes=args.classes, \
            ratio=args.ratio, k=args.k, prev_grid_size=args.grid_size, sigma=1.0, num_layers=args.num_layers, stem_transformer=args.stem_transformer)

    elif args.arch == 'swin3d_transformer':
        
        from model.swin3d_transformer import Swin

        args.patch_size = args.grid_size * args.patch_size
        args.window_sizes = [args.patch_size * args.window_size * (2**i) for i in range(args.num_layers)]
        args.grid_sizes = [args.patch_size * (2**i) for i in range(args.num_layers)]
        args.quant_sizes = [args.quant_size * (2**i) for i in range(args.num_layers)]

        model = Swin(args.depths, args.channels, args.num_heads, \
            args.window_sizes, args.up_k, args.grid_sizes, args.quant_sizes, rel_query=args.rel_query, \
            rel_key=args.rel_key, rel_value=args.rel_value, drop_path_rate=args.drop_path_rate, \
            concat_xyz=args.concat_xyz, num_classes=args.classes, \
            ratio=args.ratio, k=args.k, prev_grid_size=args.grid_size, sigma=1.0, num_layers=args.num_layers, stem_transformer=args.stem_transformer)

    elif args.arch == 'boundary_transformer':

        from model.boundary_transformer import Stratified

        args.patch_size = args.grid_size * args.patch_size
        args.window_size = [args.patch_size * args.window_size * (2**i) for i in range(args.num_layers)]
        args.grid_sizes = [args.patch_size * (2**i) for i in range(args.num_layers)]
        args.quant_sizes = [args.quant_size * (2**i) for i in range(args.num_layers)]

        model = Stratified(args.downsample_scale, args.depths, args.channels, args.num_heads, args.window_size, \
            args.up_k, args.grid_sizes, args.quant_sizes, rel_query=args.rel_query, \
            rel_key=args.rel_key, rel_value=args.rel_value, drop_path_rate=args.drop_path_rate, concat_xyz=args.concat_xyz, num_classes=args.classes, \
            ratio=args.ratio, k=args.k, prev_grid_size=args.grid_size, sigma=1.0, num_layers=args.num_layers, stem_transformer=args.stem_transformer)

    else:
        raise Exception('architecture {} not supported yet'.format(args.arch))
    
    # set loss func 
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda()
    # criterion = nn.CrossEntropyLoss().cuda()
    
    # set optimizer
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':     # Adamw 即 Adam + weight decate ,效果与 Adam + L2正则化相同,但是计算效率更高
        transformer_lr_scale = args.get("transformer_lr_scale", 0.1)
        param_dicts = [
            {"params": [p for n, p in model.named_parameters() if "blocks" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model.named_parameters() if "blocks" in n and p.requires_grad],
                "lr": args.base_lr * transformer_lr_scale,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=args.base_lr, weight_decay=args.weight_decay)

    if main_process():
        global logger, writer
        logger = get_logger(args.save_folder)
        # writer = SummaryWriter(args.save_path)
        logger.info(args)
        logger.info("=> creating model ...")
        logger.info("Classes: {}".format(args.classes))
        logger.info(model)
        logger.info('#Model parameters: {}'.format(sum([x.nelement() for x in model.parameters()])))
        # if args.get("max_grad_norm", None):
        #     logger.info("args.max_grad_norm = {}".format(args.max_grad_norm))

    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        if args.sync_bn:
            if main_process():
                logger.info("use SyncBN")
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)
    else:
        model = torch.nn.DataParallel(model.cuda())

    # if args.weight:
    #     if os.path.isfile(args.weight):
    #         if main_process():
    #             logger.info("=> loading weight '{}'".format(args.weight))
    #         checkpoint = torch.load(args.weight)
    #         model.load_state_dict(checkpoint['state_dict'])
    #         if main_process():
    #             logger.info("=> loaded weight '{}'".format(args.weight))
    #     else:
    #         logger.info("=> no weight found at '{}'".format(args.weight))

    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         if main_process():
    #             logger.info("=> loading checkpoint '{}'".format(args.resume))
    #         checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
    #         args.start_epoch = checkpoint['epoch']
    #         model.load_state_dict(checkpoint['state_dict'], strict=True)
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         scheduler_state_dict = checkpoint['scheduler']
    #         best_iou = checkpoint['best_iou']
    #         if main_process():
    #             logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    #     else:
    #         if main_process():
    #             logger.info("=> no checkpoint found at '{}'".format(args.resume))

    # if os.path.isfile(args.model_path):
    #     if main_process():
    #         logger.info("=> loading checkpoint '{}'".format(args.model_path))
    #     checkpoint = torch.load(args.model_path)
    #     state_dict = checkpoint['state_dict']
    #     new_state_dict = collections.OrderedDict()
    #     for k, v in state_dict.items():
    #         name = k[7:]
    #         new_state_dict[name.replace("item", "stem")] = v
    #     model.load_state_dict(new_state_dict, strict=True)
    #     if main_process():
    #         logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.model_path, checkpoint['epoch']))
    #     args.epoch = checkpoint['epoch']
    # else:
    #     raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))
    
    if args.model_path:
        if os.path.isfile(args.model_path):
            if main_process():
                logger.info("=> loading weight '{}'".format(args.model_path))
            checkpoint = torch.load(args.model_path)
            model.load_state_dict(checkpoint['state_dict'])
            if main_process():
                logger.info("=> loaded weight '{}'".format(args.model_path))
        else:
            raise RuntimeError("=> no weight found at '{}'".format(args.model_path))


    # if args.data_name == 's3dis':
    #     train_transform = None
    #     if args.aug:
    #         jitter_sigma = args.get('jitter_sigma', 0.01)
    #         jitter_clip = args.get('jitter_clip', 0.05)
    #         if main_process():
    #             logger.info("augmentation all")
    #             logger.info("jitter_sigma: {}, jitter_clip: {}".format(jitter_sigma, jitter_clip))
    #         train_transform = transform.Compose([
    #             transform.RandomRotate(along_z=args.get('rotate_along_z', True)),
    #             transform.RandomScale(scale_low=args.get('scale_low', 0.8), scale_high=args.get('scale_high', 1.2)),
    #             transform.RandomJitter(sigma=jitter_sigma, clip=jitter_clip),
    #             transform.RandomDropColor(color_augment=args.get('color_augment', 0.0))
    #         ])
    #     train_data = S3DIS(split='train', data_root=args.data_root, test_area=args.test_area, voxel_size=args.voxel_size, voxel_max=args.voxel_max, transform=train_transform, shuffle_index=True, loop=args.loop)
    # elif args.data_name == 'scannetv2':
    #     train_transform = None
    #     if args.aug:
    #         if main_process():
    #             logger.info("use Augmentation")
    #         train_transform = transform.Compose([
    #             transform.RandomRotate(along_z=args.get('rotate_along_z', True)),
    #             transform.RandomScale(scale_low=args.get('scale_low', 0.8), scale_high=args.get('scale_high', 1.2)),
    #             transform.RandomDropColor(color_augment=args.get('color_augment', 0.0))
    #         ])
            
    #     train_split = args.get("train_split", "train")
    #     if main_process():
    #         logger.info("scannet. train_split: {}".format(train_split))

    #     train_data = Scannetv2(split=train_split, data_root=args.data_root, voxel_size=args.voxel_size, voxel_max=args.voxel_max, transform=train_transform, shuffle_index=True, loop=args.loop)

    if args.data_name == 'abc':
        # train_transform = None
        # if args.aug:
        #     jitter_sigma = args.get('jitter_sigma', 0.01)
        #     jitter_clip = args.get('jitter_clip', 0.05)
        #     if main_process():
        #         logger.info("augmentation all")
        #         logger.info("jitter_sigma: {}, jitter_clip: {}".format(jitter_sigma, jitter_clip))
        #     train_transform = transform.Compose([
        #         transform.RandomRotate(along_z=args.get('rotate_along_z', True)),
        #         transform.RandomScale(scale_low=args.get('scale_low', 0.8), scale_high=args.get('scale_high', 1.2)),
        #         transform.RandomJitter(sigma=jitter_sigma, clip=jitter_clip),
        #         transform.RandomDropColor(color_augment=args.get('color_augment', 0.0))
        #     ])
        train_data = ABC_Dataset(split='train', data_root=args.data_root, voxel_size=args.voxel_size, voxel_max=args.voxel_max, shuffle_index=True, loop=args.loop)
    else:
        raise ValueError("The dataset {} is not supported.".format(args.data_name))

    if main_process():
            logger.info("train_data samples: '{}'".format(len(train_data)))
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, \
        pin_memory=True, sampler=train_sampler, drop_last=True, collate_fn=partial(collate_fn_limit, max_batch_points=args.max_batch_points, logger=logger if main_process() else None))

    val_transform = None
    # if args.data_name == 's3dis':
    #     val_data = S3DIS(split='val', data_root=args.data_root, test_area=args.test_area, voxel_size=args.voxel_size, voxel_max=800000, transform=val_transform)
    # elif args.data_name == 'scannetv2':
    #     val_data = Scannetv2(split='val', data_root=args.data_root, voxel_size=args.voxel_size, voxel_max=800000, transform=val_transform)
    if args.data_name == 'abc':
        val_data = ABC_Dataset(split='val', data_root=args.data_root, voxel_size=args.voxel_size, voxel_max=800000, loop=0.06)
    else:
        raise ValueError("The dataset {} is not supported.".format(args.data_name))

    if args.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
    else:
        val_sampler = None
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False, num_workers=args.workers, \
            pin_memory=True, sampler=val_sampler, collate_fn=collate_fn)
    
    # set scheduler
    # if args.scheduler == "MultiStepWithWarmup":
    #     assert args.scheduler_update == 'step'
    #     if main_process():
    #         logger.info("scheduler: MultiStepWithWarmup. scheduler_update: {}".format(args.scheduler_update))
    #     iter_per_epoch = len(train_loader)
    #     milestones = [int(args.epochs*0.6) * iter_per_epoch, int(args.epochs*0.8) * iter_per_epoch]
    #     scheduler = MultiStepWithWarmup(optimizer, milestones=milestones, gamma=0.1, warmup=args.warmup, \
    #         warmup_iters=args.warmup_iters, warmup_ratio=args.warmup_ratio)
    # elif args.scheduler == 'MultiStep':
    #     assert args.scheduler_update == 'epoch'
    #     milestones = [int(x) for x in args.milestones.split(",")] if hasattr(args, "milestones") else [int(args.epochs*0.4), int(args.epochs*0.8)]
    #     gamma = args.gamma if hasattr(args, 'gamma') else 0.1
    #     if main_process():
    #         logger.info("scheduler: MultiStep. scheduler_update: {}. milestones: {}, gamma: {}".format(args.scheduler_update, milestones, gamma))
    #     scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)     # 当前epoch数满足设定值时，调整学习率
    # elif args.scheduler == 'Poly':
    #     if main_process():
    #         logger.info("scheduler: Poly. scheduler_update: {}".format(args.scheduler_update))
    #     if args.scheduler_update == 'epoch':
    #         scheduler = PolyLR(optimizer, max_iter=args.epochs, power=args.power)
    #     elif args.scheduler_update == 'step':
    #         iter_per_epoch = len(train_loader)
    #         scheduler = PolyLR(optimizer, max_iter=args.epochs*iter_per_epoch, power=args.power)
    #     else:
    #         raise ValueError("No such scheduler update {}".format(args.scheduler_update))
    # else:
    #     raise ValueError("No such scheduler {}".format(args.scheduler))

    # if args.resume and os.path.isfile(args.resume):
    #     scheduler.load_state_dict(scheduler_state_dict)
    #     print("resume scheduler")

    ###################
    # start training #
    ###################

    # if args.use_amp:    # 自动混合精度训练 —— 节省显存并加快推理速度
    #     scaler = torch.cuda.amp.GradScaler()
    # else:
    #     scaler = None
    
    # for epoch in range(args.start_epoch, args.epochs):
    #     if args.distributed:
    #         train_sampler.set_epoch(epoch)

    #     if main_process():
    #         logger.info("lr: {}".format(scheduler.get_last_lr()))
            
    #     # loss_train, mIoU_train, mAcc_train, allAcc_train = train(train_loader, model, criterion, optimizer, epoch, scaler, scheduler)
    #     feat_loss_train, type_loss_train, boundary_loss_train = train(train_loader, model, criterion, optimizer, epoch, scaler, scheduler)
    #     if args.scheduler_update == 'epoch':
    #         scheduler.step()
    #     epoch_log = epoch + 1
        
    #     if main_process():
    #         writer.add_scalar('feat_loss_train', feat_loss_train, epoch_log)
    #         writer.add_scalar('type_loss_train', type_loss_train, epoch_log)
    #         writer.add_scalar('boundary_loss_train', boundary_loss_train, epoch_log)
            # writer.add_scalar('loss_train', loss_train, epoch_log)
            # writer.add_scalar('mIoU_train', mIoU_train, epoch_log)
            # writer.add_scalar('mAcc_train', mAcc_train, epoch_log)
            # writer.add_scalar('allAcc_train', allAcc_train, epoch_log)

        # is_best = False
        # if args.evaluate and (epoch_log % args.eval_freq == 0):
            # loss_val, mIoU_val, mAcc_val, allAcc_val = validate(val_loader, model, criterion)
    with torch.no_grad():
        s_miou, p_miou, feat_loss_val, type_loss_val, boundary_loss_val = validate(val_loader, model, criterion)

            # if main_process():
            #     writer.add_scalar('feat_loss_val', feat_loss_val, epoch_log)
            #     writer.add_scalar('type_loss_val', type_loss_val, epoch_log)
            #     writer.add_scalar('boundary_loss_val', boundary_loss_val, epoch_log)
            #     writer.add_scalar('s_miou', s_miou, epoch_log)
            #     writer.add_scalar('p_miou', p_miou, epoch_log)
            #     # writer.add_scalar('loss_val', loss_val, epoch_log)
            #     # writer.add_scalar('mIoU_val', mIoU_val, epoch_log)
            #     # writer.add_scalar('mAcc_val', mAcc_val, epoch_log)
            #     # writer.add_scalar('allAcc_val', allAcc_val, epoch_log)
            #     is_best = s_miou > best_iou
            #     best_iou = max(best_iou, s_miou)

        # if (epoch_log % args.save_freq == 0) and main_process():
        #     if not os.path.exists(args.save_path + "/model/"):
        #         os.makedirs(args.save_path + "/model/")
        #     filename = args.save_path + '/model/model_last.pth'
        #     logger.info('Saving checkpoint to: ' + filename)
        #     torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
        #                 'scheduler': scheduler.state_dict(), 'best_iou': best_iou, 'is_best': is_best}, filename)
        #     if is_best:
        #         shutil.copyfile(filename, args.save_path + '/model/model_best.pth')

        # if main_process():
        #     # writer.close()
        #     logger.info('Val result: Seg_mIoU/Type_mIoU {:.4f}/{:.4f}.'.format(s_miou, p_miou))
        #     logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')


def train(train_loader, model, criterion, optimizer, epoch, scaler, scheduler):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # loss_meter = AverageMeter()
    # intersection_meter = AverageMeter()
    # union_meter = AverageMeter()
    # target_meter = AverageMeter()
    feat_loss_meter = AverageMeter()
    type_loss_meter = AverageMeter()
    boundary_loss_meter = AverageMeter()
    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    for i, (coord, normals, boundary, label, semantic, param, offset) in enumerate(train_loader):  # (n, 3), (n, c), (n), (b)
        data_time.update(time.time() - end)

        offset_ = offset.clone()
        offset_[1:] = offset_[1:] - offset_[:-1]
        batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long()

        sigma = 1.0
        radius = 2.5 * args.grid_size * sigma
        neighbor_idx = tp.ball_query(radius, args.max_num_neighbors, coord, coord, mode="partial_dense", batch_x=batch, batch_y=batch)[0]
    
        coord, normals, boundary, label, semantic, param, offset = coord.cuda(non_blocking=True), normals.cuda(non_blocking=True), boundary.cuda(non_blocking=True), \
                                label.cuda(non_blocking=True), semantic.cuda(non_blocking=True), param.cuda(non_blocking=True), offset.cuda(non_blocking=True)
        batch = batch.cuda(non_blocking=True)
        neighbor_idx = neighbor_idx.cuda(non_blocking=True)
        assert batch.shape[0] == normals.shape[0]
        
        if args.concat_xyz:
            feat = torch.cat([normals, coord], 1)

        use_amp = args.use_amp
        with torch.cuda.amp.autocast(enabled=use_amp):
            primitive_embedding, type_per_point, boundary_pred = model(feat, coord, offset, batch, neighbor_idx)
            assert type_per_point.shape[1] == args.classes
            if semantic.shape[-1] == 1:
                semantic = semantic[:, 0]  # for cls
            # loss = criterion(output, target)

            feat_loss, pull_loss, push_loss = compute_embedding_loss(primitive_embedding, label, offset)
            type_loss = criterion(type_per_point, semantic)
            boundary_loss = criterion(boundary_pred, boundary)
            loss = feat_loss + type_loss + boundary_loss
            
        optimizer.zero_grad()
        
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if args.scheduler_update == 'step':
            scheduler.step()

        # output = output.max(1)[1]
        # n = coord.size(0)
        # if args.multiprocessing_distributed:
        #     loss *= n
        #     count = target.new_tensor([n], dtype=torch.long)
        #     dist.all_reduce(loss), dist.all_reduce(count)
        #     n = count.item()
        #     loss /= n
        # intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        # if args.multiprocessing_distributed:
        #     dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        # intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        # intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        # accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        # loss_meter.update(loss.item(), n)

        # All Reduce loss
        if args.multiprocessing_distributed:
            dist.all_reduce(feat_loss.div_(torch.cuda.device_count()))
            dist.all_reduce(type_loss.div_(torch.cuda.device_count()))
            dist.all_reduce(boundary_loss.div_(torch.cuda.device_count()))
        feat_loss_, type_loss_, boundary_loss_ = feat_loss.data.cpu().numpy(), type_loss.data.cpu().numpy(), boundary_loss.data.cpu().numpy()
        feat_loss_meter.update(feat_loss_.item())
        type_loss_meter.update(type_loss_.item())
        boundary_loss_meter.update(boundary_loss_.item())
        batch_time.update(time.time() - end)
        end = time.time()

        # calculate remain time
        current_iter = epoch * len(train_loader) + i + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            lr = scheduler.get_last_lr()
            if isinstance(lr, list):
                lr = [round(x, 8) for x in lr]
            elif isinstance(lr, float):
                lr = round(lr, 8)
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        # 'Loss {loss_meter.val:.4f} '
                        'Feat_Loss {feat_loss_meter.val:.4f} '
                        'Type_Loss {type_loss_meter.val:.4f}.'
                        'Boundary_Loss {boundary_loss_meter.val:.4f}.'
                        'Lr: {lr} '.format(epoch+1, args.epochs, i + 1, len(train_loader),
                                                          batch_time=batch_time, data_time=data_time,
                                                          remain_time=remain_time,
                                                          feat_loss_meter=feat_loss_meter,
                                                          type_loss_meter=type_loss_meter,
                                                          boundary_loss_meter=boundary_loss_meter,
                                                          lr=lr))
        if main_process():
            # writer.add_scalar('loss_train_batch', loss_meter.val, current_iter)
            writer.add_scalar('feat_loss_train_batch', feat_loss_meter.val, current_iter)
            writer.add_scalar('type_loss_train_batch', type_loss_meter.val, current_iter)
            writer.add_scalar('boundary_loss_train_batch', boundary_loss_meter.val, current_iter)
            # writer.add_scalar('mIoU_train_batch', np.mean(intersection / (union + 1e-10)), current_iter)
            # writer.add_scalar('mAcc_train_batch', np.mean(intersection / (target + 1e-10)), current_iter)
            # writer.add_scalar('allAcc_train_batch', accuracy, current_iter)

    # iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    # accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    # mIoU = np.mean(iou_class)
    # mAcc = np.mean(accuracy_class)
    # allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    # if main_process():
    #     logger.info('Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch+1, args.epochs, mIoU, mAcc, allAcc))
    # return loss_meter.avg, mIoU, mAcc, allAcc
    return feat_loss_meter.avg, type_loss_meter.avg, boundary_loss_meter.avg


def validate(val_loader, model, criterion):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # loss_meter = AverageMeter()
    # intersection_meter = AverageMeter()
    # union_meter = AverageMeter()
    # target_meter = AverageMeter()
    feat_loss_meter = AverageMeter()
    type_loss_meter = AverageMeter()
    boundary_loss_meter = AverageMeter()
    s_iou_meter = AverageMeter()
    type_iou_meter = AverageMeter()

    torch.cuda.empty_cache()

    model.eval()
    end = time.time()
    for i, (coord, normals, boundary, label, semantic, param, offset) in enumerate(val_loader):
        data_time.update(time.time() - end)
    
        offset_ = offset.clone()
        offset_[1:] = offset_[1:] - offset_[:-1]
        batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long()

        sigma = 1.0
        radius = 2.5 * args.grid_size * sigma
        neighbor_idx = tp.ball_query(radius, args.max_num_neighbors, coord, coord, mode="partial_dense", batch_x=batch, batch_y=batch)[0]
    
        coord, normals, boundary, label, semantic, param, offset = coord.cuda(non_blocking=True), normals.cuda(non_blocking=True), boundary.cuda(non_blocking=True), \
                                label.cuda(non_blocking=True), semantic.cuda(non_blocking=True), param.cuda(non_blocking=True), offset.cuda(non_blocking=True)
        batch = batch.cuda(non_blocking=True)
        neighbor_idx = neighbor_idx.cuda(non_blocking=True)
        assert batch.shape[0] == normals.shape[0]
        
        if semantic.shape[-1] == 1:
            semantic = semantic[:, 0]  # for cls

        if args.concat_xyz:
            feat = torch.cat([normals, coord], 1)

        with torch.no_grad():
            primitive_embedding, type_per_point, boundary_pred = model(feat, coord, offset, batch, neighbor_idx)
            # loss = criterion(output, target)

            feat_loss, pull_loss, push_loss = compute_embedding_loss(primitive_embedding, label, offset)
            type_loss = criterion(type_per_point, semantic)
            boundary_loss = criterion(boundary_pred, boundary)
            loss = feat_loss + type_loss + boundary_loss

        # output = output.max(1)[1]
        # n = coord.size(0)
        # if args.multiprocessing_distributed:
        #     loss *= n
        #     count = target.new_tensor([n], dtype=torch.long)
        #     dist.all_reduce(loss), dist.all_reduce(count)
        #     n = count.item()
        #     loss /= n

        # intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        # if args.multiprocessing_distributed:
        #     dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        # intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        # intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        # accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        # loss_meter.update(loss.item(), n)

        spec_cluster_pred = mean_shift_gpu(primitive_embedding, offset, bandwidth=args.bandwidth)
        s_iou, p_iou = compute_iou(label, spec_cluster_pred, type_per_point, semantic, offset)

        if args.visual:
            softmax = torch.nn.Softmax(dim=1)
            boundary_pred_ = softmax(boundary_pred)
            boundary_pred_ = (boundary_pred_[:,1] > 0.5).data.cpu().numpy().astype('int32')
            bound_color = np.array([[0.41176,0.41176,0.41176], [1,0,0]])
            for k in range(len(offset)):
                if k == 0:
                    pb = boundary_pred_[0:offset[k]]
                    pc = coord[0:offset[k]]
                else:
                    pb = boundary_pred_[offset[k-1]:offset[k]]
                    pc = coord[offset[k-1]:offset[k]]

                fp = open('/home/fz20/Project/Prim-Stratified-Transformer/visual/%s_%s.obj'%(i,k), 'w')
                for j in range(pc.shape[0]):
                    v = pc[j]
                    if pb[j] < 0:
                        p = np.array([0,0,0])
                    else:
                        p = bound_color[pb[j]]
                    fp.write('v %f %f %f %f %f %f\n'%(v[0],v[1],v[2],p[0],p[1],p[2]))
                fp.close()
        # All Reduce loss
        if args.multiprocessing_distributed:
            dist.all_reduce(feat_loss.div_(torch.cuda.device_count()))
            dist.all_reduce(type_loss.div_(torch.cuda.device_count()))
            dist.all_reduce(boundary_loss.div_(torch.cuda.device_count()))
            # dist.all_reduce(s_iou.div_(torch.cuda.device_count()))
            # dist.all_reduce(p_iou.div_(torch.cuda.device_count()))
        feat_loss_, type_loss_, boundary_loss_ = feat_loss.data.cpu().numpy(), type_loss.data.cpu().numpy(), boundary_loss.data.cpu().numpy()
        feat_loss_meter.update(feat_loss_.item())
        type_loss_meter.update(type_loss_.item())
        boundary_loss_meter.update(boundary_loss_.item())
        s_iou_meter.update(s_iou)
        type_iou_meter.update(p_iou)
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        # 'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'Feat_Loss {feat_loss_meter.val:.4f} ({feat_loss_meter.avg:.4f}) '
                        'Type_Loss {type_loss_meter.val:.4f} ({type_loss_meter.avg:.4f}) '
                        'Boundary_Loss {boundary_loss_meter.val:.4f} ({boundary_loss_meter.avg:.4f}) '
                        'Seg_IoU {s_iou_meter.val:.4f} ({s_iou_meter.avg:.4f}) '
                        'Type_IoU {type_iou_meter.val:.4f} ({type_iou_meter.avg:.4f}).'.format(i + 1, len(val_loader),
                                                          data_time=data_time,
                                                          batch_time=batch_time,
                                                          feat_loss_meter=feat_loss_meter,
                                                          type_loss_meter=type_loss_meter,
                                                          boundary_loss_meter=boundary_loss_meter,
                                                          s_iou_meter=s_iou_meter,
                                                          type_iou_meter=type_iou_meter))

    # iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    # accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    # mIoU = np.mean(iou_class)
    # mAcc = np.mean(accuracy_class)
    # allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        # logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        # for i in range(args.classes):
        #     logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
        logger.info('Val result: Seg_mIoU/Type_mIoU {:.4f}/{:.4f}.'.format(s_iou_meter.avg, type_iou_meter.avg))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    
    # return loss_meter.avg, mIoU, mAcc, allAcc
    return s_iou_meter.avg, type_iou_meter.avg, feat_loss_meter.avg, type_loss_meter.avg, boundary_loss_meter.avg


if __name__ == '__main__':
    import gc
    gc.collect()
    main()