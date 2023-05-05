# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Train a YOLOv5 model on a custom dataset.
Models and datasets download automatically from the latest YOLOv5 release.

Usage - Single-GPU training:
    $ python train.py --data coco128.yaml --weights yolov5s.pt --img 640  # from pretrained (recommended)
    $ python train.py --data coco128.yaml --weights '' --cfg yolov5s.yaml --img 640  # from scratch

Usage - Multi-GPU DDP training:
    $ python -m torch.distributed.run --nproc_per_node 4 --master_port 1 train.py --data coco128.yaml --weights yolov5s.pt --img 640 --device 0,1,2,3

Models:     https://github.com/ultralytics/yolov5/tree/master/models
Datasets:   https://github.com/ultralytics/yolov5/tree/master/data
Tutorial:   https://docs.ultralytics.com/yolov5/tutorials/train_custom_data
"""
"""
train.py: 训练YOLOv5模型
1. 数据;
2. 模型
3. 学习率
4. 优化器
5. 训练
"""
import argparse  # 解析命令行参数模块
import math  # 数学公式模块
import os  # 与操作系统进行交互的模块 包含文件路径操作和解析
import random  # 生成随机数模块
import subprocess
import sys  # sys系统模块 包含了与Python解释器和它的环境有关的函数
import time  # 时间模块 更底层
from copy import deepcopy  # 深度拷贝模块
from datetime import datetime  # 日期时间模块
from pathlib import Path  # Path将str转换为Path对象 使字符串路径易于操作的模块

import numpy as np  # 数组操作模块
import torch  # PyTorch模块
import torch.distributed as dist  # 分布式训练模块
import torch.nn as nn  # 神经网络模块
import yaml  # YAML文件读写模块
from torch.optim import lr_scheduler  # 学习率调整模块
from tqdm import tqdm  # 进度条模块

FILE = Path(__file__).resolve()  # 将 yolov5/train.py 转换为绝对路径
ROOT = FILE.parents[0]  # YOLOv5 root directory: yolov5/
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import val as validate  # 验证模块
from models.experimental import attempt_load  # 模型加载模块
from models.yolo import Model  # YOLO模型模块

from utils.autoanchor import check_anchors  # 检查锚点模块
from utils.autobatch import check_train_batch_size  # 检查训练批次大小模块
from utils.callbacks import Callbacks  # 回调模块
from utils.dataloaders import create_dataloader  # 创建数据加载器模块
from utils.downloads import attempt_download, is_url  # 下载模块
# 通用模块
from utils.general import (LOGGER, TQDM_BAR_FORMAT, check_amp,  # 日志、进度条、检查amp、
                           check_dataset, check_file, check_git_info,  # 检查数据集、检查文件、检查git信息、
                           check_git_status, check_img_size,  # 检查git状态、检查图片大小、
                           check_requirements, check_suffix, check_yaml,  # 检查要求、检查后缀、检查yaml、
                           colorstr, get_latest_run, increment_path,  # 颜色字符串、获取最新运行、增加路径、
                           init_seeds, intersect_dicts, labels_to_class_weights,  # 初始化种子、交集字典、标签转类别权重、
                           labels_to_image_weights, methods, one_cycle,  # 标签转图像权重、方法、单周期、
                           print_args, print_mutation, strip_optimizer, yaml_save)  # 打印参数、打印突变、剥离优化器、yaml保存
from utils.loggers import Loggers  # 日志模块
from utils.loggers.comet.comet_utils import check_comet_resume  # 检查comet恢复模块
from utils.loss import ComputeLoss  # 计算损失模块
from utils.metrics import fitness  # 适应度模块
from utils.plots import plot_evolve, plot_lr_scheduler  # 绘制进化图模块
from utils.torch_utils import (EarlyStopping, ModelEMA,  # 提前停止模块、指数移动平均模块
                               de_parallel, select_device,  # 取消并行模块、选择设备模块
                               smart_DDP, smart_optimizer,  # 智能DDP模块、智能优化器模块
                               smart_resume, torch_distributed_zero_first)  # 智能恢复模块、分布式训练模块

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
GIT_INFO = check_git_info()


def train(hyp, opt, device, callbacks) -> tuple:
    """训练模型
    :param hyp: 超参数
    :param opt: 命令行参数
    :param device: 设备
    :param callbacks: 回调函数
    :return: results: 训练结果
    """
    """--------------------------------------------- 初始化参数和配置信息 ---------------------------------------------"""
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
            opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze
    callbacks.run('on_pretrain_routine_start')

    # Directories
    w = save_dir / 'weights'  # weights dir
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # 新建权重文件夹, 如果是进化训练, 则在父文件夹下新建
    last, best = w / 'last.pt', w / 'best.pt'

    # Hyperparameters 超参数
    if isinstance(hyp, str):  # 如果超参数是字符串, 则读取超参数文件
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict 读取超参数字典
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    opt.hyp = hyp.copy()  # for saving hyps to checkpoints 保存超参数到检查点

    # Save run settings
    if not evolve:
        yaml_save(save_dir / 'hyp.yaml', hyp)
        yaml_save(save_dir / 'opt.yaml', vars(opt))

    # Loggers
    data_dict = None
    if RANK in {-1, 0}:
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  # loggers instance

        # Register actions
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))

        # Process custom dataset artifact link
        data_dict = loggers.remote_dataset
        if resume:  # If resuming runs from remote artifact
            weights, epochs, hyp, batch_size = opt.weights, opt.epochs, opt.hyp, opt.batch_size

    # Config
    plots = not evolve and not opt.noplots  # create plots. 不是进化训练, 且不是不绘制图像
    cuda = device.type != 'cpu'  # 是否使用cuda
    init_seeds(opt.seed + 1 + RANK, deterministic=True)  # 初始化种子
    with torch_distributed_zero_first(LOCAL_RANK):  # 分布式训练
        data_dict = data_dict or check_dataset(data)  # check if None
    train_path, val_path = data_dict['train'], data_dict['val']  # 训练集和验证集路径
    nc = 1 if single_cls else int(data_dict['nc'])  # number of classes 如果是单类, 则类别数为1, 否则为数据集类别数
    names = {0: 'item'} if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    is_coco = isinstance(val_path, str) and val_path.endswith('coco/val2017.txt')  # COCO dataset 是否是COCO数据集

    """---------------------------------------------------- 模型 ----------------------------------------------------"""
    # Model
    check_suffix(weights, '.pt')  # check weights
    pretrained = weights.endswith('.pt')
    if pretrained:  # 如果是预训练模型
        with torch_distributed_zero_first(LOCAL_RANK):
            weights = attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
        model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(csd, strict=False)  # load
        LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report
    else:  # 如果不是预训练模型
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # Create model
    amp = check_amp(model)  # check AMP（自动混合精度）

    # Freeze 冻结权重
    # 这里只是给了冻结权重层的一个例子, 但是作者并不建议冻结权重层, 训练全部层参数, 可以得到更好的性能, 当然也会更慢
    freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # layers to freeze. 冻结层
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers  训练所有层
        # v.register_hook(lambda x: torch.nan_to_num(x))  NaN to 0 (commented for erratic training results)
        if any(x in k for x in freeze):
            LOGGER.info(f'freezing {k}')
            v.requires_grad = False

    # Image size
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

    """--------------------------------------------------- 优化器 ---------------------------------------------------"""
    # Optimizer
    # Batch size
    if RANK == -1 and batch_size == -1:  # single-GPU only, estimate best batch size
        batch_size = check_train_batch_size(model, imgsz, amp)
        loggers.on_params_update({'batch_size': batch_size})

    # nbs 标称的batch_size,模拟的batch_size 比如默认的话上面设置的opt.batch_size=16 -> nbs=64
    # 也就是模型梯度累计 64/16=4(accumulate) 次之后就更新一次模型 等于变相的扩大了batch_size
    nbs = 64  # nominal batch size  标称的batch_size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing  模拟的batch_size
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    optimizer = smart_optimizer(model, opt.optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])

    # Scheduler  学习率调度器
    if opt.cos_lr:
        # 使用one cycle 学习率  https://arxiv.org/pdf/1803.09820.pdf
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    else:
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    plot_lr_scheduler(optimizer, scheduler, epochs, save_dir=loggers.save_dir)  # plot lr schedule

    """------------------------------------------------- 训练前准备 -------------------------------------------------"""
    # EMA  指数移动平均
    # 单卡训练: 使用EMA（指数移动平均）对模型的参数做平均, 一种给予近期数据更高权重的平均方法, 以求提高测试指标并增加模型鲁棒。
    ema = ModelEMA(model) if RANK in {-1, 0} else None

    # Resume  恢复训练
    best_fitness, start_epoch = 0.0, 0
    if pretrained:
        if resume:
            best_fitness, start_epoch, epochs = smart_resume(ckpt, optimizer, ema, weights, epochs, resume)
        del ckpt, csd

    # DP mode 数据并行模式
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:  # check DP mode 如果是单卡训练, 则不需要DP模式
        LOGGER.warning(
            'WARNING ⚠️ DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n'
            'See Multi-GPU Tutorial at https://docs.ultralytics.com/yolov5/tutorials/multi_gpu_training to get started.'
        )
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm 同步BN
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info('Using SyncBatchNorm()')

    """-------------------------------------------------- 数据加载 --------------------------------------------------"""
    # 加载训练集dataloader、dataset + 参数(mlc、nb) + 加载验证集testloader + 如果不使用断点续训，设置labels相关参数(labels、c) ，
    # plots可视化数据集labels信息，检查anchors(k-means + 遗传进化算法)，model半精度
    # Trainloader
    train_loader, dataset = create_dataloader(train_path,
                                              imgsz,
                                              batch_size // WORLD_SIZE,
                                              gs,
                                              single_cls,
                                              hyp=hyp,
                                              augment=True,
                                              cache=None if opt.cache == 'val' else opt.cache,
                                              rect=opt.rect,
                                              rank=LOCAL_RANK,
                                              workers=workers,
                                              image_weights=opt.image_weights,
                                              quad=opt.quad,
                                              prefix=colorstr('train: '),
                                              shuffle=True,
                                              seed=opt.seed)
    labels = np.concatenate(dataset.labels, 0)
    mlc = int(labels[:, 0].max())  # max label class
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'

    # Process 0
    if RANK in {-1, 0}:
        val_loader = create_dataloader(val_path,
                                       imgsz,
                                       batch_size // WORLD_SIZE * 2,
                                       gs,
                                       single_cls,
                                       hyp=hyp,
                                       cache=None if noval else opt.cache,
                                       rect=True,
                                       rank=-1,
                                       workers=workers * 2,
                                       pad=0.5,
                                       prefix=colorstr('val: '))[0]

        if not resume:
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)  # run AutoAnchor
            model.half().float()  # pre-reduce anchor precision

        callbacks.run('on_pretrain_routine_end', labels, names)

    # DDP mode
    if cuda and RANK != -1:
        model = smart_DDP(model)

    """---------------------------------------------------- 训练 ----------------------------------------------------"""
    # 设置/初始化一些训练要用的参数(hyp[‘box’]、hyp[‘cls’]、hyp[‘obj’]、hyp[‘label_smoothing’]、model.nc、model.hyp、model.gr、
    # 从训练样本标签得到类别权重model.class_weights、model.names、热身迭代的次数iterationsnw、last_opt_step、初始化maps和results、
    # 学习率衰减所进行到的轮次scheduler.last_epoch +
    # 设置amp混合精度训练scaler +
    # 初始化损失函数compute_loss + 打印日志信息) +
    # 开始训练(注意五点：图片采样策略 + Warmup热身训练 + multi_scale多尺度训练 + amp混合精度训练 + accumulate 梯度更新策略) +
    # 打印训练相关信息(包括当前epoch、显存、损失(box、obj、cls、total)、当前batch的target的数量和图片的size等 +
    # Plot 前三次迭代的barch的标签框再图片中画出来并保存 + wandb ) +
    # validation(调整学习率、scheduler.step() 、emp val.run()得到results, maps相关信息、
    # 将测试结果results写入result.txt中、wandb_logger、Update best mAP 以加权mAP fitness为衡量标准、Save model)
    # Model attributes
    nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)
    hyp['box'] *= 3 / nl  # scale to layers
    hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names

    # Start training
    t0 = time.time()
    nb = len(train_loader)  # number of batches
    nw = max(round(hyp['warmup_epochs'] * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  limit warmup to < 1/2 of training
    last_opt_step = -1
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    stopper, stop = EarlyStopping(patience=opt.patience), False
    compute_loss = ComputeLoss(model)  # init loss class
    callbacks.run('on_train_start')
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting training for {epochs} epochs...')
    """================================================ Epoch begin ================================================"""
    for epoch in range(start_epoch, epochs):
        callbacks.run('on_train_epoch_start')
        model.train()

        # Update image weights (optional, single-GPU only)
        if opt.image_weights:
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx

        # Update mosaic border (optional)
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  height, width borders

        mloss = torch.zeros(3, device=device)  # mean losses
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)
        LOGGER.info(('\n' + '%11s' * 7) % ('Epoch', 'GPU_mem', 'box_loss', 'obj_loss', 'cls_loss', 'Instances', 'Size'))
        if RANK in {-1, 0}:
            pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)  # progress bar
        optimizer.zero_grad()
        """============================================== Batch begin =============================================="""
        for i, (imgs, targets, paths, _) in pbar:
            callbacks.run('on_train_batch_start')
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 0 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(int(imgsz * 0.5), int(imgsz * 1.5) + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            with torch.cuda.amp.autocast(amp):
                pred = model(imgs)  # forward
                loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                if RANK != -1:
                    loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                if opt.quad:
                    loss *= 4.

            # Backward
            scaler.scale(loss).backward()

            # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
            if ni - last_opt_step >= accumulate:
                scaler.unscale_(optimizer)  # unscale gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni

            # Log
            if RANK in {-1, 0}:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('%11s' * 2 + '%11.4g' * 5) %
                                     (f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
                callbacks.run('on_train_batch_end', model, ni, imgs, targets, paths, list(mloss))
                if callbacks.stop_training:
                    return
        """=============================================== Batch end ==============================================="""

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
        scheduler.step()

        if RANK in {-1, 0}:
            # mAP
            callbacks.run('on_train_epoch_end', epoch=epoch)
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            if not noval or final_epoch:  # Calculate mAP
                results, maps, _ = validate.run(data_dict,
                                                batch_size=batch_size // WORLD_SIZE * 2,
                                                imgsz=imgsz,
                                                half=amp,
                                                model=ema.ema,
                                                single_cls=single_cls,
                                                dataloader=val_loader,
                                                save_dir=save_dir,
                                                plots=False,
                                                callbacks=callbacks,
                                                compute_loss=compute_loss)

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            stop = stopper(epoch=epoch, fitness=fi)  # early stop check
            if fi > best_fitness:
                best_fitness = fi
            log_vals = list(mloss) + list(results) + lr
            callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)

            # Save model
            if (not nosave) or (final_epoch and not evolve):  # if save
                ckpt = {
                    'epoch': epoch,
                    'best_fitness': best_fitness,
                    'model': deepcopy(de_parallel(model)).half(),
                    'ema': deepcopy(ema.ema).half(),
                    'updates': ema.updates,
                    'optimizer': optimizer.state_dict(),
                    'opt': vars(opt),
                    'git': GIT_INFO,  # {remote, branch, commit} if a git repo
                    'date': datetime.now().isoformat()}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    torch.save(ckpt, w / f'epoch{epoch}.pt')
                del ckpt
                callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)

        # EarlyStopping
        if RANK != -1:  # if DDP training
            broadcast_list = [stop if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
            if RANK != 0:
                stop = broadcast_list[0]
        if stop:
            break  # must break all DDP ranks
    """================================================= Epoch end ================================================="""
    # end training -----------------------------------------------------------------------------------------------------
    if RANK in {-1, 0}:  # normal mode or DDP rank 0 mode
        LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is best:
                    LOGGER.info(f'\nValidating {f}...')
                    results, _, _ = validate.run(
                        data_dict,
                        batch_size=batch_size // WORLD_SIZE * 2,
                        imgsz=imgsz,
                        model=attempt_load(f, device).half(),
                        iou_thres=0.65 if is_coco else 0.60,  # best pycocotools at iou 0.65
                        single_cls=single_cls,
                        dataloader=val_loader,
                        save_dir=save_dir,
                        save_json=is_coco,
                        verbose=True,
                        plots=plots,
                        callbacks=callbacks,
                        compute_loss=compute_loss)  # val best model with plots
                    if is_coco:
                        callbacks.run('on_fit_epoch_end', list(mloss) + list(results) + lr, epoch, best_fitness, fi)

        callbacks.run('on_train_end', last, best, epoch, results)

    torch.cuda.empty_cache()
    return results


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    """-------------------------------------------------- 常用参数 --------------------------------------------------"""
    # weights: 权重文件路径
    parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s.pt', help='initial weights path')
    # cfg: 模型配置文件 包括nc、depth_multiple、width_multiple、anchors、backbone、head等
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    # data: 数据集配置文件 包括path、train、val、test、nc、names、download等
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    # hyp: 初始超参文件
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    # epochs: 训练轮次
    parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
    # batch-size: 训练批次大小
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
    # img-size: 输入网络的图片分辨率大小
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    # resume: 断点续训, 从上次打断的训练结果处接着训练  默认False
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    # nosave: 不保存模型  默认False(保存)      True: only test final epoch
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    # noval: 不验证
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    # noplots: 不保存训练过程中的图片
    parser.add_argument('--noplots', action='store_true', help='save no plot files')
    # device: 训练的设备
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # single-cls: 数据集是否只有一个类别 默认False
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    # workers: dataloader中的最大work数（线程个数）
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    """------------------------------------------------- 数据增强参数 -------------------------------------------------"""
    # rect: 是否使用矩形训练
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    # noautoanchor: 不自动调整anchor
    parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
    # evolve: 进化超参
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    # multi-scale: 是否使用多尺度训练
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    # label-smoothing: 标签平滑, 0.0为不使用, 默认0.0
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    # optimize: 是否使用优化器, 默认使用SGD(随机梯度下降)
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    # sync-bn: 是否使用同步BN, 只在DDP模式下可用
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    # cos-lr: 是否使用余弦学习率
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    # cache-images: 缓存图片
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='image --cache ram/disk')
    # image-weights: 使用图片权重
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    """-------------------------------------------------- 其他参数 --------------------------------------------------"""
    # bucket: gsutil bucket, i.e. gs://my-bucket/
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    # save dir: 存储路径, default: runs/train
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    # name: 项目名称, default: exp
    parser.add_argument('--name', default='exp', help='save to project/name')
    # exist-ok: 是否允许覆盖已存在的项目
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    # quad: 是否使用四路数据加载器
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    # save-period: 每隔多少个epoch保存一次模型, 默认-1(不保存)
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    # patience: 早停参数, 默认100
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    # freeze: 冻结层, 默认[0](不冻结)
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    # seed: 全局训练种子, 默认0
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    # local rank: 自动DDP多GPU参数, 不要修改, 默认-1
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')

    # Logger arguments
    parser.add_argument('--artifact_alias', type=str, default='latest', help='Version of dataset artifact to use')
    """--------------------------------------------- 三个W&B(wandb)参数 ---------------------------------------------"""
    # entity: W&B实体
    parser.add_argument('--entity', default=None, help='Entity')
    # upload_dataset: 上传数据集, 默认False
    parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='Upload data, "val" option')
    # bbox_interval: 设置bounding-box图片记录间隔, 默认-1(不记录)
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval')

    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt, callbacks=Callbacks()):
    """主函数
    :param opt: 参数
    :param callbacks: 回调函数
    :return:
    """
    """---------------------------------------------logging和wandb初始化---------------------------------------------"""
    # Checks
    if RANK in {-1, 0}:  # 如果是主进程
        print_args(vars(opt))  # 打印参数 utils/general.py
        check_git_status()  # 检查git状态 utils/general.py
        check_requirements()  # 检查requirements.txt是否满足 utils/general.py

    """---------------------------------------判断是否使用断点续训resume, 读取参数---------------------------------------"""
    # Resume (from specified or most recent last.pt)
    # 恢复训练（从指定的或最近的last.pt）
    if opt.resume and not check_comet_resume(opt) and not opt.evolve:  # opt.resume: 是否从断点处恢复训练
        # 如果opt.resume为True, 则从最近的last.pt恢复训练, 否则从指定的last.pt恢复训练
        last = Path(check_file(opt.resume) if isinstance(opt.resume, str) else get_latest_run())
        opt_yaml = last.parent.parent / 'opt.yaml'  # train options yaml
        opt_data = opt.data  # original dataset
        if opt_yaml.is_file():
            with open(opt_yaml, errors='ignore') as f:
                d = yaml.safe_load(f)
        else:
            d = torch.load(last, map_location='cpu')['opt']
        opt = argparse.Namespace(**d)  # replace
        opt.cfg, opt.weights, opt.resume = '', str(last), True  # reinstate
        if is_url(opt_data):
            opt.data = check_file(opt_data)  # avoid HUB resume auth timeout
    else:
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
            check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project)  # checks
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        if opt.evolve:
            if opt.project == str(ROOT / 'runs/train'):  # if default project name, rename to runs/evolve
                opt.project = str(ROOT / 'runs/evolve')
            opt.exist_ok, opt.resume = opt.resume, False  # pass resume to exist_ok and disable resume
        if opt.name == 'cfg':
            opt.name = Path(opt.cfg).stem  # use model.yaml as name
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    """------------------------------------------------DDP mode 设置------------------------------------------------"""
    # DDP（分布式数据并行）模式
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:  # 如果是DDP模式
        msg = 'is not compatible with YOLOv5 Multi-GPU DDP training'
        assert not opt.image_weights, f'--image-weights {msg}'
        assert not opt.evolve, f'--evolve {msg}'
        assert opt.batch_size != -1, f'AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size'
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)  # 设置当前GPU
        device = torch.device('cuda', LOCAL_RANK)  # 设置当前设备
        dist.init_process_group(backend='nccl' if dist.is_nccl_available() else 'gloo')  # 初始化进程组, 用于多进程通信

    """不进化算法，正常训练"""
    """遗传进化算法，边进化边训练"""
    # Train
    if not opt.evolve:
        train(opt.hyp, opt, device, callbacks)

    # Evolve hyperparameters (optional)
    # opt.evolve: 是否进行超参数进化
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        # 超参数进化元数据(变异规模0-1, 下限, 上限)
        meta = {  # 优化器超参数
            # initial learning rate (SGD=1E-2, Adam=1E-3) 初始学习率(SGD=1E-2, Adam=1E-3)
            'lr0': (1, 1e-5, 1e-1),
            # final OneCycleLR learning rate (lr0 * lrf) 最终的OneCycleLR学习率(lr0 * lrf)
            'lrf': (1, 0.01, 1.0),
            # SGD momentum/Adam beta1 SGD动量/Adam beta1
            'momentum': (0.3, 0.6, 0.98),
            # optimizer weight decay 优化器权重衰减
            'weight_decay': (1, 0.0, 0.001),
            # warmup epochs (fractions ok) 热身周期(分数ok)
            'warmup_epochs': (1, 0.0, 5.0),
            # warmup initial momentum 热身初始动量
            'warmup_momentum': (1, 0.0, 0.95),
            # warmup initial bias lr 热身初始偏置lr
            'warmup_bias_lr': (1, 0.0, 0.2),
            # box loss gain box损失增益
            'box': (1, 0.02, 0.2),
            # cls loss gain cls损失增益
            'cls': (1, 0.2, 4.0),
            # cls BCELoss positive_weight cls BCELoss正权重
            'cls_pw': (1, 0.5, 2.0),
            # obj loss gain (scale with pixels) obj损失增益(与像素缩放)
            'obj': (1, 0.2, 4.0),
            # obj BCELoss positive_weight obj BCELoss正权重
            'obj_pw': (1, 0.5, 2.0),
            # IoU training threshold IoU训练阈值
            'iou_t': (0, 0.1, 0.7),
            # anchor-multiple threshold 锚点多阈值
            'anchor_t': (1, 2.0, 8.0),
            # anchors per output grid (0 to ignore) 输出网格的锚点(0忽略)
            'anchors': (2, 2.0, 10.0),
            # focal loss gamma (efficientDet default gamma=1.5) 焦点损失伽马(efficientDet默认伽马=1.5)
            'fl_gamma': (0, 0.0, 2.0),
            # image HSV-Hue augmentation (fraction) 图像HSV-Hue增强(分数)
            'hsv_h': (1, 0.0, 0.1),
            # image HSV-Saturation augmentation (fraction) 图像HSV-Saturation增强(分数)
            'hsv_s': (1, 0.0, 0.9),
            # image HSV-Value augmentation (fraction) 图像HSV-Value增强(分数)
            'hsv_v': (1, 0.0, 0.9),
            # image rotation (+/- deg) 图像旋转(+/- deg)
            'degrees': (1, 0.0, 45.0),
            # image translation (+/- fraction) 图像平移(+/- fraction)
            'translate': (1, 0.0, 0.9),
            # image scale (+/- gain) 图像缩放(+/- gain)
            'scale': (1, 0.0, 0.9),
            # image shear (+/- deg) 图像剪切(+/- deg)
            'shear': (1, 0.0, 10.0),
            # image perspective (+/- fraction), range 0-0.001 图像透视(+/- fraction), 范围0-0.001
            'perspective': (0, 0.0, 0.001),
            # image flip up-down (probability) 图像上下翻转(概率)
            'flipud': (1, 0.0, 1.0),
            # image flip left-right (probability) 图像左右翻转(概率)
            'fliplr': (0, 0.0, 1.0),
            # image mixup (probability) 图像混合(概率)
            'mosaic': (1, 0.0, 1.0),
            # image mixup (probability) 图像混合(概率)
            'mixup': (1, 0.0, 1.0),
            # segment copy-paste (probability) 段复制粘贴(概率)
            'copy_paste': (1, 0.0, 1.0)}

        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict 加载超参数字典
            if 'anchors' not in hyp:  # anchors commented in hyp.yaml
                hyp['anchors'] = 3
        if opt.noautoanchor:  # 使用默认锚点
            del hyp['anchors'], meta['anchors']
        opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)  # only val/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  evolvable indices
        evolve_yaml, evolve_csv = save_dir / 'hyp_evolve.yaml', save_dir / 'evolve.csv'
        if opt.bucket:
            # download evolve.csv if exists
            subprocess.run([
                'gsutil',
                'cp',
                f'gs://{opt.bucket}/evolve.csv',
                str(evolve_csv), ])

        for _ in range(opt.evolve):  # generations to evolve 进化的代数, 默认300
            if evolve_csv.exists():
                # if evolve.csv exists: select best hyps and mutate 如果evolve.csv存在: 选择最好的hyps并进行变异
                # Select parent(s)
                # 从evolve.csv中选择最好的hyps, 作为父母进行变异
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt(evolve_csv, ndmin=2, delimiter=',', skiprows=1)  # 加载csv文件
                n = min(5, len(x))  # number of previous results to consider 要考虑的先前结果的数量
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations 顶部n个变异
                w = fitness(x) - fitness(x).min() + 1E-6  # weights (sum > 0) 权重(总和>0)
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  random selection 随机选择
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection 加权选择
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination 加权组合

                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma 变异概率, 标准差
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([meta[k][0] for k in hyp.keys()])  # gains 0-1 增益0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates) 变异直到发生变化(防止重复)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300) to view mutation distribution 查看变异分布
                    hyp[k] = float(x[i + 7] * v[i])  # mutate hyper-parameters 变异超参数

            # Constrain to limits 限制超参再规定范围
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit 下限
                hyp[k] = min(hyp[k], v[2])  # upper limit 上限
                hyp[k] = round(hyp[k], 5)  # significant digits 有效数字

            # Train mutation 训练变异
            results = train(hyp.copy(), opt, device, callbacks)
            callbacks = Callbacks()
            # Write mutation results 写入变异结果
            # 将结果写入results 并将对应的hyp写到evolve.txt evolve.txt中每一行为一次进化的结果
            # 每行前七个数字 (P, R, mAP, F1, test_losses(GIOU, obj, cls)) 之后为hyp
            keys = ('metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95', 'val/box_loss',
                    'val/obj_loss', 'val/cls_loss')
            print_mutation(keys, results, hyp.copy(), save_dir, opt.bucket)

        # Plot results
        plot_evolve(evolve_csv)
        LOGGER.info(f'Hyperparameter evolution finished {opt.evolve} generations\n'
                    f"Results saved to {colorstr('bold', save_dir)}\n"
                    f'Usage example: $ python train.py --hyp {evolve_yaml}')


def run(**kwargs):
    """Train a model with a given set of hyper-parameters 使用给定的一组超参数训练模型
    :param kwargs: key=value pairs are added to opt
    :return: opt
    """
    # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt


if __name__ == '__main__':
    opt = parse_opt()  # parse arguments
    main(opt)  # run
