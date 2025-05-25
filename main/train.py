import os
import torch
import time
import torch.nn as nn
import logging
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import DataLoader
from . import GasDataLoader
from . import preprocess_sample
from . import SwinTransformer3D
from . import BCEDiceLoss,calculate_dice
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
import torch.amp as amp
   # 在文件顶部导入库后添加
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

results = {
        'mask_video': '/home/chenli/weber/GasSwin/simulated_gas/labels',
        'filename': '/home/chenli/weber/GasSwin/simulated_gas/gas',
        'model_path':'/home/chenli/weber/GasSwin/main/checkpoints/models',#模型保存路径
        'pretrained':'/home/chenli/weber/GasSwin/pretrained/swin_tiny_patch244_window877_kinetics400_1k.pth',
        #'pretrained':'/home/chenli/weber/GasSwin/main/checkpoints/models/best_model.pth',
        # 视频处理相关字段
        'video_readers': [],     # 视频读取器列表
        'total_frames': [],      # 各视频总帧数
        'video_paths':[],
        'frame_inds': [],        # 采样帧索引列表
        'frame_size': 10,        # 采样窗口大小
        'frame_stride': 5,      # 帧间隔
        # 图像数据存储
        'imgs': [],              # 原始视频帧列表
        'original_shape': [],    # 原始视频尺寸
        # 掩码处理相关
        'mask_paths': [],        # 掩码视频路径列表
        'gt_seg_maps': [],       # 二值化掩码数据
        # 预处理状态
        'img_shape': [],         # 当前处理尺寸
        'flip': [],              # 翻转状态
        # 数据加载参数
        'clip_len': 16,         # 采样窗口大小
        'frame_interval': 1,    # 帧间隔
        'batch_size': 2,
        'test_size': 0.2,
        'val_size': 0.1,
        'num_workers': 4,
        'video_indices':[],
        'mask_location':[],
        'video_paths':[],
        'patch_size':(2,4,4),
        'window_size':(2,7,7),
        'frozen_stages':3,
        'frozen_epochs':10,
        'head_lr': 5e-4,
        'backbone_lr': 1e-4,
        'epochs':50,
        'patience':10,
        'pretrained2d':False,
        'gradient_accumulation_steps':4
    }

def train_epoch(model,train_loader,optimizer,criterion,device,epoch,is_distributed=False):
    model.train()
    train_loss=0.0
    train_dice=0.0
    if is_distributed:#设置 sampler 的 epoch，以保证每个 epoch 的 shuffle 不同
        train_loader.sampler.set_epoch(epoch)
    scaler=amp.GradScaler()
        # 只在主进程显示进度条
    should_print = not is_distributed or dist.get_rank() == 0
    # 创建进度条
    progress_bar = tqdm(
        total=len(train_loader), 
        desc=f"训练轮次 {epoch+1}", 
        unit="batch",
        disable=not should_print,
        ncols=100,
        leave=False
    )
    accumulation_step=results['gradient_accumulation_steps']
    current_step=0
    for data,label in train_loader:
        current_step+=1
        data=data.to(device)
        label=label.to(device)
        if current_step%accumulation_step==1:
            optimizer.zero_grad()

        with amp.autocast(device_type="cuda"):
             output=model(data)#开始前向传播
             loss=criterion(output,label)/accumulation_step#因为分母不同所以计算损失同时补偿，正常分母应为总batch
        #loss.backward()#计算梯度
        #optimizer.step()#更新
        # 使用scaler处理反向传播和优化器步骤
        scaler.scale(loss).backward()
        if current_step % accumulation_step==0 or current_step==len(train_loader):
            scaler.step(optimizer)
            scaler.update()

        #同步loss和dice
        current_loss=loss.detach()*accumulation_step
        with torch.no_grad():
            current_dice=calculate_dice(output,label).detach()#避免混合精度影响计算dice
        if is_distributed:
            dist.all_reduce(current_loss,op=dist.ReduceOp.SUM)
            dist.all_reduce(current_dice,op=dist.ReduceOp.SUM)#同步广播相加求和
            world_size=dist.get_world_size()
            train_loss+=current_loss.item()/world_size
            train_dice+=current_dice.item()/world_size
        else:
            train_loss+=current_loss.item()
            train_dice+=current_dice.item()
        # 更新进度条信息
        if should_print:
            progress_bar.set_postfix({
                "loss": f"{current_loss.item():.4f}",
                "dice": f"{current_dice.item():.4f}"
            })
            progress_bar.update()
    
    progress_bar.close()
    return train_loss/len(train_loader),train_dice/len(train_loader)
#len(train_loader）代表有多少批次


def evaluate(model,val_loader,criterion,device,is_distributed=False):
    model.eval()
    total_loss=0
    total_dice=0


    # 只在主进程显示进度条
    should_print = not is_distributed or dist.get_rank() == 0
    # 创建进度条
    progress_bar = tqdm(
        total=len(val_loader), 
        desc="验证", 
        unit="batch",
        disable=not should_print,
        ncols=100,
        leave=False
    )
   
    with torch.no_grad():
        for data,label in val_loader:
            data=data.to(device)
            label=label.to(device)
            with amp.autocast(device_type="cuda"):
                output=model(data)
                loss=criterion(output,label)
            current_loss=loss.detach()
            current_dice=calculate_dice(output,label).detach()
            if is_distributed:
                dist.all_reduce(current_loss,op=dist.ReduceOp.SUM)
                dist.all_reduce(current_dice,op=dist.ReduceOp.SUM)
                world_size=dist.get_world_size()
                total_loss+=current_loss.item()/world_size
                total_dice+=current_dice.item()/world_size
            else:
                total_loss+=current_loss.item()
                total_dice+=current_dice.item()

            # 更新进度条信息
            if should_print:
                progress_bar.set_postfix({
                    "loss": f"{current_loss.item():.4f}",
                    "dice": f"{current_dice.item():.4f}"
                })
                progress_bar.update()
    
    progress_bar.close()
    return total_loss/len(val_loader),total_dice/len(val_loader)
            


def main():
    # torchrun
    env_local_rank = os.environ.get("LOCAL_RANK")
    # 添加日志记录
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    #设置DDP初始化
    is_distributed=False
    rank = 0  # 单进程运行时，rank 为 0 
    world_size = 1
    global results
    if env_local_rank is not None:
        env_local_rank=int(env_local_rank)
        torch.cuda.set_device(env_local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')#建立全局连接
        is_distributed = True
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        print(f"DDP: Initialized process {rank}/{world_size} on GPU {env_local_rank}")
    #设置device
    if is_distributed:
        device=torch.device(f"cuda:{env_local_rank}")
    else:
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #判断主进程
    is_master_process=(not is_distributed) or (rank==0)
    #广播文件
    data_to_broadcast = [None]
    #数据处理及缓存部分
    if is_master_process:
        cache_file = os.path.join(results['model_path'], 'preprocessed_data.pt')
        if os.path.exists(cache_file):
            print(f">>> 加载预处理缓存: {cache_file}")
            data_cache = torch.load(cache_file)
            results['imgs']        = [img for img in data_cache['imgs']]
            results['gt_seg_maps'] = [m for m in data_cache['masks']]
            results['img_shape']      = data_cache['img_shape']
            results['original_shape'] = data_cache['original_shape']
        else:
            print(">>> 缓存不存在，执行数据预处理并保存到缓存")
            results = preprocess_sample(results)
            # 去掉 VideoReader 以便序列化
            results.pop('video_readers', None)
            os.makedirs(results['model_path'], exist_ok=True)
            torch.save({
                'imgs':           results['imgs'],
                'masks':          results['gt_seg_maps'],
                'img_shape':      results['img_shape'],
                'original_shape': results['original_shape'],
            }, cache_file)
            print(f">>> 已缓存预处理数据到: {cache_file}")
        data_to_broadcast[0] = results
    if is_distributed:
        dist.broadcast_object_list(data_to_broadcast,src=0)
        results = data_to_broadcast[0]
        dist.barrier()
    #缓存部分
    if is_master_process:
        print(f"数据形状: {results['imgs'][0].shape}")
        print(f"标签形状: {results['gt_seg_maps'][0].shape}")
        print(f"数据维度数: {results['imgs'][0].dim()}")
    #加载数据
    train_dataset,val_dataset,test_dataset=GasDataLoader(results)()
    #创建DistributedSampler:多分布核心就是sampler使用distributedsampler
    if is_distributed:
        train_sampler=DistributedSampler(train_dataset,shuffle=True)
        val_sampler=DistributedSampler(val_dataset,shuffle=False)
        test_sampler=DistributedSampler(test_dataset,shuffle=False)
    else:
        train_sampler=None
        val_sampler=None
        test_sampler=None
    #创建DataLoader，验证时不打乱，训练时由sampler控制，也设定为False
    train_loader=DataLoader(train_dataset,batch_size=results['batch_size'],sampler=train_sampler,
    shuffle=(train_sampler is None),num_workers=results['num_workers'],pin_memory=True)
    val_loader=DataLoader(val_dataset,batch_size=results['batch_size'],sampler=val_sampler,
    shuffle=(False),num_workers=results['num_workers'],pin_memory=True)
    test_loader=DataLoader(test_dataset,batch_size=results['batch_size'],sampler=test_sampler,
    shuffle=(False),num_workers=results['num_workers'],pin_memory=True)

    
    #定义模型
    model=SwinTransformer3D(
                 pretrained=results['pretrained'],pretrained2d=results['pretrained2d'],
                 frozen_stages=results['frozen_stages'],use_checkpoint=False,
                 patch_size=results['patch_size'],window_size=results['window_size'],patch_norm=True).to(device)
    
    #使用DDP包装模型
    if is_distributed:
        # 将模型中所有BatchNorm转换为SyncBatchNorm
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model=DDP(model,device_ids=[env_local_rank],output_device=env_local_rank,find_unused_parameters=True)
    #为不同参数构建不同学习率
    model_c = model.module if is_distributed else model
    head_params, backbone_params = [], []
    for n, p in model_c.named_parameters():
        if not p.requires_grad:            # 已被冻结的直接跳过
            continue
        if 'head' in n:                    # 'head.' 来自 FPNHead
            head_params.append(p)
        else:
            backbone_params.append(p)
    param_groups = []
    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr': results['backbone_lr']})
    if head_params:
        param_groups.append({'params': head_params,     'lr': results['head_lr']})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-2)

    #显示总参数
    if is_master_process:
        total_params=sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params}")



    #定义学习率调度器
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=5)
    cosine  = CosineAnnealingLR(optimizer, T_max=results['frozen_epochs']-5, eta_min=1e-4)
    scheduler = SequentialLR(optimizer, [warmup, cosine], [5])
    #损失函数
    criterion = BCEDiceLoss(weight_bce=0.5, weight_dice=0.5) 
    #训练历史记录
    history={
        'train_loss':[],
        'val_loss':[],
        'train_dice':[],
        'val_dice':[]
    }
    #早停设置
    best_val_loss=float('inf')#初始化最佳验证损失
    best_epoch=0              #记录最佳模型对应轮次
    patience_counter=0        #计数器，记录连续未改进的轮次
    #开始训练时间
    start_time=time.time()
    
    # ===== 日志文件 =====
    # 仅主进程负责写日志文件，避免多进程冲突
    log_file_path = None
    if (not is_distributed) or (dist.get_rank() == 0):
        os.makedirs(results['model_path'], exist_ok=True)
        log_file_path = os.path.join(results['model_path'], 'train.log')
        file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)
        logging.info("========== 开始新的训练任务 ==========")
    
    for epoch in range(results['epochs']):
        
        if epoch==results['frozen_epochs']:
            if is_master_process:
             print("开始解冻")
            model_c.frozen_stages=-1
            for m in model_c.modules():
                m.train()
            for p in model_c.parameters():
                p.requires_grad=True
            head_params,backbone_params= [], []
            for n,p in model_c.named_parameters():
                if 'head' in n:
                    head_params.append(p)
                else:
                    backbone_params.append(p)
            results['backbone_lr']=results['head_lr']*0.1
            optimizer = torch.optim.AdamW(
                [
                    {'params': backbone_params, 'lr': results['backbone_lr']},
                    {'params': head_params,     'lr': results['head_lr']}
                ],
                weight_decay=1e-2)
            warm2 = LinearLR(optimizer, start_factor=0.2, total_iters=3)  # 让 LR 先爬上来
            cos2  = CosineAnnealingWarmRestarts(optimizer,T_0=8,T_mult=2,eta_min=2e-5)
            scheduler = SequentialLR(optimizer,[warm2,cos2],[3])
            if is_distributed:
                dist.barrier()
        if is_master_process:
            print(f"\n轮次{epoch+1}/{results['epochs']}开始训练")
            logging.info(f"轮次{epoch+1}/{results['epochs']}开始训练")
        #训练
        train_loss,train_dice=train_epoch(model,train_loader,optimizer,criterion,device,epoch,is_distributed)
        #评估
        val_loss,val_dice=evaluate(model,val_loader,criterion,device,is_distributed)

        #更新学习率，在每个epoch结束时更新而且不需要输入loss
        scheduler.step()
        stop_signal=torch.tensor(False,device=device)
        #只有主进程执行下面操作
        if is_master_process:
            for i, param_group in enumerate(optimizer.param_groups):
                print(f"Epoch {epoch}  group{i} lr={param_group['lr']:.6e}")
        #记录训练历史
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_dice'].append(train_dice)
            history['val_dice'].append(val_dice)
            #打印训练信息
            print(f"轮次{epoch+1}/{results['epochs']}训练损失:{train_loss:.4f},训练dice:{train_dice:.4f},验证损失:{val_loss:.4f},验证dice:{val_dice:.4f}")
            logging.info(f"轮次{epoch+1}/{results['epochs']} 训练loss={train_loss:.4f}, dice={train_dice:.4f}; 验证loss={val_loss:.4f}, dice={val_dice:.4f}")
            #保存模型
            if val_loss<best_val_loss:
                best_val_loss=val_loss
                best_val_dice=val_dice
                best_epoch=epoch
                patience_counter=0
                current_model_path=os.path.join(results['model_path'],f'best_model_{epoch+1}.pth')
                best_model_path=os.path.join(results['model_path'],'best_model.pth')
                #DDP：此时主模型已经被包装成DDP，所以需要获取.module
                model_to_save=model.module if is_distributed else model
                torch.save(model_to_save.state_dict(),current_model_path)
                torch.save(model_to_save.state_dict(),best_model_path)
                print(f"轮次{epoch+1}模型已经保存，验证损失为{val_loss:.4f}")
                logging.info(f"轮次{epoch+1} 新最佳模型已保存  val_loss={val_loss:.4f}")
            else:
                patience_counter+=1
                print(f"轮次{epoch+1}验证损失没有下降，当前patience为{patience_counter}/{results['patience']}")
                logging.info(f"轮次{epoch+1} 验证损失未改进  patience={patience_counter}/{results['patience']}")
                if patience_counter>=results['patience']:
                    print(f"验证损失连续{results['patience']}轮没有下降，训练结束")
                    logging.info(f"早停: 验证损失连续{results['patience']}轮没有下降")
                    stop_signal=torch.tensor(True,device=device)
        if is_distributed:
            dist.broadcast(stop_signal,src=0)
            if stop_signal.item()==True:
                break
            dist.barrier()
            #测试集评估
    test_loss,test_dice=evaluate(model,test_loader,criterion,device,is_distributed )
   # 训练时间
    if is_master_process:
        total_time=time.time()-start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        print(f"训练结束，总训练时间: {hours}小时 {minutes}分钟 {seconds}秒")
        logging.info(f"训练结束，总训练时间: {hours}h {minutes}m {seconds}s")
        print(f"最佳验证损失为{best_val_loss:.4f},最佳验证dice为{best_val_dice:.4f},最佳训练轮次为{best_epoch+1}")
        logging.info(f"最佳验证损失 {best_val_loss:.4f}, dice {best_val_dice:.4f}, 轮次 {best_epoch+1}")
        print(f"验证集损失为{test_loss:.4f},验证集dice为{test_dice:.4f}")
        logging.info(f"测试集: loss {test_loss:.4f}, dice {test_dice:.4f}")
    #DDP：主进程结束
    if is_distributed:
        dist.destroy_process_group()
if __name__=='__main__':
    main()



    


 