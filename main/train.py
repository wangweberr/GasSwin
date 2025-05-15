import os
import torch
import time
import torch.nn as nn
import argparse
import logging
import torch.distributed as dist
from torch.utils.data import DataLoader
from . import GasDataLoader
from . import preprocess_sample
from . import SwinTransformer3D
from . import BCEDiceLoss,calculate_dice
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

   # 在文件顶部导入库后添加
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

results = {
        'mask_video': '/home/chenli/weber/Video-Swin-Transformer/simulated_gas/labels',
        'filename': '/home/chenli/weber/Video-Swin-Transformer/simulated_gas/gas',
        'model_path':'/home/chenli/weber/Video-Swin-Transformer/main/checkpoints/models',
        'pretrained':'/home/chenli/weber/Video-Swin-Transformer/pretrained/swin_tiny_patch244_window877_kinetics400_1k.pth',
        # 视频处理相关字段
        'video_readers': [],     # 视频读取器列表
        'total_frames': [],      # 各视频总帧数
        'frame_inds': [],        # 采样帧索引列表
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
        'clip_len': 32,         # 采样窗口大小
        'frame_interval': 1,    # 帧间隔
        'batch_size': 16,
        'test_size': 0.2,
        'val_size': 0.1,
        'num_workers': 4,
        'video_indices':[],
        'mask_location':[],
        'video_paths':[],
        'window_size':32,
        'stride':16,
        'epochs':20,
        'patience':10,
        'pretrained2d':False
    }

def train_epoch(model,train_loader,optimizer,criterion,device,epoch,is_distributed=False):
    model.train()
    train_loss=0.0
    train_dice=0.0
    if is_distributed:#设置 sampler 的 epoch，以保证每个 epoch 的 shuffle 不同
        train_loader.sampler.set_epoch(epoch)
    for data,label in train_loader:
        data=data.to(device)
        label=label.to(device)
        optimizer.zero_grad()
        output=model(data)#开始前向传播
        loss=criterion(output,label)#计算损失
        loss.backward()#计算梯度
        optimizer.step()#更新
        #同步loss和dice
        current_loss=loss.detach()
        current_dice=calculate_dice(output,label).detach()
        if is_distributed:
            dist.all_reduce(current_loss,op=dist.ReduceOp.SUM)
            dist.all_reduce(current_dice,op=dist.ReduceOp.SUM)#同步广播相加求和
            world_size=dist.get_world_size()
            train_loss+=current_loss.item()/world_size
            train_dice+=current_dice.item()/world_size
        else:
            train_loss+=current_loss.item()
            train_dice+=current_dice.item()
    return train_loss/len(train_loader),train_dice/len(train_loader)
#len(train_loader）代表有多少批次


def evaluate(model,val_loader,criterion,device,is_distributed=False):
    model.eval()
    total_loss=0
    total_dice=0
    with torch.no_grad():
        for data,label in val_loader:
            data=data.to(device)
            label=label.to(device)
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
    return total_loss/len(val_loader),total_dice/len(val_loader)
            


def main():
    parser=argparse.ArgumentParser()#创建解释器
    parser.add_argument('--local_rank',type=int,default=-1,help="DDP local rank")#创建解释器的命令
    args=parser.parse_args()#输入赋值给args
    # 添加日志记录
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    #设置DDP初始化
    is_distributed=False
    global results
    if args.local_rank !=-1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')#建立全局连接
        is_distributed = True
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        print(f"DDP: Initialized process {rank}/{world_size} on GPU {args.local_rank}")
    #设置device
    if is_distributed:
        device=torch.device(f"cuda:{args.local_rank}")
    else:
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #判断主进程
    is_master_process=(not is_distributed) or (rank==0)
    #保存目录
    data_to_broadcast = [None]
    if is_master_process:
        results=preprocess_sample(results)
        results.pop('video_readers',None)
        data_to_broadcast[0] = results
        os.makedirs(results['model_path'],exist_ok=True)
    if is_distributed:
        dist.broadcast_object_list(data_to_broadcast,src=0)
        results = data_to_broadcast[0]



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
                 pretrained=results['pretrained'],pretrained2d=results['pretrained2d']).to(device)
    #使用DDP包装模型
    if is_distributed:
        model=DDP(model,device_ids=[args.local_rank],output_device=args.local_rank)
    
    #显示总参数
    if is_master_process:
        total_params=sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params}")
    #定义优化器
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
    #定义学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=20,  # 一个完整余弦周期的 epoch 数
        eta_min=1e-6  # 最小学习率
    )
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
    
    for epoch in range(results['epochs']):
        print(f"\n轮次{epoch+1}/{results['epochs']}开始训练")
        #训练
        train_loss,train_dice=train_epoch(model,train_loader,optimizer,criterion,device,epoch,is_distributed)
        #评估
        val_loss,val_dice=evaluate(model,val_loader,criterion,device,is_distributed)

        #更新学习率，在每个epoch结束时更新而且不需要输入loss
        scheduler.step()
        #只有主进程执行下面操作
        if is_master_process:
        #记录训练历史
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_dice'].append(train_dice)
            history['val_dice'].append(val_dice)
            #打印训练信息
            print(f"轮次{epoch+1}/{results['epochs']}训练损失:{train_loss:.4f},训练dice:{train_dice:.4f},验证损失:{val_loss:.4f},验证dice:{val_dice:.4f}")
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
            else:
                patience_counter+=1
                print(f"轮次{epoch+1}验证损失没有下降，当前patience为{patience_counter}/{results['patience']}")
                if patience_counter>=results['patience']:
                    print(f"验证损失连续{results['patience']}轮没有下降，训练结束")
                    break
            #测试集评估
    test_loss,test_dice=evaluate(model,test_loader,criterion,device,is_distributed )
   # 训练时间超过一小时后不易读
    if is_master_process:
        total_time=time.time()-start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        print(f"训练结束，总训练时间: {hours}小时 {minutes}分钟 {seconds}秒")
        print(f"最佳验证损失为{best_val_loss:.4f},最佳验证dice为{best_val_dice:.4f},最佳训练轮次为{best_epoch+1}")
        print(f"验证集损失为{test_loss:.4f},验证集dice为{test_dice:.4f}")
    #DDP：主进程结束
    if is_distributed:
        dist.destroy_process_group()
if __name__=='__main__':
    main()



    


 