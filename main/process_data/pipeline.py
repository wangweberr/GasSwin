# custom_pipeline.py

import os
import random
import numpy as np
import torch
import cv2
import decord
from decord import VideoReader, cpu
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split



class SlidingWindowSampler:
    """动态滑窗采样，固定取 window_size 帧"""
    def __init__(self, window_size, stride, test_mode=False):
        self.window_size = window_size
        self.stride = stride
        self.test_mode = test_mode
    def __call__(self, results):
        if 'total_frames' not in results:
            results['total_frames'] = 451
        inds=[]
        Video_incdice=[]
        for i,total in enumerate(results['total_frames']):
            video_inds = []
            starts = list(range(0, max(1, total - self.window_size + 1), self.stride))
            if not starts:
                starts = [0]
            if self.test_mode:
                # 测试模式取所有窗口
                video_inds = [np.arange(s, s+self.window_size) for s in starts]
            else:
                # 训练模式随机选择多个窗口
                max_samples = max(1, (total - self.window_size) // self.stride + 1)
                selected_starts = random.choices(starts, k=max_samples)
                video_inds = [np.arange(s, s+self.window_size) for s in selected_starts]
            video_inds = [np.clip(ind, 0, total - 1) for ind in video_inds]
            inds.extend(video_inds) 
            for s in video_inds:
                results['video_indices'].append(i)
           
        # 保证不越界
        results['frame_inds'] = inds
        results['clip_len'] = self.window_size
        results['frame_interval'] = 1
        return results




    """class LoadAnnotations:
    按索引读取分割 mask
    def __init__(self, seg_prefix, filename_tmpl='{:05d}.png'):
        self.seg_prefix = seg_prefix
        self.filename_tmpl = filename_tmpl

    def __call__(self, results):
        inds = results['frame_inds']
        segs = []
        for idx in inds:
            path = os.path.join(self.seg_prefix, self.filename_tmpl.format(int(idx)))
            m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if m is None:
                raise FileNotFoundError(f'找不到 mask：{path}')
            segs.append(m)
        # T×H×W
        results['gt_seg_maps'] = np.stack(segs, axis=0)
        return results"""


class Resize:
    """统一 resize 到指定 (h, w)"""
    def __init__(self, scale):
        # scale: (h, w)
        self.h, self.w = scale

    def __call__(self, results):
        processed_imgs = []
        processed_segs = []
        
        for vid_imgs in results['imgs']:
            T = vid_imgs.shape[0]
            resized = [cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_LINEAR) 
                      for img in vid_imgs]
            processed_imgs.append(np.stack(resized, axis=0))
            
            if 'gt_seg_maps' in results:
                vid_segs = [cv2.resize(m, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
                           for m in results['gt_seg_maps'][results['imgs'].index(vid_imgs)]]
                processed_segs.append(np.stack(vid_segs, axis=0))
        
        results['imgs'] = processed_imgs
        if processed_segs:
            results['gt_seg_maps'] = processed_segs
        results['img_shape'] = [(self.h, self.w)] * len(processed_imgs)
        return results


class PadTo:
    """按 size_divisor 在时空和空间维度上 pad"""
    def __init__(self, size_divisor):
        # size_divisor: (div_d, div_h, div_w)
        self.dd, self.dh, self.dw = size_divisor

    def __call__(self, results):
        padded_imgs = []
        padded_segs = []
        
        for vid_imgs in results['imgs']:
            T, H, W = vid_imgs.shape[:3]
            pd = (self.dd - T % self.dd) % self.dd
            ph = (self.dh - H % self.dh) % self.dh
            pw = (self.dw - W % self.dw) % self.dw
            
            vid_padded = np.pad(vid_imgs,
                ((0, pd), (0, ph), (0, pw), (0, 0)),
                mode='constant', constant_values=0)
            padded_imgs.append(vid_padded)
            
            if 'gt_seg_maps' in results:
                vid_segs = results['gt_seg_maps'][results['imgs'].index(vid_imgs)]
                seg_padded = np.pad(vid_segs,
                    ((0, pd), (0, ph), (0, pw)),
                    mode='constant', constant_values=0)
                padded_segs.append(seg_padded)
        
        results['imgs'] = padded_imgs
        if padded_segs:
            results['gt_seg_maps'] = padded_segs
        return results


class RandomCrop:
    """随机裁剪到指定大小"""
    def __init__(self, size,ratio=0.5):
        # size: int 或 (h, w)
        self.h, self.w = (size, size) if isinstance(size, int) else size
        self.ratio=ratio
    def __call__(self, results):
        if random.random()<self.ratio:
            return results
        
        cropped_imgs = []
        cropped_segs = []
        
        for vid_imgs in results['imgs']:
            T, H, W = vid_imgs.shape[:3]
            top = random.randint(0, max(0, H - self.h))
            left = random.randint(0, max(0, W - self.w))
            cropped_imgs.append(vid_imgs[:, top:top+self.h, left:left+self.w, :])
            
            if 'gt_seg_maps' in results:
                vid_segs = results['gt_seg_maps'][results['imgs'].index(vid_imgs)]
                cropped_segs.append(vid_segs[:, top:top+self.h, left:left+self.w])
        
        results['imgs'] = cropped_imgs
        if cropped_segs:
            results['gt_seg_maps'] = cropped_segs
        results['img_shape'] = [(self.h, self.w)] * len(cropped_imgs)
        return results


class Flip:
    """随机水平翻转"""
    def __init__(self, flip_ratio=0.5):
        self.flip_ratio = flip_ratio

    def __call__(self, results):
        flip_flags = []
        
        for i in range(len(results['imgs'])):
            if random.random() < self.flip_ratio:
                # 水平翻转当前视频
                results['imgs'][i] = results['imgs'][i][:, :, ::-1, :]
                if 'gt_seg_maps' in results:
                    results['gt_seg_maps'][i] = results['gt_seg_maps'][i][:, :, ::-1]
                flip_flags.append(True)
            else:
                flip_flags.append(False)
        
        results['flip'] = flip_flags
        return results


class Normalize:
    """减均值除方差"""
    def __init__(self, mean=0, std=1, to_bgr=False):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_bgr = to_bgr

    def __call__(self, results):
        for i,vid_imgs in enumerate(results['imgs']):
            vid_imgs = vid_imgs.astype(np.float32)
            if self.to_bgr:
                vid_imgs = vid_imgs[..., ::-1]
            for c in range(vid_imgs.shape[-1]):#对每个通道进行归一化，我这里都默认是(0,1)
                vid_imgs[..., c] = (vid_imgs[..., c] - self.mean[c]) / self.std[c]
            results['imgs'][i]=vid_imgs
        return results


class ToTensor:
    """把 numpy 转成 torch.Tensor"""
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        for k in self.keys:
            if isinstance(results[k], list):
                # 处理多视频列表
                tensor_list = []
                for vid_arr in results[k]:
                    if isinstance(vid_arr, np.ndarray):
                        if k == 'imgs':
                            vid_arr = vid_arr.transpose(3, 0, 1, 2)
                        tensor_list.append(torch.from_numpy(vid_arr))
                results[k] = tensor_list
        return results


class FormatShape:
    """补充 input_shape，保持 Output 统一"""
    def __init__(self, input_format='NCTHW'):
        assert input_format in ('NCTHW',)
        self.input_format = input_format

    def __call__(self, results):
        imgs_list = results['imgs']
        processed_imgs = []
        input_shapes = []
        masks_list = results['gt_seg_maps']
        processed_masks = []
        
        for imgs in imgs_list:
            # 为每个视频添加批次维度
            processed = imgs.unsqueeze(0)  # 1×C×T×H×W
            processed_imgs.append(processed)
            input_shapes.append(tuple(processed.shape))
        
        for mask in masks_list:
                # mask 形状为 [T,H,W]，添加通道维度和批次维度
            processed = mask.unsqueeze(0).unsqueeze(0)  # [1,1,T,H,W]
            processed_masks.append(processed)
            
        results['gt_seg_maps'] = processed_masks
        results['imgs'] = processed_imgs
        results['input_shape'] = input_shapes
        return results





class DecordInit:
    """Using decord to initialize the video_reader.

    Decord: https://github.com/dmlc/decord

    Required keys are "filename",
    added or modified keys are "video_reader" and "total_frames".
    """
    def __init__(self, num_threads=1):
        self.num_threads = num_threads
    def __call__(self, results):
        """Perform the Decord initialization.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        try:
            import decord
        except ImportError:
            raise ImportError(
                'Please run "pip install decord" to install Decord first.')
        all_file=os.listdir(results['filename'])
        mp4_file=[f for f in all_file if f.endswith('.mp4')]
        full_path=[]
     # 直接用文件路径解码，无需 FileClient/io.BytesIO
        for file_name in mp4_file:
            full_path.append(os.path.join(results['filename'],file_name))
        results['video_paths']=full_path
        for file in full_path: 
            container = VideoReader(file,
                                ctx=cpu(0),
                                num_threads=self.num_threads)
            results['video_readers'].append(container)
            results['total_frames'].append(len(container))
        return results

    
    



class DecordDecode:
    """Using decord to decode the video.

    Decord: https://github.com/dmlc/decord

    Required keys are "video_reader", "filename" and "frame_inds",
    added or modified keys are "imgs" and "original_shape".
    """

    def __call__(self, results):
        """Perform the Decord decoding.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        
        for ind,index in zip(results['frame_inds'],results['video_indices']):
            container = results['video_readers'][index] 
            # ind: T or ndarr
            if ind.ndim != 1:
               ind = np.squeeze(ind)
             # Generate frame index mapping in order
            imgs = container.get_batch(ind).asnumpy()
            results['imgs'].append(imgs)
            results['original_shape'].append(imgs[0].shape[:2])
            results['img_shape'].append(imgs[0].shape[:2])

        return results

    

class LoadMaskFromVideo:
    """按索引读取灰度分割后的视频帧并二值化为 0/1 掩码"""
    def __init__(self,
                 mask_key='mask_video',
                 frame_inds_key='frame_inds',
                 bin_mask='gt_seg_maps'):
        self.mask_key = mask_key
        self.frame_inds_key = frame_inds_key
        self.bin_mask = bin_mask

    def __call__(self, results):
        # 1. 获取 mask 视频路径 和 需要的帧索引
        all_mask=os.listdir(results[self.mask_key])
        mask_file=[f for f in all_mask if f.endswith('.mp4')]
        inds = results[self.frame_inds_key]
        for file in mask_file:
            results['mask_location'].append(os.path.join(results[self.mask_key],file))
        for index,ind in zip(results['video_indices'],inds):
            # 2. 用 Decord 一次性批量读取指定帧
            vr = decord.VideoReader(results['mask_location'][index], ctx=cpu(0))
            frames = vr.get_batch(ind).asnumpy()  # shape: (T, H, W, C) 或 (T, H, W)

            # 3. 转灰度（若是三通道）并二值化（>0 视作前景）
            if  frames.ndim == 4 and frames.shape[-1] == 1:
                gray = frames[..., 0]#通道剔除
            else:
                gray = frames  # 本身就 (T,H,W)

            bin_mask = (gray > 0).astype(np.uint8)  # (T, H, W)，值为 0 或 1

            # 4. 写回 results，供后续 Resize/Pad/... 使用
            results[self.bin_mask].append(bin_mask)
        return results


class Gasdata(Dataset):
    def __init__(self,data,label):
        self.data=data
        self.label=label
    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        return self.data[index],self.label[index]

class GasDataLoader():
    def __init__(self,results):
        self.num_workers=results['num_workers']
        self.imgs=results['imgs']
        self.labels=results['gt_seg_maps']
        self.batch_size=results['batch_size']
        self.test_size=results['test_size']
        self.val_size=results['val_size']
    def __call__(self):
        data_temp, data_test, label_temp, label_test = train_test_split(
        self.imgs, self.labels, test_size=self.test_size, random_state=42)
        #分割训练验证集
        data_train, data_val, label_train, label_val = train_test_split(
        data_temp, label_temp, test_size=self.val_size, random_state=42)
        #创建数据集
        train_dataset=Gasdata(data_train,label_train)
        val_dataset=Gasdata(data_val,label_val)
        test_dataset=Gasdata(data_test,label_test)
        #创建加载器,加载器就是用来控制输入数量的
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers, 
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers, 
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers, 
            pin_memory=True
        )
        return train_loader,val_loader,test_loader
        

        

      