# custom_pipeline.py


import random
import numpy as np
import torch
import cv2
import decord
from decord import VideoReader, cpu



class SlidingWindowSampler:
    """动态滑窗采样，固定取 window_size 帧"""
    def __init__(self, window_size, stride, test_mode=False):
        self.window_size = window_size
        self.stride = stride
        self.test_mode = test_mode

    def __call__(self, results):
        if 'total_frames' not in results:
            results['total_frames'] = 451
        total = results['total_frames']
        starts = list(range(0, max(1, total - self.window_size + 1), self.stride))
        if not starts:
            starts = [0]
        if self.test_mode:
            start = starts[len(starts) // 2]
        else:
            start = random.choice(starts)
        inds = np.arange(start, start + self.window_size)
        # 保证不越界
        inds = np.clip(inds, 0, total - 1)
        results['frame_inds'] = inds
        results['clip_len'] = self.window_size
        results['frame_interval'] = 1
        results['num_clips'] = 1
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
        imgs = results['imgs']  # T×H×W
        T = imgs.shape[0]
        resized = [cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
                   for img in imgs]
        results['imgs'] = np.stack(resized, axis=0)
        if 'gt_seg_maps' in results:
            segs = results['gt_seg_maps']
            resizedm = [cv2.resize(m, (self.w, self.h),
                                   interpolation=cv2.INTER_NEAREST)
                        for m in segs]
            results['gt_seg_maps'] = np.stack(resizedm, axis=0)
        # 更新 img_shape
        results['img_shape'] = (self.h, self.w)
        return results


class PadTo:
    """按 size_divisor 在时空和空间维度上 pad"""
    def __init__(self, size_divisor):
        # size_divisor: (div_d, div_h, div_w)
        self.dd, self.dh, self.dw = size_divisor

    def __call__(self, results):
        imgs = results['imgs']  # T×H×W×C
        T, H, W = imgs.shape
        pd = (self.dd - T % self.dd) % self.dd
        ph = (self.dh - H % self.dh) % self.dh
        pw = (self.dw - W % self.dw) % self.dw
        imgs = np.pad(imgs,
                      ((0, pd), (0, ph), (0, pw), (0, 0)),
                      mode='constant', constant_values=0)
        results['imgs'] = imgs
        if 'gt_seg_maps' in results:
            segs = results['gt_seg_maps']  # T×H×W
            segs = np.pad(segs,
                          ((0, pd), (0, ph), (0, pw)),
                          mode='constant', constant_values=0)
            results['gt_seg_maps'] = segs
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
        imgs = results['imgs']
        T, H, W = imgs.shape
        top = random.randint(0, max(0, H - self.h))
        left = random.randint(0, max(0, W - self.w))
        results['imgs'] = imgs[:, top:top+self.h, left:left+self.w, :]
        if 'gt_seg_maps' in results:
            segs = results['gt_seg_maps']
            results['gt_seg_maps'] = segs[:, top:top+self.h, left:left+self.w]
        results['img_shape'] = (self.h, self.w)
        return results


class Flip:
    """随机水平翻转"""
    def __init__(self, flip_ratio=0.5):
        self.flip_ratio = flip_ratio

    def __call__(self, results):
        if random.random() < self.flip_ratio:
            imgs = results['imgs']
            results['imgs'] = imgs[:, :, ::-1, :]
            if 'gt_seg_maps' in results:
                segs = results['gt_seg_maps']
                results['gt_seg_maps'] = segs[:, :, ::-1]
            results['flip'] = True
        else:
            results['flip'] = False
        return results


class Normalize:
    """减均值除方差"""
    def __init__(self, mean=0, std=1, to_bgr=False):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_bgr = to_bgr

    def __call__(self, results):
        imgs = results['imgs'].astype(np.float32)  # T×H×W×C
        if self.to_bgr:
            imgs = imgs[..., ::-1]
        # 逐通道归一化
        for c in range(imgs.shape[-1]):
            imgs[..., c] = (imgs[..., c] - self.mean[c]) / self.std[c]
        results['imgs'] = imgs
        return results


class ToTensor:
    """把 numpy 转成 torch.Tensor"""
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        for k in self.keys:
            arr = results[k]
            if isinstance(arr, np.ndarray):
                # segs: T×H×W → T×H×W（long）
                # imgs: T×H×W×C → C×T×H×W
                if k == 'imgs':
                    # transpose to C×T×H×W
                    arr = arr.transpose(3, 0, 1, 2)
                results[k] = torch.from_numpy(arr)
        return results


class FormatShape:
    """补充 input_shape，保持 Output 统一"""
    def __init__(self, input_format='NCTHW'):
        assert input_format in ('NCTHW',)
        self.input_format = input_format

    def __call__(self, results):
        imgs = results['imgs']  # Tensor C×T×H×W
        # N=1 时直接 unsqueeze
        results['imgs'] = imgs.unsqueeze(0)  # 1×C×T×H×W
        results['input_shape'] = tuple(results['imgs'].shape)#touple显式转换
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
        
     # 直接用文件路径解码，无需 FileClient/io.BytesIO
        from decord import VideoReader, cpu
        container = VideoReader(results['filename'],
                                ctx=cpu(0),
                                num_threads=self.num_threads)

        results['video_reader'] = container
        results['total_frames'] = len(container)
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
        container = results['video_reader']

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        frame_inds = results['frame_inds']
        # Generate frame index mapping in order
        imgs = container.get_batch(frame_inds).asnumpy()

        results['video_reader'] = None
        del container

        results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]

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
        mask_path = results[self.mask_key]
        inds = results[self.frame_inds_key]

        # 2. 用 Decord 一次性批量读取指定帧
        vr = decord.VideoReader(mask_path, ctx=cpu(0))
        frames = vr.get_batch(inds).asnumpy()  # shape: (T, H, W, C) 或 (T, H, W)

        # 3. 转灰度（若是三通道）并二值化（>0 视作前景）
        if  frames.ndim == 4 and frames.shape[-1] == 1:
            gray = frames[..., 0]
        else:
            gray = frames  # 本身就 (T,H,W)

        bin_mask = (gray > 0).astype(np.uint8)  # (T, H, W)，值为 0 或 1

        # 4. 写回 results，供后续 Resize/Pad/... 使用
        results[self.bin_mask] = bin_mask
        return results