"""前处理。两平台共用，保证输入完全一致。

返回 (blob, meta)：
- blob: np.ndarray，NCHW，float32，已归一化，C-contiguous（后端可直接喂）
- meta: dict，保存还原坐标所需信息（ratio / pad / orig_shape），供后处理使用

后续第 5 周的预处理硬件加速（TX2 NX 用 CUDA Kernel、RK3588 用 RGA）只需替换
__call__ 内部实现，接口保持不变。
"""
from typing import Tuple

import cv2
import numpy as np


def letterbox(
    img: np.ndarray,
    new_shape: Tuple[int, int],
    color: Tuple[int, int, int] = (114, 114, 114),
):
    """等比缩放 + 居中填充。new_shape = (H, W)。返回 (padded, ratio, (pad_w, pad_h))。"""
    h, w = img.shape[:2]
    nh, nw = new_shape
    r = min(nh / h, nw / w)
    rw, rh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_LINEAR)
    pad_w, pad_h = nw - rw, nh - rh
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return padded, r, (left, top)


class YoloPreprocessor:
    """YOLOv8 前处理：letterbox -> BGR2RGB -> /255 -> NCHW。"""

    def __init__(self, to_rgb: bool = True):
        self.to_rgb = to_rgb

    def __call__(self, img_bgr: np.ndarray, input_size: Tuple[int, int]):
        padded, r, (pad_w, pad_h) = letterbox(img_bgr, input_size)
        if self.to_rgb:
            padded = padded[:, :, ::-1]
        blob = padded.astype(np.float32) / 255.0
        blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[None])  # (1,3,H,W)
        meta = {
            "ratio": r,
            "pad": (pad_w, pad_h),
            "orig_shape": img_bgr.shape[:2],  # (H, W)
        }
        return blob, meta


class ResnetPreprocessor:
    """ResNet50 前处理：短边 resize 到 resize_size -> center crop -> ImageNet 归一化。"""

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, resize_size: int = 256, to_rgb: bool = True):
        self.resize_size = resize_size
        self.to_rgb = to_rgb

    def __call__(self, img_bgr: np.ndarray, input_size: Tuple[int, int]):
        nh, nw = input_size  # 通常 (224, 224)
        h, w = img_bgr.shape[:2]
        scale = self.resize_size / min(h, w)
        rw, rh = int(round(w * scale)), int(round(h * scale))
        img = cv2.resize(img_bgr, (rw, rh), interpolation=cv2.INTER_LINEAR)
        # center crop
        x0 = max(0, (rw - nw) // 2)
        y0 = max(0, (rh - nh) // 2)
        img = img[y0:y0 + nh, x0:x0 + nw]
        if self.to_rgb:
            img = img[:, :, ::-1]
        blob = img.astype(np.float32) / 255.0
        blob = (blob - self.IMAGENET_MEAN) / self.IMAGENET_STD
        blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[None])
        meta = {"orig_shape": img_bgr.shape[:2]}
        return blob, meta
