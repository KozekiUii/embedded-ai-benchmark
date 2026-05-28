"""TensorRT INT8 熵校准器（IInt8EntropyCalibrator2）。在 TX2 NX 上执行。

用校准集（COCO 子集）跑前向，统计每层激活分布，生成 INT8 量化所需的 scale。
预处理必须和推理时完全一致（复用 YoloPreprocessor），否则量化分布失真。
"""
import glob
import os

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda


class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, image_dir, preprocessor, input_size, cache_file,
                 batch_size=1, max_images=500):
        super().__init__()
        self.preprocessor = preprocessor
        self.input_size = input_size            # (H, W)
        self.cache_file = cache_file
        self.batch_size = batch_size

        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        files = []
        for e in exts:
            files += glob.glob(os.path.join(image_dir, "**", e), recursive=True)
        self.files = sorted(files)[:max_images]
        if not self.files:
            raise FileNotFoundError(f"校准集为空: {image_dir}")
        print(f"[calib] {len(self.files)} images, batch={batch_size}")

        self.idx = 0
        self.device_input = None  # 延迟分配

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.idx + self.batch_size > len(self.files):
            return None  # 校准结束
        blobs = []
        for f in self.files[self.idx:self.idx + self.batch_size]:
            img = cv2.imread(f)
            if img is None:
                # 跳过坏图，用零填充保证 batch 完整
                img = np.zeros((self.input_size[0], self.input_size[1], 3), np.uint8)
            blob, _ = self.preprocessor(img, self.input_size)  # (1,3,H,W) float32
            blobs.append(blob)
        arr = np.ascontiguousarray(np.concatenate(blobs, axis=0).astype(np.float32))

        if self.device_input is None:
            self.device_input = cuda.mem_alloc(arr.nbytes)
        cuda.memcpy_htod(self.device_input, arr)

        self.idx += self.batch_size
        if self.idx % (50 * self.batch_size) == 0:
            print(f"[calib] {self.idx}/{len(self.files)}")
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            print(f"[calib] use cache {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
        print(f"[calib] cache saved -> {self.cache_file}")
