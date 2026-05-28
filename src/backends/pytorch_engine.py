"""PyTorch baseline 后端。两平台都能跑（有 CUDA 走 CUDA，否则 CPU）。

作为精度基准：第 2 周 TensorRT / RKNN 的输出都要和它对齐（mAP / top-1 差异 <0.5%）。
"""
from typing import List

import numpy as np

from src.core.base_infer import InferenceEngine


class PyTorchEngine(InferenceEngine):
    def __init__(self, model_cfg, preprocessor, postprocessor, precision="fp32", device=None):
        super().__init__(model_cfg, preprocessor, postprocessor, precision)
        import torch  # 延迟导入，避免无 torch 的平台报错
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_cfg["type"]           # "yolov8" | "resnet50"
        self.weight = model_cfg.get("pytorch_weight")  # 权重路径或名称
        self.model = None

    def _load(self) -> None:
        torch = self._torch
        if self.model_type == "yolov8":
            from ultralytics import YOLO
            # 取出底层 nn.Module，直接拿检测头原始输出，便于和 TRT/RKNN 对齐
            self.model = YOLO(self.weight).model.to(self.device).eval()
        elif self.model_type == "resnet50":
            import torchvision
            weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
            self.model = torchvision.models.resnet50(weights=weights).to(self.device).eval()
        else:
            raise ValueError(f"unknown model type: {self.model_type}")

        if self.precision == "fp16" and self.device == "cuda":
            self.model = self.model.half()

    def _forward(self, blob: np.ndarray) -> List[np.ndarray]:
        torch = self._torch
        with torch.no_grad():
            x = torch.from_numpy(blob).to(self.device)
            if self.precision == "fp16" and self.device == "cuda":
                x = x.half()
            y = self.model(x)
            if self.device == "cuda":
                torch.cuda.synchronize()

        # ultralytics YOLO 的 nn.Module 推理返回 (preds, ...)，取首元素
        if isinstance(y, (tuple, list)):
            y = y[0]
        return [y.float().cpu().numpy()]

    def _free(self) -> None:
        self.model = None
        if self.device == "cuda":
            self._torch.cuda.empty_cache()
