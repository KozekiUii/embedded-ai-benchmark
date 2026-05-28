"""统一推理接口。

设计要点：
- 所有后端（PyTorch / TensorRT / RKNN）只实现 _load / _forward / _free 三个方法；
- 前处理、后处理、计时全部由基类编排，保证两平台逻辑完全一致，对比数据可比；
- _forward 拿到的是已经预处理好的 NCHW float32 blob，返回原始输出张量列表（list[np.ndarray]）。

注意：GPU/NPU 推理是异步的，各后端的 _forward 内部必须自行同步（如
torch.cuda.synchronize() / cudaStreamSynchronize），否则 inference_ms 不准。
"""
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, List

import numpy as np


@dataclass
class Timing:
    """单次推理三段耗时（毫秒）。"""
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.preprocess_ms + self.inference_ms + self.postprocess_ms


@dataclass
class InferResult:
    """一次推理的结果。

    outputs 的具体类型由后处理器决定：
    - 检测：np.ndarray，形状 (N, 6) -> [x1, y1, x2, y2, score, cls]
    - 分类：np.ndarray，形状 (topk, 2) -> [class_id, prob]
    """
    outputs: Any = None
    timing: Timing = field(default_factory=Timing)
    raw: List[np.ndarray] = None


class InferenceEngine(ABC):
    """推理引擎抽象基类。"""

    def __init__(
        self,
        model_cfg: dict,
        preprocessor: Callable,
        postprocessor: Callable,
        precision: str = "fp32",
    ):
        self.cfg = model_cfg
        self.pre = preprocessor
        self.post = postprocessor
        self.precision = precision.lower()
        # input_size 统一存成 (H, W)
        h, w = model_cfg["input_size"]
        self.input_size = (int(h), int(w))
        self._loaded = False

    # ---- 子类必须实现 ----
    @abstractmethod
    def _load(self) -> None:
        """加载模型 / 反序列化 engine。"""

    @abstractmethod
    def _forward(self, blob: np.ndarray) -> List[np.ndarray]:
        """对预处理后的 NCHW blob 做一次前向，返回原始输出张量列表。内部需同步。"""

    @abstractmethod
    def _free(self) -> None:
        """释放显存 / 资源。"""

    # ---- 基类编排 ----
    def load(self) -> "InferenceEngine":
        if not self._loaded:
            self._load()
            self._loaded = True
        return self

    def warmup(self, n: int = 10) -> None:
        """用全零图预热，避免首帧 lazy 初始化污染计时。"""
        dummy = np.zeros((self.input_size[0], self.input_size[1], 3), dtype=np.uint8)
        for _ in range(n):
            self.infer(dummy)

    def infer(self, image_bgr: np.ndarray) -> InferResult:
        """端到端推理：预处理 -> 前向 -> 后处理，并分段计时。"""
        if not self._loaded:
            self.load()

        t = Timing()
        t0 = time.perf_counter()
        blob, meta = self.pre(image_bgr, self.input_size)
        t1 = time.perf_counter()
        raw = self._forward(blob)
        t2 = time.perf_counter()
        out = self.post(raw, meta)
        t3 = time.perf_counter()

        t.preprocess_ms = (t1 - t0) * 1000.0
        t.inference_ms = (t2 - t1) * 1000.0
        t.postprocess_ms = (t3 - t2) * 1000.0
        return InferResult(outputs=out, timing=t, raw=raw)

    def release(self) -> None:
        if self._loaded:
            self._free()
            self._loaded = False

    def __enter__(self) -> "InferenceEngine":
        return self.load()

    def __exit__(self, *exc) -> None:
        self.release()
