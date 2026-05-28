"""TensorRT 推理后端（TX2 NX）。tensorrt + pycuda，不依赖 torch。

假定静态 batch=1 的 engine（export 时未开 --dynamic）。动态 shape 需另行 set_binding_shape。
GPU 异步执行，_forward 内用 stream.synchronize() 保证计时准确。
"""
from typing import List

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda

from src.core.base_infer import InferenceEngine


class TensorRTEngine(InferenceEngine):
    def __init__(self, model_cfg, preprocessor, postprocessor, precision="fp32"):
        super().__init__(model_cfg, preprocessor, postprocessor, precision)
        self.engine_path = model_cfg["engine_path"].format(precision=precision)
        self.cuda_ctx = None
        self.engine = None
        self.context = None
        self.stream = None
        self.inputs = []
        self.outputs = []
        self.bindings = []

    def _load(self) -> None:
        cuda.init()
        # 显式创建并压入 CUDA context（不用 pycuda.autoinit，便于 release 时干净弹出）
        self.cuda_ctx = cuda.Device(0).make_context()

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(self.engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"deserialize engine failed: {self.engine_path}")
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        for i in range(self.engine.num_bindings):
            shape = self.engine.get_binding_shape(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            size = trt.volume(shape)
            host = cuda.pagelocked_empty(size, dtype)
            dev = cuda.mem_alloc(host.nbytes)
            self.bindings.append(int(dev))
            entry = {"host": host, "dev": dev, "shape": tuple(shape), "dtype": dtype}
            if self.engine.binding_is_input(i):
                self.inputs.append(entry)
            else:
                self.outputs.append(entry)

    def _forward(self, blob: np.ndarray) -> List[np.ndarray]:
        inp = self.inputs[0]
        np.copyto(inp["host"], blob.ravel().astype(inp["dtype"], copy=False))
        cuda.memcpy_htod_async(inp["dev"], inp["host"], self.stream)
        self.context.execute_async_v2(self.bindings, self.stream.handle)
        for o in self.outputs:
            cuda.memcpy_dtoh_async(o["host"], o["dev"], self.stream)
        self.stream.synchronize()
        return [o["host"].reshape(o["shape"]).copy() for o in self.outputs]

    def _free(self) -> None:
        self.inputs.clear()
        self.outputs.clear()
        self.bindings.clear()
        self.context = None
        self.engine = None
        if self.cuda_ctx is not None:
            self.cuda_ctx.pop()
            self.cuda_ctx = None
