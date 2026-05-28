"""后端 / 前后处理工厂。按 config 选择，懒加载避免跨平台依赖冲突。

backend 与平台依赖对应关系：
- pytorch  -> torch (+ ultralytics / torchvision)，两平台均可
- tensorrt -> 仅 TX2 NX（tensorrt + pycuda）
- rknn     -> 仅 RK3588（rknnlite）
"""
from src.core.preprocess import YoloPreprocessor, ResnetPreprocessor
from src.core.postprocess import YoloV8Postprocessor, ClassifyPostprocessor


def build_processors(model_cfg: dict):
    mtype = model_cfg["type"]
    if mtype == "yolov8":
        pre = YoloPreprocessor(to_rgb=True)
        post = YoloV8Postprocessor(
            conf_thres=model_cfg.get("conf_thres", 0.25),
            iou_thres=model_cfg.get("iou_thres", 0.45),
            num_classes=model_cfg.get("num_classes", 80),
        )
    elif mtype == "resnet50":
        pre = ResnetPreprocessor(resize_size=model_cfg.get("resize_size", 256))
        post = ClassifyPostprocessor(topk=model_cfg.get("topk", 5))
    else:
        raise ValueError(f"unknown model type: {mtype}")
    return pre, post


def build_engine(backend: str, model_cfg: dict, precision: str = "fp32", **kwargs):
    pre, post = build_processors(model_cfg)
    backend = backend.lower()

    if backend == "pytorch":
        from src.backends.pytorch_engine import PyTorchEngine
        return PyTorchEngine(model_cfg, pre, post, precision=precision, **kwargs)
    if backend == "tensorrt":
        from src.backends.tensorrt_engine import TensorRTEngine  # 第 2 周实现
        return TensorRTEngine(model_cfg, pre, post, precision=precision, **kwargs)
    if backend == "rknn":
        from src.backends.rknn_engine import RKNNEngine        # 第 2 周实现
        return RKNNEngine(model_cfg, pre, post, precision=precision, **kwargs)
    raise ValueError(f"unknown backend: {backend}")
