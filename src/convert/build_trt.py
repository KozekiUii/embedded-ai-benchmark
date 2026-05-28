"""ONNX -> TensorRT engine。在 TX2 NX 上执行。

用法：
    python src/convert/build_trt.py --model configs/yolov8n.yaml --precision fp32
    python src/convert/build_trt.py --model configs/yolov8n.yaml --precision fp16
    python src/convert/build_trt.py --model configs/yolov8n.yaml --precision int8 \
        --calib-dir data/calib --calib-num 128

注意：TX2 NX 内存 4GB 与 CPU 共享，workspace 默认 512MB，过大易 OOM。
sm_62 支持 INT8 DP4A，量化有真实加速。
"""
import argparse
import os
import sys

import yaml
import tensorrt as trt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def build(onnx_path, engine_path, precision, workspace_mb, calibrator=None):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError(f"failed to parse onnx: {onnx_path}")

    config = builder.create_builder_config()
    config.max_workspace_size = workspace_mb * 1024 * 1024

    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            print("[warn] platform has no fast fp16")
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        if calibrator is None:
            print("[warn] INT8 without calibrator -> 精度会崩，第 4 周接 calibrator.py")
        else:
            config.int8_calibrator = calibrator

    print(f"[trt] building {precision} engine (workspace={workspace_mb}MB)... 可能几分钟")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("engine build failed")

    data = bytes(serialized)  # IHostMemory -> bytes（TRT 8.0 不支持 len()）
    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(data)
    print(f"[trt] saved -> {engine_path}  ({len(data)/1e6:.1f} MB)")


def make_calibrator(cfg, args, engine_path):
    """构造 INT8 校准器：复用推理同款前处理。需要 pycuda context。"""
    import pycuda.autoinit  # noqa: F401  创建 CUDA context（仅 INT8 路径需要）
    from src.core.factory import build_processors
    from src.convert.calibrator import EntropyCalibrator

    pre, _ = build_processors(cfg)
    h, w = cfg["input_size"]
    cache_file = os.path.splitext(engine_path)[0] + ".cache"
    return EntropyCalibrator(
        image_dir=args.calib_dir,
        preprocessor=pre,
        input_size=(int(h), int(w)),
        cache_file=cache_file,
        batch_size=args.calib_batch,
        max_images=args.calib_num,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "int8"])
    ap.add_argument("--workspace", type=int, default=512, help="workspace MB")
    ap.add_argument("--calib-dir", default="data/calib", help="INT8 校准集目录")
    ap.add_argument("--calib-num", type=int, default=128, help="校准图片数量")
    ap.add_argument("--calib-batch", type=int, default=1)
    args = ap.parse_args()

    with open(args.model, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    onnx_path = cfg["onnx_path"]
    engine_path = cfg["engine_path"].format(precision=args.precision)
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"{onnx_path} 不存在，先在 azure-vm 跑 export_onnx.py 并 scp 过来")

    calibrator = make_calibrator(cfg, args, engine_path) if args.precision == "int8" else None
    build(onnx_path, engine_path, args.precision, args.workspace, calibrator)


if __name__ == "__main__":
    main()
