"""PyTorch -> ONNX。在 x86 + Python3.8+ 环境（如 azure-vm）执行，产物 scp 给设备。

用法：
    python src/convert/export_onnx.py --model configs/yolov8n.yaml
    python src/convert/export_onnx.py --model configs/resnet50.yaml --opset 12
"""
import argparse
import os
import shutil
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def export_yolov8(cfg, opset, dynamic, out_path):
    from ultralytics import YOLO
    imgsz = list(cfg["input_size"])  # [H, W]
    model = YOLO(cfg["pytorch_weight"])
    exported = model.export(
        format="onnx", opset=opset, imgsz=imgsz, simplify=True, dynamic=dynamic
    )
    shutil.move(str(exported), out_path)


def export_resnet50(cfg, opset, dynamic, out_path):
    import torch
    import torchvision
    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    model = torchvision.models.resnet50(weights=weights).eval()
    h, w = cfg["input_size"]
    dummy = torch.randn(1, 3, h, w)
    dyn = {"input": {0: "batch"}, "output": {0: "batch"}} if dynamic else None
    torch.onnx.export(
        model, dummy, out_path, opset_version=opset,
        input_names=["input"], output_names=["output"], dynamic_axes=dyn,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--opset", type=int, default=12, help="TRT 8.0 建议 11~13")
    ap.add_argument("--dynamic", action="store_true", help="动态 batch（默认静态 batch=1）")
    args = ap.parse_args()

    with open(args.model, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_path = cfg["onnx_path"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if cfg["type"] == "yolov8":
        export_yolov8(cfg, args.opset, args.dynamic, out_path)
    elif cfg["type"] == "resnet50":
        export_resnet50(cfg, args.opset, args.dynamic, out_path)
    else:
        raise ValueError(cfg["type"])

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"[onnx] saved -> {out_path}  ({size_mb:.1f} MB, opset={args.opset}, dynamic={args.dynamic})")


if __name__ == "__main__":
    main()
