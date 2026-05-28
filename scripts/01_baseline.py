"""第 1 周收尾：跑通 PyTorch baseline，验证整条 预处理->推理->后处理 链路。

用法（在仓库根目录执行）：
    python scripts/01_baseline.py --model configs/yolov8n.yaml --image data/samples/bus.jpg
    python scripts/01_baseline.py --model configs/yolov8n.yaml --image data/samples/bus.jpg --runs 50
"""
import argparse
import os
import sys

import cv2
import numpy as np
import yaml

# 把仓库根目录加入 import 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.factory import build_engine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="模型 config yaml")
    ap.add_argument("--image", required=True, help="测试图路径")
    ap.add_argument("--backend", default="pytorch")
    ap.add_argument("--precision", default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--runs", type=int, default=20, help="计时循环次数")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--save", default="", help="可视化结果输出路径（仅检测）")
    args = ap.parse_args()

    with open(args.model, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)

    engine = build_engine(args.backend, model_cfg, precision=args.precision)
    with engine:
        print(f"[load] backend={args.backend} precision={args.precision} "
              f"device={getattr(engine, 'device', 'n/a')}")
        engine.warmup(args.warmup)

        # 计时
        pre_t, inf_t, post_t = [], [], []
        last = None
        for _ in range(args.runs):
            res = engine.infer(img)
            pre_t.append(res.timing.preprocess_ms)
            inf_t.append(res.timing.inference_ms)
            post_t.append(res.timing.postprocess_ms)
            last = res

        def stat(a):
            a = np.array(a)
            return f"mean={a.mean():.2f} p50={np.percentile(a,50):.2f} p95={np.percentile(a,95):.2f}"

        total = np.array(pre_t) + np.array(inf_t) + np.array(post_t)
        print(f"[pre ] {stat(pre_t)} ms")
        print(f"[infer] {stat(inf_t)} ms")
        print(f"[post] {stat(post_t)} ms")
        print(f"[total] {stat(total.tolist())} ms  ->  {1000.0/total.mean():.1f} FPS")

        if model_cfg["type"] == "yolov8":
            print(f"[result] detections: {last.outputs.shape[0]}")
            if args.save:
                vis = img.copy()
                for x1, y1, x2, y2, score, cls in last.outputs:
                    cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(vis, f"{int(cls)}:{score:.2f}", (int(x1), int(y1) - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imwrite(args.save, vis)
                print(f"[result] saved -> {args.save}")
        else:
            print(f"[result] top-k (class_id, prob): {last.outputs.tolist()}")


if __name__ == "__main__":
    main()
