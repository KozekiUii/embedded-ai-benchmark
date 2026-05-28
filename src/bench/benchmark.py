"""统一性能评测：延迟 / 吞吐 / 内存 / 温度 / 功耗 / 能效比，结果写 CSV。

两平台共用。功耗读 sysfs INA（TX2 在 2-0040 iio:device0），需 root 可读；
读不到时自动跳过功耗/能效，其余指标照常输出。

用法：
    python src/bench/benchmark.py --model configs/yolov8n.yaml \
        --backend tensorrt --precision int8 --image data/samples/bus.jpg --runs 100
"""
import argparse
import csv
import glob
import os
import sys
import threading
import time

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.factory import build_engine

# TX2 NX INA3221 三路功耗（mW），需 root 可读
INA_DIR = "/sys/devices/3180000.i2c/i2c-2/2-0040/iio:device0"


def read_power_mw():
    """读 INA 三路功耗之和(mW)。读不到返回 None。"""
    total = 0
    found = False
    for i in range(3):
        p = os.path.join(INA_DIR, "in_power%d_input" % i)
        try:
            with open(p) as f:
                total += int(f.read().strip())
                found = True
        except Exception:
            return None
    return total if found else None


def read_ram_used_mb():
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":")
                info[k] = int(v.strip().split()[0])  # kB
        return (info["MemTotal"] - info["MemAvailable"]) / 1024.0
    except Exception:
        return None


def read_temp_c():
    """所有 thermal_zone 的最高温(℃)。"""
    temps = []
    for f in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
        try:
            with open(f) as fh:
                temps.append(int(fh.read().strip()) / 1000.0)
        except Exception:
            pass
    return max(temps) if temps else None


def read_gpu_load_pct():
    for p in ("/sys/devices/gpu.0/load", "/sys/devices/57000000.gpu/load"):
        try:
            with open(p) as f:
                return int(f.read().strip()) / 10.0  # 千分比 -> 百分比
        except Exception:
            continue
    return None


class Sampler(threading.Thread):
    """后台采样功耗/内存/温度/GPU 占用。"""
    def __init__(self, interval=0.05):
        super().__init__(daemon=True)
        self.interval = interval
        self._stop_evt = threading.Event()
        self.power, self.ram, self.temp, self.gpu = [], [], [], []

    def run(self):
        while not self._stop_evt.is_set():
            p = read_power_mw()
            if p is not None:
                self.power.append(p)
            r = read_ram_used_mb()
            if r is not None:
                self.ram.append(r)
            t = read_temp_c()
            if t is not None:
                self.temp.append(t)
            g = read_gpu_load_pct()
            if g is not None:
                self.gpu.append(g)
            time.sleep(self.interval)

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--backend", default="tensorrt")
    ap.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "int8"])
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--platform", default="tx2nx", help="平台标签，写进 CSV")
    ap.add_argument("--csv", default="results/raw/benchmark.csv")
    args = ap.parse_args()

    with open(args.model, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)

    engine = build_engine(args.backend, cfg, precision=args.precision)
    with engine:
        engine.warmup(args.warmup)

        sampler = Sampler()
        sampler.start()
        pre, inf, post = [], [], []
        t_start = time.perf_counter()
        for _ in range(args.runs):
            res = engine.infer(img)
            pre.append(res.timing.preprocess_ms)
            inf.append(res.timing.inference_ms)
            post.append(res.timing.postprocess_ms)
        wall = time.perf_counter() - t_start
        sampler.stop()

    pre, inf, post = map(np.array, (pre, inf, post))
    total = pre + inf + post
    fps = args.runs / wall

    power_w = float(np.mean(sampler.power)) / 1000.0 if sampler.power else None
    ram_peak = max(sampler.ram) if sampler.ram else None
    temp_max = max(sampler.temp) if sampler.temp else None
    gpu_mean = float(np.mean(sampler.gpu)) if sampler.gpu else None
    energy_j = (power_w / fps) if power_w else None          # 每帧能耗 J
    fpj = (fps / power_w) if power_w else None               # frames per joule

    def m(a):
        return (float(a.mean()), float(np.percentile(a, 50)), float(np.percentile(a, 95)))

    print("==== %s | %s | %s ====" % (args.platform, args.backend, args.precision))
    print("pre   mean/p50/p95 ms: %.2f / %.2f / %.2f" % m(pre))
    print("infer mean/p50/p95 ms: %.2f / %.2f / %.2f" % m(inf))
    print("post  mean/p50/p95 ms: %.2f / %.2f / %.2f" % m(post))
    print("total mean ms: %.2f  ->  %.1f FPS (wall)" % (total.mean(), fps))
    print("RAM peak: %s MB | temp max: %s C | GPU mean: %s %%" %
          (round(ram_peak, 1) if ram_peak else "n/a",
           round(temp_max, 1) if temp_max else "n/a",
           round(gpu_mean, 1) if gpu_mean is not None else "n/a"))
    if power_w:
        print("power mean: %.2f W | energy/frame: %.4f J | frames/joule: %.2f" %
              (power_w, energy_j, fpj))
    else:
        print("power: n/a (INA 不可读，需 sudo 解锁 %s/in_power*_input)" % INA_DIR)

    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": args.platform, "model": cfg.get("type"), "backend": args.backend,
        "precision": args.precision, "runs": args.runs,
        "pre_ms": round(float(pre.mean()), 3), "infer_ms": round(float(inf.mean()), 3),
        "post_ms": round(float(post.mean()), 3), "total_ms": round(float(total.mean()), 3),
        "fps": round(fps, 2),
        "ram_used_mb_peak": round(ram_peak, 1) if ram_peak else "",
        "temp_c_max": round(temp_max, 1) if temp_max else "",
        "gpu_pct_mean": round(gpu_mean, 1) if gpu_mean is not None else "",
        "power_w_mean": round(power_w, 2) if power_w else "",
        "energy_per_frame_j": round(energy_j, 4) if energy_j else "",
        "frames_per_joule": round(fpj, 2) if fpj else "",
    }
    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    new = not os.path.exists(args.csv)
    with open(args.csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)
    print("[csv] appended -> %s" % args.csv)


if __name__ == "__main__":
    main()
