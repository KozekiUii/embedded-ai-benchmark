# TX2 NX 三精度对比 — YOLOv8n

平台：Jetson TX2 NX（Pascal sm_62, JetPack 4.6.1, TensorRT 8.0.1, CUDA 10.2）
模型：YOLOv8n 640×640，测试图 bus.jpg，80 次推理 + 20 次预热，**同一会话背靠背测**。

| 精度 | 预处理 ms | 推理 ms | 后处理 ms | 总延迟 ms | FPS | engine 大小 | 检测数 |
|------|----------|---------|-----------|-----------|-----|------------|--------|
| FP32 | ~17.8 | 35.75 | ~8 | 61.54 | 16.2 | 30 MB | 5 |
| FP16 | ~17.8 | 30.37 | ~8 | 56.00 | 17.9 | 15 MB | 5 |
| INT8 | ~17.8 | 29.20 | ~8 | 54.81 | 18.2 | 15 MB | 5 |

INT8 校准集：COCO128（128 张），TensorRT IInt8EntropyCalibrator2（PTQ）。
检测结果三精度一致（同图同框），精度视觉无损。

## 关键发现

1. **INT8 仅比 FP16 快约 4%，远没到理论 2×。** 原因：TX2 的 Pascal 架构**没有 INT8 Tensor Core**（Turing 起才有），INT8 只能靠 DP4A，吞吐与 FP16×2 相近；加上 INT8↔FP16 的 reformat 开销，净收益很小。对比之下，Orin/Xavier 有 INT8 DLA/Tensor Core，INT8 才会明显领先。
2. **FP16 是这台设备的性价比甜点**：比 FP32 快约 17%、engine 体积减半、且无需校准。
3. **Benchmark 方法学坑**：最初跨会话测出「INT8 比 FP16 慢」，是热/频率状态不同导致的假象；改为同一会话、统一预热背靠背测后，INT8 才稳定略快。→ 嵌入式性能测试必须控制会话/温度变量。
4. **预处理(~18ms)+后处理(~8ms) 约 26ms 的 CPU 固定开销**已超过三种精度之间的推理差异，成为总延迟主导项 → 第 5 周用 CUDA Kernel 加速预处理的必要性由数据直接论证。

## 待办
- [ ] 正式 benchmark.py：加内存/功耗(tegrastats)/能效比，写 CSV
- [ ] 精度对齐：与 PyTorch baseline 跑 COCO mAP（当前仅单图视觉验证）
- [ ] 锁频对比：jetson_clocks 开/关对延迟的影响
