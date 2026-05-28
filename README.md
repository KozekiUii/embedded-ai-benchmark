# 嵌入式 AI 模型部署性能对比研究

Jetson TX2 NX（TensorRT 8.2）与 RK3588（RKNN 2.x）双平台 YOLOv8n / ResNet50 推理优化与性能对比。

## 设计

统一接口 + 平台后端分离：所有后端实现 `InferenceEngine.infer()` 同一接口，前处理 /
后处理 / benchmark 两平台共用，保证对比数据可比。

```
src/core/      抽象接口、前后处理、工厂
src/backends/  pytorch / tensorrt(TX2 NX) / rknn(RK3588)
src/convert/   PyTorch->ONNX->engine/rknn（含 INT8 校准）
src/bench/     统一评测：延迟/吞吐/内存/功耗/能效比
scripts/       baseline / convert / bench / 画图
```

## 环境

- TX2 NX：`pip install -r requirements/jetson.txt`（torch/tensorrt 用 JetPack 自带或 NVIDIA wheel）
- RK3588：`pip install -r requirements/rk3588.txt` + 瑞芯微 `rknn_toolkit_lite2` whl

## 快速验证（第 1 周）

```bash
# 跑通 PyTorch baseline，确认链路通
python scripts/01_baseline.py --model configs/yolov8n.yaml --image data/samples/bus.jpg --save out.jpg
```

## 进度

- [x] 第 1 周：环境 + baseline（core 接口 / pytorch 后端 / 前后处理）
- [ ] 第 2 周：模型转换（ONNX / TensorRT / RKNN）+ 精度对齐
- [ ] 第 3 周：FP16 + benchmark
- [ ] 第 4 周：INT8 量化
- [ ] 第 5 周：工程优化（CUDA Kernel / RGA / 零拷贝）
- [ ] 第 6 周：ResNet50 复现 + 可视化
- [ ] 第 8 周：RK3588 端侧 LLM 探索（llm_on_rk3588/）
