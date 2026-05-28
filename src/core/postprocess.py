"""后处理。两平台共用。

YOLOv8 检测头原始输出形状为 (1, 4+nc, num_anchors)，例如 640 输入下 COCO 为
(1, 84, 8400)。无单独 objectness，4 个为 cx,cy,w,h（letterbox 后的像素坐标），
其余 nc 个为各类别分数。后端输出可能已 squeeze 或转置，这里做鲁棒 reshape。
"""
from typing import List

import cv2
import numpy as np


class YoloV8Postprocessor:
    def __init__(self, conf_thres: float = 0.25, iou_thres: float = 0.45, num_classes: int = 80):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.nc = num_classes

    def _as_anchors_last(self, arr: np.ndarray) -> np.ndarray:
        """统一成 (num_anchors, 4+nc)。"""
        arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise ValueError(f"unexpected yolo output shape: {arr.shape}")
        # 若第 0 维是通道（4+nc），转置
        if arr.shape[0] == 4 + self.nc and arr.shape[1] != 4 + self.nc:
            arr = arr.T
        return arr

    def __call__(self, raw: List[np.ndarray], meta: dict) -> np.ndarray:
        pred = self._as_anchors_last(raw[0])  # (M, 4+nc)
        boxes_xywh = pred[:, :4]
        scores_all = pred[:, 4:]
        cls = scores_all.argmax(axis=1)
        conf = scores_all.max(axis=1)

        keep = conf >= self.conf_thres
        boxes_xywh, conf, cls = boxes_xywh[keep], conf[keep], cls[keep]
        if boxes_xywh.shape[0] == 0:
            return np.zeros((0, 6), dtype=np.float32)

        # xywh(center) -> xyxy
        xy = boxes_xywh[:, :2]
        wh = boxes_xywh[:, 2:4]
        x1y1 = xy - wh / 2
        x2y2 = xy + wh / 2
        boxes = np.concatenate([x1y1, x2y2], axis=1)

        # NMS（cv2 接受 [x, y, w, h]）
        nms_boxes = np.concatenate([x1y1, wh], axis=1).tolist()
        idxs = cv2.dnn.NMSBoxes(nms_boxes, conf.tolist(), self.conf_thres, self.iou_thres)
        if len(idxs) == 0:
            return np.zeros((0, 6), dtype=np.float32)
        idxs = np.array(idxs).flatten()
        boxes, conf, cls = boxes[idxs], conf[idxs], cls[idxs]

        # 坐标从 letterbox 空间还原到原图
        r = meta["ratio"]
        pad_w, pad_h = meta["pad"]
        boxes[:, [0, 2]] -= pad_w
        boxes[:, [1, 3]] -= pad_h
        boxes /= r
        oh, ow = meta["orig_shape"]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, ow)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, oh)

        out = np.concatenate(
            [boxes, conf[:, None], cls[:, None].astype(np.float32)], axis=1
        )
        return out.astype(np.float32)


class ClassifyPostprocessor:
    """分类后处理：softmax + top-k。返回 (topk, 2) -> [class_id, prob]。"""

    def __init__(self, topk: int = 5):
        self.topk = topk

    def __call__(self, raw: List[np.ndarray], meta: dict) -> np.ndarray:
        logits = np.squeeze(raw[0]).astype(np.float32)
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        idx = np.argsort(probs)[::-1][: self.topk]
        return np.stack([idx.astype(np.float32), probs[idx]], axis=1)
