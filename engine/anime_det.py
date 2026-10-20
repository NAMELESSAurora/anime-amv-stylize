# -*- coding: utf-8 -*-
"""
动漫脸 / 人物检测（deepghs 的 YOLOv8 ONNX：models/anime_face_detection/face_detect_v1.4_s、
models/anime_person_detection/person_detect_v1.3_s，阈值用模型自带 threshold.json）。

v2 风格层的锚点：causasui 星座节点挂在人物轮廓上、Raf0x00 准星锁人物 / 脸；护脸 mask（眼口干净）也从这里来。
用法: from anime_det import Det; Det("face")(rgb) -> [(x0, y0, x1, y1, conf), ...]（原图坐标，按置信度降序）
"""
import json, os
import numpy as np
import cv2
import onnxruntime as ort

ROOT = r"D:\A\provant"
MODELS = {"face": "anime_face_detection/face_detect_v1.4_s", "person": "anime_person_detection/person_detect_v1.3_s",
          "head": "anime_head_detection/head_detect_v0.5_s"}   # head：面甲 / 侧脸 / 背影时脸检测不到，用头部框


class Det:
    def __init__(self, kind, size=640):
        d = os.path.join(ROOT, "models", MODELS[kind])
        so = ort.SessionOptions(); so.intra_op_num_threads = int(os.environ.get("DET_THREADS", "2"))
        prov = ["CPUExecutionProvider"]
        if "DmlExecutionProvider" in ort.get_available_providers():      # .venv_gpu：显卡推理
            so.enable_mem_pattern = False; so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            prov = ["DmlExecutionProvider", "CPUExecutionProvider"]
        self.sess = ort.InferenceSession(os.path.join(d, "model.onnx"), so, providers=prov)
        self.inp = self.sess.get_inputs()[0].name
        self.th = json.load(open(os.path.join(d, "threshold.json")))["threshold"]
        self.size = size

    def __call__(self, rgb, th=None, iou=0.5):
        h, w = rgb.shape[:2]
        s = self.size / max(h, w)
        nw, nh = int(round(w * s)), int(round(h * s))
        pad = np.full((self.size, self.size, 3), 114, np.uint8)
        pad[:nh, :nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        x = pad.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        o = self.sess.run(None, {self.inp: x})[0][0]          # (5, N): cx, cy, w, h, score
        sc = o[4]
        k = sc > (self.th if th is None else th)
        if not k.any():
            return []
        cx, cy, bw, bh, sc = o[0][k], o[1][k], o[2][k], o[3][k], sc[k]
        boxes = np.stack([cx - bw / 2, cy - bh / 2, bw, bh], 1) / s
        keep = cv2.dnn.NMSBoxes(boxes.tolist(), sc.tolist(), 0.0, iou)
        keep = np.array(keep).reshape(-1)
        out = [(float(boxes[i, 0]), float(boxes[i, 1]), float(boxes[i, 0] + boxes[i, 2]), float(boxes[i, 1] + boxes[i, 3]),
                float(sc[i])) for i in keep]
        return sorted(out, key=lambda b: -b[4])
