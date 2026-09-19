"""SCRFD ONNX detector — re-implementation of the official InsightFace inference math.

Source traced: deepinsight/insightface @ 1480e705287bc5d59f923b46c260ec6e3e4150f6,
python-package/insightface/model_zoo/scrfd.py (sha256 c06275fe...). Reproduced exactly:
  * letterbox into a fixed square input (here 320x320, spec §4): keep aspect ratio, cv2.resize
    default (INTER_LINEAR), place at top-left, zero-pad; det_scale = new_height / img_h
  * blob = cv2.dnn.blobFromImage(det_img, 1/128, (320,320), (127.5,)*3, swapRB=True)
    -> input is RGB, (x - 127.5) / 128, NCHW float32; the source image is BGR (cv2)
  * 9-output KPS models: strides [8,16,32], 2 anchors; distance2bbox / distance2kps with pred*stride
  * keep scores >= threshold (spec 0.50); boxes/kps / det_scale -> original-frame coordinates
  * stable sort by score, NMS IoU threshold 0.4 (official default; spec silent), official +1 area
Model file (Q-02 = RESOLVED_BY_OWNER_SELECTION, byte-verified): scrfd_10g_bnkps.onnx, the detector of
the official InsightFace `antelopev2` model pack (release v0.7). Its sha256 was confirmed EXACTLY equal
to the member `antelopev2/scrfd_10g_bnkps.onnx` of the official antelopev2.zip. The unselected candidate
det_2.5g.onnx (SCRFD_2.5G_KPS, buffalo_m) is refused here and kept only as registry history.
Nothing is imported from the insightface package.
"""
from __future__ import annotations

import hashlib

import cv2
import numpy as np

NMS_THRESH = 0.4            # official InsightFace default (IMPLEMENTATION_DETAIL; spec silent)
INPUT_MEAN, INPUT_STD = 127.5, 128.0

# Q-02 owner selection: SCRFD_10G_KPS (scrfd_10g_bnkps.onnx), byte-identical to the official
# antelopev2 pack member. Any other detector file is refused by this class.
SELECTED_VARIANT = "SCRFD_10G_KPS"
SELECTED_WEIGHT_SHA256 = "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91"
NOT_SELECTED_SHA256 = {
    "041f73f47371333d1d17a6fee6c8ab4e6aecabefe398ff32cca4e2d5eaee0af9":
        "det_2.5g.onnx (SCRFD_2.5G_KPS, buffalo_m); NOT_SELECTED_FOR_FINAL_M2",
}


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def distance2bbox(points, d):
    return np.stack([points[:, 0] - d[:, 0], points[:, 1] - d[:, 1], points[:, 0] + d[:, 2], points[:, 1] + d[:, 3]], axis=-1)


def distance2kps(points, d):
    out = []
    for i in range(0, d.shape[1], 2):
        out.append(points[:, i % 2] + d[:, i])
        out.append(points[:, i % 2 + 1] + d[:, i + 1])
    return np.stack(out, axis=-1)


def nms(dets, thresh=NMS_THRESH):
    x1, y1, x2, y2, s = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.argsort(-s, kind="stable")
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
        xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(ovr <= thresh)[0] + 1]
    return keep


class SCRFD:
    def __init__(self, model_path, expected_sha256: str | None = None, input_size: int = 320, providers=("CPUExecutionProvider",)):
        """`expected_sha256` defaults to the owner-selected 10G weights; an unselected candidate
        (e.g. det_2.5g.onnx) is refused with an explicit message and can never become active
        by passing its own hash."""
        import onnxruntime as ort
        self.sha256 = _sha256(model_path)
        if self.sha256 in NOT_SELECTED_SHA256:
            raise ValueError(f"refusing a detector the owner did not select: {NOT_SELECTED_SHA256[self.sha256]}")
        expected = expected_sha256 or SELECTED_WEIGHT_SHA256
        if self.sha256 != expected:
            raise ValueError(f"SCRFD weight hash mismatch: {self.sha256} != {expected}")
        self.variant = SELECTED_VARIANT
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1          # determinism
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(model_path), sess_options=so, providers=list(providers))
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        if len(self.output_names) != 9:
            raise ValueError("expected a 9-output SCRFD *_KPS model (strides 8/16/32, 2 anchors)")
        self.strides, self.num_anchors, self.fmc = [8, 16, 32], 2, 3
        self.input_size = (int(input_size), int(input_size))

    def letterbox(self, img_bgr):
        W, H = self.input_size
        im_ratio = img_bgr.shape[0] / img_bgr.shape[1]
        if im_ratio > H / W:
            nh, nw = H, int(H / im_ratio)
        else:
            nw, nh = W, int(W * im_ratio)
        det_scale = nh / img_bgr.shape[0]
        det = np.zeros((H, W, 3), dtype=np.uint8)
        det[:nh, :nw, :] = cv2.resize(img_bgr, (nw, nh))
        return det, det_scale, (nw, nh)

    def detect(self, img_bgr, threshold: float):
        det_img, det_scale, _ = self.letterbox(img_bgr)
        blob = cv2.dnn.blobFromImage(det_img, 1.0 / INPUT_STD, self.input_size, (INPUT_MEAN,) * 3, swapRB=True)
        outs = self.session.run(self.output_names, {self.input_name: np.ascontiguousarray(blob, np.float32)})
        H, W = blob.shape[2], blob.shape[3]
        scores_l, boxes_l, kps_l = [], [], []
        for idx, stride in enumerate(self.strides):
            scores = outs[idx]
            bp = outs[idx + self.fmc] * stride
            kp = outs[idx + 2 * self.fmc] * stride
            h, w = H // stride, W // stride
            centers = (np.stack(np.mgrid[:h, :w][::-1], axis=-1).astype(np.float32) * stride).reshape(-1, 2)
            centers = np.stack([centers] * self.num_anchors, axis=1).reshape(-1, 2)
            pos = np.where(scores >= threshold)[0]
            scores_l.append(scores[pos])
            boxes_l.append(distance2bbox(centers, bp)[pos])
            kps_l.append(distance2kps(centers, kp).reshape(-1, 5, 2)[pos])
        scores = np.vstack(scores_l).ravel()
        if scores.size == 0:
            return np.empty((0, 5), np.float32), np.empty((0, 5, 2), np.float32)
        order = np.argsort(-scores, kind="stable")
        boxes = np.vstack(boxes_l) / det_scale
        kpss = np.vstack(kps_l) / det_scale
        pre = np.hstack((boxes, scores[:, None])).astype(np.float32)[order]
        kpss = kpss[order]
        keep = nms(pre)
        return pre[keep], kpss[keep]
