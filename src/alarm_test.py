"""
独立检测+OCR+绘制脚本
支持本地图片/视频输入，YOLO RKNN 检测 + PaddleOCR RKNN 舷号识别，结果绘制并保存。

用法:
  # 图片检测
  python standalone_detect.py --input ./test.jpg --output ./result.jpg

  # 视频检测（保存为视频）
  python standalone_detect.py --input ./test.mp4 --output ./result.mp4

  # 视频检测（保存为帧图片）
  python standalone_detect.py --input ./test.mp4 --output ./frames/ --save-frames

  # 关闭 OCR
  python standalone_detect.py --input ./test.jpg --no-ocr

  # 关闭显示
  python standalone_detect.py --input ./test.mp4 --no-display

  # 调整检测阈值
  python standalone_detect.py --input ./test.jpg --conf 0.3 --iou 0.5
"""

import cv2
import sys
import argparse
import numpy as np
import time
import os
import re

# ======================== 模型路径（按需修改） ========================
CLS_MODEL_PATH = "/home/Tronlong/Desktop/rknn_model_zoo_and_models/yolov11n-cls/cls_1-best_255_fp.rknn"
DET_DAY_MODEL_PATH = "/home/Tronlong/Desktop/rknn_model_zoo_and_models/yolov11n-day/day_5-best_255_i8.rknn"
DET_NIGHT_MODEL_PATH = "/home/Tronlong/Desktop/rknn_model_zoo_and_models/yolov11n-night/night_2-best_255_i8.rknn"

# ======================== OCR RKNN 模型路径 ========================
OCR_DET_MODEL_PATH = "/home/Tronlong/Desktop/rknn_model_zoo_and_models/PPOCR_rknn/det/ppocrv4_det.rknn"
OCR_REC_MODEL_PATH = "/home/Tronlong/Desktop/rknn_model_zoo_and_models/PPOCR_rknn/rec/ppocrv4_rec.rknn"
OCR_DICT_PATH = "/home/Tronlong/Desktop/rknn_model_zoo_and_models/PPOCR_rknn/ppocr_keys_v1.txt"

# ======================== 检测配置 ========================
CLS_INPUT_SIZE = 256
DET_INPUT_SIZE = 640
CLS_INTERVAL = 5
# 告警像素占比阈值（基准分辨率 1920×1280，适配16:9任意分辨率）
SHIP_ALARM_THRESH = 0.00185    # 船舶告警：5000 / (1920×1280) = 0.002
LIGHT_ALARM_THRESH = 0.0004  # 灯光告警：1200 / (1920×1280) = 0.0005

# ======================== OCR 配置 ========================
OCR_CONF_THRESH = 0.5
OCR_DET_INPUT_SHAPE = [480, 480]
OCR_REC_INPUT_SHAPE = [48, 320]

# ======================== 绘制配置 ========================
FONT_SCALE = 1
FONT_MULTIPALE = 20
FONT_THICKNESS = 2
ALARM_COLOR = (0, 0, 255)
NORMAL_COLOR = (0, 255, 0)
TEXT_COLOR = (255, 255, 255)
TEXT_BG_COLOR = (0, 0, 0)
CORNER_RADIUS = 8
PADDING = 2
BOX_THICKNESS = 2
OCR_TEXT_COLOR = (255, 255, 255)
OCR_BG_COLOR = (0, 0, 0)
OCR_BOX_COLOR = (255, 165, 0)
LABEL_OFFSET = 8

# ======================== 中文字体路径 ========================
CHINESE_FONT_PATH = None
font_candidates = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/msyh.ttc",
    "/System/Library/Fonts/PingFang.ttc",
]
for font_path in font_candidates:
    if os.path.exists(font_path):
        CHINESE_FONT_PATH = font_path
        break

# ======================== Freetype 混合渲染 ========================
_ft_handle = None

def _get_ft():
    global _ft_handle, CHINESE_FONT_PATH
    if _ft_handle is None and CHINESE_FONT_PATH and hasattr(cv2, "freetype"):
        _ft_handle = cv2.freetype.createFreeType2()
        _ft_handle.loadFontData(fontFileName=CHINESE_FONT_PATH, id=0)
    return _ft_handle

def _needs_freetype(c):
    cp = ord(c)
    return not (0x20 <= cp <= 0x7E)

def _put_mixed_text(img, text, org, font_height, color, thickness):
    ft = _get_ft()
    if ft is None:
        scale = font_height / 30.0
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                    color, thickness, cv2.LINE_AA)
        return img
    x, y = org
    i = 0
    while i < len(text):
        if _needs_freetype(text[i]):
            j = i
            while j < len(text) and _needs_freetype(text[j]):
                j += 1
            segment = text[i:j]
            ft.putText(img, segment, (x, y), fontHeight=font_height,
                       color=color, thickness=thickness, line_type=cv2.LINE_AA,
                       bottomLeftOrigin=True)
            size, _ = ft.getTextSize(segment, fontHeight=font_height, thickness=thickness)
            x += size[0]
            i = j
        else:
            j = i
            while j < len(text) and not _needs_freetype(text[j]):
                j += 1
            segment = text[i:j]
            scale = font_height / 30.0
            cv2.putText(img, segment, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                        color, thickness, cv2.LINE_AA)
            (tw, _), _ = cv2.getTextSize(segment, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
            x += tw
            i = j
    return img

def _get_mixed_text_size(text, font_height, thickness):
    ft = _get_ft()
    if ft is None:
        scale = font_height / 30.0
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        return (tw, th)
    total_w = 0
    i = 0
    while i < len(text):
        if _needs_freetype(text[i]):
            j = i
            while j < len(text) and _needs_freetype(text[j]):
                j += 1
            size, _ = ft.getTextSize(text[i:j], fontHeight=font_height, thickness=thickness)
            total_w += size[0]
            i = j
        else:
            j = i
            while j < len(text) and not _needs_freetype(text[j]):
                j += 1
            scale = font_height / 30.0
            (tw, _), _ = cv2.getTextSize(text[i:j], cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
            total_w += tw
            i = j
    return total_w, font_height

# ======================== 绘制函数 ========================
def draw_rounded_rectangle(img, top_left, bottom_right, color, thickness=-1, radius=10):
    x1, y1 = top_left
    x2, y2 = bottom_right
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = max(x1 + 1, int(x2)), max(y1 + 1, int(y2))
    radius = min(radius, max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2))

    if thickness == -1:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    else:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)

    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)
    return img

def draw_chinese_text(img, text, pos, font_size=0.8, color=(255, 255, 255), thickness=2):
    font_height = max(12, int(font_size * FONT_MULTIPALE))
    x, y_top = pos
    y_baseline = y_top + font_height
    _put_mixed_text(img, text, (x, y_baseline), font_height, color, thickness)
    return img

def get_text_size(text, font_size=0.8):
    font_height = max(12, int(font_size * FONT_MULTIPALE))
    w, h = _get_mixed_text_size(text, font_height, FONT_THICKNESS)
    return (w + 2, h + 2)

def draw_custom_boxes(frame, post_results, weather_info, model_names):
    h, w = frame.shape[:2]
    weather_label, weather_conf = weather_info

    # 绘制天气标签
    weather_text = f"Weather: {weather_label}({weather_conf:.2f})"
    text_w, text_h = get_text_size(weather_text, FONT_SCALE)
    tl_x, tl_y = 5, 5
    br_x = min(tl_x + text_w + 2 * PADDING, w - 5)
    br_y = min(tl_y + text_h + 2 * PADDING, h - 5)
    frame = draw_rounded_rectangle(frame, (tl_x, tl_y), (br_x, br_y),
                                   TEXT_BG_COLOR, thickness=-1, radius=CORNER_RADIUS)
    frame = draw_chinese_text(frame, weather_text, (tl_x + PADDING, tl_y + PADDING),
                              font_size=FONT_SCALE, color=TEXT_COLOR, thickness=FONT_THICKNESS)

    for (x1, y1, x2, y2, conf, cls, area, is_alarm) in post_results:
        box_color = ALARM_COLOR if is_alarm else NORMAL_COLOR
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, BOX_THICKNESS)

        cls_name = model_names.get(int(cls), "unknown")
        alarm_text = "ALARM" if is_alarm else "NORMAL"
        label_text = f"{cls_name}({conf:.2f})| Area:{area}| {alarm_text}"

        label_w, label_h = get_text_size(label_text, FONT_SCALE)
        label_height = label_h + 2 * PADDING
        label_width = label_w + 2 * PADDING

        label_tl_y = y1 - label_height - LABEL_OFFSET
        if label_tl_y < 0:
            label_tl_y = y2 + LABEL_OFFSET
        label_tl_y = max(0, min(label_tl_y, h - label_height - 5))
        label_tl_x = max(0, min(x1, w - label_width - 5))
        label_br_x = label_tl_x + label_width
        label_br_y = label_tl_y + label_height - 2 * PADDING
        if label_br_x > w:
            label_tl_x = max(0, w - label_width - 5)
            label_br_x = label_tl_x + label_width

        frame = draw_rounded_rectangle(frame, (label_tl_x, label_tl_y),
                                       (label_br_x, label_br_y),
                                       TEXT_BG_COLOR, thickness=-1, radius=CORNER_RADIUS)
        frame = draw_chinese_text(frame, label_text,
                                  (label_tl_x + PADDING, label_tl_y + PADDING),
                                  font_size=FONT_SCALE, color=TEXT_COLOR, thickness=FONT_THICKNESS)
    return frame

def draw_ocr_results(frame, ocr_results):
    h, w = frame.shape[:2]
    for ocr_res in ocr_results:
        text = ocr_res["text"]
        conf = ocr_res["conf"]
        text_x1, text_y1, text_x2, text_y2 = ocr_res["bbox"]
        ship_x1, ship_y1, ship_x2, ship_y2 = ocr_res["ship_bbox"]

        display_text = f"{text}({conf:.2f})"
        text_w, text_h = get_text_size(display_text, FONT_SCALE)

        label_tl_x = int(text_x1)
        label_tl_y = int(text_y1 - text_h - 2 * PADDING - LABEL_OFFSET)
        label_br_x = int(label_tl_x + text_w + 2 * PADDING)
        label_br_y = int(label_tl_y + text_h + 2 * PADDING)

        if label_br_x > w:
            label_tl_x = max(0, int(w - text_w - 2 * PADDING - 10))
            label_br_x = int(label_tl_x + text_w + 2 * PADDING)
        if label_br_y > h:
            label_tl_y = max(0, int(h - text_h - 2 * PADDING - 10))
            label_br_y = int(label_tl_y + text_h + 2 * PADDING)
        if label_tl_y < 0:
            label_tl_y = int(ship_y2 + LABEL_OFFSET)
            label_br_y = int(label_tl_y + text_h + 2 * PADDING)

        frame = draw_rounded_rectangle(frame, (label_tl_x, label_tl_y),
                                       (label_br_x, label_br_y),
                                       OCR_BG_COLOR, thickness=-1, radius=CORNER_RADIUS)
        frame = draw_chinese_text(frame, display_text,
                                  (label_tl_x + PADDING, label_tl_y + PADDING),
                                  font_size=FONT_SCALE, color=OCR_TEXT_COLOR,
                                  thickness=FONT_THICKNESS)

        cv2.rectangle(frame, (int(text_x1), int(text_y1)),
                      (int(text_x2), int(text_y2)), OCR_BOX_COLOR, max(1, BOX_THICKNESS // 2))
    return frame

# ======================== OCR 工具函数 ========================
def is_valid_text(text, min_length=2, min_chinese_count=1, min_digit_count=1):
    if not text or not isinstance(text, str):
        return False
    text = text.strip()
    if len(text) < min_length:
        return False

    meaningless_patterns = [
        '-', '.', ',', ' ', '', 'ane', 'one', 'the', 'and', 'or',
        '。', '，', '、', '；', '：', '？', '！',
        '一', '二', '三', '四', '五', '六', '七', '八', '九', '十',
    ]
    if text in meaningless_patterns:
        return False
    if re.match(r'^[\W_]+$', text):
        return False

    chinese_count = sum(1 for c in text if '一' <= c <= '鿿')
    digit_count = sum(1 for c in text if c.isdigit())
    letter_count = sum(1 for c in text if c.isalpha() and not ('一' <= c <= '鿿'))

    if letter_count > 0 and chinese_count == 0 and digit_count == 0:
        return False
    if len(text) == 1 and text.isalpha():
        return False

    valid_combinations = [
        chinese_count > 0 and digit_count > 0,
        chinese_count > 0 and letter_count > 0,
        chinese_count > 0 and digit_count > 0 and letter_count > 0,
        digit_count > 0 and len(text) >= 4,
        chinese_count > 0 and chinese_count >= 2,
    ]
    if any(valid_combinations):
        return True
    if digit_count / max(len(text), 1) > 0.6 and len(text) >= 3:
        return True
    return False

# ======================== RKNN 模型封装 ========================
from rknnlite.api import RKNNLite

class RKNNModel:
    """通用 RKNN 模型加载与推理"""
    def __init__(self, model_path):
        self.rknn_lite = RKNNLite()
        ret = self.rknn_lite.load_rknn(model_path)
        if ret != 0:
            print(f'❌ 加载模型失败: {model_path}')
            exit(ret)
        ret = self.rknn_lite.init_runtime()
        if ret != 0:
            print('❌ 初始化 runtime 失败!')
            exit(ret)
        print(f"✅ 模型加载成功: {model_path}")

    def run(self, inputs, data_format=None):
        return self.rknn_lite.inference(inputs=inputs, data_format=data_format)


class YOLO_RKNN:
    """简化版 YOLO RKNN 封装（仅推理）"""
    def __init__(self, model_path):
        self.rknn = RKNNModel(model_path)

    def run(self, inputs, data_format=None):
        return self.rknn.run(inputs, data_format=data_format)


class YOLOv11RKNN:
    """YOLOv11 RKNN 检测/分类模型"""
    CONF_THRESH = 0.45
    IOU_THRESH = 0.25

    def __init__(self, model_path, model_type="det", input_size=640, names=None,
                 conf_thresh=None, iou_thresh=None):
        self.model_path = model_path
        self.model_type = model_type
        self.input_size = input_size
        self.rknn = YOLO_RKNN(model_path)

        if conf_thresh is not None:
            self.CONF_THRESH = conf_thresh
        if iou_thresh is not None:
            self.IOU_THRESH = iou_thresh

        if names is not None:
            self.names = names
        else:
            if self.model_type == "cls":
                self.names = {0: "cloudy", 1: "foggy", 2: "night", 3: "rainy", 4: "sunny"}
            else:
                self.names = {0: "ship"}

        print(f"✅ RKNN 模型类型={model_type}, 尺寸={input_size}, 类别={self.names}")

    # ── 预处理 ──
    def _letterbox(self, img, new_size):
        h0, w0 = img.shape[:2]
        r = min(new_size / h0, new_size / w0)
        new_w = int(round(w0 * r))
        new_h = int(round(h0 * r))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        dw = (new_size - new_w) / 2
        dh = (new_size - new_h) / 2
        top = int(round(dh - 0.1))
        bottom = int(round(dh + 0.1))
        left = int(round(dw - 0.1))
        right = int(round(dw + 0.1))
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return padded, r, (dw, dh)

    def _preprocess_det(self, img):
        padded, r, (dw, dh) = self._letterbox(img, self.input_size)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        blob = np.expand_dims(rgb, axis=0)  # (1, H, W, 3) NHWC
        return blob, r, (dw, dh)

    def _preprocess_cls(self, img):
        h0, w0 = img.shape[:2]
        r = self.input_size / min(h0, w0)
        new_w = int(round(w0 * r))
        new_h = int(round(h0 * r))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        cy, cx = new_h // 2, new_w // 2
        half = self.input_size // 2
        cropped = resized[
            max(0, cy - half):cy - half + self.input_size,
            max(0, cx - half):cx - half + self.input_size
        ]
        if cropped.shape[:2] != (self.input_size, self.input_size):
            cropped = cv2.resize(cropped, (self.input_size, self.input_size))
        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        chw = rgb.transpose(2, 0, 1)
        blob = np.expand_dims(chw.astype(np.uint8), axis=0)
        return blob

    # ── 后处理 ──
    @staticmethod
    def _softmax(x, axis=None):
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    @staticmethod
    def _dfl_numpy(position):
        n, c, h, w = position.shape
        p_num = 4
        mc = c // p_num
        y = position.reshape(n, p_num, mc, h, w)
        y = YOLOv11RKNN._softmax(y, axis=2)
        acc_matrix = np.arange(mc).reshape(1, 1, mc, 1, 1).astype(float)
        y = (y * acc_matrix).sum(2)
        return y

    @staticmethod
    def _box_process(position, grid_h, grid_w, input_size=640):
        col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
        col = col.reshape(1, 1, grid_h, grid_w)
        row = row.reshape(1, 1, grid_h, grid_w)
        grid = np.concatenate((col, row), axis=1)
        stride = np.array([input_size // grid_h, input_size // grid_w]).reshape(1, 2, 1, 1)
        position = YOLOv11RKNN._dfl_numpy(position)
        box_xy = grid + 0.5 - position[:, 0:2, :, :]
        box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
        xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)
        return xyxy

    def _postprocess_det(self, outputs, orig_shape, r, dw, dh):
        if outputs is None or len(outputs) == 0:
            return []

        layers = {}
        for out in outputs:
            h, w = out.shape[2:4]
            if (h, w) not in layers:
                layers[(h, w)] = []
            layers[(h, w)].append(out)

        all_boxes, all_scores, all_class_ids = [], [], []
        num_classes = len(self.names)

        for (h, w), tensors in layers.items():
            box_t = score_t = obj_t = None
            for t in tensors:
                c = t.shape[1]
                if c == 64:
                    box_t = t
                elif c == num_classes:
                    score_t = t
                elif c == 1:
                    obj_t = t

            if box_t is None or score_t is None:
                continue

            s = score_t if obj_t is None else score_t * obj_t
            s = s.transpose(0, 2, 3, 1).reshape(-1, num_classes)
            b = self._box_process(box_t, h, w, self.input_size).transpose(0, 2, 3, 1).reshape(-1, 4)

            max_scores = np.max(s, axis=1)
            max_class_ids = np.argmax(s, axis=1)

            idx = max_scores > 0.05
            all_boxes.append(b[idx])
            all_scores.append(max_scores[idx])
            all_class_ids.append(max_class_ids[idx])

        if not all_boxes:
            return []

        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        class_ids = np.concatenate(all_class_ids, axis=0)

        nms_boxes = []
        for box in boxes:
            nms_boxes.append([box[0], box[1], box[2] - box[0], box[3] - box[1]])

        indices = cv2.dnn.NMSBoxes(nms_boxes, scores.tolist(), self.CONF_THRESH, self.IOU_THRESH)
        if len(indices) == 0:
            return []

        if isinstance(indices, np.ndarray):
            indices = indices.flatten()

        final_boxes = boxes[indices]
        final_scores = scores[indices]
        final_cls = class_ids[indices]

        orig_h, orig_w = orig_shape
        results = []
        for box, score, cls_id in zip(final_boxes, final_scores, final_cls):
            x1 = (box[0] - dw) / r
            y1 = (box[1] - dh) / r
            x2 = (box[2] - dw) / r
            y2 = (box[3] - dh) / r
            x1n = np.clip(x1 / orig_w, 0.0, 1.0)
            y1n = np.clip(y1 / orig_h, 0.0, 1.0)
            x2n = np.clip(x2 / orig_w, 0.0, 1.0)
            y2n = np.clip(y2 / orig_h, 0.0, 1.0)
            results.append((x1n, y1n, x2n, y2n, float(score), float(cls_id)))
        return results

    def _postprocess_cls(self, outputs):
        probs = outputs[0]
        if probs.ndim == 2:
            probs = probs[0]
        top1_idx = int(np.argmax(probs))
        top1_conf = float(probs[top1_idx])

        # foggy / night 混淆修正
        if top1_idx == 1:
            night_prob = float(probs[2]) if len(probs) > 2 else 0.0
            if night_prob > 0.3 and (top1_conf - night_prob) < 0.3:
                top1_idx = 2
                top1_conf = night_prob
                print("———————————————————修复为night————————————————————")
        return top1_idx, top1_conf

    # ── 统一推理入口 ──
    def predict(self, frame):
        shape = frame.shape[:2]
        if self.model_type == "cls":
            blob = self._preprocess_cls(frame)
            out = self.rknn.run([blob])
            idx, conf = self._postprocess_cls(out)
            return [("cls", idx, conf, self.names)]
        else:
            blob, r, (dw, dh) = self._preprocess_det(frame)
            out = self.rknn.run([blob], data_format='nhwc')
            dets = self._postprocess_det(out, shape, r, dw, dh)
            return [("det", dets, self.names)]


# ======================== OCR RKNN 引擎 ========================
# 导入 OCR 后处理工具（沿用原项目 utils）
import utils.operators
from utils.db_postprocess import DBPostProcess, DetPostProcess
from utils.rec_postprocess import CTCLabelDecode

# OCR 预处理/后处理配置
DET_PRE_PROCESS_CONFIG = [
    {'DetResizeForTest': {'image_shape': OCR_DET_INPUT_SHAPE}},
    {'NormalizeImage': {'std': [1., 1., 1.], 'mean': [0., 0., 0.], 'scale': '1.', 'order': 'hwc'}}
]
DET_POSTPROCESS_CONFIG = {
    'DBPostProcess': {
        'thresh': 0.3, 'box_thresh': 0.6, 'max_candidates': 1000,
        'unclip_ratio': 1.5, 'use_dilation': False, 'score_mode': 'fast',
    }
}
REC_PRE_PROCESS_CONFIG = [
    {'NormalizeImage': {'std': [1, 1, 1], 'mean': [0, 0, 0], 'scale': '1./255.', 'order': 'hwc'}}
]
REC_POSTPROCESS_CONFIG = {
    'CTCLabelDecode': {"character_dict_path": OCR_DICT_PATH, "use_space_char": True}
}


class RKNNOCREngine:
    """PaddleOCR RKNN 引擎：检测 + 识别"""

    def __init__(self, det_model_path=None, rec_model_path=None, dict_path=None):
        det_path = det_model_path or OCR_DET_MODEL_PATH
        rec_path = rec_model_path or OCR_REC_MODEL_PATH
        dict_path = dict_path or OCR_DICT_PATH

        # 更新字典路径
        REC_POSTPROCESS_CONFIG['CTCLabelDecode']['character_dict_path'] = dict_path

        # 检测模型
        self.det_model = RKNNModel(det_path)
        self.det_preprocess = []
        for item in DET_PRE_PROCESS_CONFIG:
            for key in item:
                pclass = getattr(utils.operators, key)
                p = pclass(**item[key])
                self.det_preprocess.append(p)
        self.db_postprocess = DBPostProcess(**DET_POSTPROCESS_CONFIG['DBPostProcess'])
        self.det_postprocess = DetPostProcess()

        # 识别模型
        self.rec_model = RKNNModel(rec_path)
        self.rec_preprocess = []
        for item in REC_PRE_PROCESS_CONFIG:
            for key in item:
                pclass = getattr(utils.operators, key)
                p = pclass(**item[key])
                self.rec_preprocess.append(p)
        self.ctc_postprocess = CTCLabelDecode(**REC_POSTPROCESS_CONFIG['CTCLabelDecode'])

        print("✅ RKNN OCR 引擎初始化完成")

    def predict(self, img):
        """
        输入: 单张图片 (ROI)
        输出: [{'rec_texts': [...], 'rec_scores': [...], 'rec_polys': [...]}]
        """
        rec_texts, rec_scores, rec_polys = [], [], []
        try:
            # 1. 文字检测
            data = {'image': img}
            for p in self.det_preprocess:
                data = p(data)
            img_input = data['image'][np.newaxis, :]
            output = self.det_model.run([img_input])
            preds = {'maps': output[0].astype(np.float32)}
            result = self.db_postprocess(preds, data['shape'])
            text_boxes = self.det_postprocess.filter_tag_det_res(result[0]['points'], img.shape)
            text_boxes = sorted(text_boxes, key=lambda x: (np.min(x[:, 1]), np.min(x[:, 0])))

            # 2. 逐个文字框识别
            for box in text_boxes:
                bx1, by1 = int(np.min(box[:, 0])), int(np.min(box[:, 1]))
                bx2, by2 = int(np.max(box[:, 0])), int(np.max(box[:, 1]))
                text_crop = img[by1:by2, bx1:bx2]
                if text_crop.size == 0:
                    continue

                text_crop = cv2.resize(text_crop, (OCR_REC_INPUT_SHAPE[1], OCR_REC_INPUT_SHAPE[0]))
                data_rec = {'image': text_crop}
                for p in self.rec_preprocess:
                    data_rec = p(data_rec)
                img_input_rec = data_rec['image'][np.newaxis, :]
                output_rec = self.rec_model.run([img_input_rec])
                text, score = self.ctc_postprocess(output_rec[0])[0]

                rec_texts.append(text)
                rec_scores.append(score)
                rec_polys.append(box.tolist())

        except Exception as e:
            print(f"⚠️ RKNN OCR 推理异常: {e}")

        return [{'rec_texts': rec_texts, 'rec_scores': rec_scores, 'rec_polys': rec_polys}]


# ======================== 核心业务函数 ========================
# 全局模型实例（延迟初始化）
_cls_model = None
_det_day_model = None
_det_night_model = None
_ocr_engine = None
_ocr_enabled = True

# 天气分类缓存
weather_cache = {}

def _get_models(conf_thresh=None, iou_thresh=None, no_ocr=False):
    """延迟初始化模型（仅首次调用时加载）"""
    global _cls_model, _det_day_model, _det_night_model, _ocr_engine, _ocr_enabled
    _ocr_enabled = not no_ocr

    if _cls_model is None:
        _cls_model = YOLOv11RKNN(
            CLS_MODEL_PATH, model_type="cls", input_size=CLS_INPUT_SIZE,
            names={0: "cloudy", 1: "foggy", 2: "night", 3: "rainy", 4: "sunny"}
        )
    if _det_day_model is None:
        _det_day_model = YOLOv11RKNN(
            DET_DAY_MODEL_PATH, model_type="det", input_size=DET_INPUT_SIZE,
            names={0: "ship"}, conf_thresh=conf_thresh, iou_thresh=iou_thresh
        )
    if _det_night_model is None:
        _det_night_model = YOLOv11RKNN(
            DET_NIGHT_MODEL_PATH, model_type="det", input_size=DET_INPUT_SIZE,
            names={0: "light"}, conf_thresh=conf_thresh, iou_thresh=iou_thresh
        )
    if _ocr_engine is None and _ocr_enabled:
        try:
            _ocr_engine = RKNNOCREngine()
        except Exception as e:
            print(f"⚠️ OCR 引擎初始化失败: {e}，已禁用 OCR")
            _ocr_enabled = False

    return _cls_model, _det_day_model, _det_night_model, _ocr_engine


def get_weather_class(frame):
    """场景分类"""
    results = _cls_model.predict(frame)
    _, idx, conf, names = results[0]
    cls_label = names.get(idx, str(idx))
    print(f"分类结果：{cls_label} | 置信度：{conf:.3f}")
    return cls_label, conf


def get_weather_class_cached(frame, camera_id, frame_idx):
    """带缓存的场景分类"""
    cache = weather_cache.get(camera_id)
    if cache is not None and frame_idx % CLS_INTERVAL != 0:
        return cache
    result = get_weather_class(frame)
    weather_cache[camera_id] = result
    return result


def detect_target(frame, weather_cls):
    """目标检测"""
    det_model = _det_night_model if weather_cls == "night" else _det_day_model
    results = det_model.predict(frame)
    _, dets, names = results[0]

    det_results = []
    for (x1n, y1n, x2n, y2n, conf, cls_id) in dets:
        det_results.append((x1n, y1n, x2n, y2n, conf, cls_id))
    return det_results


def postprocess_detections(det_results, orig_frame_shape, weather_cls,
                           ship_alarm_thresh=None, light_alarm_thresh=None):
    """后处理：归一化坐标转像素坐标 + 告警判定"""
    if ship_alarm_thresh is None:
        ship_alarm_thresh = SHIP_ALARM_THRESH
    if light_alarm_thresh is None:
        light_alarm_thresh = LIGHT_ALARM_THRESH

    h, w = orig_frame_shape
    post_results = []
    has_alarm = False

    for (x1n, y1n, x2n, y2n, conf, cls_id) in det_results:
        x1 = int(x1n * w)
        y1 = int(y1n * h)
        x2 = int(x2n * w)
        y2 = int(y2n * h)
        area = max(0, x2 - x1) * max(0, y2 - y1)
        total_area = w * h
        area_ratio = area / total_area if total_area > 0 else 0.0

        if weather_cls == "night":
            is_alarm = area_ratio >= light_alarm_thresh
        else:
            is_alarm = area_ratio >= ship_alarm_thresh

        if is_alarm:
            has_alarm = True

        post_results.append((x1, y1, x2, y2, conf, cls_id, area, is_alarm))
    return post_results, has_alarm


def process_ship_ocr(frame, post_results, weather_label, camera_id=0):
    """对告警船只 ROI 进行 OCR 识别"""
    ocr_results = []
    if not _ocr_enabled or _ocr_engine is None or weather_label in ["night", "夜晚"]:
        return ocr_results

    alarm_ship_rois = [res for res in post_results if len(res) >= 8 and res[7]]
    if len(alarm_ship_rois) == 0:
        return ocr_results

    h, w = frame.shape[:2]

    for roi_res in alarm_ship_rois:
        if len(roi_res) < 8:
            continue
        x1, y1, x2, y2, conf, cls, area, is_alarm = roi_res

        roi_x1 = max(0, x1 - 5)
        roi_y1 = max(0, y1 - 5)
        roi_x2 = min(w, x2 + 5)
        roi_y2 = min(h, y2 + 5)
        if (roi_x2 - roi_x1) < 10 or (roi_y2 - roi_y1) < 10:
            continue

        ship_roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        if ship_roi.size == 0:
            continue

        try:
            result = _ocr_engine.predict(ship_roi)
            if not (isinstance(result, list) and len(result) > 0):
                continue
            first_result = result[0]
            if not isinstance(first_result, dict):
                continue

            rec_texts = first_result.get('rec_texts', [])
            rec_scores = first_result.get('rec_scores', [])
            rec_polys = first_result.get('rec_polys', [])
        except Exception as e:
            print(f"⚠️ OCR 推理异常（ROI区域）：{str(e)}")
            continue

        if len(rec_texts) == 0:
            continue

        # 筛选最佳结果
        best_text, best_score, best_poly = None, 0, None
        for idx in range(len(rec_texts)):
            text = rec_texts[idx]
            score = float(rec_scores[idx]) if idx < len(rec_scores) else 0
            poly = rec_polys[idx] if idx < len(rec_polys) else []

            if score < OCR_CONF_THRESH:
                continue
            if not is_valid_text(text):
                continue
            if score > best_score:
                best_text, best_score, best_poly = text, score, poly

        if best_text:
            print(f"✅ OCR 识别: '{best_text}' (置信度: {best_score:.3f})")

            # 计算 OCR 文本在原图上的坐标
            try:
                if best_poly is not None and len(best_poly) >= 4:
                    poly_flat = [coord for point in best_poly for coord in point]
                    text_x1 = min(poly_flat[::2]) + roi_x1
                    text_y1 = min(poly_flat[1::2]) + roi_y1
                    text_x2 = max(poly_flat[::2]) + roi_x1
                    text_y2 = max(poly_flat[1::2]) + roi_y1
                else:
                    text_x1, text_y1, text_x2, text_y2 = x1, y1, x2, y2
            except Exception:
                text_x1, text_y1, text_x2, text_y2 = x1, y1, x2, y2

            text_x1 = max(0, text_x1)
            text_y1 = max(0, text_y1)
            text_x2 = min(w, text_x2)
            text_y2 = min(h, text_y2)

            ocr_results.append({
                "text": best_text,
                "conf": best_score,
                "bbox": (text_x1, text_y1, text_x2, text_y2),
                "ship_bbox": (x1, y1, x2, y2),
                "camera_id": camera_id
            })

    return ocr_results


# ======================== 单帧处理流水线 ========================
def process_frame(frame, frame_idx=0, camera_id=0, do_ocr=True):
    """完整的单帧处理：分类 → 检测 → 后处理 → OCR → 绘制"""
    global _ocr_enabled
    _ocr_enabled = do_ocr

    # 场景分类
    weather_label, weather_conf = get_weather_class_cached(frame, camera_id, frame_idx)
    weather_info = (weather_label, weather_conf)

    # 目标检测
    det_results = detect_target(frame, weather_label)

    # 后处理
    post_results, has_alarm = postprocess_detections(det_results, frame.shape[:2], weather_label)

    # OCR
    ocr_results = []
    if do_ocr and has_alarm and _ocr_engine is not None:
        ocr_results = process_ship_ocr(frame, post_results, weather_label, camera_id)

    # 绘制
    model_names = _det_night_model.names if weather_label == "night" else _det_day_model.names
    draw_frame = draw_custom_boxes(frame.copy(), post_results, weather_info, model_names)
    draw_frame = draw_ocr_results(draw_frame, ocr_results)

    return draw_frame, post_results, ocr_results, weather_info


# ======================== 主程序 ========================
def process_image(input_path, output_path, conf_thresh, iou_thresh, no_ocr, no_display):
    """处理单张图片"""
    print(f"\n📷 处理图片: {input_path}")
    frame = cv2.imread(input_path)
    if frame is None:
        print(f"❌ 无法读取图片: {input_path}")
        return

    _get_models(conf_thresh, iou_thresh, no_ocr)

    draw_frame, post_results, ocr_results, weather_info = process_frame(
        frame, frame_idx=0, camera_id=0, do_ocr=not no_ocr
    )

    # 打印结果
    print(f"\n📊 检测结果：{len(post_results)} 个目标")
    for (x1, y1, x2, y2, conf, cls, area, is_alarm) in post_results:
        status = "🚨 ALARM" if is_alarm else "✓"
        print(f"  {status} [{x1},{y1},{x2},{y2}] area={area} conf={conf:.2f}")

    if ocr_results:
        print(f"\n🔤 OCR 结果：{len(ocr_results)} 个识别")
        for r in ocr_results:
            print(f"  舷号: {r['text']} (置信度: {r['conf']:.3f})")

    # 保存
    print(f"\n💾 保存结果: {output_path}")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    cv2.imwrite(output_path, draw_frame)
    print(f"✅ 已保存到 {output_path}")

    # 显示
    if not no_display:
        cv2.imshow("Detection Result", draw_frame)
        print("按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_video(input_path, output_path, conf_thresh, iou_thresh, no_ocr, no_display,
                  save_frames=False):
    """处理视频"""
    print(f"\n🎬 处理视频: {input_path}")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {input_path}")
        return

    _get_models(conf_thresh, iou_thresh, no_ocr)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        fps = 25.0
    print(f"  分辨率: {width}x{height}, FPS: {fps:.1f}, 总帧数: {total_frames}")

    # 输出
    writer = None
    frames_dir = None
    if save_frames:
        frames_dir = output_path
        os.makedirs(frames_dir, exist_ok=True)
        print(f"  保存帧到目录: {frames_dir}")
    else:
        # 确保输出目录存在
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"  输出视频: {output_path}")

    frame_idx = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        draw_frame, post_results, ocr_results, weather_info = process_frame(
            frame, frame_idx=frame_idx, camera_id=0, do_ocr=not no_ocr
        )

        # 保存
        if save_frames and frames_dir:
            frame_path = os.path.join(frames_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(frame_path, draw_frame)
        elif writer:
            writer.write(draw_frame)

        # 显示
        if not no_display:
            cv2.imshow("Detection", draw_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                print("用户手动退出")
                break

        frame_idx += 1
        if frame_idx % 30 == 0:
            elapsed = time.time() - start_time
            current_fps = frame_idx / elapsed if elapsed > 0 else 0
            print(f"  已处理 {frame_idx}/{total_frames} 帧, {current_fps:.1f} fps")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    print(f"\n✅ 处理完成: {frame_idx} 帧, 耗时 {elapsed:.1f}s, 平均 {frame_idx/elapsed:.1f} fps")
    if not save_frames:
        print(f"💾 视频已保存到: {output_path}")
    else:
        print(f"💾 {frame_idx} 帧已保存到: {frames_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='独立船舶检测+OCR+绘制脚本 — 支持本地图片/视频',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python standalone_detect.py --input ./test.jpg
  python standalone_detect.py --input ./test.jpg --output ./result.jpg
  python standalone_detect.py --input ./test.mp4 --output ./result.mp4
  python standalone_detect.py --input ./test.mp4 --output ./frames/ --save-frames
  python standalone_detect.py --input ./test.mp4 --no-ocr --no-display
  python standalone_detect.py --input ./test.jpg --conf 0.3 --iou 0.5
        """
    )

    parser.add_argument('--input', '-i', type=str, default="/home/Tronlong/Desktop/pipeline/test/alarm_test2.png",
                        help='输入图片或视频路径')
    parser.add_argument('--output', '-o', type=str, default="/home/Tronlong/Desktop/pipeline/output/alarm_test2_result.png",
                        help='输出路径（图片/视频/目录）。默认：图片输出为 input_result.jpg，视频输出为 input_result.mp4')
    parser.add_argument('--conf', type=float, default=0.45,
                        help='检测置信度阈值 (默认 0.45)')
    parser.add_argument('--iou', type=float, default=0.25,
                        help='NMS IoU 阈值 (默认 0.25)')
    parser.add_argument('--no-ocr', action='store_true',
                        help='禁用 OCR 舷号识别')
    parser.add_argument('--no-display', action='store_true',
                        help='不显示 GUI 窗口')
    parser.add_argument('--save-frames', action='store_true',
                        help='视频处理时保存为帧图片而非视频文件')

    args = parser.parse_args()

    # 检查输入
    if not os.path.exists(args.input):
        print(f"❌ 输入文件不存在: {args.input}")
        sys.exit(1)

    # 判断输入类型
    ext = os.path.splitext(args.input)[1].lower()
    is_video = ext in ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm')

    # 自动生成输出路径
    if args.output is None:
        base = os.path.splitext(args.input)[0]
        if is_video:
            args.output = f"{base}_result.mp4"
        else:
            args.output = f"{base}_result.jpg"

    if is_video:
        process_video(args.input, args.output, args.conf, args.iou,
                      args.no_ocr, args.no_display, args.save_frames)
    else:
        process_image(args.input, args.output, args.conf, args.iou,
                      args.no_ocr, args.no_display)
