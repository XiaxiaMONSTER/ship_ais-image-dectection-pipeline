#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""本地视频单次逐帧推理：保留原尺寸/帧时间戳/时长，每个输入仅输出一个MP4。

用法：python3 23-2-fourth_ais.py --videos /path/a.mp4 /path/b.mp4
使用原有 RKNN/OpenCV/numpy/OCR utils 环境，另需 PyAV（python3 -m pip install av）。
不会导入 third 脚本，不开启串口、网络发送、SD备份、定时拍照或告警素材归档。
"""
import argparse
import hashlib
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import unicodedata
from datetime import datetime
from fractions import Fraction

for _thread_var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_thread_var, '1')

# 可直接填入一个或多个本地文件；命令行 --videos 优先。不扫描目录、不读取摄像头。
VIDEO_PATHS = [
    #"/root/Desktop/230-60m.mp4",  #场景长视频
    #"/root/Desktop/pipeline/test/多云-近处小船（P船号）.mp4",     #多云  
    #"/root/Desktop/140-230m.mp4",  #场景长视频
    "/root/Desktop/pipeline/test/夜晚-带月亮.mp4",   #夜晚  不正常，识别成foggy
]
OUTPUT_ROOT = '/root/Desktop/local_video_results'
VERBOSE = False
_ft_handle = None
CHINESE_FONT_PATH = None
weather_cache = {}
_ocr_watermark_regions = {}
cls_model = det_day_model = det_night_model = ocr_engine = None
CLS_MODEL_PATH = "/root/Desktop/rknn_model_zoo_and_models/yolov11n-cls/cls_2-best_255_fp.rknn"
DET_DAY_MODEL_PATH = "/root/Desktop/rknn_model_zoo_and_models/yolov11n-day/day_8.5_tiny-best_255_fp.rknn"
DET_NIGHT_MODEL_PATH = "/root/Desktop/rknn_model_zoo_and_models/yolov11n-night/night_2-best_255_i8.rknn"
ENABLE_NIGHT_MODEL = True    #是否夜晚启用
CLS_INTERVAL = 5
CLS_INPUT_SIZE = 256
DET_INPUT_SIZE = 640

SHIP_AREA_SCALE = 2.0 / 3.0

SHIP_ALARM_THRESH = 0.00185 * SHIP_AREA_SCALE

LIGHT_ALARM_THRESH = 0.0004

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

ENABLE_OCR = True

OCR_CONF_THRESH = 0.5

DET_MODEL_PATH = "/root/Desktop/rknn_model_zoo_and_models/PPOCR_rknn/det/ppocrv4_det.rknn"

REC_MODEL_PATH = "/root/Desktop/rknn_model_zoo_and_models/PPOCR_rknn/rec/ppocrv4_rec.rknn"

DICT_PATH = "/root/Desktop/rknn_model_zoo_and_models/PPOCR_rknn/ppocr_keys_v1.txt"

DET_INPUT_SHAPE = [480, 480]

DET_PRE_PROCESS_CONFIG = [
    {'DetResizeForTest': {'image_shape': DET_INPUT_SHAPE}},
    {
        'NormalizeImage': {
            'std': [1., 1., 1.],
            'mean': [0., 0., 0.],
            'scale': '1.',
            'order': 'hwc'
        }
    }
]

DET_POSTPROCESS_CONFIG = {
    'DBPostProcess': {
        'thresh': 0.3,
        'box_thresh': 0.6,
        'max_candidates': 1000,
        'unclip_ratio': 1.5,
        'use_dilation': False,
        'score_mode': 'fast',
    }
}

REC_INPUT_SHAPE = [48, 320]

REC_PRE_PROCESS_CONFIG = [
    {
        'NormalizeImage': {
            'std': [1, 1, 1],
            'mean': [0, 0, 0],
            'scale': '1./255.',
            'order': 'hwc'
        }
    }
]

REC_POSTPROCESS_CONFIG = {
    'CTCLabelDecode': {
        "character_dict_path": DICT_PATH,
        "use_space_char": True
    }
}

CAM_HFOV = 93.1

SHIP_LENGTH = 10.0

SHIP_DISTANCE_COEFFICIENT = (15.0 / SHIP_LENGTH) * math.sqrt(SHIP_AREA_SCALE)

NIGHT_TARGET_LENGTH = 15.0

OCR_WATERMARK_CACHE_SECONDS = 300.0

def _get_ft():
    global _ft_handle, CHINESE_FONT_PATH
    if _ft_handle is None and CHINESE_FONT_PATH and hasattr(cv2, "freetype"):
        _ft_handle = cv2.freetype.createFreeType2()
        _ft_handle.loadFontData(fontFileName=CHINESE_FONT_PATH, id=0)
    return _ft_handle

def _needs_freetype(c):
    """True if cv2.putText Hershey fonts can't render this character."""
    cp = ord(c)
    return not (0x20 <= cp <= 0x7E)

def _put_mixed_text(img, text, org, font_height, color, thickness):
    """Draw `text` in-place on `img`. ASCII→putText, CJK/symbols→freetype."""
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
    """Return (width, height) of mixed text, for layout calculations."""
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

def estimate_distance_by_bbox(bbox_width_px, image_width_px, use_night_detector=False):
    """单目测距：根据检测框像素宽度 + 固定船长，估算船只到摄像头直线距离（米）。
    原理：D = L * f / w * k，焦距 f = (W/2) / tan(HFOV/2)。
    白天k约1.224745：同宽高比下，面积阈值缩为2/3后仍对应原距离。
    同一个未缩小的检测框，其白天估算距离会变为原来的sqrt(2/3)。
    夜间灯光模型仍用原版15米标尺，不应用白天补偿。
    分辨率自适应：image_width_px 传入当前帧实际宽度，动态计算焦距。"""
    if bbox_width_px is None or bbox_width_px <= 0 or image_width_px <= 0:
        return None
    focal = (image_width_px / 2.0) / np.tan(np.radians(CAM_HFOV / 2.0))
    effective_length = (NIGHT_TARGET_LENGTH if use_night_detector
                        else SHIP_LENGTH * SHIP_DISTANCE_COEFFICIENT)
    return effective_length * focal / bbox_width_px

def _normalize_ocr_watermark_text(text):
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text).strip()
    text = text.translate(str.maketrans({"。": ".", "–": "-", "—": "-"}))
    return re.sub(r"\s+", " ", text)

def is_ocr_camera_watermark(text):
    """过滤 Camera01~04 及接近 camera 的 OCR 拼写；只匹配完整标签。"""
    value = _normalize_ocr_watermark_text(text).casefold()
    # 兼容全角、大小写、空格、标签外括号以及常见编号分隔符。
    value = value.strip("[](){}<>【】（）")
    value = re.sub(r"[\s_.:\-]+", "", value)
    match = re.fullmatch(r"([a-z]{4,8}?)([0o]?[1-4il])", value)
    if match is None:
        return False
    label, channel = match.groups()
    channel = channel.translate(str.maketrans({"o": "0", "i": "1", "l": "1"}))
    if channel not in {"1", "2", "3", "4", "01", "02", "03", "04"}:
        return False
    # m 常被拆成 rn；其余允许一次字符替换、漏字或多字，例如 tamera/camra。
    label = label.replace("rn", "m")
    expected = "camera"
    if abs(len(label) - len(expected)) > 1:
        return False
    previous = list(range(len(expected) + 1))
    for row, char in enumerate(label, 1):
        current = [row]
        for column, target in enumerate(expected, 1):
            current.append(min(current[-1] + 1, previous[column] + 1,
                               previous[column - 1] + (char != target)))
        previous = current
    return previous[-1] <= 1

def is_ocr_time_watermark(text):
    """识别明确日期/时间；不把任意纯数字船号当作时间。"""
    value = _normalize_ocr_watermark_text(text)
    if not value:
        return False
    # O/0、I/l/1 混淆只用于水印判断，不改动返回的船号文字。
    value = value.translate(str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1"}))
    # 日期本身足以确定水印，后面的时间允许冒号漏识别/误识别。
    if re.search(r"(?<!\d)(?:19|20)\d{2}\s*[-/.年]\s*(?:0?[1-9]|1[0-2])"
                 r"\s*[-/.月]\s*(?:0?[1-9]|[12]\d|3[01])(?:日|(?!\d))", value):
        return True
    value = value.replace(" ", "")
    # 部分摄像头日期与时间之间无空格，或只显示时分。
    if re.fullmatch(r"(?:19|20)\d{2}[-/.](?:0[1-9]|1[0-2])[-/.](?:0[1-9]|[12]\d|3[01])"
                    r"T?(?:[01]\d|2[0-3])[:8B]?[0-5]\d[:8B]?[0-5]\d", value):
        return True
    if re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", value):
        return True
    # 完整时分秒；支持一个冒号被识别为 8、B 或丢失，至少保留一个分隔符。
    hour, minute, second = r"(?:[01]?\d|2[0-3])", r"[0-5]\d", r"[0-5]\d"
    clock = hour + r"[:时]" + minute + r"(?:[:分8B]|)" + second
    clock_alt = hour + r"(?:[8B]|)" + minute + r"[:分]" + second
    if re.fullmatch(r"(?:" + clock + "|" + clock_alt + r")(?:秒|[.,]\d{1,3})?", value):
        return True
    # 连写的完整年月日时分秒需通过日历校验；6~8位纯数字保留给位置判断。
    if re.fullmatch(r"(?:19|20)\d{12}", value):
        try:
            datetime.strptime(value, "%Y%m%d%H%M%S")
            return True
        except ValueError:
            pass
    return False

def _ocr_watermark_bbox(poly, roi_x1, roi_y1, frame_width, frame_height):
    """把 ROI 内 OCR 框转换成原图归一化坐标，兼容列表和 ndarray。"""
    try:
        if poly is None or len(poly) < 4 or frame_width <= 0 or frame_height <= 0:
            return None
        xs = [(float(point[0]) + roi_x1) / frame_width for point in poly]
        ys = [(float(point[1]) + roi_y1) / frame_height for point in poly]
        box = (max(0.0, min(xs)), max(0.0, min(ys)),
               min(1.0, max(xs)), min(1.0, max(ys)))
        if not all(math.isfinite(v) for v in box):
            return None
        return box if box[2] > box[0] and box[3] > box[1] else None
    except (TypeError, ValueError, IndexError, OverflowError):
        return None

def _ocr_inside_watermark(box, region):
    if box is None:
        return False
    # 仅允许很小的稳像/检测框抖动；要求候选框至少80%落在已确认水印区。
    pad = 0.005
    left, top = max(box[0], region[0] - pad), max(box[1], region[1] - pad)
    right, bottom = min(box[2], region[2] + pad), min(box[3], region[3] + pad)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    return intersection / ((box[2] - box[0]) * (box[3] - box[1])) >= 0.8

def filter_ocr_watermark_indices(texts, scores, polys, camera_id,
                                 roi_x1, roi_y1, frame_width, frame_height):
    """先学习明确水印的位置，再过滤同一位置的数字碎片，避免候选顺序影响。"""
    now = time.monotonic()
    regions = [(box, seen) for box, seen in _ocr_watermark_regions.get(camera_id, [])
               if now - seen <= OCR_WATERMARK_CACHE_SECONDS]
    rejected = set()
    boxes = [_ocr_watermark_bbox(poly, roi_x1, roi_y1, frame_width, frame_height)
             for poly in polys]
    for index, (text, score, box) in enumerate(zip(texts, scores, boxes)):
        # 摄像头名称直接排除，不用于扩展时间水印的位置缓存。
        if is_ocr_camera_watermark(text):
            rejected.add(index)
            continue
        if not is_ocr_time_watermark(text):
            continue
        rejected.add(index)
        if box is not None and float(score) >= OCR_CONF_THRESH:
            # 只由明确日期/时间刷新缓存，误识别数字不能让旧位置永久有效。
            regions = [(old, seen) for old, seen in regions
                       if not (_ocr_inside_watermark(box, old) and _ocr_inside_watermark(old, box))]
            regions.append((box, now))
    regions = regions[-8:]  # 每路上限，位置变化时也不会无限增长
    if regions:
        _ocr_watermark_regions[camera_id] = regions
    else:
        _ocr_watermark_regions.pop(camera_id, None)
    for index, (text, box) in enumerate(zip(texts, boxes)):
        value = _normalize_ocr_watermark_text(text).replace(" ", "")
        if (re.fullmatch(r"[0-9OoIl:./\-]+", value)
                and sum(c.isdigit() for c in value) >= 2
                and any(_ocr_inside_watermark(box, region) for region, _ in regions)):
            rejected.add(index)
    return rejected

def is_valid_text(text, min_length=2, min_chinese_count=1, min_digit_count=1):
    """
    判断文本是否为有效文本（过滤无意义字符）
    新增：过滤掉纯英文文本（无中文、无数字）
    """
    if not text or not isinstance(text, str):
        return False
    
    text = text.strip()
    if is_ocr_time_watermark(text) or is_ocr_camera_watermark(text):
        return False
    
    # 1. 长度过滤
    if len(text) < min_length:
        return False
    
    # 2. 常见无意义字符过滤
    meaningless_patterns = [
        '-', '.', ',', ' ', '', 'ane', 'one', 'the', 'and', 'or',
        '。', '，', '、', '；', '：', '？', '！',
        '一', '二', '三', '四', '五', '六', '七', '八', '九', '十',
    ]
    if text in meaningless_patterns:
        return False
    
    # 3. 全是标点符号或特殊字符
    if re.match(r'^[\W_]+$', text):
        return False
    
    # 4. 统计字符类型
    chinese_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    digit_count = sum(1 for c in text if c.isdigit())
    letter_count = sum(1 for c in text if c.isalpha() and not ('\u4e00' <= c <= '\u9fff'))
    
    # ========== 新增：过滤纯英文文本 ==========
    # 如果只包含字母（允许空格、连字符等，但无中文和数字），一律过滤
    if letter_count > 0 and chinese_count == 0 and digit_count == 0:
        return False
    
    # 5. 船舶号码的常见特征
    has_chinese = chinese_count > 0
    has_digits = digit_count > 0
    has_letters = letter_count > 0
    
    # 排除纯单个字母的情况（已在上方纯英文过滤中包含，此处仅作为安全余量）
    if len(text) == 1 and text.isalpha():
        return False
    
    # 排除纯英文单词（该规则已无实际作用，但保留注释）
    if has_letters and not has_chinese and not has_digits and len(text) < 5:
        common_english_words = ['and', 'the', 'for', 'you', 'are', 'was', 'not', 'but', 'had']
        if text.lower() in common_english_words:
            return False
        # 此处已通过上方纯英文过滤，不会执行到，故可直接移除
        # 保留仅用于兼容原逻辑
    
    # 船舶号码的有效组合
    valid_combinations = [
        has_chinese and has_digits,
        has_chinese and has_letters,
        has_chinese and has_digits and has_letters,
        has_digits and len(text) >= 4,
        has_chinese and chinese_count >= 2,
    ]
    if any(valid_combinations):
        return True
    
    # 数字占比过高可能是编号
    if digit_count / len(text) > 0.6 and len(text) >= 3:
        return True
    
    # 纯字母但长度足够（已在上方被过滤，不会执行到此处）
    if has_letters and len(text) >= 3 and not has_chinese and not has_digits:
        if text.islower() and len(text) <= 3:
            return False
        return True
    
    return False

class RKNNModel:
    def __init__(self, model_path):
        self.rknn_lite = RKNNLite()
        ret = self.rknn_lite.load_rknn(model_path)
        if ret != 0:
            print(f'Load {model_path} failed!')
            exit(ret)
        ret = self.rknn_lite.init_runtime()
        if ret != 0:
            print('Init runtime failed!')
            exit(ret)
        print(f"模型加载成功: {model_path}")
    def run(self, inputs):
        return self.rknn_lite.inference(inputs=inputs)

class RKNNOCREngine:
    def __init__(self):
        # 初始化检测模型
        self.det_model = RKNNModel(DET_MODEL_PATH)
        self.det_preprocess = []
        for item in DET_PRE_PROCESS_CONFIG:
            for key in item:
                pclass = getattr(utils.operators, key)
                p = pclass(**item[key])
                self.det_preprocess.append(p)
        self.db_postprocess = DBPostProcess(**DET_POSTPROCESS_CONFIG['DBPostProcess'])
        self.det_postprocess = DetPostProcess()

        # 初始化识别模型
        self.rec_model = RKNNModel(REC_MODEL_PATH)
        self.rec_preprocess = []
        for item in REC_PRE_PROCESS_CONFIG:
            for key in item:
                pclass = getattr(utils.operators, key)
                p = pclass(**item[key])
                self.rec_preprocess.append(p)
        self.ctc_postprocess = CTCLabelDecode(**REC_POSTPROCESS_CONFIG['CTCLabelDecode'])

    def predict(self, img):
        """
        接口完全兼容你原版PaddleOCR
        输入: 单张图片(ROI)
        输出: [{'rec_texts': [...], 'rec_scores': [...], 'rec_polys': [...]}]
        """
        rec_texts = []
        rec_scores = []
        rec_polys = []

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
            
            # 排序：从上到下
            text_boxes = sorted(text_boxes, key=lambda x: (np.min(x[:, 1]), np.min(x[:, 0])))

            # 2. 逐个文字框识别
            for box in text_boxes:
                # 裁剪文字区域
                bx1, by1 = int(np.min(box[:, 0])), int(np.min(box[:, 1]))
                bx2, by2 = int(np.max(box[:, 0])), int(np.max(box[:, 1]))
                text_crop = img[by1:by2, bx1:bx2]
                if text_crop.size == 0:
                    continue

                # 识别
                text_crop = cv2.resize(text_crop, (REC_INPUT_SHAPE[1], REC_INPUT_SHAPE[0]))
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
            pass

        # 返回和你原版PaddleOCR完全一致的格式
        return [{'rec_texts': rec_texts, 'rec_scores': rec_scores, 'rec_polys': rec_polys}]

def draw_chinese_text(img, text, pos, font_size=0.8, color=(255,255,255), thickness=2):
    """
    绘制支持中文的文本（基于 freetype 混合渲染，零全图拷贝）
    :param img: OpenCV BGR图像（原地修改）
    :param text: 要绘制的文本（支持中文）
    :param pos: 文本位置 (x, y) - 左上角坐标
    :param font_size: 字体大小
    :param color: 文本颜色 (B,G,R)
    :param thickness: 线条厚度
    :return: 绘制后的OpenCV图像（同img）
    """
    font_height = max(12, int(font_size * FONT_MULTIPALE))
    x, y_top = pos
    # PIL top-left origin → cv2/freetype bottom-left origin
    y_baseline = y_top + font_height
    _put_mixed_text(img, text, (x, y_baseline), font_height, color, thickness)
    return img

def get_text_size(text, font_size=0.8):
    """
    计算文本尺寸（支持中文，基于 freetype 混合测量，无 PIL 依赖）
    :return: (width, height)
    """
    font_height = max(12, int(font_size * FONT_MULTIPALE))
    w, h = _get_mixed_text_size(text, font_height, FONT_THICKNESS)
    return (w + 2, h + 2)

def process_ship_ocr(frame, post_results, weather_label, camera_id):
    """
    仅对白天告警的船只ROI进行OCR识别，返回纯结果，不绘制任何内容
    :param frame: 干净的稳像帧（无任何绘制）
    :param post_results: 后处理后的检测结果
    :param weather_label: 场景分类标签
    :param camera_id: 相机编号
    :return: OCR结果列表
    """
    ocr_results = []
    if not ENABLE_OCR or ocr_engine is None or weather_label in ["night", "夜晚"]:
        return ocr_results
    
    alarm_ship_rois = []
    for res in post_results:
        if len(res) < 8:
            continue
        x1, y1, x2, y2, conf, cls, area, is_alarm = res
        if is_alarm:
            alarm_ship_rois.append(res)
    
    if len(alarm_ship_rois) == 0:
        return ocr_results
    
    h, w = frame.shape[:2]
    
    for roi_res in alarm_ship_rois:
        if len(roi_res) < 8:
            continue
        x1, y1, x2, y2, conf, cls, area, is_alarm = roi_res
        
        # 扩展ROI边界
        roi_x1 = max(0, x1 - 5)
        roi_y1 = max(0, y1 - 5)
        roi_x2 = min(w, x2 + 5)
        roi_y2 = min(h, y2 + 5)
        if (roi_x2 - roi_x1) < 10 or (roi_y2 - roi_y1) < 10:
            continue
        
        ship_roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        if ship_roi.size == 0:
            continue
        
        # OCR推理（在干净的ROI上）
        rec_texts, rec_scores, rec_polys = [], [], []  # 每个ROI独立，避免空返回复用上次结果
        try:
            result = ocr_engine.predict(ship_roi)
            
            if isinstance(result, list) and len(result) > 0:
                first_result = result[0]
                if isinstance(first_result, dict):
                    rec_texts = first_result.get('rec_texts', [])
                    rec_scores = first_result.get('rec_scores', [])
                    rec_polys = first_result.get('rec_polys', [])
                    
                    # print(f"找到 {len(rec_texts)} 个识别文本: {rec_texts}")
                    # print(f"对应置信度: {rec_scores}")
        
        except Exception as e:
            print(f"⚠️ OCR推理异常（ROI区域）：{str(e)}")
            continue
        
        if len(rec_texts) == 0:
            # print(f"ℹ️ 当前ROI无有效识别文字（x1:{x1},y1:{y1},x2:{x2},y2:{y2}）")
            continue
        elif len(rec_texts) != len(rec_scores) or len(rec_texts) != len(rec_polys):
            print(f"⚠️ OCR结果字段长度不匹配：文本数={len(rec_texts)}, 置信度数={len(rec_scores)}, 坐标数={len(rec_polys)}")
            continue
        
        # 水印先于置信度排序过滤，避免水印压过真实船号。
        watermark_indices = filter_ocr_watermark_indices(
            rec_texts, rec_scores, rec_polys, camera_id, roi_x1, roi_y1, w, h)

        # 筛选最佳结果
        best_text = None
        best_score = 0
        best_poly = None
        
        for idx in range(len(rec_texts)):
            if idx in watermark_indices:
                continue
            text = rec_texts[idx]
            score = float(rec_scores[idx]) if idx < len(rec_scores) else 0
            poly = rec_polys[idx] if idx < len(rec_polys) else []
            
            if score < OCR_CONF_THRESH:
                # print(f"  ✗ 跳过低置信度文本: '{text}' ({score:.3f})")
                continue
            
            if not is_valid_text(text):
                # print(f"  ✗ 跳过无效文本: '{text}'")
                continue
            
            if score > best_score:
                best_text = text
                best_score = score
                best_poly = poly
        
        if best_text:
            if VERBOSE:
                print(f"✅ 选择最佳识别结果: '{best_text}' (置信度: {best_score:.3f})")
            
            # 计算OCR文本在原图上的坐标
            try:
                if best_poly is not None and len(best_poly) >= 4:
                    poly_flat = [coord for point in best_poly for coord in point]
                    text_x1 = min(poly_flat[::2])
                    text_y1 = min(poly_flat[1::2])
                    text_x2 = max(poly_flat[::2])
                    text_y2 = max(poly_flat[1::2])
                    
                    # 转换为原图坐标
                    text_x1 += roi_x1
                    text_y1 += roi_y1
                    text_x2 += roi_x1
                    text_y2 += roi_y1
                    
                    text_x1 = max(0, text_x1)
                    text_y1 = max(0, text_y1)
                    text_x2 = min(w, text_x2)
                    text_y2 = min(h, text_y2)
                else:
                    # 如果没有多边形坐标，使用船舶检测框的位置
                    text_x1 = x1
                    text_y1 = y1
                    text_x2 = x2
                    text_y2 = y2
                    
            except Exception as e:
                print(f"⚠️ 坐标解析异常：{str(e)}")
                text_x1 = x1
                text_y1 = y1
                text_x2 = x2
                text_y2 = y2
            
            # 保存OCR结果
            ocr_results.append({
                "text": best_text,
                "conf": best_score,
                "bbox": (text_x1, text_y1, text_x2, text_y2),
                "ship_bbox": (x1, y1, x2, y2),
                "camera_id": camera_id
            })
        else:
            if VERBOSE:
                print(f"ℹ️ 当前ROI无有效识别文字（经过过滤）")
    
    return ocr_results

def draw_ocr_results(frame, ocr_results):
    """
    在已绘制检测框的帧上，叠加绘制OCR结果
    :param frame: 已画好检测框的帧
    :param ocr_results: process_ship_ocr返回的OCR结果列表
    :return: 绘制完成的帧
    """
    h, w = frame.shape[:2]
    
    for ocr_res in ocr_results:
        text = ocr_res["text"]
        conf = ocr_res["conf"]
        text_x1, text_y1, text_x2, text_y2 = ocr_res["bbox"]
        ship_x1, ship_y1, ship_x2, ship_y2 = ocr_res["ship_bbox"]
        
        display_text = f"{text}({conf:.2f})"
        text_w, text_h = get_text_size(display_text, FONT_SCALE)
        
        # 计算OCR标签位置（在船舶检测框上方）
        label_tl_x = int(text_x1)
        label_tl_y = int(text_y1 - text_h - 2 * PADDING - LABEL_OFFSET)
        label_br_x = int(label_tl_x + text_w + 2 * PADDING)
        label_br_y = int(label_tl_y + text_h + 2 * PADDING)
        
        # 边界检查
        if label_br_x > w:
            label_tl_x = max(0, int(w - text_w - 2 * PADDING - 10))
            label_br_x = int(label_tl_x + text_w + 2 * PADDING)
        if label_br_y > h:
            label_tl_y = max(0, int(h - text_h - 2 * PADDING - 10))
            label_br_y = int(label_tl_y + text_h + 2 * PADDING)
        if label_tl_y < 0:
            # 如果上方放不下，放到船舶检测框下方
            label_tl_y = int(ship_y2 + LABEL_OFFSET)
            label_br_y = int(label_tl_y + text_h + 2 * PADDING)
        
        # 绘制OCR标签背景
        frame = draw_rounded_rectangle(
            frame,
            (label_tl_x, label_tl_y),
            (label_br_x, label_br_y),
            OCR_BG_COLOR,
            thickness=-1,
            radius=CORNER_RADIUS
        )
        
        # 绘制OCR文字
        text_pos_x = label_tl_x + PADDING
        text_pos_y = label_tl_y + PADDING
        
        frame = draw_chinese_text(
            frame,
            display_text,
            (text_pos_x, text_pos_y),
            font_size=FONT_SCALE,
            color=OCR_TEXT_COLOR,
            thickness=FONT_THICKNESS
        )
        
        # 可选：绘制OCR文本框边框
        cv2.rectangle(frame, (int(text_x1), int(text_y1)),
                      (int(text_x2), int(text_y2)), OCR_BOX_COLOR, max(1, BOX_THICKNESS//2))
    
    return frame

def draw_rounded_rectangle(img, top_left, bottom_right, color, thickness=-1, radius=10):
    """
    绘制圆角矩形
    :param img: 画布
    :param top_left: 左上角坐标 (x1, y1)
    :param bottom_right: 右下角坐标 (x2, y2)
    :param color: 颜色 (B,G,R)
    :param thickness: -1=填充，>0=边框厚度
    :param radius: 圆角半径
    """
    x1, y1 = top_left
    x2, y2 = bottom_right
    
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = max(x1+1, int(x2)), max(y1+1, int(y2))
    
    radius = min(radius, max(1, (x2-x1)//2), max(1, (y2-y1)//2))
    
    if thickness == -1:
        cv2.rectangle(img, (x1+radius, y1), (x2-radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1+radius), (x2, y2-radius), color, -1)
    else:
        cv2.rectangle(img, (x1+radius, y1), (x2-radius, y2), color, thickness)
        cv2.rectangle(img, (x1, y1+radius), (x2, y2-radius), color, thickness)
    
    cv2.ellipse(img, (x1+radius, y1+radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2-radius, y1+radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1+radius, y2-radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2-radius, y2-radius), (radius, radius), 0, 0, 90, color, thickness)
    return img

class YOLO_RKNN:
    def __init__(self, model_path):
        from rknnlite.api import RKNNLite
        self.rknn_lite = RKNNLite()
        ret = self.rknn_lite.load_rknn(model_path)
        if ret != 0:
            print(f'❌ RKNN 加载失败: {model_path}')
            exit(ret)
        ret = self.rknn_lite.init_runtime()
        if ret != 0:
            print(f'❌ RKNN 初始化失败: {model_path}')
            exit(ret)
        print(f'✅ RKNN 模型就绪: {model_path}')

    def run(self, inputs, data_format=None):   # 添加 data_format 参数
        return self.rknn_lite.inference(inputs=inputs, data_format=data_format)

class YOLOv11RKNN:
    # 检测置信度 / NMS 阈值（与参考代码一致）
    CONF_THRESH = 0.45
    IOU_THRESH = 0.25

    def __init__(self, model_path, model_type="det", input_size=640, names=None, conf_thresh=None):
        """
        :param names: 自定义类别映射字典，如 {0:"ship"}
        :param conf_thresh: 实例级置信度阈值（白天/夜晚模型可不同），None 时用类默认 CONF_THRESH
        """
        self.model_path = model_path
        self.model_type = model_type
        self.input_size = input_size
        self.conf_thresh = conf_thresh if conf_thresh is not None else self.CONF_THRESH
        self.rknn = YOLO_RKNN(model_path)

        # 类别映射（外部可指定，否则使用默认）
        if names is not None:
            self.names = names
        else:
            if self.model_type == "cls":
                self.names = {0: "cloudy", 1: "foggy", 2: "night", 3: "rainy", 4: "sunny"}
            else:
                self.names = {0: "ship"}  # 默认单类 ship

        print(f"✅ RKNN 模型类型={model_type}, 尺寸={input_size}, 类别={self.names}")

    # ============== 预处理（分类保持原样，检测改为 NHWC） ==============
    def _letterbox(self, img, new_size):
        h0, w0 = img.shape[:2]
        r = min(new_size / h0, new_size / w0)
        new_w = int(round(w0 * r))
        new_h = int(round(h0 * r))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        dw = (new_size - new_w) / 2
        dh = (new_size - new_h) / 2
        top    = int(round(dh - 0.1))
        bottom = int(round(dh + 0.1))
        left   = int(round(dw - 0.1))
        right  = int(round(dw + 0.1))

        # 填充 0（黑色），与参考代码一致
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        return padded, r, (dw, dh)

    def _preprocess_det(self, img):
        """检测预处理：RGB 顺序，HWC 格式，uint8"""
        padded, r, (dw, dh) = self._letterbox(img, self.input_size)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        blob = np.expand_dims(rgb, axis=0)          # (1, H, W, 3) NHWC
        return blob, r, (dw, dh)

    def _preprocess_cls(self, img):
        """分类预处理：保持原版 NCHW uint8"""
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
        chw = rgb.transpose(2, 0, 1)               # (3, H, W)
        blob = np.expand_dims(chw.astype(np.uint8), axis=0)  # (1, 3, H, W) NCHW
        return blob

    # ============== 检测后处理（Decoupled Head + DFL） ==============
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
    def _box_process(position, grid_h, grid_w):
        col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
        col = col.reshape(1, 1, grid_h, grid_w)
        row = row.reshape(1, 1, grid_h, grid_w)
        grid = np.concatenate((col, row), axis=1)

        stride = np.array([640 // grid_h, 640 // grid_w]).reshape(1, 2, 1, 1)

        position = YOLOv11RKNN._dfl_numpy(position)
        box_xy  = grid + 0.5 - position[:, 0:2, :, :]
        box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
        xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)
        return xyxy

    def _postprocess_det(self, outputs, orig_shape, r, dw, dh):
        """
        处理多个不同尺度的 decoupled head 输出
        outputs: list of feature maps
        每个特征图层可能包含 box (64ch), score (类别数ch), obj (1ch)
        """
        if outputs is None or len(outputs) == 0:
            return []

        # 按 (h, w) 分组
        layers = {}
        for out in outputs:
            h, w = out.shape[2:4]
            if (h, w) not in layers:
                layers[(h, w)] = []
            layers[(h, w)].append(out)

        all_boxes = []
        all_scores = []
        all_class_ids = []
        num_classes = len(self.names)   # 1 (ship 或 light)

        for (h, w), tensors in layers.items():
            box_t = score_t = obj_t = None

            for t in tensors:
                c = t.shape[1]
                if c == 64:                # box 分支
                    box_t = t
                elif c == num_classes:     # 类别分支（单类时通道=1）
                    score_t = t
                elif c == 1:               # obj 分支（若有）
                    obj_t = t

            if box_t is None or score_t is None:
                continue

            # 得分 = class_score * obj_score（若 obj 分支存在）
            s = score_t
            if obj_t is not None:
                s = s * obj_t

            # 转为 (N, num_classes) 和 (N, 4)
            s = s.transpose(0, 2, 3, 1).reshape(-1, num_classes)
            b = self._box_process(box_t, h, w).transpose(0, 2, 3, 1).reshape(-1, 4)

            max_scores = np.max(s, axis=1)
            max_class_ids = np.argmax(s, axis=1)

            # 粗过滤，加速 NMS
            idx = max_scores > 0.05
            all_boxes.append(b[idx])
            all_scores.append(max_scores[idx])
            all_class_ids.append(max_class_ids[idx])

        if not all_boxes:
            return []

        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        class_ids = np.concatenate(all_class_ids, axis=0)

        # 构造 cv2.dnn.NMSBoxes 需要的 (x,y,w,h) 格式
        nms_boxes = []
        for box in boxes:
            nms_boxes.append([box[0], box[1], box[2] - box[0], box[3] - box[1]])

        indices = cv2.dnn.NMSBoxes(
            nms_boxes, scores.tolist(),
            self.conf_thresh, self.IOU_THRESH
        )

        if len(indices) == 0:
            return []

        if isinstance(indices, np.ndarray):
            indices = indices.flatten()

        final_boxes  = boxes[indices]
        final_scores = scores[indices]
        final_cls    = class_ids[indices]

        orig_h, orig_w = orig_shape
        results = []
        for box, score, cls_id in zip(final_boxes, final_scores, final_cls):
            # box 坐标在 640x640 空间 (xyxy)，映射回原图
            x1 = (box[0] - dw) / r
            y1 = (box[1] - dh) / r
            x2 = (box[2] - dw) / r
            y2 = (box[3] - dh) / r

            # 归一化到 0~1，便于后续处理
            x1n = np.clip(x1 / orig_w, 0.0, 1.0)
            y1n = np.clip(y1 / orig_h, 0.0, 1.0)
            x2n = np.clip(x2 / orig_w, 0.0, 1.0)
            y2n = np.clip(y2 / orig_h, 0.0, 1.0)

            results.append((x1n, y1n, x2n, y2n, float(score), float(cls_id)))

        return results

    # ============== 分类后处理（原版保持） ==============
    def _postprocess_cls(self, outputs):
        probs = outputs[0]
        if probs.ndim == 2:
            probs = probs[0]

        top1_idx  = int(np.argmax(probs))
        top1_conf = float(probs[top1_idx])

        # ---- 修正 foggy / night 混淆 ----
        if top1_idx == 1:                     # 当前判为 foggy
            night_prob = float(probs[2]) if len(probs) > 2 else 0.0
            # 阈值可根据实际调试调整
            if night_prob > 0.3 and (top1_conf - night_prob) < 0.3:
                top1_idx  = 2                 # 修正为 night
                top1_conf = night_prob
                print("———————————————————修复为night————————————————————")
        return top1_idx, top1_conf

    # ============== 统一推理入口 ==============
    def predict(self, frame):
        shape = frame.shape[:2]
        if self.model_type == "cls":
            blob = self._preprocess_cls(frame)
            # 分类模型 NCHW 输入，不指定 data_format
            out = self.rknn.run([blob])
            idx, conf = self._postprocess_cls(out)
            return [_ClsResult(idx, conf, self.names)]
        else:
            blob, r, (dw, dh) = self._preprocess_det(frame)
            # 检测模型 NHWC 输入
            out = self.rknn.run([blob], data_format='nhwc')
            dets = self._postprocess_det(out, shape, r, dw, dh)
            return [_DetResult(dets, self.names)]

class _ClsProbs:
    def __init__(self, top1_idx, top1_conf):
        self.top1     = top1_idx
        self.top1conf = _FloatWrapper(top1_conf)

class _FloatWrapper:
    def __init__(self, val):
        self._val = val

    def item(self):
        return self._val

    def __float__(self):
        return float(self._val)

class _ClsResult:
    def __init__(self, top1_idx, top1_conf, names):
        self.probs = _ClsProbs(top1_idx, top1_conf)
        self.names = names

class _Box:
    def __init__(self, x1n, y1n, x2n, y2n, conf, cls_id):
        self.xyxyn = [_ToListWrapper([x1n, y1n, x2n, y2n])]
        self.conf  = [_FloatWrapper(conf)]
        self.cls   = [_FloatWrapper(cls_id)]

class _ToListWrapper:
    def __init__(self, data):
        self._data = data

    def tolist(self):
        return list(self._data)

class _Boxes:
    def __init__(self, dets):
        self._boxes = [_Box(*d) for d in dets]

    def __iter__(self):
        return iter(self._boxes)

    def __len__(self):
        return len(self._boxes)

class _DetResult:
    def __init__(self, dets, names):
        self.boxes = _Boxes(dets) if dets else None
        self.names = names

def configure_night_model(enabled):
    """Startup switch: disabled by default; enabled mode loads the model once."""
    global ENABLE_NIGHT_MODEL, det_night_model
    if enabled and det_night_model is None:
        det_night_model = YOLOv11RKNN(
            DET_NIGHT_MODEL_PATH, model_type="det", input_size=DET_INPUT_SIZE,
            names={0: "light"}
        )
    ENABLE_NIGHT_MODEL = bool(enabled)
    print('[model] Night light detector: %s' %
          ('enabled (scene-dependent)' if ENABLE_NIGHT_MODEL else
           'disabled; day ship detector used for all scenes'))

def _use_night_detector(weather_cls):
    return ENABLE_NIGHT_MODEL and weather_cls == "night"

def _select_detection_model(weather_cls):
    if _use_night_detector(weather_cls):
        if det_night_model is None:
            raise RuntimeError('Night detector enabled but not initialized')
        return det_night_model
    return det_day_model

def get_weather_class(frame):
    """场景分类：移除多余参数，仅传frame"""
    #cls_frame = cv2.resize(frame, (320, 320), interpolation=cv2.INTER_LINEAR)
    results = cls_model.predict(frame)
    
    result = results[0]
    top1_idx = result.probs.top1
    top1_conf = result.probs.top1conf.item()
    cls_label = result.names.get(top1_idx, str(top1_idx))
    
    if VERBOSE:
        print(f"分类结果：{cls_label} | 置信度：{top1_conf:.3f}")
    return cls_label, top1_conf

def get_weather_class_cached(frame, camera_id, frame_idx):
    cache = weather_cache.get(camera_id)
    if cache is not None and frame_idx % CLS_INTERVAL != 0:
        return cache

    result = get_weather_class(frame)
    weather_cache[camera_id] = result
    return result

def detect_target(frame, weather_cls):
    """目标检测：移除多余参数，仅传frame"""
    det_model = _select_detection_model(weather_cls)
    
    results = det_model.predict(frame)
    
    boxes = results[0].boxes
    det_results = []
    if boxes is not None:
        for box in boxes:
            x1n, y1n, x2n, y2n = box.xyxyn[0].tolist()
            conf = box.conf[0].item()
            cls = box.cls[0].item()
            det_results.append((x1n, y1n, x2n, y2n, conf, cls))
    return det_results

def postprocess_detections(det_results, orig_frame_shape, weather_cls):
    h, w = orig_frame_shape
    post_results = []
    has_alarm = False

    for (x1n, y1n, x2n, y2n, conf, cls) in det_results:
        x1 = int(x1n * w)
        y1 = int(y1n * h)
        x2 = int(x2n * w)
        y2 = int(y2n * h)
        
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        area = width * height
        total_area = w * h
        area_ratio = area / total_area if total_area > 0 else 0.0

        if _use_night_detector(weather_cls):
            is_alarm = area_ratio >= LIGHT_ALARM_THRESH
        else:
            is_alarm = area_ratio >= SHIP_ALARM_THRESH
        
        if is_alarm:
            has_alarm = True
        
        post_results.append((x1, y1, x2, y2, conf, cls, area, is_alarm))
    
    return post_results, has_alarm

def draw_custom_boxes(frame, post_results, weather_info, model_names):
    """
    优化后的绘制函数：
    1. 修复文字位置，确保在框内显示
    2. 优化标签框位置计算，避免超出画面
    3. 统一视觉样式
    """
    h, w = frame.shape[:2]
    weather_label, weather_conf = weather_info

    weather_text = f"Weather: {weather_label}({weather_conf:.2f})"
    text_w, text_h = get_text_size(weather_text, FONT_SCALE)
    tl_x = 5
    tl_y = 5
    br_x = tl_x + text_w + 2 * PADDING
    br_y = tl_y + text_h + 2 * PADDING
    
    br_x = min(br_x, w - 5)
    br_y = min(br_y, h - 5)
    
    frame = draw_rounded_rectangle(
        frame, 
        (tl_x, tl_y), 
        (br_x, br_y), 
        TEXT_BG_COLOR, 
        thickness=-1,
        radius=CORNER_RADIUS
    )
    text_x = tl_x + PADDING
    text_y = tl_y + PADDING
    frame = draw_chinese_text(frame, weather_text, (text_x, text_y),
                             font_size=FONT_SCALE, color=TEXT_COLOR, thickness=FONT_THICKNESS)

    for (x1, y1, x2, y2, conf, cls, area, is_alarm) in post_results:
        box_color = ALARM_COLOR if is_alarm else NORMAL_COLOR
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, BOX_THICKNESS)

        cls_name = model_names.get(int(cls), "unknown")
        alarm_text = "ALARM" if is_alarm else "NORMAL"
        distance = estimate_distance_by_bbox(
            x2 - x1, w, use_night_detector=_use_night_detector(weather_label)
        )
        distance_text = f"{distance:.1f}m" if distance is not None else "N/A"
        label_text = (f"{cls_name}({conf:.2f})| Area:{area}| "
                      f"Distance:{distance_text}| {alarm_text}")
        
        label_w, label_h = get_text_size(label_text, FONT_SCALE)
        label_height = label_h + 2 * PADDING
        label_width = label_w + 2 * PADDING
        
        label_tl_y = y1 - label_height - LABEL_OFFSET
        if label_tl_y < 0:
            label_tl_y = y2 + LABEL_OFFSET
        
        label_tl_y = max(0, min(label_tl_y, h - label_height - 5))
        label_tl_x = max(0, min(x1, w - label_width - 5))
        
        label_br_x = label_tl_x + label_width
        #label_br_y = label_tl_y + label_height - 2 * PADDING
        label_br_y = label_tl_y + label_height

        if label_br_x > w:
            label_tl_x = max(0, w - label_width - 5)
            label_br_x = label_tl_x + label_width
        
        frame = draw_rounded_rectangle(
            frame, 
            (label_tl_x, label_tl_y), 
            (label_br_x, label_br_y), 
            TEXT_BG_COLOR, 
            thickness=-1, 
            radius=CORNER_RADIUS
        )
        
        label_text_x = label_tl_x + PADDING
        label_text_y = label_tl_y + PADDING
        frame = draw_chinese_text(frame, label_text, (label_text_x, label_text_y),
                                 font_size=FONT_SCALE, color=TEXT_COLOR, thickness=FONT_THICKNESS)
    
    return frame

def initialize_inference(args):
    """模型只在验证输入之后加载；--help 不需要 RKNN 或视频依赖。"""
    global cv2, np, RKNNLite, utils, DBPostProcess, DetPostProcess, CTCLabelDecode
    global cls_model, det_day_model, ocr_engine, ENABLE_OCR, CLS_INTERVAL, VERBOSE
    global CHINESE_FONT_PATH
    import cv2
    import numpy as np
    cv2.setNumThreads(1)
    VERBOSE = args.verbose
    ENABLE_OCR = not args.no_ocr
    CLS_INTERVAL = args.cls_interval
    for candidate in ('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
                      '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'):
        if Path(candidate).is_file():
            CHINESE_FONT_PATH = candidate
            break
    cls_model = YOLOv11RKNN(args.cls_model, model_type='cls', input_size=CLS_INPUT_SIZE,
                           names={0: 'cloudy', 1: 'foggy', 2: 'night', 3: 'rainy', 4: 'sunny'})
    det_day_model = YOLOv11RKNN(args.det_model, model_type='det', input_size=DET_INPUT_SIZE,
                               names={0: 'ship'}, conf_thresh=0.5)
    configure_night_model(args.night_model)
    if ENABLE_OCR:
        from rknnlite.api import RKNNLite
        import utils.operators
        from utils.db_postprocess import DBPostProcess, DetPostProcess
        from utils.rec_postprocess import CTCLabelDecode
        ocr_engine = RKNNOCREngine()


def release_inference():
    for model in (cls_model, det_day_model, det_night_model):
        if model is not None:
            model.rknn.rknn_lite.release()
    if ocr_engine is not None:
        ocr_engine.det_model.rknn_lite.release()
        ocr_engine.rec_model.rknn_lite.release()


def annotate_frame(frame, frame_index, video_id):
    weather = get_weather_class_cached(frame, video_id, frame_index)
    detections = detect_target(frame, weather[0])
    results, _ = postprocess_detections(detections, frame.shape[:2], weather[0])
    ocr = process_ship_ocr(frame, results, weather[0], video_id)
    drawn = draw_custom_boxes(frame.copy(), results, weather,
                              _select_detection_model(weather[0]).names)
    return draw_ocr_results(drawn, ocr)


def validate_inputs(values):
    inputs, seen = [], set()
    for value in values:
        if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*://', str(value)):
            raise ValueError('仅允许本地视频文件，不支持网络地址：' + str(value))
        path = Path(value).expanduser().resolve(strict=True)
        if not path.is_file():
            raise ValueError('请指定视频文件，不扫描文件夹：' + str(path))
        identity = (path.stat().st_dev, path.stat().st_ino)
        if identity in seen:
            print('[skip] 重复输入只处理一次：' + str(path))
            continue
        seen.add(identity)
        inputs.append(path)
    if not inputs:
        raise ValueError('请通过 --videos 指定本地视频，或在 VIDEO_PATHS 中填写路径。')
    return inputs


def plan_outputs(inputs, directory):
    planned, used = [], set()
    for source in inputs:
        index = 1
        while True:
            suffix = '' if index == 1 else '_' + str(index)
            target = directory / (source.stem + '_detected' + suffix + '.mp4')
            if target not in used and not target.exists() and target != source:
                break
            index += 1
        planned.append(target)
        used.add(target)
    return planned


def _timeline_hash_update(digest, timestamp):
    digest.update((str(Fraction(timestamp)) + '\n').encode('ascii'))


def _ticks(value, time_base):
    value = Fraction(value) / Fraction(time_base)
    if value.denominator != 1:
        raise ValueError('时间戳不能按原时间基精确表示，停止以避免播放时长变化。')
    return value.numerator


def verify_output(av, path, expected_count, size, expected_digest, expected_duration, tolerance):
    """只解码检查，不再运行检测。核对每帧时间戳、帧数、尺寸、时长。"""
    digest = hashlib.sha256()
    count = 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            if (frame.width, frame.height) != size or frame.pts is None:
                raise RuntimeError('输出尺寸或时间戳异常。')
            _timeline_hash_update(digest, frame.pts * frame.time_base)
            count += 1
        if count != expected_count or digest.hexdigest() != expected_digest:
            raise RuntimeError('输出帧数或逐帧时间戳不一致，拒绝发布不完整视频。')
        if stream.duration is None:
            raise RuntimeError('无法核验输出视频时长。')
        duration = stream.duration * stream.time_base
        if abs(duration - expected_duration) > tolerance:
            raise RuntimeError(f'输出时长不一致：{float(duration):.6f} vs {float(expected_duration):.6f} 秒')
    return count, duration


def _source_frame_duration(frame, packet_duration, fallback, stamp, source_end, tolerance):
    """用帧/包的时长核对末尾，禁止把缺失尾段补成长时间的最后一帧。"""
    ticks = getattr(frame, 'duration', 0) or 0
    observed = ticks * frame.time_base if ticks > 0 else packet_duration
    duration = observed if observed and observed > 0 else fallback
    if source_end is not None:
        remaining = source_end - stamp
        if remaining <= 0:
            raise ValueError('源视频时长元数据早于最后一帧，无法保证原时间轴。')
        if abs(remaining - duration) > tolerance:
            raise ValueError(
                f'源视频结尾时间轴不一致：末帧结束={float(stamp + duration):.6f}s，'
                f'文件声明结束={float(source_end):.6f}s；可能尾部缺失或时长元数据异常。'
                '不会通过延长最后一帧掩盖差异，请检查原视频。')
        # 允许封装层时长舍入，不改变真正的帧间时间戳。
        return remaining
    return duration


def _partial_video_path(target):
    candidate = target.with_name(target.stem + '_partial' + target.suffix)
    index = 2
    while candidate.exists():
        candidate = target.with_name(target.stem + f'_partial_{index}' + target.suffix)
        index += 1
    return candidate


def _trim_partial_audio(av, path, video_duration, time_base):
    """部分视频结束后不再保留尾部音频；仅重新封装，不再检测或编码画面。"""
    fd, name = tempfile.mkstemp(prefix='.partial_audio_', suffix='.mp4', dir=str(path.parent))
    os.close(fd)
    trimmed = Path(name)
    try:
        with av.open(str(path)) as source:
            with av.open(str(trimmed), 'w', format='mp4',
                         options={'video_track_timescale': str(time_base.denominator)}) as dest:
                mapping = {}
                for stream in source.streams:
                    if hasattr(dest, 'add_stream_from_template'):
                        mapping[stream.index] = dest.add_stream_from_template(stream)
                    else:
                        mapping[stream.index] = dest.add_stream(template=stream)
                for packet in source.demux():
                    if packet.dts is None:
                        continue
                    if packet.stream.type == 'audio':
                        stamp = (packet.pts if packet.pts is not None else packet.dts) * packet.time_base
                        if stamp >= video_duration:
                            continue
                        remaining = max(1, int((video_duration - stamp) / packet.time_base))
                        if packet.duration and packet.duration > remaining:
                            packet.duration = remaining
                    packet.stream = mapping[packet.stream.index]
                    dest.mux(packet)
        os.replace(trimmed, path)
    finally:
        if trimmed.exists():
            trimmed.unlink()


def process_local_video(source, target, video_id, args, annotate=annotate_frame):
    """顺序解码，最多暂存一帧；按源PTS编码，永不按处理用时抽帧/循环。"""
    import av
    # 独立状态，避免上一段视频的天气或水印位置影响下一段。
    weather_cache.clear()
    _ocr_watermark_regions.clear()
    count = 0
    digest = hashlib.sha256()
    origin = None
    pending = None
    pending_audio = []
    durations = {}
    input_durations = {}
    last_interval = None
    partial_reasons = []
    partial_events = 0

    def mark_partial(reason):
        nonlocal partial_events
        partial_events += 1
        if reason not in partial_reasons and len(partial_reasons) < 10:
            partial_reasons.append(reason)
            print(f'[recovery] {source.name}: {reason}', flush=True)

    began = time.monotonic()
    last_progress = began
    fd, temporary_name = tempfile.mkstemp(prefix='.' + target.stem + '_', suffix='.partial.mp4', dir=str(target.parent))
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with av.open(str(source), mode='r') as reader:
            if not reader.streams.video:
                raise ValueError('文件没有视频流：' + str(source))
            video = reader.streams.video[0]
            # 元数据帧数只作参考；解码错误仍必须显式失败，不允许忽略损坏。
            video.codec_context.options = dict(video.codec_context.options, err_detect='explode')
            size = (video.codec_context.width, video.codec_context.height)
            rate = video.average_rate or video.base_rate or video.guessed_rate
            if rate is None or rate <= 0 or not video.time_base or min(size) <= 0:
                raise ValueError('无法读取有效的源分辨率、帧率或时间基：' + str(source))
            rate = Fraction(rate)
            # 分子为1可精确表达源时间戳；保持源时基分母，不四舍五入29.97等帧率。
            tb = Fraction(1, video.time_base.denominator)
            declared_frames = video.frames
            source_end = ((video.start_time or 0) + video.duration) * video.time_base if video.duration else None
            print(f'[input] {source}\n        {size[0]}x{size[1]}, metadata_fps={rate}, metadata_frames={declared_frames or "unknown"}')
            with av.open(str(temporary), mode='w', format='mp4',
                         options={'video_track_timescale': str(tb.denominator)}) as writer:
                output = writer.add_stream('libx264', rate=rate)
                output.width, output.height = size
                output.pix_fmt = 'yuv420p' if size[0] % 2 == 0 and size[1] % 2 == 0 else 'yuv444p'
                output.time_base = tb
                output.codec_context.time_base = tb
                output.codec_context.max_b_frames = 0
                output.options = {'crf': str(args.crf), 'preset': 'veryfast', 'tune': 'zerolatency'}
                if video.sample_aspect_ratio:
                    output.codec_context.sample_aspect_ratio = video.sample_aspect_ratio
                audio_map = {}
                if not args.no_audio:
                    for stream in reader.streams.audio:
                        if hasattr(writer, 'add_stream_from_template'):
                            audio_map[stream.index] = writer.add_stream_from_template(stream)
                        else:
                            audio_map[stream.index] = writer.add_stream(template=stream)

                def mux_audio(packet):
                    if packet.dts is None:
                        return
                    # 音视频使用相同起点，保留音频包，不进行声音识别/二次压缩。
                    shift = round(origin / packet.time_base)
                    packet.pts = packet.pts - shift if packet.pts is not None else None
                    packet.dts -= shift
                    packet.stream = audio_map[packet.stream.index]
                    writer.mux(packet)

                def mux_video(packet):
                    stamp = packet.pts * packet.time_base
                    duration = durations.pop(stamp)
                    packet.duration = _ticks(duration, packet.time_base)
                    writer.mux(packet)

                def emit(frame, stamp, duration):
                    nonlocal count, last_progress
                    if (frame.width, frame.height) != size or duration <= 0:
                        raise ValueError('源视频中途改变分辨率或包含无效帧时长。')
                    if getattr(frame, 'is_corrupt', False):
                        raise ValueError('检测到损坏帧，停止而不静默丢帧。')
                    image = frame.to_ndarray(format='bgr24')
                    drawn = annotate(image, count, video_id)
                    if drawn.shape != image.shape:
                        raise RuntimeError('检测绘制改变了原图尺寸。')
                    encoded = av.VideoFrame.from_ndarray(drawn, format='bgr24')
                    relative = stamp - origin
                    encoded.pts, encoded.time_base = _ticks(relative, tb), tb
                    durations[relative] = duration
                    for packet in output.encode(encoded):
                        mux_video(packet)
                    _timeline_hash_update(digest, relative)
                    count += 1
                    now = time.monotonic()
                    if now - last_progress >= 5:
                        print(f'[progress] {source.name}: 已处理{count}帧（元数据参考{declared_frames or "?"}帧），'
                              f'视频位置 {float(relative):.1f}s，处理速度 {count / (now - began):.2f} 帧/s', flush=True)
                        last_progress = now

                streams = [video] + [s for s in reader.streams.audio if s.index in audio_map]
                packets = iter(reader.demux(streams))
                recovering = False
                while True:
                    try:
                        packet = next(packets)
                    except StopIteration:
                        break
                    except (av.error.FFmpegError, OSError) as exc:
                        mark_partial(f'读取源视频中断，保存已恢复部分：{exc}')
                        break
                    if getattr(packet, 'is_corrupt', False):
                        mark_partial('跳过损坏数据包')
                        if packet.stream.index == video.index:
                            recovering = True
                        continue
                    if packet.stream.index in audio_map:
                        if origin is None:
                            pending_audio.append(packet)
                            if sum(p.size for p in pending_audio) > 32 * 1024 * 1024:
                                raise ValueError('视频首帧前音频数据过多，无法确定时间轴。')
                        else:
                            mux_audio(packet)
                        continue
                    if recovering:
                        # 损坏参考帧后的P/B帧不可信；等待后续关键帧重新同步。
                        if not packet.is_keyframe or not packet.size:
                            continue
                        video.codec_context.flush_buffers()
                        input_durations.clear()
                        recovering = False
                    if packet.pts is not None and packet.duration and packet.time_base:
                        input_durations[packet.pts * packet.time_base] = packet.duration * packet.time_base
                    try:
                        frames = packet.decode()
                    except av.error.FFmpegError as exc:
                        mark_partial(f'解码失败，等待下一关键帧：{exc}')
                        recovering = True
                        continue
                    for frame in frames:
                        if getattr(frame, 'is_corrupt', False):
                            mark_partial('跳过损坏画面，等待下一关键帧')
                            recovering = True
                            break
                        if frame.pts is None or not frame.time_base:
                            mark_partial('跳过缺少时间戳的画面')
                            continue
                        stamp = frame.pts * frame.time_base
                        if (frame.width, frame.height) != size:
                            mark_partial('跳过分辨率变化的画面')
                            continue
                        if pending is not None and stamp <= pending[1]:
                            mark_partial('跳过时间戳倒退或重复的画面')
                            continue
                        if origin is None:
                            origin = stamp
                            for audio_packet in pending_audio:
                                mux_audio(audio_packet)
                            pending_audio.clear()
                        if pending is not None:
                            previous, previous_stamp, _ = pending
                            last_interval = stamp - previous_stamp
                            emit(previous, previous_stamp, last_interval)
                        pending = (frame, stamp, input_durations.pop(stamp, None))
                        input_durations = {pts: span for pts, span in input_durations.items() if pts > stamp}
                if pending is None:
                    raise ValueError('源视频没有可解码的画面。')
                final_frame, final_stamp, final_packet_duration = pending
                fallback_duration = last_interval or Fraction(1, 1) / rate
                # 损坏/截断结尾也保存最后一个正常帧，不把缺失尾段补成静止画面。
                try:
                    final_duration = _source_frame_duration(
                        final_frame, final_packet_duration, fallback_duration, final_stamp, source_end,
                        max(tb * 2, Fraction(1, 1000)))
                except ValueError as exc:
                    mark_partial(str(exc))
                    final_duration = _source_frame_duration(
                        final_frame, final_packet_duration, Fraction(1, 1) / rate,
                        final_stamp, None, tb)
                emit(final_frame, final_stamp, final_duration)
                expected_duration = final_stamp - origin + final_duration
                if declared_frames and count != declared_frames:
                    mark_partial(f'文件声明{declared_frames}帧，实际保存{count}帧；'
                                 '无法确认完整性，保守标记为partial')
                for packet in output.encode():
                    mux_video(packet)
                if durations:
                    raise RuntimeError('编码器未输出所有帧。')
        if partial_events and audio_map:
            _trim_partial_audio(av, temporary, expected_duration, tb)
        print('[verify] 核验已保存画面的帧数、原尺寸与逐帧时间戳……', flush=True)
        verify_output(av, temporary, count, size, digest.hexdigest(), expected_duration, tb)
        if partial_events:
            target = _partial_video_path(target)
        if target.exists():
            raise FileExistsError('输出文件已存在，不覆盖：' + str(target))
        temporary.rename(target)
        status = 'partial' if partial_events else 'done'
        print(f'[{status}] {target}\n       {count} 帧，{float(expected_duration):.6f}s，{size[0]}x{size[1]}，源声明平均帧率={rate}，实际输出平均帧率={float(count / expected_duration):.6f}', flush=True)
        return {'frames': count, 'duration': expected_duration, 'size': size, 'rate': rate, 'path': target,
                'partial': bool(partial_events), 'recovery_events': partial_events, 'reasons': partial_reasons}
    finally:
        # 失败/中断只清理由本次创建的临时文件，绝不删除输入或旧结果。
        if temporary.exists():
            temporary.unlink()


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description='本地视频只检测一遍；完整输出原尺寸、原时间轴的带框MP4；不发送。')
    parser.add_argument('inputs', nargs='*', help='本地视频文件，也可使用 --videos')
    parser.add_argument('--videos', nargs='+', help='指定一个或多个本地视频文件')
    parser.add_argument('--output-dir', help='唯一输出文件夹；默认在 local_video_results 下创建本次运行文件夹')
    parser.add_argument('--no-ocr', action='store_true', help='关闭OCR，保留船舶检测')
    parser.add_argument('--no-audio', action='store_true', help='不保留音轨；默认原样封装源音轨（需MP4兼容）')
    night_group = parser.add_mutually_exclusive_group()
    night_group.add_argument('--night-model', dest='night_model', action='store_true',
                             help='启用夜间光点模型，覆盖静态设置')
    night_group.add_argument('--no-night-model', dest='night_model', action='store_false',
                             help='关闭夜间光点模型，覆盖静态设置')
    parser.set_defaults(night_model=ENABLE_NIGHT_MODEL)  # 未传参数时遵循静态配置
    parser.add_argument('--cls-interval', type=int, default=CLS_INTERVAL, help='场景分类间隔帧数；目标检测始终逐帧')
    parser.add_argument('--cls-model', default=CLS_MODEL_PATH)
    parser.add_argument('--det-model', default=DET_DAY_MODEL_PATH)
    parser.add_argument('--crf', type=int, default=18, help='H.264画质参数0~51，越低画质越高；不限制文件大小')
    parser.add_argument('--verbose', action='store_true', help='打印逐次分类/OCR结果')
    args = parser.parse_args(argv)
    if args.cls_interval < 1 or not 0 <= args.crf <= 51:
        parser.error('--cls-interval 必须>=1，--crf 必须为0~51')
    if args.inputs and args.videos:
        parser.error('位置参数和 --videos 只选一种，避免误选输入')
    return args


def main(argv=None):
    args = parse_arguments(argv)
    try:
        inputs = validate_inputs(args.videos or args.inputs or VIDEO_PATHS)
        try:
            import av
        except ImportError as exc:
            raise RuntimeError('缺少PyAV。请在运行本脚本的Python环境安装：python3 -m pip install av') from exc
        # 先检查编码能力和全部输入，再初始化RKNN，不因错误输入创建归档/连接串口。
        av.codec.Codec('libx264', 'w')
        for source in inputs:
            with av.open(str(source)) as reader:
                if not reader.streams.video:
                    raise ValueError('文件没有视频流：' + str(source))
        initialize_inference(args)
        if args.output_dir:
            directory = Path(args.output_dir).expanduser().resolve()
            directory.mkdir(parents=True, exist_ok=True)
        else:
            root = Path(OUTPUT_ROOT)
            root.mkdir(parents=True, exist_ok=True)
            directory = Path(tempfile.mkdtemp(prefix=datetime.now().strftime('%Y%m%d_%H%M%S_'), dir=str(root)))
        outputs = plan_outputs(inputs, directory)
        print('[output] 唯一保存目录：' + str(directory), flush=True)
        failures = 0
        partials = 0
        for video_id, (source, target) in enumerate(zip(inputs, outputs), 1):
            try:
                result = process_local_video(source, target, video_id, args)
                partials += int(result['partial'])
            except Exception as exc:
                failures += 1
                print(f'[failed] {source}: {exc}', file=sys.stderr, flush=True)
        print(f'[finished] 完整保存 {len(inputs) - failures - partials}；部分保存 {partials}；失败 {failures}；'
              '程序退出，不循环播放。')
        return 1 if failures else (2 if partials else 0)
    except KeyboardInterrupt:
        print('\n[stopped] 用户中断；当前未完成视频不作为完整结果保存。', file=sys.stderr)
        return 130
    except Exception as exc:
        print('[error] ' + str(exc), file=sys.stderr)
        return 1
    finally:
        release_inference()


if __name__ == '__main__':
    sys.exit(main())
