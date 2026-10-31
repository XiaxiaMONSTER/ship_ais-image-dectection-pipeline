"""
这个yolo测试onnx版本可以成功加载，但返回处理有bug，或者说转换过程中模型功能失效。

"""


import cv2
import sys
import numpy as np
import time
import onnxruntime as ort
from collections import defaultdict
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os
import shutil
import re

# ======================== PIL导入（可视化兼容处理） ========================
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️ 未安装PIL库，中文显示可能异常，请执行：pip install pillow")


# 统一模型路径：只把 .pt 改成 .onnx，其余完全不变
CLS_MODEL_PATH = "./models/yolov11n-cls/1-best.onnx"
DET_DAY_MODEL_PATH = "./models/yolov11n-day/5-best.onnx"
DET_NIGHT_MODEL_PATH = "./models/yolov11n-night/2-best.onnx"

# ======================== 调整：适配单类检测模型的类别字典 ========================
CLS_NAMES = {
    0: "cloudy",
    1: "foggy",
    2: "night",
    3: "rainy",
    4: "sunny"
}
# 单类检测模型：白天=ship（ID 0），夜晚=light（ID 0）
DET_DAY_NAMES = {0: "ship"}   # 白天单类：船只
DET_NIGHT_NAMES = {0: "light"} # 夜晚单类：灯光

# 4路视频路径 —— 完全不动
VIDEO_PATHS = [
    r"E:\数据集\批次2（夜晚）\718save_carmera\2025year_07month_18day_01_32_41_index001_CT-00000000639386512.mp4",
    r"P:\.vscode\Ship_Detection\ultralytics-yolo11-main\dataset\detect_video\近处小船（P船号）.mp4",
    r"P:\.vscode\Ship_Detection\ultralytics-yolo11-main\dataset\detect_video\晴天短视频.mp4",
    r"E:\数据集\批次3\723save_carmera（雾天）\2025year_07month_23day_09_57_06_index001_CT000000000452202992.mp4"
]

# 模型输入尺寸 —— 完全不动
CLS_INPUT_SIZE = 256
DET_INPUT_SIZE = 640

# 告警阈值 —— 完全不动
SHIP_ALARM_THRESH = 5000
LIGHT_ALARM_THRESH = 1200

# 工作模式 —— 完全不动
DEFAULT_WORK_DURATION = 120
continuous_mode = False
start_work_time = time.time()

# 绘制配置 —— 完全不动
FONT = cv2.FONT_HERSHEY_SIMPLEX
CHINESE_FONT_PATH = None
font_candidates = [
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/msyh.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
]
for font_path in font_candidates:
    if os.path.exists(font_path):
        CHINESE_FONT_PATH = font_path
        break

FONT_SCALE = 1
FONT_MULTIPALE = 20
FONT_THICKNESS = 1
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

# ======================== 稳像相关 —— 完全不动 ========================
ENABLE_STABILIZATION = True
STAB_SMOOTHING_RADIUS = 30
STAB_CROP_RATIO = 0.95
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ======================== PaddleOCR V5 配置 —— 完全不动 ========================
ENABLE_OCR = True
OCR_USE_GPU = True
OCR_MODEL_VERSION = "v5"
OCR_LANG = "ch"
OCR_CONF_THRESH = 0.5

# ======================== ROI图片保存配置 —— 完全不动 ========================
SAVE_OCR_ROI = False
ROI_SAVE_ROOT = "./ocr_roi_images"
ROI_SAVE_FORMAT = "jpg"
ROI_QUALITY = 95

# ======================== 加载模型：优化ONNX加载（修复点1） ========================
def load_onnx(model_path):
    try:
        # 增加ONNX运行时优化，提升稳定性
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return ort.InferenceSession(model_path, sess_options=sess_options, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    except Exception as e:
        print(f"CUDA加载失败，使用CPU：{e}")
        return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

cls_model = load_onnx(CLS_MODEL_PATH)
det_day_model = load_onnx(DET_DAY_MODEL_PATH)
det_night_model = load_onnx(DET_NIGHT_MODEL_PATH)

# ======================== 核心函数：修复分类/检测逻辑（关键修复点） ========================
def get_weather_class(frame):
    h, w = frame.shape[:2]
    scale = min(CLS_INPUT_SIZE/w, CLS_INPUT_SIZE/h)
    new_w, new_h = int(w*scale), int(h*scale)
    resized = cv2.resize(frame, (new_w, new_h))
    pad_x = (CLS_INPUT_SIZE - new_w)//2
    pad_y = (CLS_INPUT_SIZE - new_h)//2
    padded = cv2.copyMakeBorder(resized, pad_y, CLS_INPUT_SIZE-new_h-pad_y, pad_x, CLS_INPUT_SIZE-new_w-pad_x, cv2.BORDER_CONSTANT, value=0)
    blob = padded.astype(np.float32) / 255.0
    blob = blob.transpose(2,0,1)[None]
    out = cls_model.run(None, {cls_model.get_inputs()[0].name: blob})[0][0]
    top1_idx = np.argmax(out)
    top1_conf = float(out[top1_idx])
    # 修改：用字典映射类别名（匹配你的分类结果）
    cls_label = CLS_NAMES.get(top1_idx, f"class_{top1_idx}")
    print(f"分类结果：{cls_label} | 置信度：{top1_conf:.3f}")
    return cls_label, top1_conf

def detect_target(frame, weather_cls):
    model = det_night_model if weather_cls == "night" else det_day_model
    h, w = frame.shape[:2]
    scale = min(DET_INPUT_SIZE / w, DET_INPUT_SIZE / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))
    pad_x = (DET_INPUT_SIZE - new_w) // 2
    pad_y = (DET_INPUT_SIZE - new_h) // 2
    padded = cv2.copyMakeBorder(resized, pad_y, DET_INPUT_SIZE-new_h-pad_y, pad_x, DET_INPUT_SIZE-new_w-pad_x, cv2.BORDER_CONSTANT, value=0)
    
    blob = padded.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[None]
    
    pred = model.run(None, {model.get_inputs()[0].name: blob})[0]
    
    det_results = []
    for box in pred[0]:
        x1, y1, x2, y2, conf = box[:5]
        if conf < 0.25:
            continue
        
        cls = 0

        # ✅ 正确还原：原图绝对坐标（不再是 0~1）
        x1_ori = (x1 - pad_x) / scale
        y1_ori = (y1 - pad_y) / scale
        x2_ori = (x2 - pad_x) / scale
        y2_ori = (y2 - pad_y) / scale

        # 限制在图像内
        x1_ori = np.clip(x1_ori, 0, w)
        y1_ori = np.clip(y1_ori, 0, h)
        x2_ori = np.clip(x2_ori, 0, w)
        y2_ori = np.clip(y2_ori, 0, h)

        det_results.append((float(x1_ori), float(y1_ori), float(x2_ori), float(y2_ori), float(conf), float(cls)))
    return det_results

def postprocess_detections(det_results, orig_frame_shape, weather_cls):
    h, w = orig_frame_shape
    post_results = []
    has_alarm = False
    for (x1n, y1n, x2n, y2n, conf, cls) in det_results:
        # ❌ 删掉这里的 *w *h，这是坐标全在左上角的元凶
        x1 = int(round(x1n))
        y1 = int(round(y1n))
        x2 = int(round(x2n))
        y2 = int(round(y2n))

        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        area = width * height

        if weather_cls == "night":
            is_alarm = area >= LIGHT_ALARM_THRESH
        else:
            is_alarm = area >= SHIP_ALARM_THRESH

        if is_alarm:
            has_alarm = True
        post_results.append((x1, y1, x2, y2, conf, cls, area, is_alarm))
    return post_results, has_alarm

# 以下函数（draw_custom_boxes及后续）完全不动
def draw_custom_boxes(frame, post_results, weather_info, model_names):
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
    frame = draw_rounded_rectangle(frame, (tl_x, tl_y), (br_x, br_y), TEXT_BG_COLOR, thickness=-1, radius=CORNER_RADIUS)
    text_x = tl_x + PADDING
    text_y = tl_y + PADDING
    frame = draw_chinese_text(frame, weather_text, (text_x, text_y), font_size=FONT_SCALE, color=TEXT_COLOR, thickness=FONT_THICKNESS)
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
        frame = draw_rounded_rectangle(frame, (label_tl_x, label_tl_y), (label_br_x, label_br_y), TEXT_BG_COLOR, thickness=-1, radius=CORNER_RADIUS)
        label_text_x = label_tl_x + PADDING
        label_text_y = label_tl_y + PADDING
        frame = draw_chinese_text(frame, label_text, (label_text_x, label_text_y), font_size=FONT_SCALE, color=TEXT_COLOR, thickness=FONT_THICKNESS)
    return frame

def check_work_mode(has_alarm):
    global continuous_mode
    if has_alarm:
        print("发现告警目标！！！")
        continuous_mode = True
    else:
        continuous_mode = False
    return continuous_mode

def check_work_duration():
    if continuous_mode:
        return True
    elapsed_time = time.time() - start_work_time
    return elapsed_time < DEFAULT_WORK_DURATION

# ======================== 稳像类/ROI/OCR相关函数：完全不动 ========================
class RealTimeVideoStabilizer:
    def __init__(self, smoothing_radius=30, crop_ratio=0.9):
        self.smoothing_radius = smoothing_radius
        self.crop_ratio = crop_ratio
        self.prev_frame = None
        self.prev_gray = None
        self.prev_points = None
        self.trajectory = []
        self.smoothed_trajectory = []
        self.cumulative_transform = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        self.initialized = False

    def extract_features(self, frame, max_corners=200):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        feature_params = dict(maxCorners=max_corners, qualityLevel=0.01, minDistance=30, blockSize=7)
        corners = cv2.goodFeaturesToTrack(gray, mask=None,** feature_params)
        return corners

    def calculate_optical_flow(self, prev_gray, curr_gray, prev_points):
        lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        curr_points, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_points, None, **lk_params)
        return curr_points, status

    def estimate_rigid_transform(self, prev_points, curr_points):
        transform_matrix, inliers = cv2.estimateAffinePartial2D(prev_points, curr_points, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if transform_matrix is None:
            transform_matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        return transform_matrix

    def decompose_transform(self, transform):
        dx = transform[0, 2]
        dy = transform[1, 2]
        da = np.arctan2(transform[1, 0], transform[0, 0])
        return dx, dy, da

    def smooth_trajectory(self, trajectory):
        if len(trajectory) <= 1:
            return trajectory
        window_size = 2 * self.smoothing_radius + 1
        start_idx = max(0, len(trajectory) - window_size)
        recent_traj = np.array(trajectory[start_idx:])
        smoothed = np.copy(trajectory)
        if len(recent_traj) > 1:
            kernel = np.ones(len(recent_traj)) / len(recent_traj)
            for i in range(3):
                smoothed[-1, i] = np.sum(recent_traj[:, i] * kernel)
        return smoothed

    def apply_transform(self, frame, transform):
        h, w = frame.shape[:2]
        stabilized = cv2.warpAffine(frame, transform, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        crop_h = int(h * self.crop_ratio)
        crop_w = int(w * self.crop_ratio)
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2
        cropped = stabilized[start_h:start_h+crop_h, start_w:start_w+crop_w]
        return cropped

    def process_frame(self, curr_frame):
        if not self.initialized:
            self.prev_frame = curr_frame.copy()
            self.prev_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            self.prev_points = self.extract_features(self.prev_gray)
            self.initialized = True
            return curr_frame
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        if self.prev_points is None or len(self.prev_points) < 4:
            self.prev_points = self.extract_features(self.prev_gray)
            if self.prev_points is None or len(self.prev_points) < 4:
                self.prev_frame = curr_frame.copy()
                self.prev_gray = curr_gray
                return curr_frame
        curr_points, status = self.calculate_optical_flow(self.prev_gray, curr_gray, self.prev_points)
        valid_prev = self.prev_points[status == 1]
        valid_curr = curr_points[status == 1]
        if len(valid_prev) < 4:
            self.prev_points = self.extract_features(self.prev_gray)
            self.prev_frame = curr_frame.copy()
            self.prev_gray = curr_gray
            return curr_frame
        transform = self.estimate_rigid_transform(valid_prev, valid_curr)
        self.cumulative_transform = np.vstack([self.cumulative_transform @ np.vstack([transform, [0, 0, 1]]),])[:2, :]
        dx, dy, da = self.decompose_transform(self.cumulative_transform)
        self.trajectory.append([dx, dy, da])
        self.smoothed_trajectory = self.smooth_trajectory(self.trajectory)
        if len(self.smoothed_trajectory) > 0 and len(self.trajectory) > 0:
            diff_dx = self.smoothed_trajectory[-1][0] - self.trajectory[-1][0]
            diff_dy = self.smoothed_trajectory[-1][1] - self.trajectory[-1][1]
            diff_da = self.smoothed_trajectory[-1][2] - self.trajectory[-1][2]
            comp_transform = np.array([[np.cos(diff_da), -np.sin(diff_da), diff_dx],[np.sin(diff_da), np.cos(diff_da), diff_dy]], dtype=np.float32)
            stabilized_frame = self.apply_transform(curr_frame, comp_transform)
        else:
            stabilized_frame = curr_frame.copy()
        self.prev_frame = curr_frame.copy()
        self.prev_gray = curr_gray
        self.prev_points = self.extract_features(curr_gray)
        return stabilized_frame

    def reset(self):
        self.prev_frame = None
        self.prev_gray = None
        self.prev_points = None
        self.trajectory = []
        self.smoothed_trajectory = []
        self.cumulative_transform = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        self.initialized = False

def create_camera_dir(root_dir, camera_id):
    camera_dir = os.path.join(root_dir, f"camera_{camera_id}")
    os.makedirs(camera_dir, exist_ok=True)
    return camera_dir

def generate_roi_filename(camera_id, x1, y1, x2, y2):
    timestamp = int(time.time() * 1000)
    filename = f"roi_camera{camera_id}_{timestamp}_x{x1}y{y1}x{x2}y{y2}.{ROI_SAVE_FORMAT}"
    return filename

def init_paddle_ocr_v5():
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(
            text_detection_model_dir=r"P:\.vscode\Ship_Detection\pipeline\models\paddleocr_v5\PP-OCRv5_server_det",
            text_recognition_model_dir=r"P:\.vscode\Ship_Detection\pipeline\models\paddleocr_v5\PP-OCRv5_server_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            ocr_version="PP-OCRv5",
            device="gpu"
        )
        print("✅ PaddleOCR V5 初始化成功")
        return ocr
    except ImportError:
        print("❌ 未安装PaddleOCR，请执行：pip install paddleocr>=2.7.0")
        sys.exit(1)
    except Exception as e:
        print(f"❌ PaddleOCR V5 初始化失败：{str(e)}")
        sys.exit(1)

if ENABLE_OCR:
    ocr_engine = init_paddle_ocr_v5()
else:
    ocr_engine = None

def is_valid_text(text, min_length=2, min_chinese_count=1, min_digit_count=1):
    if not text or not isinstance(text, str):
        return False
    text = text.strip()
    if len(text) < min_length:
        return False
    meaningless_patterns = ['-', '.', ',', ' ', '', 'ane', 'one', 'the', 'and', 'or','。', '，', '、', '；', '：', '？', '！','一', '二', '三', '四', '五', '六', '七', '八', '九', '十']
    if text in meaningless_patterns:
        return False
    if re.match(r'^[\W_]+$', text):
        return False
    chinese_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    digit_count = sum(1 for c in text if c.isdigit())
    letter_count = sum(1 for c in text if c.isalpha() and not ('\u4e00' <= c <= '\u9fff'))
    has_chinese = chinese_count > 0
    has_digits = digit_count > 0
    has_letters = letter_count > 0
    if len(text) == 1 and text.isalpha():
        return False
    if has_letters and not has_chinese and not has_digits and len(text) < 5:
        common_english_words = ['and', 'the', 'for', 'you', 'are', 'was', 'not', 'but', 'had']
        if text.lower() in common_english_words:
            return False
    valid_combinations = [has_chinese and has_digits, has_chinese and has_letters, has_chinese and has_digits and has_letters, has_digits and len(text) >= 4, has_chinese and chinese_count >= 2]
    if any(valid_combinations):
        return True
    if digit_count / len(text) > 0.6 and len(text) >= 3:
        return True
    if has_letters and len(text) >= 3 and not has_chinese and not has_digits:
        if text.islower() and len(text) <= 3:
            return False
        return True
    return False

def draw_chinese_text(img, text, pos, font_size=0.8, color=(255,255,255), thickness=2):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    try:
        if CHINESE_FONT_PATH and PIL_AVAILABLE:
            font_height = max(12, int(font_size * FONT_MULTIPALE))
            font = ImageFont.truetype(CHINESE_FONT_PATH, font_height)
        else:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    pil_color = (color[2], color[1], color[0])
    draw.text(pos, text, font=font, fill=pil_color, stroke_width = 0.5)
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img_bgr

def get_text_size(text, font_size=0.8):
    if not PIL_AVAILABLE or not CHINESE_FONT_PATH:
        (w, h), _ = cv2.getTextSize(text, FONT, font_size, FONT_THICKNESS)
        return (w, h)
    try:
        font_height = max(12, int(font_size * FONT_MULTIPALE))
        font = ImageFont.truetype(CHINESE_FONT_PATH, font_height)
        dummy_img = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(dummy_img)
        try:
            left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        except AttributeError:
            left, top, right, bottom = font.getbbox(text)
        w = right - left
        h = bottom - top
        return (w + 2, h + 2)
    except:
        (w, h), _ = cv2.getTextSize(text, FONT, font_size, FONT_THICKNESS)
        return (w, h)

def process_ship_ocr(frame, post_results, weather_label, camera_id):
    ocr_results = []
    # 修复点7：匹配英文分类结果（night）
    if not ENABLE_OCR or ocr_engine is None or weather_label == "night":
        return ocr_results, frame
    alarm_ship_rois = []
    for res in post_results:
        if len(res) < 8:
            continue
        x1, y1, x2, y2, conf, cls, area, is_alarm = res
        if is_alarm:
            alarm_ship_rois.append(res)
    if len(alarm_ship_rois) == 0:
        return ocr_results, frame
    rec_texts = []
    rec_scores = []
    rec_polys = []
    for roi_res in alarm_ship_rois:
        if len(roi_res) < 8:
            continue
        x1, y1, x2, y2, conf, cls, area, is_alarm = roi_res
        h, w = frame.shape[:2]
        roi_x1 = max(0, x1 - 5)
        roi_y1 = max(0, y1 - 5)
        roi_x2 = min(w, x2 + 5)
        roi_y2 = min(h, y2 + 5)
        if (roi_x2 - roi_x1) < 10 or (roi_y2 - roi_y1) < 10:
            continue
        ship_roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        if ship_roi.size == 0:
            continue
        if SAVE_OCR_ROI:
            try:
                camera_dir = create_camera_dir(ROI_SAVE_ROOT, camera_id)
                roi_filename = generate_roi_filename(camera_id, x1, y1, x2, y2)
                save_path = os.path.join(camera_dir, roi_filename)
                if ROI_SAVE_FORMAT.lower() == "jpg":
                    cv2.imwrite(save_path, ship_roi, [cv2.IMWRITE_JPEG_QUALITY, ROI_QUALITY])
                elif ROI_SAVE_FORMAT.lower() == "png":
                    cv2.imwrite(save_path, ship_roi, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                print(f"✅ 保存ROI图片：{save_path}")
            except Exception as e:
                print(f"⚠️ 保存ROI图片失败：{str(e)}")
        try:
            result = ocr_engine.predict(ship_roi)
            if isinstance(result, list) and len(result) > 0:
                first_result = result[0]
                if isinstance(first_result, dict):
                    rec_texts = first_result.get('rec_texts', [])
                    rec_scores = first_result.get('rec_scores', [])
                    rec_polys = first_result.get('rec_polys', [])
                    print(f"找到 {len(rec_texts)} 个识别文本: {rec_texts}")
                    print(f"对应置信度: {rec_scores}")
        except Exception as e:
            print(f"⚠️ OCR推理异常（ROI区域）：{str(e)}")
            continue
        if len(rec_texts) == 0 and len(rec_scores) == 0 and len(rec_polys) == 0:
            print(f"ℹ️ 当前ROI无有效识别文字（x1:{x1},y1:{y1},x2:{x2},y2:{y2}）")
            continue
        elif len(rec_texts) != len(rec_scores) or len(rec_texts) != len(rec_polys):
            print(f"⚠️ OCR结果字段长度不匹配：文本数={len(rec_texts)}, 置信度数={len(rec_scores)}, 坐标数={len(rec_polys)}")
            continue
        best_text = None
        best_score = 0
        best_poly = None
        best_idx = -1
        for idx in range(len(rec_texts)):
            text = rec_texts[idx]
            score = float(rec_scores[idx]) if idx < len(rec_scores) else 0
            poly = rec_polys[idx] if idx < len(rec_polys) else []
            if score < OCR_CONF_THRESH:
                print(f"  ✗ 跳过低置信度文本: '{text}' ({score:.3f})")
                continue
            if not is_valid_text(text):
                print(f"  ✗ 跳过无效文本: '{text}'")
                continue
            if score > best_score:
                best_text = text
                best_score = score
                best_poly = poly
                best_idx = idx
        if best_text:
            print(f"✅ 选择最佳识别结果: '{best_text}' (置信度: {best_score:.3f})")
            try:
                if best_poly is not None and len(best_poly) >= 4:
                    poly_flat = [coord for point in best_poly for coord in point]
                    text_x1 = min(poly_flat[::2])
                    text_y1 = min(poly_flat[1::2])
                    text_x2 = max(poly_flat[::2])
                    text_y2 = max(poly_flat[1::2])
                    text_x1 += roi_x1
                    text_y1 += roi_y1
                    text_x2 += roi_x1
                    text_y2 += roi_y1
                    text_x1 = max(0, text_x1)
                    text_y1 = max(0, text_y1)
                    text_x2 = min(w, text_x2)
                    text_y2 = min(h, text_y2)
                else:
                    text_w, text_h = get_text_size(f"{best_text} ({best_score:.2f})", FONT_SCALE)
                    label_height = text_h + 2 * PADDING
                    label_tl_y = y1 - label_height - LABEL_OFFSET
                    if label_tl_y < 0:
                        label_tl_y = y2 + LABEL_OFFSET
                    text_x1 = x1
                    text_y1 = label_tl_y
                    text_x2 = x2
                    text_y2 = label_tl_y + label_height
            except Exception as e:
                print(f"⚠️ 坐标解析异常：{str(e)}")
                text_w, text_h = get_text_size(f"{best_text} ({best_score:.2f})", FONT_SCALE)
                label_height = text_h + 2 * PADDING
                label_tl_y = y1 - label_height - LABEL_OFFSET if y1 - label_height - LABEL_OFFSET > 0 else y2 + LABEL_OFFSET
                text_x1 = x1
                text_y1 = label_tl_y
                text_x2 = x2
                text_y2 = label_tl_y + label_height
            display_text = f"{best_text}({best_score:.2f})"
            text_w, text_h = get_text_size(display_text, FONT_SCALE)
            label_tl_x = text_x1
            label_tl_y = text_y1 - text_h - 2 * PADDING - LABEL_OFFSET
            label_br_x = label_tl_x + text_w + 2 * PADDING
            label_br_y = label_tl_y + text_h + 2 * PADDING
            if label_br_x > w:
                label_tl_x = max(0, w - text_w - 2 * PADDING - 10)
                label_br_x = label_tl_x + text_w + 2 * PADDING
            if label_br_y > h:
                label_tl_y = max(0, h - text_h - 2 * PADDING - 10)
                label_br_y = label_tl_y + text_h + 2 * PADDING
            ocr_results.append({"text": best_text, "conf": best_score, "bbox": (text_x1, text_y1, text_x2, text_y2), "ship_bbox": (x1, y1, x2, y2), "camera_id": camera_id})
            try:
                frame = draw_rounded_rectangle(frame,(label_tl_x, label_tl_y),(label_br_x, label_br_y),OCR_BG_COLOR,thickness=-1,radius=CORNER_RADIUS)
                text_pos_x = label_tl_x + PADDING
                text_pos_y = label_tl_y + PADDING
                frame = draw_chinese_text(frame,display_text,(text_pos_x, text_pos_y),font_size=FONT_SCALE,color=OCR_TEXT_COLOR,thickness=FONT_THICKNESS)
                if best_poly is not None and len(best_poly) >= 4:
                    cv2.rectangle(frame, (text_x1, text_y1), (text_x2, text_y2), OCR_BOX_COLOR, BOX_THICKNESS//2)
            except Exception as e:
                print(f"⚠️ OCR绘制异常：{str(e)}")
        else:
            print(f"ℹ️ 当前ROI无有效识别文字（经过过滤）")
    return ocr_results, frame

def draw_rounded_rectangle(img, top_left, bottom_right, color, thickness=-1, radius=10):
    x1, y1 = top_left
    x2, y2 = bottom_right
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = max(x1+1, x2), max(y1+1, y2)
    radius = min(radius, (x2-x1)//2, (y2-y1)//2)
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

# ======================== 主程序：仅匹配英文分类结果 ========================
if __name__ == "__main__":
    if not PIL_AVAILABLE:
        print("📦 正在安装PIL库...")
        os.system("pip install pillow -q")
        try:
            from PIL import Image, ImageDraw, ImageFont
            PIL_AVAILABLE = True
            print("✅ PIL库安装成功")
        except:
            print("❌ PIL库安装失败，中文显示可能异常")

    caps = []
    stabilizers = []
    for path in VIDEO_PATHS:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"警告：无法打开视频 {path}")
            caps.append(None)
            stabilizers.append(None)
        else:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            caps.append(cap)
            if ENABLE_STABILIZATION:
                stabilizer = RealTimeVideoStabilizer(smoothing_radius=STAB_SMOOTHING_RADIUS, crop_ratio=STAB_CROP_RATIO)
                stabilizers.append(stabilizer)
            else:
                stabilizers.append(None)

    while True:
        if not check_work_duration():
            print(f"工作时长已达{DEFAULT_WORK_DURATION}秒，退出程序")
            break

        for idx, (cap, stabilizer) in enumerate(zip(caps, stabilizers)):
            if cap is None:
                continue
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                if stabilizer is not None:
                    stabilizer.reset()
                continue

            if ENABLE_STABILIZATION and stabilizer is not None:
                stabilized_frame = stabilizer.process_frame(frame)
            else:
                stabilized_frame = frame

            weather_label, weather_conf = get_weather_class(stabilized_frame)
            weather_info = (weather_label, weather_conf)
            det_results = detect_target(stabilized_frame, weather_label)
            orig_h, orig_w = stabilized_frame.shape[:2]
            post_results, has_alarm = postprocess_detections(det_results, (orig_h, orig_w), weather_label)
            check_work_mode(has_alarm)
            ocr_results, stabilized_frame = process_ship_ocr(stabilized_frame.copy(), post_results, weather_label, idx+1)
            if ocr_results:
                print(f"❗相机{idx+1} OCR识别结果：{[item['text'] for item in ocr_results]}")

            # 修复点8：匹配英文分类结果（night）
            model_names = DET_NIGHT_NAMES if weather_label == "night" else DET_DAY_NAMES
            frame_drawn = draw_custom_boxes(stabilized_frame, post_results, weather_info, model_names)
            cv2.imshow(f"Camera {idx+1} (Stabilized+OCR)", cv2.resize(frame_drawn, (1344, 760)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("用户手动退出程序")
            break

    for cap in caps:
        if cap is not None:
            cap.release()
    cv2.destroyAllWindows()