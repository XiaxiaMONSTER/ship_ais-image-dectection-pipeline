"""
deepseek给出的,绘制不对，没用
"""


import cv2
import sys
import numpy as np
import time
from ultralytics import YOLO
from collections import defaultdict
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os
import shutil
import re
import onnxruntime as ort  # 新增导入

# ======================== PIL导入（可视化兼容处理） ========================
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️ 未安装PIL库，中文显示可能异常，请执行：pip install pillow")


# 统一模型路径
CLS_MODEL_PATH = "./models/yolov11n-cls/1-best.onnx"  # 场景分类模型
DET_DAY_MODEL_PATH = "./models/yolov11n-day/5-best.onnx"  # 白天船舶检测模型
DET_NIGHT_MODEL_PATH = "./models/yolov11n-night/2-best.onnx"  # 夜晚灯光检测模型

# 4路视频路径
VIDEO_PATHS = [
    r"E:\数据集\批次2（夜晚）\718save_carmera\2025year_07month_18day_01_32_41_index001_CT-00000000639386512.mp4",  #夜晚   正常
    #0  # 摄像头1（默认摄像头）
    #r"P:\.vscode\Ship_Detection\ultralytics-yolo11-main\dataset\detect_video\近处小船（P船号）.mp4",     #多云    sunny
    #r"P:\.vscode\Ship_Detection\ultralytics-yolo11-main\dataset\detect_video\晴天短视频.mp4",  #晴天    night
    #r"E:\数据集\批次3\723save_carmera（雾天）\2025year_07month_23day_09_57_06_index001_CT000000000452202992.mp4"  #雾天  正常
]

# 模型输入尺寸
CLS_INPUT_SIZE = 256  
DET_INPUT_SIZE = 640

# 告警阈值
SHIP_ALARM_THRESH = 5000
LIGHT_ALARM_THRESH = 1200

# 工作模式
DEFAULT_WORK_DURATION = 120
continuous_mode = False
# 全局计时起点
start_work_time = time.time()

# 绘制配置
FONT = cv2.FONT_HERSHEY_SIMPLEX
# 中文字体路径
CHINESE_FONT_PATH = None
font_candidates = [
    "C:/Windows/Fonts/simhei.ttf",    # Windows 黑体
    "C:/Windows/Fonts/msyh.ttc",      # Windows 微软雅黑
    "/System/Library/Fonts/PingFang.ttc",  # Mac 苹方
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # Linux
]
for font_path in font_candidates:
    if os.path.exists(font_path):
        CHINESE_FONT_PATH = font_path
        break

# 统一样式参数
FONT_SCALE = 1       # 文本大小
FONT_MULTIPALE = 20   # PIL文字计算倍数
FONT_THICKNESS = 1     # 文本线条厚度
ALARM_COLOR = (0, 0, 255)       # 告警目标：红色
NORMAL_COLOR = (0, 255, 0)       # 正常目标：绿色
TEXT_COLOR = (255, 255, 255)     # 文本颜色：白色
TEXT_BG_COLOR = (0, 0, 0)        # 文本背景：黑色（半透明优化）
CORNER_RADIUS = 8                # 圆角半径（越大越圆润）
PADDING = 2                      # 文本框内边距（增加内边距，避免文字贴边）
BOX_THICKNESS = 2                # 检测框线条厚度
OCR_TEXT_COLOR = (255, 255, 255) # OCR识别文字颜色：白色
OCR_BG_COLOR = (0, 0, 0)         # OCR文字背景色：黑色
OCR_BOX_COLOR = (255, 165, 0)    # OCR文本框边框色：蓝色
LABEL_OFFSET = 8                 # 标签框与检测框的间距

# ======================== 稳像相关配置 ========================
ENABLE_STABILIZATION = True  # 稳像功能总开关
STAB_SMOOTHING_RADIUS = 30   # 平滑窗口半径（越大越稳，实时场景建议10-30）
STAB_CROP_RATIO = 0.95       # 裁剪比例（实时场景建议0.9-0.95，减少黑边）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 设置字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ======================== PaddleOCR V5 配置 ========================
ENABLE_OCR = True  # OCR功能总开关
OCR_USE_GPU = True  # 启用GPU加速
OCR_MODEL_VERSION = "v5"  # 指定PaddleOCR V5版本
# 仅识别白天告警船只的ROI，可配置识别语言（中文/英文/数字）
OCR_LANG = "ch"  # ch=中文, en=英文, ch_en=中英混合
# OCR置信度阈值（过滤低置信度结果）
OCR_CONF_THRESH = 0.5

# ======================== ROI图片保存配置 ========================
SAVE_OCR_ROI = False  # 是否保存OCR检测的ROI图片
ROI_SAVE_ROOT = "./ocr_roi_images"  # ROI保存根目录
ROI_SAVE_FORMAT = "jpg"  # 保存格式（jpg/png）
ROI_QUALITY = 95  # JPG保存质量（0-100）

# ======================== 稳像类（实时逐帧处理） ========================
class RealTimeVideoStabilizer:
    """
    实时逐帧视频稳像类（基于特征点跟踪+运动补偿）
    适配实时视频流场景，而非批量处理整段视频
    """
    def __init__(self, smoothing_radius=30, crop_ratio=0.9):
        self.smoothing_radius = smoothing_radius
        self.crop_ratio = crop_ratio
        
        # 帧缓存
        self.prev_frame = None
        self.prev_gray = None
        self.prev_points = None
        
        # 运动轨迹缓存
        self.trajectory = []
        self.smoothed_trajectory = []
        
        # 累积变换矩阵
        self.cumulative_transform = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        
        # 初始化完成标记
        self.initialized = False

    def extract_features(self, frame, max_corners=200):
        """提取图像特征点（Shi-Tomasi角点检测）"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        feature_params = dict(
            maxCorners=max_corners,
            qualityLevel=0.01,
            minDistance=30,
            blockSize=7
        )
        corners = cv2.goodFeaturesToTrack(gray, mask=None,** feature_params)
        return corners

    def calculate_optical_flow(self, prev_gray, curr_gray, prev_points):
        """Lucas-Kanade光流法计算特征点运动"""
        lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        curr_points, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_points, None, **lk_params
        )
        return curr_points, status

    def estimate_rigid_transform(self, prev_points, curr_points):
        """估计刚性变换矩阵（平移+旋转+缩放）"""
        transform_matrix, inliers = cv2.estimateAffinePartial2D(
            prev_points, curr_points, method=cv2.RANSAC, ransacReprojThreshold=3.0
        )
        if transform_matrix is None:
            transform_matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        return transform_matrix

    def decompose_transform(self, transform):
        """分解变换矩阵为平移、旋转分量"""
        dx = transform[0, 2]
        dy = transform[1, 2]
        da = np.arctan2(transform[1, 0], transform[0, 0])
        return dx, dy, da

    def smooth_trajectory(self, trajectory):
        """实时轨迹平滑（简化版移动平均）"""
        if len(trajectory) <= 1:
            return trajectory
        
        # 仅保留最近的2*radius+1个轨迹点进行平滑
        window_size = 2 * self.smoothing_radius + 1
        start_idx = max(0, len(trajectory) - window_size)
        recent_traj = np.array(trajectory[start_idx:])
        
        smoothed = np.copy(trajectory)
        if len(recent_traj) > 1:
            # 简单移动平均
            kernel = np.ones(len(recent_traj)) / len(recent_traj)
            for i in range(3):
                smoothed[-1, i] = np.sum(recent_traj[:, i] * kernel)
        
        return smoothed

    def apply_transform(self, frame, transform):
        """应用变换矩阵并裁剪黑边"""
        h, w = frame.shape[:2]
        # 应用运动补偿
        stabilized = cv2.warpAffine(
            frame, transform, (w, h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )
        # 裁剪边缘去除黑边
        crop_h = int(h * self.crop_ratio)
        crop_w = int(w * self.crop_ratio)
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2
        cropped = stabilized[start_h:start_h+crop_h, start_w:start_w+crop_w]
        return cropped

    def process_frame(self, curr_frame):
        """
        实时处理单帧图像
        :param curr_frame: 输入的BGR帧
        :return: 稳像后的BGR帧
        """
        # 第一帧初始化
        if not self.initialized:
            self.prev_frame = curr_frame.copy()
            self.prev_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            self.prev_points = self.extract_features(self.prev_gray)
            self.initialized = True
            return curr_frame  # 第一帧无稳像，直接返回
        
        # 转换为灰度图
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # 特征点检测与光流计算
        if self.prev_points is None or len(self.prev_points) < 4:
            self.prev_points = self.extract_features(self.prev_gray)
            if self.prev_points is None or len(self.prev_points) < 4:
                # 无足够特征点，返回原帧
                self.prev_frame = curr_frame.copy()
                self.prev_gray = curr_gray
                return curr_frame
        
        # 计算光流
        curr_points, status = self.calculate_optical_flow(
            self.prev_gray, curr_gray, self.prev_points
        )
        
        # 筛选有效特征点
        valid_prev = self.prev_points[status == 1]
        valid_curr = curr_points[status == 1]
        
        if len(valid_prev) < 4:
            self.prev_points = self.extract_features(self.prev_gray)
            self.prev_frame = curr_frame.copy()
            self.prev_gray = curr_gray
            return curr_frame
        
        # 估计变换矩阵
        transform = self.estimate_rigid_transform(valid_prev, valid_curr)
        
        # 累积变换
        self.cumulative_transform = np.vstack([
            self.cumulative_transform @ np.vstack([transform, [0, 0, 1]]),
        ])[:2, :]
        
        # 分解变换并更新轨迹
        dx, dy, da = self.decompose_transform(self.cumulative_transform)
        self.trajectory.append([dx, dy, da])
        
        # 轨迹平滑
        self.smoothed_trajectory = self.smooth_trajectory(self.trajectory)
        
        # 计算补偿变换
        if len(self.smoothed_trajectory) > 0 and len(self.trajectory) > 0:
            diff_dx = self.smoothed_trajectory[-1][0] - self.trajectory[-1][0]
            diff_dy = self.smoothed_trajectory[-1][1] - self.trajectory[-1][1]
            diff_da = self.smoothed_trajectory[-1][2] - self.trajectory[-1][2]
            
            comp_transform = np.array([
                [np.cos(diff_da), -np.sin(diff_da), diff_dx],
                [np.sin(diff_da), np.cos(diff_da), diff_dy]
            ], dtype=np.float32)
            
            # 应用补偿变换
            stabilized_frame = self.apply_transform(curr_frame, comp_transform)
        else:
            stabilized_frame = curr_frame.copy()
        
        # 更新缓存
        self.prev_frame = curr_frame.copy()
        self.prev_gray = curr_gray
        self.prev_points = self.extract_features(curr_gray)  # 更新特征点
        
        return stabilized_frame

    def reset(self):
        """重置稳像器状态（用于视频循环播放）"""
        self.prev_frame = None
        self.prev_gray = None
        self.prev_points = None
        self.trajectory = []
        self.smoothed_trajectory = []
        self.cumulative_transform = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        self.initialized = False

# ======================== ROI保存辅助函数========================
def create_camera_dir(root_dir, camera_id):
    """
    创建相机专属文件夹
    :param root_dir: 根目录
    :param camera_id: 相机编号（1/2/3/4）
    :return: 相机文件夹路径
    """
    camera_dir = os.path.join(root_dir, f"camera_{camera_id}")
    # 递归创建文件夹（不存在则创建，存在则忽略）
    os.makedirs(camera_dir, exist_ok=True)
    return camera_dir

def generate_roi_filename(camera_id, x1, y1, x2, y2):
    """
    生成唯一的ROI文件名（避免重复）
    格式：roi_cameraID_时间戳_x1y1x2y2.jpg
    :param camera_id: 相机编号
    :param x1,y1,x2,y2: ROI坐标
    :return: 文件名
    """
    timestamp = int(time.time() * 1000)  # 毫秒级时间戳，避免重复
    filename = f"roi_camera{camera_id}_{timestamp}_x{x1}y{y1}x{x2}y{y2}.{ROI_SAVE_FORMAT}"
    return filename

# ======================== PaddleOCR V5 初始化与处理函数 ========================
def init_paddle_ocr_v5():
    """初始化PaddleOCR V5（轻量化配置）"""
    try:
        from paddleocr import PaddleOCR
        # PaddleOCR V5 轻量化配置（仅检测+识别，关闭方向分类）
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

# 初始化OCR实例
if ENABLE_OCR:
    ocr_engine = init_paddle_ocr_v5()
else:
    ocr_engine = None

def is_valid_text(text, min_length=2, min_chinese_count=1, min_digit_count=1):
    """
    判断文本是否为有效文本（过滤无意义字符）
    :param text: 识别的文本
    :param min_length: 最小长度要求
    :param min_chinese_count: 最少中文字符数
    :param min_digit_count: 最少数字数
    :return: True/False
    """
    if not text or not isinstance(text, str):
        return False
    
    text = text.strip()
    
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
    
    # 5. 船舶号码的常见特征
    has_chinese = chinese_count > 0
    has_digits = digit_count > 0
    has_letters = letter_count > 0
    
    # 排除纯单个字母的情况
    if len(text) == 1 and text.isalpha():
        return False
    
    # 排除纯英文单词（非船舶相关）
    if has_letters and not has_chinese and not has_digits and len(text) < 5:
        common_english_words = ['and', 'the', 'for', 'you', 'are', 'was', 'not', 'but', 'had']
        if text.lower() in common_english_words:
            return False
    
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
    
    # 纯字母但长度足够（可能是英文船名）
    if has_letters and len(text) >= 3 and not has_chinese and not has_digits:
        if text.islower() and len(text) <= 3:
            return False
        return True
    
    return False

def draw_chinese_text(img, text, pos, font_size=0.8, color=(255,255,255), thickness=2):
    """
    绘制支持中文的文本（基于PIL，兼容所有OpenCV版本）
    :param img: OpenCV BGR图像
    :param text: 要绘制的文本（支持中文）
    :param pos: 文本位置 (x, y) - 左上角坐标
    :param font_size: 字体大小
    :param color: 文本颜色 (B,G,R)
    :param thickness: 线条厚度
    :return: 绘制后的OpenCV图像
    """
    # 转换颜色空间：BGR -> RGB（PIL使用RGB）
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 转换为PIL图像
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    # 设置字体
    try:
        if CHINESE_FONT_PATH and PIL_AVAILABLE:
            # 计算字体大小（和OpenCV的FONT_SCALE对应）
            font_height = max(12, int(font_size * FONT_MULTIPALE))  # 给个最小值
            font = ImageFont.truetype(CHINESE_FONT_PATH, font_height)
        else:
            # 回退到默认字体
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    # 转换颜色格式：OpenCV (B,G,R) -> PIL (R,G,B)
    pil_color = (color[2], color[1], color[0])

    draw.text(pos, text, font=font, fill=pil_color,
            stroke_width = 0.5)

    # RGB -> BGR
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img_bgr

def get_text_size(text, font_size=0.8):
    """
    计算文本尺寸（支持中文）
    :return: (width, height)
    """
    if not PIL_AVAILABLE or not CHINESE_FONT_PATH:
        (w, h), _ = cv2.getTextSize(text, FONT, font_size, FONT_THICKNESS)
        return (w, h)

    try:
        font_height = max(12, int(font_size * FONT_MULTIPALE))
        font = ImageFont.truetype(CHINESE_FONT_PATH, font_height)

        # 这里用一个临时 ImageDraw 获取 textbbox，保证与 draw_chinese_text 一致
        dummy_img = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(dummy_img)
        try:
            left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        except AttributeError:
            left, top, right, bottom = font.getbbox(text)

        w = right - left
        h = bottom - top
        # 适当加 1~2 像素冗余，避免边界裁切
        return (w + 2, h + 2)
    except:
        (w, h), _ = cv2.getTextSize(text, FONT, font_size, FONT_THICKNESS)
        return (w, h)

def process_ship_ocr(frame, post_results, weather_label, camera_id):
    """
    仅对白天告警的船只ROI进行PaddleOCR V5识别（适配字典格式结果）
    :param frame: 稳像后的帧
    :param post_results: 后处理后的检测结果
    :param weather_label: 场景分类标签（白天/夜晚/雾天等）
    :param camera_id: 相机编号（1/2/3/4）
    :return: OCR结果列表、绘制后的帧
    """
    ocr_results = []
    # 1. 前置条件校验：仅处理白天+告警+OCR启用的场景
    if not ENABLE_OCR or ocr_engine is None or weather_label in ["night", "夜晚"]:
        return ocr_results, frame
    
    # 2. 筛选白天的告警船只ROI
    alarm_ship_rois = []
    for res in post_results:
        # 校验post_results格式，避免解包异常
        if len(res) < 8:
            continue
        x1, y1, x2, y2, conf, cls, area, is_alarm = res
        # 仅处理告警的船只（白天场景）
        if is_alarm:
            alarm_ship_rois.append(res)
    
    if len(alarm_ship_rois) == 0:
        return ocr_results, frame
    
    # 初始化OCR结果变量，避免未定义报错
    rec_texts = []
    rec_scores = []
    rec_polys = []
    
    # 3. 对每个告警船只ROI进行OCR识别
    for roi_res in alarm_ship_rois:
        # 再次校验ROI结果格式
        if len(roi_res) < 8:
            continue
        x1, y1, x2, y2, conf, cls, area, is_alarm = roi_res
        
        # 4. 校验ROI坐标有效性，避免裁剪异常
        h, w = frame.shape[:2]
        roi_x1 = max(0, x1 - 5)
        roi_y1 = max(0, y1 - 5)
        roi_x2 = min(w, x2 + 5)
        roi_y2 = min(h, y2 + 5)
        # 校验ROI尺寸（至少10x10像素，避免无效裁剪）
        if (roi_x2 - roi_x1) < 10 or (roi_y2 - roi_y1) < 10:
            continue
        
        ship_roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        if ship_roi.size == 0:
            continue
        
        # ======================== 保存ROI图片 ========================
        if SAVE_OCR_ROI:
            try:
                # 创建相机文件夹
                camera_dir = create_camera_dir(ROI_SAVE_ROOT, camera_id)
                # 生成唯一文件名
                roi_filename = generate_roi_filename(camera_id, x1, y1, x2, y2)
                # 拼接完整保存路径
                save_path = os.path.join(camera_dir, roi_filename)
                # 保存图片（区分jpg/png格式）
                if ROI_SAVE_FORMAT.lower() == "jpg":
                    cv2.imwrite(save_path, ship_roi, [cv2.IMWRITE_JPEG_QUALITY, ROI_QUALITY])
                elif ROI_SAVE_FORMAT.lower() == "png":
                    cv2.imwrite(save_path, ship_roi, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                print(f"✅ 保存ROI图片：{save_path}")
            except Exception as e:
                print(f"⚠️ 保存ROI图片失败：{str(e)}")
        
        # 5. PaddleOCR V5 推理（适配字典格式结果）
        try:
            # 执行OCR推理，返回字典格式结果
            result = ocr_engine.predict(ship_roi)
            
            # 解析OCR结果
            if isinstance(result, list) and len(result) > 0:
                first_result = result[0]  # 获取第一个结果
                if isinstance(first_result, dict):
                    rec_texts = first_result.get('rec_texts', [])
                    rec_scores = first_result.get('rec_scores', [])
                    rec_polys = first_result.get('rec_polys', [])
                    
                    print(f"找到 {len(rec_texts)} 个识别文本: {rec_texts}")
                    print(f"对应置信度: {rec_scores}")
        
        except Exception as e:
            print(f"⚠️ OCR推理异常（ROI区域）：{str(e)}")
            continue
        
        # 校验字段长度一致（一一对应）
        if len(rec_texts) == 0 and len(rec_scores) == 0 and len(rec_polys) == 0:
            # 空结果：仅打印调试日志（可选关闭）
            print(f"ℹ️ 当前ROI无有效识别文字（x1:{x1},y1:{y1},x2:{x2},y2:{y2}）")
            continue
        elif len(rec_texts) != len(rec_scores) or len(rec_texts) != len(rec_polys):
            # 真异常：字段长度不匹配（非空）
            print(f"⚠️ OCR结果字段长度不匹配：文本数={len(rec_texts)}, 置信度数={len(rec_scores)}, 坐标数={len(rec_polys)}")
            continue
        
        # 6. 过滤和选择最佳识别结果
        best_text = None
        best_score = 0
        best_poly = None
        best_idx = -1
        
        for idx in range(len(rec_texts)):
            text = rec_texts[idx]
            score = float(rec_scores[idx]) if idx < len(rec_scores) else 0
            poly = rec_polys[idx] if idx < len(rec_polys) else []
            
            # 跳过低置信度结果
            if score < OCR_CONF_THRESH:
                print(f"  ✗ 跳过低置信度文本: '{text}' ({score:.3f})")
                continue
            
            # 使用过滤函数检查文本有效性
            if not is_valid_text(text):
                print(f"  ✗ 跳过无效文本: '{text}'")
                continue
            
            # 更新最佳结果（置信度最高）
            if score > best_score:
                best_text = text
                best_score = score
                best_poly = poly
                best_idx = idx
        
        # 7. 如果有有效结果，进行处理
        if best_text:
            print(f"✅ 选择最佳识别结果: '{best_text}' (置信度: {best_score:.3f})")
            
            # 解析坐标框
            try:
                if best_poly is not None and len(best_poly) >= 4:
                    # poly格式：[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    poly_flat = [coord for point in best_poly for coord in point]
                    text_x1 = min(poly_flat[::2])  # 所有x坐标的最小值
                    text_y1 = min(poly_flat[1::2]) # 所有y坐标的最小值
                    text_x2 = max(poly_flat[::2])  # 所有x坐标的最大值
                    text_y2 = max(poly_flat[1::2]) # 所有y坐标的最大值
                    
                    # 还原到原帧坐标（ROI偏移）
                    text_x1 += roi_x1
                    text_y1 += roi_y1
                    text_x2 += roi_x1
                    text_y2 += roi_y1
                    
                    # 确保坐标在帧范围内
                    text_x1 = max(0, text_x1)
                    text_y1 = max(0, text_y1)
                    text_x2 = min(w, text_x2)
                    text_y2 = min(h, text_y2)
                else:
                    print("没有多边形坐标")  #调试没有输出，说明存在坐标，正常绘制了
                    # 如果没有多边形坐标，使用船只框上方位置
                    # 优化：标签框位置调整，确保在检测框上方且不超出画面
                    text_w, text_h = get_text_size(f"{best_text} ({best_score:.2f})", FONT_SCALE)
                    label_height = text_h + 2 * PADDING
                    
                    # 优先放在检测框上方
                    label_tl_y = y1 - label_height - LABEL_OFFSET
                    # 如果上方空间不足，放在检测框下方
                    if label_tl_y < 0:
                        label_tl_y = y2 + LABEL_OFFSET
                    
                    text_x1 = x1
                    text_y1 = label_tl_y
                    text_x2 = x2
                    text_y2 = label_tl_y + label_height
                    
            except Exception as e:
                print(f"⚠️ 坐标解析异常：{str(e)}")
                # 使用默认位置
                text_w, text_h = get_text_size(f"{best_text} ({best_score:.2f})", FONT_SCALE)
                label_height = text_h + 2 * PADDING
                label_tl_y = y1 - label_height - LABEL_OFFSET if y1 - label_height - LABEL_OFFSET > 0 else y2 + LABEL_OFFSET
                
                text_x1 = x1
                text_y1 = label_tl_y
                text_x2 = x2
                text_y2 = label_tl_y + label_height
            
            # 准备OCR显示文本
            display_text = f"{best_text}({best_score:.2f})"
            
            # 计算文本尺寸（支持中文）
            text_w, text_h = get_text_size(display_text, FONT_SCALE)
            
            # 计算OCR标签位置
            label_tl_x = text_x1
            label_tl_y = text_y1 - text_h - 2 * PADDING - LABEL_OFFSET
            label_br_x = label_tl_x + text_w + 2 * PADDING
            label_br_y = label_tl_y + text_h + 2 * PADDING
            
            # 调整位置，避免超出画面
            if label_br_x > w:
                label_tl_x = max(0, w - text_w - 2 * PADDING - 10)
                label_br_x = label_tl_x + text_w + 2 * PADDING
            if label_br_y > h:
                label_tl_y = max(0, h - text_h - 2 * PADDING - 10)
                label_br_y = label_tl_y + text_h + 2 * PADDING
            
            # 8. 保存OCR结果
            ocr_results.append({
                "text": best_text,
                "conf": best_score,
                "bbox": (text_x1, text_y1, text_x2, text_y2),
                "ship_bbox": (x1, y1, x2, y2),
                "camera_id": camera_id
            })
            
            # 9. 绘制OCR结果
            try:
                # 绘制圆角背景框
                frame = draw_rounded_rectangle(
                    frame,
                    (label_tl_x, label_tl_y),
                    (label_br_x, label_br_y),
                    OCR_BG_COLOR,
                    thickness=-1,
                    radius=CORNER_RADIUS
                )
                
                # 绘制文本
                text_pos_x = label_tl_x + PADDING
                text_pos_y = label_tl_y + PADDING  # 左上角坐标
                
                frame = draw_chinese_text(
                    frame,
                    display_text,
                    (text_pos_x, text_pos_y),
                    font_size=FONT_SCALE,
                    color=OCR_TEXT_COLOR,
                    thickness=FONT_THICKNESS
                )
                
                # 如果有多边形坐标，绘制OCR检测框
                if best_poly is not None and len(best_poly) >= 4:
                    cv2.rectangle(frame, (text_x1, text_y1), 
                                (text_x2, text_y2), OCR_BOX_COLOR, BOX_THICKNESS//2)
                
            except Exception as e:
                print(f"⚠️ OCR绘制异常：{str(e)}")
        else:
            print(f"ℹ️ 当前ROI无有效识别文字（经过过滤）")
    
    return ocr_results, frame

# ======================== 绘制圆角矩形 ========================
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
    
    # 确保坐标有效
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = max(x1+1, x2), max(y1+1, y2)
    
    # 确保圆角半径不超过矩形尺寸
    radius = min(radius, (x2-x1)//2, (y2-y1)//2)
    
    # 绘制直角矩形（基础）
    if thickness == -1:
        cv2.rectangle(img, (x1+radius, y1), (x2-radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1+radius), (x2, y2-radius), color, -1)
    else:
        cv2.rectangle(img, (x1+radius, y1), (x2-radius, y2), color, thickness)
        cv2.rectangle(img, (x1, y1+radius), (x2, y2-radius), color, thickness)
    
    # 绘制四个圆角（椭圆）
    cv2.ellipse(img, (x1+radius, y1+radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2-radius, y1+radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1+radius, y2-radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2-radius, y2-radius), (radius, radius), 0, 0, 90, color, thickness)
    return img

# ======================== ONNX 预处理/后处理辅助函数 ========================
def letterbox(im, new_shape=(640,640), color=(114,114,114)):
    """Resize and pad image to target size, return padded image, scale, and padding (left, top)"""
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    # Scale ratio
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (left, top)  # image, scale, (pad_w, pad_h)

def preprocess_image_cls(frame, input_size):
    """预处理分类模型输入：直接resize到目标尺寸，归一化，转换为CHW，增加batch维度"""
    img = cv2.resize(frame, (input_size, input_size))
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = np.expand_dims(img, axis=0)
    return img

def preprocess_image_det(frame, input_size):
    """预处理检测模型输入：letterbox保持宽高比，归一化，转换为CHW，增加batch维度，返回图像、缩放因子、填充"""
    img, scale, pad = letterbox(frame, (input_size, input_size))
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = np.expand_dims(img, axis=0)
    return img, scale, pad

def postprocess_detection(output, orig_shape, scale, pad, conf_thres=0.25, iou_thres=0.45):
    """
    后处理检测模型输出，返回列表，每个元素为 (x1n, y1n, x2n, y2n, conf, cls)
    output: ONNX模型输出，假设形状为 (1, num_classes+4, num_predictions) 或 (1, num_predictions, num_classes+4)
    orig_shape: 原始图像尺寸 (h, w)
    scale: 缩放因子
    pad: 填充 (left, top)
    """
    if output.ndim == 3:
        # 判断是否需要转置
        if output.shape[1] < output.shape[2]:  # 假设 (1, 84, 8400)
            output = output.transpose(0, 2, 1)  # -> (1, 8400, 84)
        predictions = output[0]  # (num_predictions, 84)
    else:
        raise ValueError("Unexpected output shape")
    
    num_classes = predictions.shape[1] - 4
    boxes = []
    scores = []
    class_ids = []
    h_orig, w_orig = orig_shape

    for pred in predictions:
        # pred: [x_center, y_center, width, height, class_scores...] (归一化到[0,1]相对于640)
        xc, yc, w, h = pred[:4]
        cls_scores = pred[4:]
        conf = np.max(cls_scores)
        if conf < conf_thres:
            continue
        cls_id = np.argmax(cls_scores)

        # 将归一化坐标转换为640坐标系下的像素坐标
        xc_640 = xc * 640
        yc_640 = yc * 640
        w_640 = w * 640
        h_640 = h * 640

        # 减去填充
        pad_w, pad_h = pad
        xc_resized = xc_640 - pad_w
        yc_resized = yc_640 - pad_h

        # 除以缩放因子得到原始图像坐标
        xc_orig = xc_resized / scale
        yc_orig = yc_resized / scale
        w_orig_pix = w_640 / scale
        h_orig_pix = h_640 / scale

        # 转换为xyxy
        x1 = xc_orig - w_orig_pix / 2
        y1 = yc_orig - h_orig_pix / 2
        x2 = xc_orig + w_orig_pix / 2
        y2 = yc_orig + h_orig_pix / 2

        # 归一化到原始图像尺寸
        x1n = max(0, min(1, x1 / w_orig))
        y1n = max(0, min(1, y1 / h_orig))
        x2n = max(0, min(1, x2 / w_orig))
        y2n = max(0, min(1, y2 / h_orig))

        boxes.append([x1n, y1n, x2n, y2n])
        scores.append(conf)
        class_ids.append(cls_id)

    if len(boxes) == 0:
        return []

    # NMS
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, iou_thres)
    if len(indices) == 0:
        return []

    det_results = []
    for i in indices.flatten():
        det_results.append((
            boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3],
            scores[i], float(class_ids[i])
        ))
    return det_results

# ======================== 加载 ONNX 模型及类别名称 ========================
# 设置 ONNX Runtime providers（优先使用CUDA）
providers = []
if 'CUDAExecutionProvider' in ort.get_available_providers():
    providers.append('CUDAExecutionProvider')
providers.append('CPUExecutionProvider')

# 分类模型
cls_session = ort.InferenceSession(CLS_MODEL_PATH.replace('.pt', '.onnx'), providers=providers)
try:
    cls_model_pt = YOLO(CLS_MODEL_PATH)  # 仅用于获取类别名称
    cls_names = cls_model_pt.names
except:
    # 如果.pt不存在，从ONNX输出形状推断类别数
    output_shape = cls_session.get_outputs()[0].shape
    num_classes = output_shape[1] if len(output_shape) == 2 else output_shape[2]
    cls_names = {i: f"class_{i}" for i in range(num_classes)}
    print("⚠️ 无法加载分类模型类别名称，使用默认名称")

# 白天检测模型
det_day_session = ort.InferenceSession(DET_DAY_MODEL_PATH.replace('.pt', '.onnx'), providers=providers)
try:
    det_day_model_pt = YOLO(DET_DAY_MODEL_PATH)
    det_day_names = det_day_model_pt.names
except:
    output_shape = det_day_session.get_outputs()[0].shape
    if output_shape[1] < output_shape[2]:
        num_classes = output_shape[1] - 4
    else:
        num_classes = output_shape[2] - 4
    det_day_names = {i: f"class_{i}" for i in range(num_classes)}
    print("⚠️ 无法加载白天检测模型类别名称，使用默认名称")

# 夜晚检测模型
det_night_session = ort.InferenceSession(DET_NIGHT_MODEL_PATH.replace('.pt', '.onnx'), providers=providers)
try:
    det_night_model_pt = YOLO(DET_NIGHT_MODEL_PATH)
    det_night_names = det_night_model_pt.names
except:
    output_shape = det_night_session.get_outputs()[0].shape
    if output_shape[1] < output_shape[2]:
        num_classes = output_shape[1] - 4
    else:
        num_classes = output_shape[2] - 4
    det_night_names = {i: f"class_{i}" for i in range(num_classes)}
    print("⚠️ 无法加载夜晚检测模型类别名称，使用默认名称")

# ======================== 核心函数 ========================
def get_weather_class(frame):
    """场景分类：使用ONNX模型推理"""
    input_tensor = preprocess_image_cls(frame, CLS_INPUT_SIZE)
    outputs = cls_session.run(None, {cls_session.get_inputs()[0].name: input_tensor})
    logits = outputs[0][0]  # 假设输出形状(1, num_classes)
    # softmax
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)
    top1_idx = np.argmax(probs)
    top1_conf = probs[top1_idx]
    cls_label = cls_names[top1_idx]
    print(f"分类结果：{cls_label} | 置信度：{top1_conf:.3f}")
    return cls_label, top1_conf

def detect_target(frame, weather_cls):
    """目标检测：使用ONNX模型推理，返回归一化检测结果列表"""
    session = det_night_session if weather_cls == "night" else det_day_session
    input_tensor, scale, pad = preprocess_image_det(frame, DET_INPUT_SIZE)
    outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})
    # 后处理
    det_results = postprocess_detection(outputs[0], frame.shape[:2], scale, pad,
                                        conf_thres=0.25, iou_thres=0.45)
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

        if weather_cls == "night":
            is_alarm = area >= LIGHT_ALARM_THRESH
        else:
            is_alarm = area >= SHIP_ALARM_THRESH
        
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

    # ---------------------- 1. 绘制场景信息（圆角文本框） ----------------------
    weather_text = f"Weather: {weather_label}({weather_conf:.2f})"
    # 计算文本尺寸（支持中文）
    text_w, text_h = get_text_size(weather_text, FONT_SCALE)
    # 计算文本框坐标（增加内边距）
    tl_x = 5
    tl_y = 5
    br_x = tl_x + text_w + 2 * PADDING
    br_y = tl_y + text_h + 2 * PADDING
    
    # 确保场景信息框在画面内
    br_x = min(br_x, w - 5)
    br_y = min(br_y, h - 5)
    
    # 绘制圆角背景框
    frame = draw_rounded_rectangle(
        frame, 
        (tl_x, tl_y), 
        (br_x, br_y), 
        TEXT_BG_COLOR, 
        thickness=-1,  # 填充背景
        radius=CORNER_RADIUS
    )
    # 绘制文本
    text_x = tl_x + PADDING
    text_y = tl_y + PADDING # 左上角坐标，draw_chinese_text会内部调整垂直位置
    frame = draw_chinese_text(frame, weather_text, (text_x, text_y), 
                             font_size=FONT_SCALE, color=TEXT_COLOR, thickness=FONT_THICKNESS)

    # ---------------------- 2. 绘制检测框（圆角边框）+ 圆角标签 ----------------------
    for (x1, y1, x2, y2, conf, cls, area, is_alarm) in post_results:
        box_color = ALARM_COLOR if is_alarm else NORMAL_COLOR
        
        # 检测框用普通直角矩形
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, BOX_THICKNESS)

        # 绘制标签文本（圆角背景）
        cls_name = model_names.get(int(cls), "unknown")
        alarm_text = "ALARM" if is_alarm else "NORMAL"
        label_text = f"{cls_name}({conf:.2f})| Area:{area}| {alarm_text}"
        
        # 计算标签尺寸（支持中文）
        label_w, label_h = get_text_size(label_text, FONT_SCALE)
        label_height = label_h + 2 * PADDING
        label_width = label_w + 2 * PADDING
        
        # 确定标签位置（关键优化）
        # 优先放在检测框上方
        label_tl_y = y1 - label_height - LABEL_OFFSET
        # 如果上方空间不足，放在检测框下方
        if label_tl_y < 0:
            label_tl_y = y2 + LABEL_OFFSET
        
        # 确保标签在画面内
        label_tl_y = max(0, min(label_tl_y, h - label_height - 5))
        label_tl_x = max(0, min(x1, w - label_width - 5))
        
        label_br_x = label_tl_x + label_width
        label_br_y = label_tl_y + label_height - 2 * PADDING
        
        # 确保标签框不超出画面
        if label_br_x > w:
            label_tl_x = max(0, w - label_width - 5)
            label_br_x = label_tl_x + label_width
        
        # 绘制圆角标签背景
        frame = draw_rounded_rectangle(
            frame, 
            (label_tl_x, label_tl_y), 
            (label_br_x, label_br_y), 
            TEXT_BG_COLOR, 
            thickness=-1, 
            radius=CORNER_RADIUS
        )
        
        # 绘制标签文本
        label_text_x = label_tl_x + PADDING
        label_text_y = label_tl_y + PADDING  # 左上角坐标
        frame = draw_chinese_text(frame, label_text, (label_text_x, label_text_y),
                                 font_size=FONT_SCALE, color=TEXT_COLOR, thickness=FONT_THICKNESS)
    
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

# ======================== 主程序 ========================
if __name__ == "__main__":
    # 安装PIL（如果未安装）
    if not PIL_AVAILABLE:
        print("📦 正在安装PIL库...")
        os.system("pip install pillow -q")
        try:
            from PIL import Image, ImageDraw, ImageFont
            PIL_AVAILABLE = True
            print("✅ PIL库安装成功")
        except:
            print("❌ PIL库安装失败，中文显示可能异常")
    
    # 初始化4路视频捕获
    caps = []
    # 初始化4路视频对应的稳像器实例（每路独立）
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
            # 初始化稳像器
            if ENABLE_STABILIZATION:
                stabilizer = RealTimeVideoStabilizer(
                    smoothing_radius=STAB_SMOOTHING_RADIUS,
                    crop_ratio=STAB_CROP_RATIO
                )
                stabilizers.append(stabilizer)
            else:
                stabilizers.append(None)
    
    # 主循环
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
                # 视频循环时重置稳像器状态
                if stabilizer is not None:
                    stabilizer.reset()
                continue
            
            # ========== 核心修改：帧预处理 - 实时稳像 ==========
            stabilized_frame = frame
            if ENABLE_STABILIZATION and stabilizer is not None:
                stabilized_frame = stabilizer.process_frame(frame)
            
            # 1. 场景分类（使用稳像后的帧）
            weather_label, weather_conf = get_weather_class(stabilized_frame)
            weather_info = (weather_label, weather_conf)
            
            # 2. 目标检测（使用稳像后的帧）
            det_results = detect_target(stabilized_frame, weather_label)
            
            # 3. 后处理
            orig_h, orig_w = stabilized_frame.shape[:2]  # 注意：使用稳像后帧的尺寸
            post_results, has_alarm = postprocess_detections(det_results, (orig_h, orig_w), weather_label)
            
            # 4. 更新工作模式
            check_work_mode(has_alarm)
            
            # 5. 仅对白天告警船只ROI做PaddleOCR V5识别（传入相机编号idx+1）
            ocr_results, stabilized_frame = process_ship_ocr(
                stabilized_frame.copy(), post_results, weather_label, idx+1
            )
            # 打印OCR结果
            if ocr_results:
                print(f"❗相机{idx+1} OCR识别结果：{[item['text'] for item in ocr_results]}")
            
            # 6. 获取类别名（从加载的names字典中获取）
            model_names = det_night_names if weather_label == "night" else det_day_names
            
            # 7. 绘制（使用优化后的圆角样式）
            frame_drawn = draw_custom_boxes(stabilized_frame, post_results, weather_info, model_names)
            
            # 显示（1344×760 匹配原视频比例，自动适配稳像后的尺寸）
            cv2.imshow(f"Camera {idx+1} (Stabilized+OCR)", cv2.resize(frame_drawn, (1344, 760)))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("用户手动退出程序")
            break
    
    # 释放资源
    for cap in caps:
        if cap is not None:
            cap.release()
    cv2.destroyAllWindows()