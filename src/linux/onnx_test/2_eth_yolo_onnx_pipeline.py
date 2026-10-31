"""
这个脚本用于测试含yolo_onnx版本的pipeline，（OCR仍为原模型）
相较于1添加了网络摄像头的视频流获取。
# 类别名称（顺序与文件夹名一致）
names:
  0: cloudy
  1: foggy
  2: night
  3: rainy
  4: sunny
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


# 统一模型路径
CLS_MODEL_PATH = "/home/Tronlong/Desktop/pipeline/models/yolov11n-cls/1-best.onnx"          # 场景分类模型
DET_DAY_MODEL_PATH = "/home/Tronlong/Desktop/pipeline/models/yolov11n-day/5-best.onnx"      # 白天船舶检测模型
DET_NIGHT_MODEL_PATH = "/home/Tronlong/Desktop/pipeline/models/yolov11n-night/2-best.onnx"  # 夜晚灯光检测模型
# 4路视频路径 - 修改为海康网络摄像头RTSP流
VIDEO_PATHS = [
    # 海康网络摄像头RTSP流格式：rtsp://用户名:密码@IP地址:端口/流路径
    "rtsp://admin:xiayuxuan113@192.168.1.64:554/stream1",  # 海康网络摄像头
    #"rtsp://admin:xiayuxuan113@192.168.1.64:554/Streaming/Channels/102",  # 子码流（推荐，占用资源少）
    #"/dev/video0",  # 主摄像头（已注释）
    #"/home/Tronlong/Desktop/pipeline/test/多云-近处小船（P船号）.mp4",     #多云    
    #"/home/Tronlong/Desktop/pipeline/test/晴天短视频.mp4",  #晴天    
    #"/home/Tronlong/Desktop/pipeline/test/雾天.mp4"  #雾天  正常
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
ENABLE_OCR = False  # OCR功能总开关
OCR_USE_GPU = False  # 启用GPU加速
OCR_MODEL_VERSION = "v5"  # 指定PaddleOCR V5版本
# 仅识别白天告警船只的ROI，可配置识别语言（中文/英文/数字）
OCR_LANG = "ch"  # ch=中文, en=英文, ch_en=中英混合
# OCR置信度阈值（过滤低置信度结果）
OCR_CONF_THRESH = 0.5

# ======================== ROI图片保存配置 ========================
SAVE_OCR_ROI = False  # 是否保存OCR检测的ROI图片
ROI_SAVE_ROOT = "/home/Tronlong/Desktop/pipeline/ocr_roi_images"  # ROI保存根目录
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
        # 修复参数警告：移除lang/ocr_version，仅保留必要参数
        ocr = PaddleOCR(
            det_model_dir="/home/Tronlong/Desktop/pipeline/models/paddleocr_v5/PP-OCRv5_server_det",
            rec_model_dir="/home/Tronlong/Desktop/pipeline/models/paddleocr_v5/PP-OCRv5_server_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            ocr_version="PP-OCRv5",
            device="cpu"
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

# ======================== ONNX模型封装类 ========================
class YOLOOnnxModel:
    """
    YOLOv11 ONNX推理封装类
    统一封装预处理、推理、后处理，对外接口与原ultralytics YOLO保持一致
    支持分类模型（cls）和检测模型（det）两种模式
    """

    # ONNX推理置信度阈值（过滤低置信度检测框）
    CONF_THRESH = 0.25
    # NMS IoU阈值
    IOU_THRESH = 0.45

    def __init__(self, model_path, model_type="det", input_size=640):
        """
        :param model_path: ONNX模型文件路径
        :param model_type: 模型类型，"cls"=分类，"det"=检测
        :param input_size: 模型输入尺寸（正方形边长，cls=256, det=640）
        """
        self.model_path = model_path
        self.model_type = model_type  # "cls" or "det"
        self.input_size = input_size

        # ---- 初始化ONNXRuntime推理会话 ----
        # 优先使用GPU（CUDAExecutionProvider），回退到CPU
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        # 过滤掉不可用的provider，避免警告
        available_providers = ort.get_available_providers()
        self.providers = [p for p in providers if p in available_providers]
        
        self.session = ort.InferenceSession(model_path, providers=self.providers)

        # 获取输入输出节点名称
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        # 读取模型元数据中的类别名 + 兜底默认值
        self.names = self._load_class_names()
        
        # 类别名兜底（根据模型类型设置默认值）
        if not self.names:
            if self.model_type == "cls":
                # 分类模型默认类别（根据你的场景分类实际类别调整）
                # 类别名称（顺序与文件夹名一致）
                self.names = {0: "cloudy", 1: "foggy", 2: "night", 3: "rainy",4: "sunny"}
            elif self.model_type == "det":
                # 检测模型默认类别（船舶/灯光）
                self.names = {0: "ship", 1: "light"}

        print(f"✅ 加载ONNX模型: {model_path}")
        print(f"   类型={model_type}, 输入尺寸={input_size}, 类别数={len(self.names)}")
        print(f"   推理后端: {self.providers}")
        
        # 提示CUDA不可用
        if "CUDAExecutionProvider" not in self.providers:
            print("⚠️ CUDA不可用，将使用CPU推理（速度较慢）")

    def _load_class_names(self):
        """
        从ONNX模型元数据读取类别名字典
        ultralytics导出时会将 names 写入 metadata_props
        格式示例：{"0": "ship", "1": "boat"}
        回退策略：返回空字典（使用默认类别名）
        """
        import json
        names = {}
        try:
            meta = self.session.get_modelmeta().custom_metadata_map
            if "names" in meta:
                raw = meta["names"]
                # 兼容单引号/双引号格式
                raw_fixed = raw.replace("'", '"').replace('’', '"')
                parsed = json.loads(raw_fixed)
                names = {int(k): v for k, v in parsed.items()}
        except Exception as e:
            print(f"⚠️ 读取ONNX类别名失败（将使用默认类别名）：{e}")
        return names

    # ------------------------------------------------------------------ #
    #                       图像预处理（letterbox）                        #
    # ------------------------------------------------------------------ #
    def _letterbox(self, img, new_size):
        """
        等比例缩放并填充到 new_size×new_size（灰色填充，值=114）
        与ultralytics内部预处理完全一致
        :param img: BGR numpy图像
        :param new_size: 目标正方形边长
        :return: (处理后图像, 缩放比例r, (dw填充宽, dh填充高))
        """
        h0, w0 = img.shape[:2]
        r = min(new_size / h0, new_size / w0)
        new_w = int(round(w0 * r))
        new_h = int(round(h0 * r))

        # 等比例缩放
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 计算填充量（居中填充）
        dw = (new_size - new_w) / 2
        dh = (new_size - new_h) / 2
        top    = int(round(dh - 0.1))
        bottom = int(round(dh + 0.1))
        left   = int(round(dw - 0.1))
        right  = int(round(dw + 0.1))

        # 灰色填充（114是ultralytics默认填充值）
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        return padded, r, (dw, dh)

    def _preprocess_det(self, img):
        """
        检测模型预处理：letterbox → BGR2RGB → HWC→CHW → 归一化 → 增加batch维
        :return: (blob float32[1,3,H,W], 缩放比r, 填充(dw,dh))
        """
        padded, r, (dw, dh) = self._letterbox(img, self.input_size)
        # BGR -> RGB
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        # HWC -> CHW
        chw = rgb.transpose(2, 0, 1)
        # 归一化到[0,1]
        blob = chw.astype(np.float32) / 255.0
        # 增加batch维
        blob = np.expand_dims(blob, axis=0)  # [1, 3, H, W]
        return blob, r, (dw, dh)

    def _preprocess_cls(self, img):
        """
        分类模型预处理：resize → BGR2RGB → HWC→CHW → 归一化 → 增加batch维
        分类模型使用中心裁剪+resize（ultralytics cls默认策略）
        :return: blob float32[1,3,H,W]
        """
        # 先等比例缩放到 input_size，再中心裁剪
        h0, w0 = img.shape[:2]
        # 短边对齐 input_size
        r = self.input_size / min(h0, w0)
        new_w = int(round(w0 * r))
        new_h = int(round(h0 * r))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 中心裁剪到 input_size × input_size
        cy, cx = new_h // 2, new_w // 2
        half = self.input_size // 2
        cropped = resized[
            max(0, cy - half): cy - half + self.input_size,
            max(0, cx - half): cx - half + self.input_size
        ]
        # 若裁剪后尺寸不足（边缘情况），直接resize补齐
        if cropped.shape[0] != self.input_size or cropped.shape[1] != self.input_size:
            cropped = cv2.resize(cropped, (self.input_size, self.input_size),
                                 interpolation=cv2.INTER_LINEAR)

        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(chw, axis=0)  # [1, 3, H, W]
        return blob

    # ------------------------------------------------------------------ #
    #                       检测后处理（NMS）                              #
    # ------------------------------------------------------------------ #
    def _postprocess_det(self, outputs, orig_shape, r, dw, dh):
        """
        YOLOv11检测模型ONNX输出后处理
        ONNX输出格式：[1, 4+num_cls, 8400]
          - 前4行：cx, cy, w, h（相对于letterbox图的绝对像素坐标）
          - 后num_cls行：各类别置信度（已经过sigmoid，无需再做）
        :param outputs: onnxruntime推理原始输出列表
        :param orig_shape: 原始图像尺寸 (h, w)
        :param r: letterbox缩放比
        :param dw: 水平填充像素数（单侧）
        :param dh: 垂直填充像素数（单侧）
        :return: list of (x1n, y1n, x2n, y2n, conf, cls_id)
                 坐标为相对于原图的归一化坐标 [0,1]
        """
        # outputs[0] shape: [1, 4+num_cls, 8400]
        pred = outputs[0]  # numpy array
        if pred.ndim == 3:
            pred = pred[0]  # [4+num_cls, 8400]

        # 转置为 [8400, 4+num_cls]，方便逐行处理
        pred = pred.T  # [8400, 4+num_cls]

        num_cls = pred.shape[1] - 4

        # 分离坐标和置信度
        boxes_cxcywh = pred[:, :4]          # [8400, 4] cx,cy,w,h（letterbox像素坐标）
        cls_scores   = pred[:, 4:]           # [8400, num_cls]

        # 取每个anchor的最大类别置信度和类别索引
        conf = cls_scores.max(axis=1)        # [8400]
        cls_id = cls_scores.argmax(axis=1)   # [8400]

        # 置信度过滤
        mask = conf >= self.CONF_THRESH
        if mask.sum() == 0:
            return []

        boxes_cxcywh = boxes_cxcywh[mask]
        conf         = conf[mask]
        cls_id       = cls_id[mask]

        # cx,cy,w,h → x1,y1,x2,y2（letterbox像素坐标）
        cx, cy, bw, bh = boxes_cxcywh[:, 0], boxes_cxcywh[:, 1], \
                         boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        # 去除letterbox填充，还原到缩放后图像坐标
        x1 = (x1 - dw) / r
        y1 = (y1 - dh) / r
        x2 = (x2 - dw) / r
        y2 = (y2 - dh) / r

        # 归一化到 [0, 1]（相对于原图）
        orig_h, orig_w = orig_shape
        x1n = np.clip(x1 / orig_w, 0.0, 1.0)
        y1n = np.clip(y1 / orig_h, 0.0, 1.0)
        x2n = np.clip(x2 / orig_w, 0.0, 1.0)
        y2n = np.clip(y2 / orig_h, 0.0, 1.0)

        # 组合为 [N, 6]：x1n, y1n, x2n, y2n, conf, cls_id
        dets = np.stack([x1n, y1n, x2n, y2n, conf, cls_id.astype(np.float32)], axis=1)

        # ---- NMS（按类别分别做）----
        results = []
        for c in np.unique(cls_id):
            c_mask = dets[:, 5] == c
            c_dets = dets[c_mask]
            # opencv NMS 需要 [x,y,w,h] 格式（像素）和分数列表
            bboxes_xywh = []
            scores_list = []
            for d in c_dets:
                _x1n, _y1n, _x2n, _y2n, _conf, _ = d
                # 还原为像素坐标用于NMS计算
                px1 = int(_x1n * orig_w)
                py1 = int(_y1n * orig_h)
                pw  = int((_x2n - _x1n) * orig_w)
                ph  = int((_y2n - _y1n) * orig_h)
                bboxes_xywh.append([px1, py1, pw, ph])
                scores_list.append(float(_conf))

            indices = cv2.dnn.NMSBoxes(
                bboxes_xywh, scores_list,
                self.CONF_THRESH, self.IOU_THRESH
            )
            if len(indices) == 0:
                continue
            # cv2.dnn.NMSBoxes 返回格式在不同版本有差异，统一展平
            indices = np.array(indices).flatten()
            for i in indices:
                results.append(tuple(c_dets[i].tolist()))

        return results  # list of (x1n, y1n, x2n, y2n, conf, cls_id)

    def _postprocess_cls(self, outputs):
        """
        分类模型后处理
        ONNX输出格式：[1, num_classes]，已经过softmax
        :return: (top1_idx, top1_conf, names_dict)
        """
        probs = outputs[0]  # [1, num_classes]
        if probs.ndim == 2:
            probs = probs[0]  # [num_classes]

        top1_idx  = int(np.argmax(probs))
        top1_conf = float(probs[top1_idx])
        return top1_idx, top1_conf

    # ------------------------------------------------------------------ #
    #                       统一推理入口                                   #
    # ------------------------------------------------------------------ #
    def predict(self, frame):
        """
        对外统一推理接口，模拟ultralytics YOLO.predict()返回格式
        :param frame: BGR numpy图像
        :return: 封装后的结果对象列表（仅含results[0]有效）
        """
        orig_shape = frame.shape[:2]  # (h, w)

        if self.model_type == "cls":
            blob = self._preprocess_cls(frame)
            outputs = self.session.run(self.output_names, {self.input_name: blob})
            top1_idx, top1_conf = self._postprocess_cls(outputs)
            return [_ClsResult(top1_idx, top1_conf, self.names)]

        elif self.model_type == "det":
            blob, r, (dw, dh) = self._preprocess_det(frame)
            outputs = self.session.run(self.output_names, {self.input_name: blob})
            dets = self._postprocess_det(outputs, orig_shape, r, dw, dh)
            return [_DetResult(dets, self.names)]

        else:
            raise ValueError(f"未知模型类型: {self.model_type}")


# ======================== 推理结果封装类（模拟ultralytics接口） ========================
class _ClsProbs:
    """模拟 ultralytics result.probs 接口"""
    def __init__(self, top1_idx, top1_conf):
        self.top1     = top1_idx
        self.top1conf = _FloatWrapper(top1_conf)


class _FloatWrapper:
    """模拟 .item() 调用（兼容 tensor.item() 接口）"""
    def __init__(self, val):
        self._val = val

    def item(self):
        return self._val

    def __float__(self):
        return float(self._val)


class _ClsResult:
    """
    模拟 ultralytics 分类结果对象
    原代码访问方式：
        result.probs.top1          → top1类别索引
        result.probs.top1conf.item() → top1置信度
        result.names[top1_idx]     → 类别名称
    """
    def __init__(self, top1_idx, top1_conf, names):
        self.probs = _ClsProbs(top1_idx, top1_conf)
        self.names = names


class _Box:
    """
    模拟单个检测框对象
    原代码访问方式：
        box.xyxyn[0].tolist()  → [x1n, y1n, x2n, y2n]
        box.conf[0].item()     → 置信度
        box.cls[0].item()      → 类别索引
    """
    def __init__(self, x1n, y1n, x2n, y2n, conf, cls_id):
        # xyxyn[0] 需要支持 .tolist()
        self.xyxyn = [_ToListWrapper([x1n, y1n, x2n, y2n])]
        self.conf  = [_FloatWrapper(conf)]
        self.cls   = [_FloatWrapper(cls_id)]


class _ToListWrapper:
    """模拟 tensor.tolist() 调用"""
    def __init__(self, data):
        self._data = data

    def tolist(self):
        return list(self._data)


class _Boxes:
    """
    模拟 ultralytics result.boxes 接口
    原代码访问方式：
        boxes = results[0].boxes
        if boxes is not None:
            for box in boxes: ...
    """
    def __init__(self, dets):
        """
        :param dets: list of (x1n, y1n, x2n, y2n, conf, cls_id)
        """
        self._boxes = [_Box(*d) for d in dets]

    def __iter__(self):
        return iter(self._boxes)

    def __len__(self):
        return len(self._boxes)


class _DetResult:
    """
    模拟 ultralytics 检测结果对象
    原代码访问方式：
        results[0].boxes  → _Boxes对象，可迭代
        results[0].names  → 类别名字典
    """
    def __init__(self, dets, names):
        self.boxes = _Boxes(dets) if dets else None
        self.names = names

# ======================== 加载模型 ========================
# 加载分类模型（ONNX）
cls_model = YOLOOnnxModel(CLS_MODEL_PATH, model_type="cls", input_size=CLS_INPUT_SIZE)

# 加载检测模型（ONNX）
det_day_model = YOLOOnnxModel(DET_DAY_MODEL_PATH, model_type="det", input_size=DET_INPUT_SIZE)
det_night_model = YOLOOnnxModel(DET_NIGHT_MODEL_PATH, model_type="det", input_size=DET_INPUT_SIZE)

# ======================== 核心函数 ========================
def get_weather_class(frame):
    """场景分类：移除多余参数，仅传frame"""
    results = cls_model.predict(frame)  # 仅传frame，移除所有多余参数
    
    result = results[0]
    top1_idx = result.probs.top1
    top1_conf = result.probs.top1conf.item()
    # 兜底：如果类别名不存在，使用索引
    cls_label = result.names.get(top1_idx, str(top1_idx))  
    
    print(f"分类结果：{cls_label} | 置信度：{top1_conf:.3f}")
    return cls_label, top1_conf

def detect_target(frame, weather_cls):
    """目标检测：移除多余参数，仅传frame"""
    det_model = det_night_model if weather_cls == "night" else det_day_model
    
    results = det_model.predict(frame)  # 仅传frame，移除所有多余参数
    
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
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减小缓冲区，提升实时性
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
        # 检查工作时长限制
        if not check_work_duration():
            print(f"工作时长已达{DEFAULT_WORK_DURATION}秒，退出程序")
            break
        
        # 逐路处理视频流
        for idx, (cap, stabilizer) in enumerate(zip(caps, stabilizers)):
            camera_id = idx + 1  # 相机编号1-4
            if cap is None:
                continue
            
            ret, frame = cap.read()
            if not ret:
                # 视频播放完毕，循环播放
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                # 视频循环时重置稳像器状态（避免累积误差）
                if stabilizer is not None:
                    stabilizer.reset()
                continue
            
            # ========== 核心步骤1：实时稳像处理 ==========
            stabilized_frame = frame
            if ENABLE_STABILIZATION and stabilizer is not None:
                stabilized_frame = stabilizer.process_frame(frame)
            
            # ========== 核心步骤2：场景分类（白天/夜晚/雾天等） ==========
            weather_label, weather_conf = get_weather_class(stabilized_frame)
            weather_info = (weather_label, weather_conf)  # 补全天气信息元组
            
            # ========== 核心步骤3：目标检测（船舶/灯光） ==========
            det_results = detect_target(stabilized_frame, weather_label)
            
            # ========== 核心步骤4：检测结果后处理（计算面积、判断告警） ==========
            post_results, has_alarm = postprocess_detections(
                det_results, 
                stabilized_frame.shape[:2], 
                weather_label
            )
            
            # ========== 核心步骤5：更新工作模式（告警时持续运行） ==========
            continuous_mode = check_work_mode(has_alarm)
            
            # ========== 核心步骤6：获取检测模型的类别名（用于绘制） ==========
            # 根据场景选择对应的检测模型名称字典
            if weather_label == "night":
                model_names = det_night_model.names
            else:
                model_names = det_day_model.names
            
            # ========== 核心步骤7：绘制检测框、场景信息、告警标签 ==========
            draw_frame = draw_custom_boxes(
                stabilized_frame.copy(), 
                post_results, 
                weather_info, 
                model_names
            )
            
            # ========== 核心步骤8：OCR识别（仅白天告警船只） ==========
            ocr_results, draw_frame = process_ship_ocr(
                draw_frame, 
                post_results, 
                weather_label, 
                camera_id
            )
            
            # ========== 核心步骤9：显示处理后的帧 ==========
            # 调整窗口大小（统一显示尺寸）
            display_frame = cv2.resize(draw_frame, (1280, 720))
            cv2.imshow(f"Ship Detection - Camera {camera_id}", display_frame)
        
        # ========== 退出逻辑：按下ESC键（27）或q键退出 ==========
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            print("用户手动退出程序")
            break
    
    # ======================== 资源释放 ========================
    # 释放所有视频捕获对象
    for cap in caps:
        if cap is not None:
            cap.release()
    # 销毁所有OpenCV窗口
    cv2.destroyAllWindows()
    print("✅ 程序正常退出，资源已释放")
