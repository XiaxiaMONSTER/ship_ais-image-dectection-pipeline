"""
基于5_round_robin_ship.py，修改了AsyncVideoCapture类，在读摄像头帧线程时维护预录缓存。并且修改了录制逻辑：告警触发时先写预录缓存，可以避免高延迟下的画面缺失。
yolo_onnx版本的pipeline，（OCR仍为原模型）
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
import subprocess
from datetime import datetime
import glob
import threading
import collections

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
    #"rtsp://admin:ADMIN123@169.254.241.100:554/stream1",  # 海康网络摄像头1
    #"rtsp://admin:xiayuxuan113@169.254.241.100:554/Streaming/Channels/102",  # 子码流（推荐，占用资源少）
    #"rtsp://admin:xiayuxuan113@169.254.242.101:554/stream1",  # 海康网络摄像头2
    #"rtsp://admin:xiayuxuan113@169.254.242.101:554/Streaming/Channels/102",  # 子码流
    #"/dev/video0",  # MSI模块摄像头
    "/home/Tronlong/Desktop/pipeline/test/多云-近处小船（P船号）.mp4",     #多云    
    "/home/Tronlong/Desktop/pipeline/test/晴天短视频.mp4",  #晴天    
    #"/home/Tronlong/Desktop/pipeline/test/雾天.mp4"  #雾天  正常
]

# ======================== 多摄像头采集优化配置 ========================
USE_GSTREAMER = True
FRAME_TIMEOUT = 2.0
CLS_INTERVAL = 5  # 每隔N帧做一次分类，其余帧复用结果
CAPTURE_RECONNECT_INTERVAL = 1.0

# ONNX Runtime线程配置（多路摄像头下避免线程争抢）
ORT_INTRA_THREADS = 2
ORT_INTER_THREADS = 1

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
ENABLE_STABILIZATION = False  # 稳像功能总开关
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

# ======================== 录制与推流核心配置 ========================
# 录制根目录
RECORD_ROOT = "/home/Tronlong/Desktop/pipeline/alarm_recordings"
# 帧保存格式
FRAME_SAVE_FORMAT = "jpg"
# 视频合成参数
VIDEO_FPS = 25  # 视频帧率
VIDEO_CODEC = cv2.VideoWriter_fourcc(*'mp4v')  # MP4编码
VIDEO_EXT = "mp4"
# 告警录制配置
# ======================== 全局告警状态机配置 ========================
ALARM_STATE = False                # 全局告警状态（True=任意一路正在告警，所有路都要录）
ALARM_LAST_TRIGGER_TIME = None     # 最后一次任意一路触发告警的时间
ALARM_STOP_DELAY = 5.0             # 告警停止延迟（最后一路告警消失后，继续录5秒才停）

# ======================== 预录帧缓存配置 ========================
ENABLE_PRERECORD = False            # 开启预录功能（解决告警前几秒丢失）
PRERECORD_SECONDS = 3.0            # 预录时长（秒），建议3-5秒
PRERECORD_FPS = 25                 # 计算需要缓存的最大帧数（假设25fps）
# 每路摄像头的预录帧缓存：{camera_id: deque([(timestamp, frame), ...])}
prerecord_buffers = defaultdict(collections.deque)

# 原有变量（兼容 create_alarm_record_dir 等函数）
RECORD_FRAMES = defaultdict(list)
RECORD_START_TIME = None

# RTSP推流配置（测试验证的稳定方案）
# ======================== RTSP推流核心配置（修复版） ========================
RTSP_ENABLE = True
RTSP_BASE_URL = "rtsp://192.168.0.104:8554/camera_"
RTSP_FPS = 15  # 降低帧率，更稳定
RTSP_RESOLUTION = (640, 480)

# 存储推流进程 + 状态监控
rtsp_processes = {}
# 连续失败次数，超过阈值强制重启
rtsp_fail_count = defaultdict(int)
MAX_RETRY_COUNT = 3

# 【修复】FFmpeg命令：降低延迟、增加超时、关闭冗余日志
FFMPEG_PUSH_PARAMS = [
    'ffmpeg', '-y',
    '-hide_banner', '-loglevel', 'error',
    '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'bgr24',
    '-s', f"{RTSP_RESOLUTION[0]}x{RTSP_RESOLUTION[1]}",
    '-r', str(RTSP_FPS), '-i', '-',
    '-c:v', 'libx264',
    '-preset', 'ultrafast',
    '-tune', 'zerolatency',
    '-b:v', '300k',
    '-f', 'rtsp', '-rtsp_transport', 'udp',
    '-profile:v', 'baseline',
    '-pkt_size', '1000'
]


# ======================== GStreamer / 异步采集工具 ========================
def opencv_has_gstreamer():
    try:
        build_info = cv2.getBuildInformation()
        # 修复：模糊匹配，只要包含 GStreamer + YES 就返回 True
        return "GStreamer:" in build_info and "YES" in build_info
    except Exception:
        return False

def build_gstreamer_rtsp_pipeline(rtsp_url):
    # ✅ RK3576 硬解专用管道（H.265，你摄像头真正的格式）
    return (
        f'rtspsrc location=\"{rtsp_url}\" latency=100 protocols=tcp ! '
        f'rtph265depay ! h265parse ! mppvideodec ! '
        f'videoconvert ! video/x-raw,format=BGR ! '
        f'appsink max-buffers=1 drop=true sync=false'
    )

def open_video_capture(path):
    if isinstance(path, str) and path.startswith("rtsp://") and USE_GSTREAMER and opencv_has_gstreamer():
        try:
            pipeline = build_gstreamer_rtsp_pipeline(path)
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                print(f"✅ RK3576 硬解启动成功: {path}")
                return cap
            else:
                print(f"❌ 硬解管道启动失败: {path}")
        except Exception as e:
            print(f"❌ 管道异常: {e}")

    cap = cv2.VideoCapture(path)
    if cap.isOpened():
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except:
            pass
    return cap

class AsyncVideoCapture:
    def __init__(self, path, camera_id):
        self.path = path
        self.camera_id = camera_id  # 绑定摄像头ID
        self.cap = None
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_ts = 0.0
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.cap = open_video_capture(self.path)
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
        return self

    def _reader(self):
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                try:
                    if self.cap is not None:
                        self.cap.release()
                except:
                    pass
                self.cap = open_video_capture(self.path)
                time.sleep(CAPTURE_RECONNECT_INTERVAL)
                continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                try:
                    self.cap.release()
                except:
                    pass
                self.cap = None
                time.sleep(CAPTURE_RECONNECT_INTERVAL)
                continue

            with self.lock:
                self.latest_frame = frame
                self.latest_ts = time.time()
            # 自动写入预录缓存（后台线程做，不占主线程）
            if ENABLE_PRERECORD:
                now = time.time()
                # 存入缓存：(时间戳, 帧副本)
                prerecord_buffers[self.camera_id].append((now, frame.copy()))
                # 计算需要缓存的最大帧数
                max_frames = int(PRERECORD_SECONDS * PRERECORD_FPS)
                # 超过长度则弹出最旧的帧
                while len(prerecord_buffers[self.camera_id]) > max_frames:
                    prerecord_buffers[self.camera_id].popleft()
    def read(self):
        with self.lock:
            if self.latest_frame is None:
                return False, None
            return True, self.latest_frame.copy()

    def is_alive(self):
        return (time.time() - self.latest_ts) < FRAME_TIMEOUT if self.latest_ts > 0 else False

    def release(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            try:
                self.cap.release()
            except:
                pass


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
        corners = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)
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
        
        window_size = 2 * self.smoothing_radius + 1
        start_idx = max(0, len(trajectory) - window_size)
        recent_traj = np.array(trajectory[start_idx:])
        
        smoothed = np.array(trajectory, dtype=np.float32)
        if len(recent_traj) > 1:
            kernel = np.ones(len(recent_traj), dtype=np.float32) / len(recent_traj)
            for i in range(3):
                smoothed[-1, i] = np.sum(recent_traj[:, i] * kernel)
        
        return smoothed.tolist()

    def apply_transform(self, frame, transform):
        """应用变换矩阵并裁剪黑边"""
        h, w = frame.shape[:2]
        stabilized = cv2.warpAffine(
            frame, transform, (w, h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )
        crop_h = int(h * self.crop_ratio)
        crop_w = int(w * self.crop_ratio)
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2
        cropped = stabilized[start_h:start_h+crop_h, start_w:start_w+crop_w]
        cropped = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        return cropped

    def process_frame(self, curr_frame):
        """
        实时处理单帧图像
        :param curr_frame: 输入的BGR帧
        :return: 稳像后的BGR帧
        """
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
        
        curr_points, status = self.calculate_optical_flow(
            self.prev_gray, curr_gray, self.prev_points
        )
        
        if curr_points is None or status is None:
            self.prev_frame = curr_frame.copy()
            self.prev_gray = curr_gray
            self.prev_points = self.extract_features(curr_gray)
            return curr_frame

        valid_prev = self.prev_points[status.flatten() == 1]
        valid_curr = curr_points[status.flatten() == 1]
        
        if len(valid_prev) < 4:
            self.prev_points = self.extract_features(self.prev_gray)
            self.prev_frame = curr_frame.copy()
            self.prev_gray = curr_gray
            return curr_frame
        
        transform = self.estimate_rigid_transform(valid_prev, valid_curr)
        
        T_prev = np.vstack([self.cumulative_transform, [0, 0, 1]])
        T_curr = np.vstack([transform, [0, 0, 1]])
        self.cumulative_transform = (T_prev @ T_curr)[:2, :]
        
        dx, dy, da = self.decompose_transform(self.cumulative_transform)
        self.trajectory.append([dx, dy, da])
        self.smoothed_trajectory = self.smooth_trajectory(self.trajectory)
        
        if len(self.smoothed_trajectory) > 0 and len(self.trajectory) > 0:
            diff_dx = self.smoothed_trajectory[-1][0] - self.trajectory[-1][0]
            diff_dy = self.smoothed_trajectory[-1][1] - self.trajectory[-1][1]
            diff_da = self.smoothed_trajectory[-1][2] - self.trajectory[-1][2]
            
            comp_transform = np.array([
                [np.cos(diff_da), -np.sin(diff_da), diff_dx],
                [np.sin(diff_da),  np.cos(diff_da), diff_dy]
            ], dtype=np.float32)
            
            stabilized_frame = self.apply_transform(curr_frame, comp_transform)
        else:
            stabilized_frame = curr_frame.copy()
        
        self.prev_frame = curr_frame.copy()
        self.prev_gray = curr_gray
        self.prev_points = self.extract_features(curr_gray)
        
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
    timestamp = int(time.time() * 1000)
    filename = f"roi_camera{camera_id}_{timestamp}_x{x1}y{y1}x{x2}y{y2}.{ROI_SAVE_FORMAT}"
    return filename


# ======================== PaddleOCR V5 初始化与处理函数 ========================
def init_paddle_ocr_v5():
    """初始化PaddleOCR V5（轻量化配置）"""
    try:
        from paddleocr import PaddleOCR
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
    
    valid_combinations = [
        has_chinese and has_digits,
        has_chinese and has_letters,
        has_chinese and has_digits and has_letters,
        has_digits and len(text) >= 4,
        has_chinese and chinese_count >= 2,
    ]
    
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

    draw.text(pos, text, font=font, fill=pil_color, stroke_width=0.5)

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
    """
    仅对白天告警的船只ROI进行PaddleOCR V5识别（适配字典格式结果）
    :param frame: 稳像后的帧
    :param post_results: 后处理后的检测结果
    :param weather_label: 场景分类标签（白天/夜晚/雾天等）
    :param camera_id: 相机编号（1/2/3/4）
    :return: OCR结果列表、绘制后的帧
    """
    ocr_results = []
    if not ENABLE_OCR or ocr_engine is None or weather_label in ["night", "夜晚"]:
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
                label_tl_y = 0
                label_br_y = int(label_tl_y + text_h + 2 * PADDING)
            
            ocr_results.append({
                "text": best_text,
                "conf": best_score,
                "bbox": (text_x1, text_y1, text_x2, text_y2),
                "ship_bbox": (x1, y1, x2, y2),
                "camera_id": camera_id
            })
            
            try:
                frame = draw_rounded_rectangle(
                    frame,
                    (label_tl_x, label_tl_y),
                    (label_br_x, label_br_y),
                    OCR_BG_COLOR,
                    thickness=-1,
                    radius=CORNER_RADIUS
                )
                
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
                
                if best_poly is not None and len(best_poly) >= 4:
                    cv2.rectangle(frame, (int(text_x1), int(text_y1)),
                                  (int(text_x2), int(text_y2)), OCR_BOX_COLOR, max(1, BOX_THICKNESS//2))
                
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


# ======================== ONNX模型封装类 ========================
class YOLOOnnxModel:
    """
    YOLOv11 ONNX推理封装类
    统一封装预处理、推理、后处理，对外接口与原ultralytics YOLO保持一致
    支持分类模型（cls）和检测模型（det）两种模式
    """

    CONF_THRESH = 0.25
    IOU_THRESH = 0.45

    def __init__(self, model_path, model_type="det", input_size=640):
        """
        :param model_path: ONNX模型文件路径
        :param model_type: 模型类型，"cls"=分类，"det"=检测
        :param input_size: 模型输入尺寸（正方形边长，cls=256, det=640）
        """
        self.model_path = model_path
        self.model_type = model_type
        self.input_size = input_size

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        available_providers = ort.get_available_providers()
        self.providers = [p for p in providers if p in available_providers]

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = ORT_INTRA_THREADS
        sess_options.inter_op_num_threads = ORT_INTER_THREADS
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.enable_mem_pattern = True
        sess_options.enable_cpu_mem_arena = True

        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=self.providers
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        self.names = self._load_class_names()
        
        if not self.names:
            if self.model_type == "cls":
                self.names = {0: "cloudy", 1: "foggy", 2: "night", 3: "rainy", 4: "sunny"}
            elif self.model_type == "det":
                self.names = {0: "ship", 1: "light"}

        print(f"✅ 加载ONNX模型: {model_path}")
        print(f"   类型={model_type}, 输入尺寸={input_size}, 类别数={len(self.names)}")
        print(f"   推理后端: {self.providers}")
        
        if "CUDAExecutionProvider" not in self.providers:
            print("⚠️ CUDA不可用，将使用CPU推理（速度较慢）")

    def _load_class_names(self):
        import json
        names = {}
        try:
            meta = self.session.get_modelmeta().custom_metadata_map
            if "names" in meta:
                raw = meta["names"]
                raw_fixed = raw.replace("'", '"').replace('’', '"')
                parsed = json.loads(raw_fixed)
                names = {int(k): v for k, v in parsed.items()}
        except Exception as e:
            print(f"⚠️ 读取ONNX类别名失败（将使用默认类别名）：{e}")
        return names

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

        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        return padded, r, (dw, dh)

    def _preprocess_det(self, img):
        padded, r, (dw, dh) = self._letterbox(img, self.input_size)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        chw = rgb.transpose(2, 0, 1)
        blob = chw.astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)
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
            max(0, cy - half): cy - half + self.input_size,
            max(0, cx - half): cx - half + self.input_size
        ]
        if cropped.shape[0] != self.input_size or cropped.shape[1] != self.input_size:
            cropped = cv2.resize(cropped, (self.input_size, self.input_size),
                                 interpolation=cv2.INTER_LINEAR)

        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(chw, axis=0)
        return blob

    def _postprocess_det(self, outputs, orig_shape, r, dw, dh):
        pred = outputs[0]
        if pred.ndim == 3:
            pred = pred[0]

        pred = pred.T
        num_cls = pred.shape[1] - 4

        boxes_cxcywh = pred[:, :4]
        cls_scores   = pred[:, 4:]

        conf = cls_scores.max(axis=1)
        cls_id = cls_scores.argmax(axis=1)

        mask = conf >= self.CONF_THRESH
        if mask.sum() == 0:
            return []

        boxes_cxcywh = boxes_cxcywh[mask]
        conf         = conf[mask]
        cls_id       = cls_id[mask]

        cx, cy, bw, bh = boxes_cxcywh[:, 0], boxes_cxcywh[:, 1], boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        x1 = (x1 - dw) / r
        y1 = (y1 - dh) / r
        x2 = (x2 - dw) / r
        y2 = (y2 - dh) / r

        orig_h, orig_w = orig_shape
        x1n = np.clip(x1 / orig_w, 0.0, 1.0)
        y1n = np.clip(y1 / orig_h, 0.0, 1.0)
        x2n = np.clip(x2 / orig_w, 0.0, 1.0)
        y2n = np.clip(y2 / orig_h, 0.0, 1.0)

        dets = np.stack([x1n, y1n, x2n, y2n, conf, cls_id.astype(np.float32)], axis=1)

        results = []
        for c in np.unique(cls_id):
            c_mask = dets[:, 5] == c
            c_dets = dets[c_mask]
            bboxes_xywh = []
            scores_list = []
            for d in c_dets:
                _x1n, _y1n, _x2n, _y2n, _conf, _ = d
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
            indices = np.array(indices).flatten()
            for i in indices:
                results.append(tuple(c_dets[i].tolist()))

        return results

    def _postprocess_cls(self, outputs):
        probs = outputs[0]
        if probs.ndim == 2:
            probs = probs[0]

        top1_idx  = int(np.argmax(probs))
        top1_conf = float(probs[top1_idx])
        return top1_idx, top1_conf

    def predict(self, frame):
        orig_shape = frame.shape[:2]

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


# ======================== 加载模型 ========================
cls_model = YOLOOnnxModel(CLS_MODEL_PATH, model_type="cls", input_size=CLS_INPUT_SIZE)
det_day_model = YOLOOnnxModel(DET_DAY_MODEL_PATH, model_type="det", input_size=DET_INPUT_SIZE)
det_night_model = YOLOOnnxModel(DET_NIGHT_MODEL_PATH, model_type="det", input_size=DET_INPUT_SIZE)


# ======================== 核心函数 ========================
def get_weather_class(frame):
    """场景分类：移除多余参数，仅传frame"""
    cls_frame = cv2.resize(frame, (320, 320), interpolation=cv2.INTER_LINEAR)
    results = cls_model.predict(cls_frame)
    
    result = results[0]
    top1_idx = result.probs.top1
    top1_conf = result.probs.top1conf.item()
    cls_label = result.names.get(top1_idx, str(top1_idx))
    
    print(f"分类结果：{cls_label} | 置信度：{top1_conf:.3f}")
    return cls_label, top1_conf

weather_cache = {}

def get_weather_class_cached(frame, camera_id, frame_idx):
    cache = weather_cache.get(camera_id)
    if cache is not None and frame_idx % CLS_INTERVAL != 0:
        return cache

    result = get_weather_class(frame)
    weather_cache[camera_id] = result
    return result

def detect_target(frame, weather_cls):
    """目标检测：移除多余参数，仅传frame"""
    det_model = det_night_model if weather_cls == "night" else det_day_model
    
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


# ======================== 录制相关核心函数 ========================
def init_recording_dirs():
    os.makedirs(RECORD_ROOT, exist_ok=True)
    for idx in range(len(VIDEO_PATHS)):
        camera_dir = os.path.join(RECORD_ROOT, f"camera_{idx+1}")
        os.makedirs(camera_dir, exist_ok=True)
    print(f"✅ 录制目录初始化完成：{RECORD_ROOT}")

def create_alarm_record_dir():
    global RECORD_START_TIME
    RECORD_START_TIME = datetime.now()
    time_str = RECORD_START_TIME.strftime("%Y%m%d_%H%M%S")
    alarm_root = os.path.join(RECORD_ROOT, f"alarm_{time_str}")
    os.makedirs(alarm_root, exist_ok=True)
    for idx in range(len(VIDEO_PATHS)):
        camera_alarm_dir = os.path.join(alarm_root, f"camera_{idx+1}")
        os.makedirs(camera_alarm_dir, exist_ok=True)
    return alarm_root

# ======================== 全局告警状态机更新函数 ========================
def update_alarm_state(current_camera_has_alarm):
    """
    核心状态机逻辑
    :param current_camera_has_alarm: 当前正在轮询的这一路摄像头是否检测到告警
    :return: (是否需要立即写入预录缓存, 当前的全局ALARM_STATE)
    """
    global ALARM_STATE, ALARM_LAST_TRIGGER_TIME, RECORD_START_TIME
    
    now = time.time()
    need_save_prerecord = False

    # 规则1：如果当前这一路检测到告警 -> 立即触发/刷新全局告警
    if current_camera_has_alarm:
        if not ALARM_STATE:
            print(f"🚨 检测到告警！启动360°全路录制！")
            need_save_prerecord = True  # 标记需要把预录缓存写进去
        
        ALARM_STATE = True
        ALARM_LAST_TRIGGER_TIME = now

    # 规则2：如果全局告警已开启 -> 检查是否超时冷却
    if ALARM_STATE:
        # 检查距离最后一次告警是否超过停止延迟时间
        if ALARM_LAST_TRIGGER_TIME and (now - ALARM_LAST_TRIGGER_TIME) > ALARM_STOP_DELAY:
            print(f"⚠️ 告警消失{ALARM_STOP_DELAY}秒，停止录制")
            ALARM_STATE = False
            # 注意：这里不立即合成视频，留到程序结束统一合成，避免中途卡顿
    
    return need_save_prerecord, ALARM_STATE

# ======================== 预录缓存写入函数 ========================
def flush_prerecord_to_disk():
    """
    告警触发瞬间调用：把所有摄像头的预录缓存写入磁盘
    """
    global RECORD_START_TIME
    if not ENABLE_PRERECORD:
        return

    if not RECORD_START_TIME:
        alarm_root = create_alarm_record_dir()
    else:
        alarm_root = os.path.join(RECORD_ROOT, f"alarm_{RECORD_START_TIME.strftime('%Y%m%d_%H%M%S')}")

    print(f"💾 写入预录缓存（最近{PRERECORD_SECONDS}秒）...")
    # 遍历所有摄像头，写入预录帧
    for cid in range(1, len(VIDEO_PATHS)+1):
        if cid in prerecord_buffers:
            count = 0
            snapshot = list(prerecord_buffers[cid])
            for (ts, prerec_frame) in snapshot:
                save_frame_for_recording(prerec_frame, cid, alarm_root)
                count += 1
            if count > 0:
                print(f"  摄像头{cid}: 写入{count}帧预录数据")

def save_frame_for_recording(frame, camera_id, alarm_root):
    frame_idx = len(RECORD_FRAMES[camera_id]) + 1
    frame_filename = f"frame_{frame_idx:05d}.{FRAME_SAVE_FORMAT}"
    frame_path = os.path.join(alarm_root, f"camera_{camera_id}", frame_filename)
    cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    RECORD_FRAMES[camera_id].append(frame_path)
    return frame_path


def compose_video_from_frames():
    print("\n📹 开始合成视频...")
    for camera_id, frame_paths in RECORD_FRAMES.items():
        if not frame_paths:
            continue
        
        sample_frame = cv2.imread(frame_paths[0])
        if sample_frame is None:
            print(f"❌ 摄像头{camera_id}无有效帧，跳过合成")
            continue
        h, w = sample_frame.shape[:2]
        
        time_str = RECORD_START_TIME.strftime("%Y%m%d_%H%M%S") if RECORD_START_TIME else "unknown"
        video_path = os.path.join(RECORD_ROOT, f"camera_{camera_id}_alarm_{time_str}.{VIDEO_EXT}")
        
        video_writer = cv2.VideoWriter(
            video_path,
            VIDEO_CODEC,
            VIDEO_FPS,
            (w, h)
        )
        
        if not video_writer.isOpened():
            print(f"❌ 摄像头{camera_id}视频写入器初始化失败")
            continue
        
        for frame_path in sorted(frame_paths):
            frame = cv2.imread(frame_path)
            if frame is not None:
                video_writer.write(frame)
        
        video_writer.release()
        print(f"✅ 摄像头{camera_id}视频合成完成：{video_path}")

# ======================== RTSP推流相关函数（修复版） ========================
def init_rtsp_processes():
    global rtsp_processes, RTSP_ENABLE
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
    except:
        print("❌ FFmpeg不可用，推流已禁用")
        RTSP_ENABLE = False
        return

    for idx in range(len(VIDEO_PATHS)):
        camera_id = idx + 1
        start_rtsp_process(camera_id)

def start_rtsp_process(camera_id):
    global rtsp_processes
    if camera_id in rtsp_processes and rtsp_processes[camera_id]:
        try:
            rtsp_processes[camera_id].stdin.close()
            rtsp_processes[camera_id].terminate()
            rtsp_processes[camera_id].wait(timeout=1)
        except:
            pass

    rtsp_url = f"{RTSP_BASE_URL}{camera_id}"
    cmd = FFMPEG_PUSH_PARAMS + [rtsp_url]
    
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            bufsize=10**7
        )
        rtsp_processes[camera_id] = proc
        rtsp_fail_count[camera_id] = 0
        print(f"✅ 摄像头{camera_id} 推流就绪: {rtsp_url}")
        return True
    except Exception as e:
        print(f"❌ 摄像头{camera_id} 推流启动失败: {e}")
        rtsp_processes[camera_id] = None
        return False

def send_frame_to_rtsp(frame, camera_id):
    """发送帧到RTSP（稳定无Broken Pipe）"""
    if not RTSP_ENABLE or camera_id not in rtsp_processes:
        return

    proc = rtsp_processes[camera_id]
    if proc is None or proc.poll() is not None:
        rtsp_fail_count[camera_id] += 1
        if rtsp_fail_count[camera_id] > MAX_RETRY_COUNT:
            print(f"⚠️ 摄像头{camera_id} 连续失败，重启推流...")
            start_rtsp_process(camera_id)
        return

    try:
        frame_resized = cv2.resize(frame, RTSP_RESOLUTION, interpolation=cv2.INTER_LINEAR)
        proc.stdin.write(frame_resized.tobytes())
    except BrokenPipeError:
        print(f"⚠️ 摄像头{camera_id} 管道断裂，准备重启...")
        start_rtsp_process(camera_id)
    except Exception as e:
        print(f"⚠️ 摄像头{camera_id} 发送失败: {str(e)}")

def release_rtsp_processes():
    global rtsp_processes
    for cam_id, proc in rtsp_processes.items():
        if proc and proc.poll() is None:
            try:
                proc.stdin.close()
                proc.terminate()
                proc.wait(timeout=2)
            except:
                pass
    rtsp_processes.clear()
    print("✅ 推流进程已全部释放")


# ======================== 主程序 ========================
if __name__ == "__main__":
    init_recording_dirs()
    
    if not PIL_AVAILABLE:
        print("📦 正在安装PIL库...")
        os.system("pip install pillow -q")
        try:
            from PIL import Image, ImageDraw, ImageFont
            PIL_AVAILABLE = True
            print("✅ PIL库安装成功")
        except:
            print("❌ PIL库安装失败，中文显示可能异常")

    print(f"OpenCV GStreamer support: {opencv_has_gstreamer()}")

    caps = []
    stabilizers = []
    frame_counters = defaultdict(int)
    # ======================== 轮流处理核心变量 ========================
    current_cam_idx = 0  # 当前正在处理的摄像头索引（0-3）
    frame_buffer = {}    # 帧缓存：存储每路摄像头最新的绘制帧（用于拼接显示）

    for cam_idx, path in enumerate(VIDEO_PATHS):
        camera_id = cam_idx + 1  # 摄像头ID从1开始
        # 传入camera_id
        cap = AsyncVideoCapture(path, camera_id).start()
        caps.append(cap)
        if ENABLE_STABILIZATION:
            stabilizer = RealTimeVideoStabilizer(
                smoothing_radius=STAB_SMOOTHING_RADIUS,
                crop_ratio=STAB_CROP_RATIO
            )
            stabilizers.append(stabilizer)
        else:
            stabilizers.append(None)
    
    if RTSP_ENABLE:
        init_rtsp_processes()
    
    try:
        while True:
            if not check_work_duration():
                print(f"工作时长已达{DEFAULT_WORK_DURATION}秒，退出程序")
                break
            
            # ======================== 1. 轮流检测单路摄像头（仅这一路跑YOLO，省算力） ========================
            idx = current_cam_idx
            cap = caps[idx]
            stabilizer = stabilizers[idx]
            camera_id = idx + 1
            current_has_alarm = False  # 仅当前这一路的检测结果

            if cap is not None:
                ret, frame = cap.read()
                if ret and frame is not None:
                    frame_counters[camera_id] += 1
                    
                    # 1. 稳像处理
                    stabilized_frame = frame
                    if ENABLE_STABILIZATION and stabilizer is not None:
                        stabilized_frame = stabilizer.process_frame(frame)
                    
                    # 2. 场景分类（带缓存）
                    weather_label, weather_conf = get_weather_class_cached(
                        stabilized_frame,
                        camera_id,
                        frame_counters[camera_id]
                    )
                    weather_info = (weather_label, weather_conf)
                    
                    # 3. 目标检测
                    det_results = detect_target(stabilized_frame, weather_label)
                    
                    # 4. 后处理 + 单路告警判断
                    post_results, current_has_alarm = postprocess_detections(
                        det_results, 
                        stabilized_frame.shape[:2], 
                        weather_label
                    )
                    
                    # 5. 工作模式切换
                    continuous_mode = check_work_mode(current_has_alarm)
                    
                    # 6. 选择检测模型
                    model_names = det_night_model.names if weather_label == "night" else det_day_model.names
                    
                    # 7. 绘制框体（仅用于当前路显示）
                    draw_frame = draw_custom_boxes(
                        stabilized_frame.copy(), 
                        post_results, 
                        weather_info, 
                        model_names
                    )
                    
                    # 8. OCR处理
                    ocr_results, draw_frame = process_ship_ocr(
                        draw_frame, 
                        post_results, 
                        weather_label, 
                        camera_id
                    )
                    
                    # 9. RTSP推流（仅推当前检测路，省带宽）
                    send_frame_to_rtsp(draw_frame, camera_id)
                    
                    # 10. 更新帧缓存（用于拼接显示）
                    frame_buffer[idx] = draw_frame

            # ======================== 2. 更新全局告警状态（调用封装函数） ========================
            need_save_prerecord, current_alarm_state = update_alarm_state(current_has_alarm)

            # 如果是告警触发的第一瞬间，先把预录缓存写进去
            if need_save_prerecord:
                flush_prerecord_to_disk()

            # ======================== 3. 执行全路录制（只要全局告警开启，所有路都录） ========================
            if current_alarm_state:
                # 确保录制目录已创建
                if not RECORD_START_TIME:
                    _ = create_alarm_record_dir()
                
                alarm_root = os.path.join(RECORD_ROOT, f"alarm_{RECORD_START_TIME.strftime('%Y%m%d_%H%M%S')}")
                
                # 遍历所有摄像头，读取最新帧并录制
                for cid, c_cap in enumerate(caps):
                    rec_camera_id = cid + 1
                    rec_ret, rec_frame = c_cap.read()
                    if rec_ret and rec_frame is not None:
                        save_frame_for_recording(rec_frame, rec_camera_id, alarm_root)

            # ======================== 4. 切换到下一个摄像头（继续轮流检测） ========================
            current_cam_idx = (current_cam_idx + 1) % len(caps)

            # ======================== 5. 统一拼接显示所有摄像头画面 ========================
            # ... (这部分保持原样，不需要改) ...
            # 从缓存中获取所有路的最新帧，无帧则用黑色占位
            all_frames = []
            for i in range(len(caps)):
                if i in frame_buffer:
                    all_frames.append(frame_buffer[i])
                else:
                    # 无帧时填充黑色画面
                    all_frames.append(np.zeros((480, 640, 3), dtype=np.uint8))

            # 原逻辑：多画面自动布局
            h, w = 480, 640
            resized_frames = [cv2.resize(f, (w, h)) for f in all_frames]
            n = len(resized_frames)

            if n == 1:
                canvas = resized_frames[0]
            elif n == 2:
                canvas = np.hstack(resized_frames)
            elif n == 3:
                top_row = np.hstack(resized_frames[:2])
                black = np.zeros((h, w, 3), dtype=np.uint8)
                bottom_row = np.hstack([resized_frames[2], black])
                canvas = np.vstack([top_row, bottom_row])
            elif n >= 4:
                row1 = np.hstack(resized_frames[:2])
                row2 = np.hstack(resized_frames[2:4])
                canvas = np.vstack([row1, row2])

            cv2.imshow("Ship Detection - ALL Cameras", canvas)

            # 退出按键
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                print("用户手动退出程序")
                break
    
    except Exception as e:
        print(f"程序异常：{str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        compose_video_from_frames()
        release_rtsp_processes()
        for cap in caps:
            if cap is not None:
                cap.release()
        cv2.destroyAllWindows()
        print("✅ 程序正常退出，资源已释放")
