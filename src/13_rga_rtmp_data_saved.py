"""
基于12_optimiser_rtmp_data_saved.py，用 rga_utils 在 Python 层把 BGR 转成 NV12，然后 pipeline 去掉 videoconvert.
  12版: appsrc(BGR) → videoconvert(CPU) → NV12 → mpph264enc → ...
  13版: appsrc(NV12) → mpph264enc → ...           # RGA 做转换
yolo_rknn，OCR为ppocrv4_rknn版本的pipeline
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
from collections import defaultdict, deque
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os
import shutil
import re
from datetime import datetime
import glob
import threading
import collections
import queue
import json

# ======================== RGA 硬件加速导入 ========================
try:
    from rga_utils import rga_bgr_to_nv12
    RGA_AVAILABLE = True
    print("✅ RGA 硬件加速已加载 (BGR→NV12)")
except Exception as e:
    RGA_AVAILABLE = False
    print(f"⚠️ RGA 不可用: {e}")

# ======================== GStreamer RK MPP 硬编码推流 ========================
try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst
    Gst.init(None)
    GST_AVAILABLE = True
except ImportError:
    GST_AVAILABLE = False
    print("⚠️ 未安装GStreamer Python绑定，MPP硬编码推流不可用")
    print("   请执行: sudo apt install python3-gi gstreamer1.0-python3-plugin-loader")


# ======================== PIL导入（可视化兼容处理） ========================
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️ 未安装PIL库，中文显示可能异常，请执行：pip install pillow")


# 统一模型路径
CLS_MODEL_PATH = "/home/Tronlong/Desktop/rknn_model_zoo_and_models/yolov11n-cls/cls_1-best_255_fp.rknn"        # 场景分类模型
DET_DAY_MODEL_PATH = "/home/Tronlong/Desktop/rknn_model_zoo_and_models/yolov11n-day/day_5-best_255_i8.rknn"    # 白天船舶检测模型
DET_NIGHT_MODEL_PATH = "/home/Tronlong/Desktop/rknn_model_zoo_and_models/yolov11n-night/night_2-best_255_i8.rknn"  # 夜晚灯光检测模型

# 4路视频路径 - 修改为海康网络摄像头RTSP流
VIDEO_PATHS = [
    # 海康网络摄像头RTSP流格式：rtsp://用户名:密码@IP地址:端口/流路径
    #"rtsp://admin:ADMIN123@169.254.241.100:554/stream1",  # 夜通航网络摄像头1
    #"rtsp://admin:ADMIN123@169.254.241.100:554/live/0/SUB",  # 子码流（推荐，占用资源少）
    #"rtsp://admin:xiayuxuan113@169.254.242.101:554/stream1",  # 海康网络摄像头2
    #"rtsp://admin:xiayuxuan113@169.254.242.101:554/Streaming/Channels/102",  # 子码流
    #"rtsp://admin:ADMIN123@169.254.243.102:554/stream1",  # 夜通航网络摄像头2
    #"rtsp://admin:ADMIN123@169.254.243.102:554/live/0/SUB",  # 子码流
    #"/dev/video0",  # MSI模块摄像头
    "/home/Tronlong/Desktop/pipeline/test/多云-近处小船（P船号）.mp4",     #多云    
    #"/home/Tronlong/Desktop/pipeline/test/晴天短视频.mp4",  #晴天    
    #"/home/Tronlong/Desktop/pipeline/test/雾天.mp4",  #雾天  正常
    #"/home/Tronlong/Desktop/pipeline/test/夜晚-带月亮.mp4",   #夜晚  不正常，识别成foggy
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
# 中文字体路径（修复 Linux 中文显示）
CHINESE_FONT_PATH = None
# 优先使用兼容Linux的中文字体
font_candidates = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Linux 标准中文字体
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",           # 文泉驿正黑
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "C:/Windows/Fonts/simhei.ttf",    # Windows
    "C:/Windows/Fonts/msyh.ttc",
    "/System/Library/Fonts/PingFang.ttc",  # Mac
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

# ======================== PaddleOCRv4_RKNN 配置 ========================
ENABLE_OCR = True  # OCR功能总开关
# 仅识别白天告警船只的ROI
# OCR置信度阈值（过滤低置信度结果）
OCR_CONF_THRESH = 0.5
# RKNN OCR模型路径
DET_MODEL_PATH = "/home/Tronlong/Desktop/rknn_model_zoo_and_models/PPOCR_rknn/det/ppocrv4_det.rknn"
REC_MODEL_PATH = "/home/Tronlong/Desktop/rknn_model_zoo_and_models/PPOCR_rknn/rec/ppocrv4_rec.rknn"
# 配套字典路径
DICT_PATH = "/home/Tronlong/Desktop/rknn_model_zoo_and_models/PPOCR_rknn/ppocr_keys_v1.txt"

# OCR预处理参数
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
VIDEO_FPS = 20  # 视频帧率
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
PRERECORD_FPS = 20                 # 计算需要缓存的最大帧数（假设20fps）
# 每路摄像头的预录帧缓存：{camera_id: deque([(timestamp, frame), ...])}
prerecord_buffers = defaultdict(collections.deque)

# 原有变量（兼容 create_alarm_record_dir 等函数）
RECORD_FRAMES = defaultdict(list)
RECORD_START_TIME = None

# ======================== RTMP推流核心配置（MPP硬编码） ========================
RTMP_ENABLE = True
RTMP_BASE_URL = "rtmp://192.168.0.104:8554/camera_"
RTMP_FPS = 20
RTMP_RESOLUTION = (640, 480)

# 存储推流管道
rtmp_pipelines = {}

# ======================== 显示配置（推流时可关闭节省CPU） ========================
ENABLE_DISPLAY = True

# ======================== 性能检测配置（部署时关闭，零开销） ========================
PROFILE_ENABLE = True            # 是否启用性能计时统计
PROFILE_PRINT_INTERVAL = 100     # 每N轮打印一次统计摘要
PROFILE_OUTPUT_PATH = "perf_log.txt"  # 统计数据输出文件（追加写入）

# ======================== 性能计时器（PROFILE_ENABLE=False时零开销） ========================
class PerfTimer:
    """轻量级逐段计时器。关闭时所有方法均为空操作。"""
    def __init__(self, enabled=True, window=200, output_path=None):
        self.enabled = enabled
        self._output_path = output_path
        # 首次写入时清空旧文件
        if enabled and output_path:
            open(output_path, 'w').close()
        if not enabled:
            return
        self._times = defaultdict(lambda: deque(maxlen=window))
        self._last = time.perf_counter()
        self._loops = deque(maxlen=window)
        self._tick = 0
        self._last_print = 0

    def tick(self):
        """标记新一轮循环开始"""
        if not self.enabled: return
        now = time.perf_counter()
        if self._tick > 0:
            self._loops.append(now - self._last)
        self._last = now
        self._tick += 1

    def lap(self, name):
        """记录自上次 tick/lap 至今的耗时(ms)"""
        if not self.enabled: return
        now = time.perf_counter()
        self._times[name].append((now - self._last) * 1000)
        self._last = now

    def summary(self):
        if not self.enabled or self._tick < 2:
            return ""
        total = sum(self._loops) or 1
        lines = [f"--- PERF [{self._tick} iters | {len(self._loops)/total:.1f} fps] ---"]
        for name in sorted(self._times):
            dq = self._times[name]
            avg = sum(dq) / len(dq) if dq else 0
            lines.append(f"  {name:>12s}: {avg:6.1f}ms avg")
        return "\n".join(lines)

    def maybe_print(self, interval=100):
        if not self.enabled: return
        if self._tick - self._last_print >= interval:
            self._last_print = self._tick
            text = self.summary()
            print(text)
            if self._output_path:
                ts = time.strftime("%Y-%m-%d %H:%M:%S")
                with open(self._output_path, 'a') as f:
                    f.write(f"[{ts}]\n{text}\n\n")


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


# ======================== PaddleOCR V4 初始化与处理函数 ========================

def is_valid_text(text, min_length=2, min_chinese_count=1, min_digit_count=1):
    """
    判断文本是否为有效文本（过滤无意义字符）
    新增：过滤掉纯英文文本（无中文、无数字）
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

# OCR_RKNN依赖
from rknnlite.api import RKNNLite
import utils.operators
from utils.db_postprocess import DBPostProcess, DetPostProcess
from utils.rec_postprocess import CTCLabelDecode

# RKNN模型基类
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

# 文本检测
class TextDetector:
    def __init__(self):
        self.model = RKNNModel(DET_MODEL_PATH)
        self.preprocess_funct = []
        for item in DET_PRE_PROCESS_CONFIG:
            for key in item:
                pclass = getattr(utils.operators, key)
                p = pclass(**item[key])
                self.preprocess_funct.append(p)
        self.db_postprocess = DBPostProcess(**DET_POSTPROCESS_CONFIG['DBPostProcess'])
        self.det_postprocess = DetPostProcess()
    def preprocess(self, img):
        data = {'image': img}
        for p in self.preprocess_funct:
            data = p(data)
        return data
    def detect(self, img):
        data = self.preprocess(img)
        img_input = data['image'][np.newaxis, :]
        output = self.model.run([img_input])
        preds = {'maps': output[0].astype(np.float32)}
        result = self.db_postprocess(preds, data['shape'])
        dt_boxes = self.det_postprocess.filter_tag_det_res(result[0]['points'], img.shape)
        return dt_boxes

# 文本识别
class TextRecognizer:
    def __init__(self):
        self.model = RKNNModel(REC_MODEL_PATH)
        self.preprocess_funct = []
        for item in REC_PRE_PROCESS_CONFIG:
            for key in item:
                pclass = getattr(utils.operators, key)
                p = pclass(**item[key])
                self.preprocess_funct.append(p)
        self.ctc_postprocess = CTCLabelDecode(**REC_POSTPROCESS_CONFIG['CTCLabelDecode'])
    def preprocess(self, img):
        data = {'image': img}
        for p in self.preprocess_funct:
            data = p(data)
        return data
    def recognize(self, img_crop):
        img = cv2.resize(img_crop, (REC_INPUT_SHAPE[1], REC_INPUT_SHAPE[0]))
        data = self.preprocess(img)
        img_input = data['image'][np.newaxis, :]
        output = self.model.run([img_input])
        text, score = self.ctc_postprocess(output[0])[0]
        return text, score

# 文本框排序 从上到下从左到右
def sort_boxes(boxes):
    boxes = sorted(boxes, key=lambda x: (np.min(x[:, 1]), np.min(x[:, 0])))
    return boxes

# ========================PaddleOCR RKNN 引擎 ========================
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

# 全局初始化
if ENABLE_OCR:
    ocr_engine = RKNNOCREngine()
else:
    ocr_engine = None

# ======================== 告警事件收集器 ========================
class AlarmCollector:
    """管理单次告警事件的所有素材与元数据"""
    def __init__(self, alarm_id, root_dir, camera_count, buoy_pos="123.4567°E, 30.1234°N"):
        self.alarm_id = alarm_id
        self.root_dir = root_dir               # 告警根目录（如 alarm_20260506_083045）
        self.camera_count = camera_count
        self.buoy_pos = buoy_pos               # 浮标位置（可从配置读取）
        self.alarm_level = "一级"               # 暂定
        self.alarm_type = "船舶入侵告警"
        self.start_time = datetime.utcnow()
        self.end_time = None
        self.weather_label = "unknown"
        self.weather_conf = 0.0
        self.ship_bbox = [0, 0, 0, 0]   	# 告警船舶边界框（默认，预制接口）
        self.ocr_text = None                    # 最佳舷号文本
        self.ship_image_paths = {}              # cam_id -> 船号图片路径
        self.bbox_image_path = None             # 船体框图像路径（代表帧）
        self.weather_image_path = None          # 天气图像路径
        self.video_paths = {}                   # cam_id -> 视频文件路径
        self.detection_json_path = None         # 检测结果JSON路径（可选）

    def set_weather(self, label, conf):
        self.weather_label = label
        self.weather_conf = conf

    def set_ocr(self, text):
        """仅当新文本比当前更长或置信度更高时更新"""
        if self.ocr_text is None or len(text) > len(self.ocr_text):
            self.ocr_text = text

    def add_ship_image(self, cam_id, img_path):
        self.ship_image_paths[cam_id] = img_path

    def set_bbox_image(self, path):
        self.bbox_image_path = path

    def set_weather_image(self, path):
        self.weather_image_path = path

    def add_video(self, cam_id, video_path):
        self.video_paths[cam_id] = video_path



def generate_alarm_json(collector):
    """根据收集器内容创建完整的告警JSON"""
    data = {
        "alarm_id": collector.alarm_id,
        "alarm_type": collector.alarm_type,
        "alarm_level": collector.alarm_level,
        "alarm_time": collector.start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "buoy_pos": collector.buoy_pos,
        "target_info": {
            "mmsi": "unknown",                             # 无AIS时占位
            "ship_name": "unknown",
            "distance": "unknown",
            "ship_bbox": "[0,0,0,0]",                      # 将在具体告警对象中填充
            "ocr_text": collector.ocr_text or "",
            "weather": collector.weather_label
        },
        "alarm_status": "结束",
        "media_url": {
            f"camera_{i+1}": (os.path.relpath(v, collector.root_dir) if v else "")
            for i in range(collector.camera_count)
            for v in [collector.video_paths.get(i+1, "")]
        },
        "ship_images": {
            f"camera_{i+1}": (os.path.relpath(p, collector.root_dir) if p else "")
            for i, p in collector.ship_image_paths.items()
        },
        "bbox_image": os.path.relpath(collector.bbox_image_path, collector.root_dir) if collector.bbox_image_path else "",
        "weather_image": os.path.relpath(collector.weather_image_path, collector.root_dir) if collector.weather_image_path else ""
    }
    json_path = os.path.join(collector.root_dir, "alarm.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ 告警JSON已保存：{json_path}")


def save_ship_roi_image(frame, ship_bbox, ocr_text, alarm_root, cam_id):
    """裁剪船舶区域并保存为船号图像"""
    x1, y1, x2, y2 = ship_bbox
    h, w = frame.shape[:2]
    # 确保坐标在范围内
    x1 = max(0, x1 - 5)
    y1 = max(0, y1 - 5)
    x2 = min(w, x2 + 5)
    y2 = min(h, y2 + 5)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    cam_dir = os.path.join(alarm_root, f"camera_{cam_id}")
    os.makedirs(cam_dir, exist_ok=True)
    filename = f"ship_{datetime.now().strftime('%H%M%S')}.jpg"
    path = os.path.join(cam_dir, filename)
    cv2.imwrite(path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return path



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
        
        # 保存ROI（可选）
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
        
        # OCR推理（在干净的ROI上）
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
        
        # 筛选最佳结果
        best_text = None
        best_score = 0
        best_poly = None
        
        for idx in range(len(rec_texts)):
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

# ======================== RKNN Lite 模型封装（兼容 YOLOOnnxModel 接口） ========================
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

    def __init__(self, model_path, model_type="det", input_size=640, names=None):
        """
        :param names: 自定义类别映射字典，如 {0:"ship"}
        """
        self.model_path = model_path
        self.model_type = model_type
        self.input_size = input_size
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
            self.CONF_THRESH, self.IOU_THRESH
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


# ======================== 加载 ONNX 模型 ========================
#cls_model = YOLOOnnxModel(CLS_MODEL_PATH, model_type="cls", input_size=CLS_INPUT_SIZE)
#det_day_model = YOLOOnnxModel(DET_DAY_MODEL_PATH, model_type="det", input_size=DET_INPUT_SIZE)
#det_night_model = YOLOOnnxModel(DET_NIGHT_MODEL_PATH, model_type="det", input_size=DET_INPUT_SIZE)

# ======================== 加载 RKNN 模型 ========================
cls_model = YOLOv11RKNN(
    CLS_MODEL_PATH, model_type="cls", input_size=CLS_INPUT_SIZE,
    names={0: "cloudy", 1: "foggy", 2: "night", 3: "rainy", 4: "sunny"}
)
det_day_model = YOLOv11RKNN(
    DET_DAY_MODEL_PATH, model_type="det", input_size=DET_INPUT_SIZE,
    names={0: "ship"}
)
det_night_model = YOLOv11RKNN(
    DET_NIGHT_MODEL_PATH, model_type="det", input_size=DET_INPUT_SIZE,
    names={0: "light"}
)

# ======================== 核心函数 ========================
def get_weather_class(frame):
    """场景分类：移除多余参数，仅传frame"""
    #cls_frame = cv2.resize(frame, (320, 320), interpolation=cv2.INTER_LINEAR)
    results = cls_model.predict(frame)
    
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

def save_representative_frames(collector, prerecord_buffers, camera_count, post_results, weather_info, model_names):
    """
    从预录缓存中取中间帧，绘制船体框并保存，同时保存天气帧。
    """
    alarm_root = collector.root_dir
    # 天气帧：取第一个摄像头的预录中间帧（或最近帧）
    for cam_id in range(1, camera_count + 1):
        if cam_id not in prerecord_buffers or len(prerecord_buffers[cam_id]) == 0:
            continue
        frames = list(prerecord_buffers[cam_id])
        # 取中间一帧作为代表
        idx = len(frames) // 2
        _, frame = frames[idx]

        # 保存天气图像（未绘制）
        weather_path = os.path.join(alarm_root, f"camera_{cam_id}", "weather.jpg")
        cv2.imwrite(weather_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        collector.set_weather_image(weather_path)

        # 如果是检测到告警的那一路，绘制船体框后保存
        if cam_id == current_alarm_camera:   # 这里需传入触发告警的摄像头ID
            draw_frame = draw_custom_boxes(frame.copy(), post_results, weather_info, model_names)
            bbox_path = os.path.join(alarm_root, f"camera_{cam_id}", "bbox_frame.jpg")
            cv2.imwrite(bbox_path, draw_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            collector.set_bbox_image(bbox_path)
        break  # 只保存一个摄像头的天气代表帧


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
    state_changed = False

    if current_camera_has_alarm:
        if not ALARM_STATE:
            print(f"🚨 检测到告警！启动360°全路录制！")
            need_save_prerecord = True
            state_changed = True
        ALARM_STATE = True
        ALARM_LAST_TRIGGER_TIME = now
    else:
        # 无告警时检查冷却
        if ALARM_STATE and (now - ALARM_LAST_TRIGGER_TIME) > ALARM_STOP_DELAY:
            print(f"⚠️ 告警消失{ALARM_STOP_DELAY}秒，停止录制")
            ALARM_STATE = False
            state_changed = True  # 增加一个输出 state_changed，标记 ALARM_STATE 是否刚发生跳变。

    return need_save_prerecord, ALARM_STATE, state_changed

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


def compose_alarm_videos(collector):
    print("\n📹 合成告警视频...")
    for camera_id, frame_paths in RECORD_FRAMES.items():
        if not frame_paths:
            continue
        sample = cv2.imread(frame_paths[0])
        if sample is None:
            continue
        h, w = sample.shape[:2]
        cam_dir = os.path.join(collector.root_dir, f"camera_{camera_id}")
        video_path = os.path.join(cam_dir, "video.mp4")
        writer = cv2.VideoWriter(video_path, VIDEO_CODEC, VIDEO_FPS, (w, h))
        for fp in sorted(frame_paths):
            f = cv2.imread(fp)
            if f is not None:
                writer.write(f)
        writer.release()
        collector.add_video(camera_id, video_path)
        print(f"  摄像头{camera_id} -> {video_path}")

def create_alarm_record_dir_with_id(alarm_id):
    alarm_root = os.path.join(RECORD_ROOT, alarm_id)
    for idx in range(len(VIDEO_PATHS)):
        os.makedirs(os.path.join(alarm_root, f"camera_{idx+1}"), exist_ok=True)
    return alarm_root

# ======================== GStreamer MPP 硬编码推流（替代FFmpeg软编码） ========================
class GstRtmpPush:
    """GStreamer RTMP 推流管道。

    用 RGA 硬件将 BGR 转 NV12 后直送 mpph264enc，管道不再走 CPU videoconvert。
    RGA 不可用时回退 CPU cv2.cvtColor + GStreamer videoconvert。
    """

    def __init__(self, camera_id, rtmp_url, width=640, height=480, fps=15):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None
        self.appsrc = None

        if not GST_AVAILABLE:
            print(f"❌ 摄像头{camera_id}: GStreamer不可用，无法创建推流管道")
            return

        # RGA 可用 → appsrc 直接吃 NV12，管道去掉 videoconvert
        if RGA_AVAILABLE:
            caps = f"video/x-raw,format=NV12,width={width},height={height},framerate={fps}/1"
            pipeline_str = (
                f'appsrc name=src is-live=true do-timestamp=true format=GST_FORMAT_TIME '
                f'max-buffers=2 drop=true '
                f'caps={caps} ! '
                f'queue max-size-buffers=2 leaky=downstream ! '
                f'mpph264enc name=enc ! '
                f'queue max-size-buffers=2 leaky=downstream ! '
                f'h264parse ! '
                f'flvmux streamable=true name=mux ! '
                f'rtmpsink name=sink sync=false'
            )
        else:
            # 回退：BGR → videoconvert(CPU) → NV12 → mpph264enc
            caps = f"video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1"
            pipeline_str = (
                f'appsrc name=src is-live=true do-timestamp=true format=GST_FORMAT_TIME '
                f'max-buffers=2 drop=true '
                f'caps={caps} ! '
                f'queue max-size-buffers=2 leaky=downstream ! '
                f'videoconvert ! video/x-raw,format=NV12 ! '
                f'queue max-size-buffers=2 leaky=downstream ! '
                f'mpph264enc name=enc ! '
                f'queue max-size-buffers=2 leaky=downstream ! '
                f'h264parse ! '
                f'flvmux streamable=true name=mux ! '
                f'rtmpsink name=sink sync=false'
            )

        try:
            self.pipeline = Gst.parse_launch(pipeline_str)
            self.appsrc = self.pipeline.get_by_name('src')

            rtmpsink = self.pipeline.get_by_name('sink')
            if rtmpsink:
                rtmpsink.set_property('location', rtmp_url)

            ret = self.pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError(f"管道启动失败 (camera {camera_id})")

            mode = "RGA NV12直送" if RGA_AVAILABLE else "CPU videoconvert"
            print(f"✅ 摄像头{camera_id} RTMP推流就绪 ({mode}): {rtmp_url}")
        except Exception as e:
            print(f"❌ 摄像头{camera_id} 推流管道创建失败: {e}")
            self.pipeline = None
            self.appsrc = None

    def push_frame(self, frame):
        """将 OpenCV BGR 帧转为 NV12 后推入 GStreamer 管道。

        RGA 可用时：硬件一步完成缩放 + BGR→NV12
        RGA 不可用时：CPU cv2.resize + cv2.cvtColor BGR→YUV_I420→NV12
        """
        if self.appsrc is None or self.pipeline is None:
            return

        try:
            if RGA_AVAILABLE:
                nv12_buf, _, _ = rga_bgr_to_nv12(
                    frame, self.width, self.height)
                data = nv12_buf.tobytes()
            else:
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))
                yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
                data = _i420_to_nv12(yuv)
                data = data.tobytes()
                # 回退路径需要重新转回 bytes — 直接用 memoryview
                # (YUV_I420 → NV12: Y 平面不变，UV 平面需要交织)

            buf = Gst.Buffer.new_allocate(None, len(data), None)
            buf.fill(0, data)

            buf.pts = Gst.CLOCK_TIME_NONE
            buf.duration = Gst.SECOND // self.fps

            self.appsrc.emit('push-buffer', buf)
        except Exception as e:
            print(f"⚠️ 摄像头{self.camera_id} 推流帧发送失败: {e}")

    def stop(self):
        if self.appsrc is not None:
            try:
                self.appsrc.emit('end-of-stream')
            except Exception:
                pass
            self.appsrc = None
        if self.pipeline is not None:
            try:
                self.pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass
            self.pipeline = None


def _i420_to_nv12(yuv_i420):
    """CPU 回退：I420 (OpenCV BGR2YUV_I420) → NV12 flat buffer。"""
    total_h = yuv_i420.shape[0]
    w = yuv_i420.shape[1]
    h = total_h * 2 // 3          # 原始图像高度
    uv_h = h // 4                  # 每个色度平面高 = h/4

    y_plane = yuv_i420[:h, :w]
    u_plane = yuv_i420[h:h + uv_h, :w // 2].ravel()
    v_plane = yuv_i420[h + uv_h:h + 2 * uv_h, :w // 2].ravel()

    uv_interleaved = np.empty(u_plane.size * 2, dtype=np.uint8)
    uv_interleaved[0::2] = u_plane
    uv_interleaved[1::2] = v_plane
    return np.concatenate([y_plane.ravel(), uv_interleaved])


def init_rtmp_pipelines():
    """初始化各摄像头的GStreamer RTMP推流管道"""
    global rtmp_pipelines, RTMP_ENABLE

    if not GST_AVAILABLE:
        print("❌ GStreamer不可用，MPP硬编码推流已禁用")
        RTMP_ENABLE = False
        return

    # 注意：不使用GLib主循环线程，避免与cv2.imshow() X11访问冲突

    for idx in range(len(VIDEO_PATHS)):
        camera_id = idx + 1
        rtmp_url = f"{RTMP_BASE_URL}{camera_id}"
        pipeline = GstRtmpPush(camera_id, rtmp_url,
                                width=RTMP_RESOLUTION[0],
                                height=RTMP_RESOLUTION[1],
                                fps=RTMP_FPS)
        rtmp_pipelines[camera_id] = pipeline


def push_frame_to_rtmp(frame, camera_id):
    """发送帧到RTMP（MPP硬编码）"""
    if not RTMP_ENABLE or camera_id not in rtmp_pipelines:
        return

    pipeline = rtmp_pipelines[camera_id]
    if pipeline is None:
        return

    pipeline.push_frame(frame)


def release_rtmp_pipelines():
    """释放所有推流管道"""
    global rtmp_pipelines
    for cam_id, pipeline in rtmp_pipelines.items():
        if pipeline is not None:
            pipeline.stop()
    rtmp_pipelines.clear()

    print("✅ 推流管道已全部释放")


# ======================== 主程序 ========================
if __name__ == "__main__":
    # ========== 初始化 ==========
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

    # 多路视频捕获与稳像器
    caps = []
    stabilizers = []
    frame_counters = defaultdict(int)
    current_cam_idx = 0                # 当前检测的摄像头索引
    frame_buffer = {}                  # 各摄像头的绘制帧缓存

    for cam_idx, path in enumerate(VIDEO_PATHS):
        camera_id = cam_idx + 1
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

    if RTMP_ENABLE:
        init_rtmp_pipelines()

    # 告警状态变量
    alarm_collector = None             # 当前告警收集器
    prev_alarm_state = False           # 用于检测告警状态变化
    current_alarm_camera = None        # 触发告警的摄像头编号

    # 启动异步上传线程（占位框架，可后续实现）
    upload_queue = queue.Queue()
    def upload_worker():
        while True:
            alarm_dir = upload_queue.get()
            if alarm_dir is None:
                break
            # TODO: 实现SFTP/HTTP上传逻辑（此处模拟归档）
            print(f"[上传] 处理 {alarm_dir} （模拟上传...）")
            time.sleep(2)
            # 移动至归档目录（示例）
            archive_dir = alarm_dir.replace("alarm_data", "archive")
            os.makedirs(os.path.dirname(archive_dir), exist_ok=True)
            shutil.move(alarm_dir, archive_dir)
            print(f"[上传] 完成，已归档至 {archive_dir}")
    threading.Thread(target=upload_worker, daemon=True).start()

    perf = PerfTimer(enabled=PROFILE_ENABLE, output_path=PROFILE_OUTPUT_PATH)

    try:
        while True:
            perf.tick()
            # 工作模式检查
            if not check_work_duration():
                print(f"工作时长已达{DEFAULT_WORK_DURATION}秒，退出程序")
                break

            # ======================== 1. 单路轮询检测 ========================
            idx = current_cam_idx
            cap = caps[idx]
            stabilizer = stabilizers[idx]
            camera_id = idx + 1
            current_has_alarm = False

            if cap is not None:
                ret, frame = cap.read()
                perf.lap('capture')
                if ret and frame is not None:
                    frame_counters[camera_id] += 1

                    # 稳像
                    stabilized_frame = frame
                    if ENABLE_STABILIZATION and stabilizer is not None:
                        stabilized_frame = stabilizer.process_frame(frame)
                    perf.lap('stabilize')

                    # 场景分类（带缓存）
                    weather_label, weather_conf = get_weather_class_cached(
                        stabilized_frame, camera_id, frame_counters[camera_id]
                    )
                    perf.lap('classify')
                    weather_info = (weather_label, weather_conf)

                    # 目标检测
                    det_results = detect_target(stabilized_frame, weather_label)
                    perf.lap('detect')

                    # 后处理 + 当前路告警标志
                    post_results, current_has_alarm = postprocess_detections(
                        det_results, stabilized_frame.shape[:2], weather_label
                    )
                    perf.lap('postproc')

                    # 工作模式
                    continuous_mode = check_work_mode(current_has_alarm)

                    # 检测模型类别名
                    model_names = det_night_model.names if weather_label == "night" else det_day_model.names

                    # OCR识别（仅告警时执行，节省CPU）
                    ocr_results = process_ship_ocr(
                        stabilized_frame, post_results, weather_label, camera_id
                    ) if current_has_alarm else []
                    perf.lap('ocr')

                    # 绘制检测框
                    draw_frame = draw_custom_boxes(
                        stabilized_frame.copy(), post_results, weather_info, model_names
                    )
                    # 绘制OCR结果
                    draw_frame = draw_ocr_results(draw_frame, ocr_results)
                    perf.lap('draw')

                    # RTMP推流（MPP硬编码）
                    push_frame_to_rtmp(draw_frame, camera_id)
                    perf.lap('rtmp_push')

                    # 更新帧缓存
                    frame_buffer[idx] = draw_frame

            # ======================== 2. 更新全局告警状态机 ========================
            need_save_prerecord, current_alarm_state, state_changed = update_alarm_state(current_has_alarm)

            # ======================== 3. 告警触发瞬间处理 ========================
            if need_save_prerecord:
                # 创建告警目录和收集器
                alarm_id = f"alarm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                alarm_root = create_alarm_record_dir_with_id(alarm_id)
                alarm_collector = AlarmCollector(alarm_id, alarm_root, len(VIDEO_PATHS))

                # 记录天气和触发摄像头
                alarm_collector.set_weather(weather_label, weather_conf)
                current_alarm_camera = camera_id

                # 提取第一个告警目标的bbox（用于JSON）
                for res in post_results:
                    x1, y1, x2, y2, conf, cls, area, is_alarm = res
                    if is_alarm:
                        alarm_collector.ship_bbox = [x1, y1, x2, y2]
                        break

                # 写入预录缓存帧
                flush_prerecord_to_disk()

                # 保存代表帧（船体框+天气）
                save_representative_frames(
                    alarm_collector, prerecord_buffers, len(VIDEO_PATHS),
                    post_results, weather_info, model_names
                )

            # ======================== 4. 告警持续期间录制与OCR记录 ========================
            if current_alarm_state:
                if alarm_collector is not None:
                    # 录制所有摄像头的最新帧
                    alarm_root = alarm_collector.root_dir
                    for cid, c_cap in enumerate(caps):
                        rec_camera_id = cid + 1
                        rec_ret, rec_frame = c_cap.read()
                        if rec_ret and rec_frame is not None:
                            save_frame_for_recording(rec_frame, rec_camera_id, alarm_root)

                    # 处理当前检测路的OCR船号
                    if ocr_results:
                        best_ocr = max(ocr_results, key=lambda x: x['conf'])
                        alarm_collector.set_ocr(best_ocr['text'])
                        # 保存船号图像（首次或高置信度更新）
                        if (camera_id not in alarm_collector.ship_image_paths or
                            best_ocr['conf'] > 0.7):
                            ship_path = save_ship_roi_image(
                                stabilized_frame, best_ocr['ship_bbox'],
                                best_ocr['text'], alarm_root, camera_id
                            )
                            if ship_path:
                                alarm_collector.add_ship_image(camera_id, ship_path)

            # ======================== 5. 告警结束处理 ========================
            if state_changed and not current_alarm_state and alarm_collector is not None:
                # 合成视频
                compose_alarm_videos(alarm_collector)
                alarm_collector.end_time = datetime.utcnow()
                # 生成alarm.json
                generate_alarm_json(alarm_collector)

                # 将告警目录加入异步上传队列
                upload_queue.put(alarm_collector.root_dir)

                # 清空录制缓存和状态
                RECORD_FRAMES.clear()
                prerecord_buffers.clear()
                alarm_collector = None
                current_alarm_camera = None
                print("✅ 告警事件归档完成")

            # ======================== 6. 切换到下一路摄像头 ========================
            current_cam_idx = (current_cam_idx + 1) % len(caps)

            # ======================== 7. 画面拼接显示 ========================
            all_frames = []
            for i in range(len(caps)):
                if i in frame_buffer:
                    all_frames.append(frame_buffer[i])
                else:
                    all_frames.append(np.zeros((480, 640, 3), dtype=np.uint8))

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

            if ENABLE_DISPLAY:
                cv2.imshow("Ship Detection - ALL Cameras", canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):
                    print("用户手动退出程序")
                    break
            else:
                # 推流模式下无需GUI事件循环，短暂sleep避免CPU空转
                time.sleep(0.001)

            perf.lap('display')
            perf.maybe_print(PROFILE_PRINT_INTERVAL)

    except Exception as e:
        print(f"程序异常：{str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # 如果有未结束的告警，强制归档
        if alarm_collector is not None:
            print("正在处理未完成的告警...")
            compose_alarm_videos(alarm_collector)
            alarm_collector.end_time = datetime.utcnow()
            generate_alarm_json(alarm_collector)
            upload_queue.put(alarm_collector.root_dir)
            RECORD_FRAMES.clear()

        # 释放推流和摄像头
        release_rtmp_pipelines()
        for cap in caps:
            if cap is not None:
                cap.release()
        cv2.destroyAllWindows()
        print("✅ 程序正常退出，资源已释放")
