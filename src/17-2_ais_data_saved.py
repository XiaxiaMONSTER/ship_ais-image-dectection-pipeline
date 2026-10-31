"""
基于16_cv2_data_saved.py，融合了AIS的处理。
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
    from rga_utils import rga_bgr_to_nv12_safe as rga_bgr_to_nv12
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


# ======================== Freetype 混合渲染（零拷贝中文绘制） ========================
# ASCII (0x20-0x7E) → cv2.putText，其余(CJK/符号/°) → cv2.freetype
_ft_handle = None

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
    "rtsp://admin:ADMIN123@169.254.243.102:554/live/0/SUB",  # 子码流
    #"/dev/video0",  # MSI模块摄像头
    #"/home/Tronlong/Desktop/pipeline/test/多云-近处小船（P船号）.mp4",     #多云    
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
FONT_THICKNESS = 2     # 文本线条厚度
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
RTMP_BASE_URL = "rtmp://192.168.1.100:1935/camera_"
RTMP_FPS = 20
RTMP_RESOLUTION = (640, 480)

# 存储推流管道
rtmp_pipelines = {}

# ======================== 显示配置（推流时可关闭节省CPU） ========================
ENABLE_DISPLAY = True

# ======================== AIS 融合配置 ========================
# AIS（船舶自动识别系统）通过串口/网络接收周围船只的实时位置广播
# 与本系统的图像检测融合，实现"图像检测 OR AIS距离<阈值"的联合告警
AIS_ENABLE = True                       # AIS 集成总开关（False=完全不加载AIS模块）
AIS_READ_INTERVAL = 0.2                 # AIS 读取尝试间隔（秒），可配置。AIS 数据更新频率本身较低（几秒一次），无需高频读取
AIS_DATA_SOURCE = 'FILE'             # 数据源：'FILE'（本地文件测试）, 'SERIAL'（串口）, 'NETWORK'（TCP）
AIS_SERIAL_PORT = '/dev/ttyS1'         # 串口设备路径（RK3576 硬件串口）
AIS_SERIAL_BAUDRATE = 115200           # 波特率（与AIS接收机一致：115200, 8-N-1）
AIS_SERIAL_BYTESIZE = 8
AIS_SERIAL_PARITY = 'N'
AIS_SERIAL_STOPBITS = 1
AIS_NETWORK_HOST = '192.168.1.10'      # 网络数据源主机
AIS_NETWORK_PORT = 8001                # 网络数据源端口
AIS_INPUT_FILE = '2点_ais.txt'          # FILE模式下的输入文件名（位于项目根目录）
AIS_BUOY_LAT = 19.564408               # 浮标（本船/观测点）纬度
AIS_BUOY_LON = 110.822627              # 浮标（本船/观测点）经度
AIS_WARNING_DISTANCE = 200.0           # AIS 告警距离阈值（米），船只进入此范围触发告警
AIS_ALARM_COOLDOWN = 300               # AIS 同一船只告警冷却时间（秒），5分钟内不重复报
# AIS 数据保存根目录（统一收敛，不再散落在 src/ 下）
_AIS_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AIS_DATA_ROOT = os.path.join(_AIS_PROJECT_ROOT, "ais_data")
AIS_LOG_FILE = os.path.join(AIS_DATA_ROOT, "ais_data_log.jsonl")       # AIS 全量数据日志
AIS_ALARM_DIR = os.path.join(AIS_DATA_ROOT, "alarms")                   # AIS 告警日志子目录
AIS_TRAJECTORY_DIR = os.path.join(AIS_DATA_ROOT, "trajectories")        # AIS 轨迹图目录
AIS_TRAJECTORY_SAVE_INTERVAL = 60.0    # 轨迹图自动保存间隔（秒），0=仅在程序退出时保存

# ======================== 性能检测配置（部署时关闭，零开销） ========================
PROFILE_ENABLE = False            # 是否启用性能计时统计
PROFILE_PRINT_INTERVAL = 100     # 每N轮打印一次统计摘要
PROFILE_OUTPUT_PATH = os.path.join(AIS_DATA_ROOT, "perf_16_log.txt")  # 统计数据输出文件

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
    def __init__(self, alarm_id, root_dir, camera_count, buoy_pos="123.4567°E, 30.1234°N", ais_data=None):
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
        self.ais_data = ais_data or {}          # AIS 融合数据：{mmsi, lat, lon, distance, level, ship_name}（融合告警时填充）
        self.alarm_trigger_source = "图像检测"   # 告警触发来源：'图像检测' / 'AIS距离' / '图像检测+AIS距离'

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


# ======================== AIS 线程安全共享状态 ========================
class AISSharedState:
    """
    线程安全的 AIS 数据共享状态。
    AIS 读取线程写入，主循环只读（非阻塞），通过 lock 保证数据一致性。
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.ships = {}              # mmsi -> {lat, lon, distance, alarm_level, in_zone, last_seen, heading, speed, ship_name}
        self.has_alarm = False       # 是否有任何船只在告警区内（距离 < AIS_WARNING_DISTANCE）
        self.alarm_ships = []        # [(mmsi, distance, level), ...] 当前在告警区内的船只列表
        self.closest_ship = None     # {mmsi, lat, lon, distance, level, ship_name} 最近的告警船只
        self.last_read_time = 0      # 最后一次成功读取 AIS 数据的时间戳
        self.total_messages = 0      # 累计处理的 AIS 报文数
        self.track_history = {}      # mmsi -> [(lat, lon, ts), ...] 用于轨迹绘制

    def update_ship(self, mmsi, lat, lon, timestamp, distance, in_zone, level,
                    heading=None, speed=None, ship_name=None):
        """由 AIS 读取线程调用：更新单艘船的信息"""
        with self.lock:
            self.ships[mmsi] = {
                'lat': lat, 'lon': lon, 'distance': distance,
                'in_zone': in_zone, 'alarm_level': level,
                'last_seen': timestamp, 'heading': heading,
                'speed': speed, 'ship_name': ship_name
            }
            # 轨迹历史（限制长度，防止内存溢出）
            if mmsi not in self.track_history:
                self.track_history[mmsi] = []
            self.track_history[mmsi].append((lat, lon, timestamp))
            if len(self.track_history[mmsi]) > 10000:
                self.track_history[mmsi] = self.track_history[mmsi][-5000:]

    def update_alarm_status(self):
        """由 AIS 读取线程调用：重新计算告警状态"""
        with self.lock:
            self.alarm_ships = [
                (mmsi, info['distance'], info['alarm_level'])
                for mmsi, info in self.ships.items()
                if info['in_zone']
            ]
            self.has_alarm = len(self.alarm_ships) > 0
            # 找到最近（距离最小）的告警船只
            if self.alarm_ships:
                self.alarm_ships.sort(key=lambda x: x[1])  # 按距离排序
                closest_mmsi, closest_dist, closest_level = self.alarm_ships[0]
                info = self.ships.get(closest_mmsi, {})
                self.closest_ship = {
                    'mmsi': closest_mmsi,
                    'lat': info.get('lat'), 'lon': info.get('lon'),
                    'distance': closest_dist, 'level': closest_level,
                    'ship_name': info.get('ship_name'),
                    'heading': info.get('heading'),
                    'speed': info.get('speed'),
                }
            else:
                self.closest_ship = None

    def get_alarm_state(self):
        """主线程调用：获取当前 AIS 告警状态（非阻塞快速读取）"""
        with self.lock:
            return self.has_alarm, list(self.alarm_ships), self.closest_ship.copy() if self.closest_ship else None

    def get_all_ships(self):
        """主线程调用：获取所有追踪船只的快照"""
        with self.lock:
            return dict(self.ships)

    def get_track_history(self):
        """主线程调用：获取轨迹历史快照"""
        with self.lock:
            return {mmsi: list(pts) for mmsi, pts in self.track_history.items()}


# ======================== AIS 数据读取线程 ========================
class AISReaderThread:
    """
    AIS 数据读取线程。
    - 低功耗设计：可配置读取间隔（AIS_READ_INTERVAL），避免高频空转
    - 非阻塞：独立线程运行，不卡主循环
    - 支持三种数据源：FILE（本地测试）、SERIAL（串口）、NETWORK（TCP）
    """

    def __init__(self, state: AISSharedState):
        self.state = state
        self.running = False
        self.thread = None
        self._alarm_cooldown = {}     # mmsi -> last_alarm_time（AIS内部冷却，防日志刷屏）
        self._log_file = None
        self._alarm_log_file = None

    def start(self):
        """启动 AIS 读取线程"""
        if not AIS_ENABLE:
            print("ℹ️ AIS 集成已禁用（AIS_ENABLE=False）")
            return
        self.running = True
        # 确保 AIS 数据目录存在
        os.makedirs(os.path.dirname(AIS_LOG_FILE), exist_ok=True)
        self._log_file = open(AIS_LOG_FILE, 'a', encoding='utf-8')
        # 告警日志单独存入 ais_data/alarms/ 子目录
        os.makedirs(AIS_ALARM_DIR, exist_ok=True)
        alarm_filename = os.path.join(AIS_ALARM_DIR, f"ais_alarm_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
        self._alarm_log_file = open(alarm_filename, 'w', encoding='utf-8')
        # 启动线程
        self.thread = threading.Thread(target=self._run, daemon=True, name="AIS-Reader")
        self.thread.start()
        print(f"✅ AIS 读取线程已启动（数据源={AIS_DATA_SOURCE}, 读取间隔={AIS_READ_INTERVAL}s, 告警距离<{AIS_WARNING_DISTANCE}m）")

    def _run(self):
        """根据配置选择数据源"""
        if AIS_DATA_SOURCE == 'FILE':
            self._run_file()
        elif AIS_DATA_SOURCE == 'SERIAL':
            self._run_serial()
        elif AIS_DATA_SOURCE == 'NETWORK':
            self._run_network()
        else:
            print(f"⚠️ 未知的 AIS 数据源类型: {AIS_DATA_SOURCE}")

    def _run_file(self):
        """从本地文件读取 AIS 数据（测试用）"""
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', AIS_INPUT_FILE)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                print(f"📂 AIS 文件读取: {file_path}")
                for line in f:
                    if not self.running:
                        break
                    self._process_line(line)
                    time.sleep(AIS_READ_INTERVAL)  # 模拟读取间隔
            print("--- AIS 文件处理完成 ---")
        except FileNotFoundError:
            print(f"⚠️ AIS 输入文件未找到: {file_path}")

    def _run_serial(self):
        """从串口读取 AIS 数据（生产环境）"""
        import serial
        while self.running:
            try:
                with serial.Serial(
                    AIS_SERIAL_PORT, AIS_SERIAL_BAUDRATE,
                    bytesize=AIS_SERIAL_BYTESIZE,
                    parity=AIS_SERIAL_PARITY,
                    stopbits=AIS_SERIAL_STOPBITS,
                    timeout=0.1  # 短超时（100ms），避免阻塞线程退出
                ) as ser:
                    print(f"✅ AIS 串口已连接: {AIS_SERIAL_PORT} @ {AIS_SERIAL_BAUDRATE}bps")
                    while self.running:
                        line_bytes = ser.readline()
                        if line_bytes:
                            self._process_line(line_bytes.decode('utf-8', errors='ignore'))
                        else:
                            # 无数据时休眠配置的间隔，降低 CPU 占用
                            time.sleep(AIS_READ_INTERVAL)
            except serial.SerialException as e:
                print(f"⚠️ AIS 串口错误: {e}，5秒后重试...")
                time.sleep(5)
            except Exception as e:
                print(f"⚠️ AIS 串口未知错误: {e}，5秒后重试...")
                time.sleep(5)

    def _run_network(self):
        """从 TCP 网络端口读取 AIS 数据"""
        import socket
        while self.running:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(1.0)
                    s.connect((AIS_NETWORK_HOST, AIS_NETWORK_PORT))
                    print(f"✅ AIS 网络已连接: {AIS_NETWORK_HOST}:{AIS_NETWORK_PORT}")
                    with s.makefile('r', encoding='utf-8', errors='ignore') as f:
                        while self.running:
                            line = f.readline()
                            if not line:
                                print("⚠️ AIS 网络连接已断开，尝试重连...")
                                break
                            self._process_line(line)
            except Exception as e:
                print(f"⚠️ AIS 网络错误: {e}，5秒后重试...")
                time.sleep(5)

    @staticmethod
    def _extract_nmea(line: str):
        """
        从原始行中提取时间戳和纯 NMEA 报文。
        兼容格式：
          - [133981662129140000]!AIVDM,...
          - !AIVDM,... （纯 NMEA）
        :return: (timestamp_ns, nmea_string) 或 (None, None)
        """
        timestamp = None
        nmea = None
        line = line.strip()
        if not line:
            return None, None
        if ']' in line and ('!' in line or '$' in line):
            try:
                parts = line.split(']', 1)
                ts_str = parts[0].strip('[')
                if ts_str.isdigit():
                    timestamp = int(ts_str)
                nmea = parts[1].strip()
            except (ValueError, IndexError):
                return None, None
        elif line.startswith(('!', '$')):
            nmea = line
            timestamp = int(time.time() * 1e9)
        return timestamp, nmea

    @staticmethod
    def _get_alarm_level(distance):
        """根据距离返回告警等级：1(一级,≤100m) / 2(二级,≤150m) / 3(三级,≤200m) / None"""
        if distance is None:
            return None
        if distance <= 100:
            return 1
        elif distance <= 150:
            return 2
        elif distance <= AIS_WARNING_DISTANCE:
            return 3
        return None

    def _process_line(self, line: str):
        """处理单行 NMEA 数据：解码 → 计算距离 → 更新共享状态 → 写日志"""
        timestamp, nmea = self._extract_nmea(line)
        if not nmea:
            return

        try:
            from pyais import decode
            msg = decode(nmea).asdict()
        except ImportError:
            # pyais 未安装时静默跳过（首次打印提示）
            if self.state.total_messages == 0:
                print("⚠️ pyais 未安装，AIS 解码不可用。请执行: pip install pyais")
            return
        except Exception:
            return  # 解码失败，静默跳过

        msg_type = msg.get("msg_type")
        # 仅处理 AIS 位置报告报文（类型 1/2/3/18/19）
        if msg_type not in (1, 2, 3, 18, 19):
            return

        lat = msg.get("lat")
        lon = msg.get("lon")
        mmsi = msg.get("mmsi")

        if not all([lat, lon, mmsi]):
            return

        # 使用 Haversine 公式计算与浮标的距离
        distance = _ais_calc_distance(AIS_BUOY_LAT, AIS_BUOY_LON, lat, lon)

        # 判断是否在告警区内
        level = self._get_alarm_level(distance)
        in_zone = level is not None

        # 提取附加信息
        heading = msg.get('heading')
        speed = msg.get('speed')
        raw_shipname = msg.get('shipname', '')
        ship_name = raw_shipname.strip('@').strip() if raw_shipname else None

        # 更新线程安全共享状态
        self.state.update_ship(mmsi, lat, lon, timestamp, distance, in_zone, level,
                               heading=heading, speed=speed, ship_name=ship_name)
        self.state.total_messages += 1
        self.state.last_read_time = time.time()
        self.state.update_alarm_status()

        # 告警时打印到控制台（带冷却，防刷屏）
        if in_zone:
            now = time.time()
            if mmsi not in self._alarm_cooldown or (now - self._alarm_cooldown[mmsi]) > AIS_ALARM_COOLDOWN:
                self._alarm_cooldown[mmsi] = now
                level_label = {1: "一级", 2: "二级", 3: "三级"}.get(level, "")
                print(f"\n    #######################################")
                print(f"    #   !!! AIS {level_label}告警 !!! MMSI:{mmsi} 距离:{round(distance, 1)}m   #")
                print(f"    #######################################\n")

        # 写入全量数据日志（JSONL）
        if self._log_file:
            log_record = {
                'log_timestamp_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'raw_nmea': nmea,
                'decoded_data': msg,
                'analysis': {
                    'distance_to_buoy_m': distance,
                    'warning_triggered': in_zone,
                    'alarm_level': level
                }
            }
            self._log_file.write(json.dumps(log_record, ensure_ascii=False) + '\n')
            self._log_file.flush()

        # 告警时单独写入告警日志
        if in_zone and self._alarm_log_file:
            alarm_record = {
                'alarm_timestamp_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'alarm_level': level,
                'mmsi': mmsi,
                'lat': lat,
                'lon': lon,
                'distance_to_buoy_m': distance,
                'ship_name': ship_name,
                'raw_nmea': nmea,
                'decoded_data': msg
            }
            self._alarm_log_file.write(json.dumps(alarm_record, ensure_ascii=False) + '\n')
            self._alarm_log_file.flush()

    def stop(self):
        """停止 AIS 读取线程并关闭日志文件"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3.0)
        if self._log_file:
            self._log_file.close()
            self._log_file = None
        if self._alarm_log_file:
            self._alarm_log_file.close()
            self._alarm_log_file = None
        print("✅ AIS 读取线程已停止")


# ======================== AIS 距离计算（Haversine 公式，内联避免跨文件依赖） ========================
def _ais_calc_distance(lat1, lon1, lat2, lon2):
    """
    使用 Haversine 公式计算两点间的大圆距离（米）。
    内联实现，避免对 AIS/utils.py 的运行时依赖。
    """
    import math
    if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
        return None
    R = 6371000  # 地球半径（米）
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return round(R * c, 2)



def generate_alarm_json(collector):
    """根据收集器内容创建完整的告警JSON（含AIS融合数据）"""
    ais = collector.ais_data
    # 从 AIS 数据或默认值填充目标船只信息
    mmsi = ais.get("mmsi", "unknown") if ais else "unknown"
    ship_name = ais.get("ship_name", "unknown") if ais else "unknown"
    distance_str = f"{ais.get('distance', 'unknown')}m" if ais and ais.get('distance') else "unknown"
    target_lat = ais.get("lat", "unknown") if ais else "unknown"
    target_lon = ais.get("lon", "unknown") if ais else "unknown"
    target_heading = ais.get("heading", "unknown") if ais else "unknown"
    target_speed = ais.get("speed", "unknown") if ais else "unknown"

    data = {
        "alarm_id": collector.alarm_id,
        "alarm_type": collector.alarm_type,
        "alarm_level": collector.alarm_level,
        "alarm_time": collector.start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "alarm_trigger_source": collector.alarm_trigger_source,
        "buoy_pos": collector.buoy_pos,
        "target_info": {
            "mmsi": mmsi,
            "ship_name": ship_name,
            "distance": distance_str,
            "target_lat": target_lat,
            "target_lon": target_lon,
            "target_heading": target_heading,
            "target_speed": target_speed,
            "ship_bbox": str(collector.ship_bbox),
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

def save_representative_frames(collector, raw_frame, draw_frame, trigger_cam_id, camera_count):
    """
    保存告警代表帧（不依赖预录缓存，直接用告警触发时刻的当前帧）。

    :param collector: AlarmCollector 实例
    :param raw_frame: 触发告警那一路的原始稳像帧（无绘制，用于 weather.jpg）
    :param draw_frame: 触发告警那一路的已绘制帧（带检测框，用于 bbox_frame.jpg）
    :param trigger_cam_id: 触发告警的摄像头编号 (1-based)
    :param camera_count: 摄像头总数
    """
    alarm_root = collector.root_dir

    # 为触发告警的摄像头保存 weather 和 bbox
    cam_dir = os.path.join(alarm_root, f"camera_{trigger_cam_id}")
    os.makedirs(cam_dir, exist_ok=True)

    weather_path = os.path.join(cam_dir, "weather.jpg")
    cv2.imwrite(weather_path, raw_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    collector.set_weather_image(weather_path)
    print(f"✅ 天气代表帧已保存: {weather_path}")

    bbox_path = os.path.join(cam_dir, "bbox_frame.jpg")
    cv2.imwrite(bbox_path, draw_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    collector.set_bbox_image(bbox_path)
    print(f"✅ 船体框代表帧已保存: {bbox_path}")


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

# ======================== 全局告警状态机更新函数（融合 AIS） ========================
def update_alarm_state(current_camera_has_alarm, ais_has_alarm=False):
    """
    核心状态机逻辑（融合图像检测 + AIS 距离告警）
    告警触发条件 = 图像检测告警 OR AIS 距离 < 200m
    一旦触发，统一的状态机启动录制和推流。

    :param current_camera_has_alarm: 当前正在轮询的这一路摄像头是否检测到告警
    :param ais_has_alarm: AIS 是否检测到船只进入告警区（距离 < AIS_WARNING_DISTANCE）
    :return: (need_save_prerecord, current_alarm_state, state_changed)
    """
    global ALARM_STATE, ALARM_LAST_TRIGGER_TIME, RECORD_START_TIME
    now = time.time()
    need_save_prerecord = False
    state_changed = False

    # ── 融合告警：图像检测 OR AIS 距离 ──
    combined_alarm = current_camera_has_alarm or ais_has_alarm

    if combined_alarm:
        if not ALARM_STATE:
            # 构造触发源描述
            sources = []
            if current_camera_has_alarm:
                sources.append("图像检测")
            if ais_has_alarm:
                sources.append("AIS距离")
            print(f"🚨 检测到告警（{' + '.join(sources)}）！启动360°全路录制！")
            need_save_prerecord = True
            state_changed = True
        ALARM_STATE = True
        ALARM_LAST_TRIGGER_TIME = now
    else:
        # 无告警时检查冷却
        if ALARM_STATE and (now - ALARM_LAST_TRIGGER_TIME) > ALARM_STOP_DELAY:
            print(f"⚠️ 告警消失{ALARM_STOP_DELAY}秒，停止录制")
            ALARM_STATE = False
            state_changed = True

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

# ======================== GStreamer MPP 硬编码推流（OpenCV VideoWriter 方式） ========================
class GstRtmpPush:
    """独立线程推流 — appsrc 直推，与命令行完全一致"""

    def __init__(self, camera_id, rtmp_url, width=640, height=480, fps=20):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None
        self.appsrc = None
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.frame_count = 0

        if not GST_AVAILABLE:
            print(f"❌ 摄像头{camera_id}: GStreamer不可用")
            return

        # 与命令行完全一致的管道（BGR → videoconvert → NV12 → mpph264enc）
        pipeline_str = (
            f'appsrc name=src is-live=true do-timestamp=true '
            f'caps="video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1" ! '
            f'videoconvert ! video/x-raw,format=NV12 ! '
            f'mpph264enc header-mode=1 ! '
            f'h264parse config-interval=-1 ! '
            f'flvmux streamable=true ! '
            f'rtmpsink location={rtmp_url} sync=false'
        )

        print(f"📝 摄像头{camera_id} 管道: {pipeline_str}")

        try:
            self.pipeline = Gst.parse_launch(pipeline_str)
            self.appsrc = self.pipeline.get_by_name('src')

            bus = self.pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect("message", self._on_bus_message)

            ret = self.pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("管道启动失败")

            # 启动独立推流线程
            self.running = True
            self.thread = threading.Thread(target=self._stream_loop, daemon=True)
            self.thread.start()

            print(f"✅ 摄像头{camera_id} MPP推流就绪: {rtmp_url}")

        except Exception as e:
            print(f"❌ 摄像头{camera_id} 推流管道失败: {e}")
            if self.pipeline:
                self.pipeline.set_state(Gst.State.NULL)
            self.pipeline = None
            self.appsrc = None

    def _on_bus_message(self, bus, msg):
        t = msg.type
        if t == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            print(f"❌ 摄像头{self.camera_id} GStreamer错误: {err.message}")
        elif t == Gst.MessageType.WARNING:
            err, debug = msg.parse_warning()
            print(f"⚠️ 摄像头{self.camera_id} GStreamer警告: {err.message}")

    def _stream_loop(self):
        """独立推流线程，持续推送最新帧"""
        black = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        last_pts = 0

        while self.running:
            with self.lock:
                frame = self.latest_frame.copy() if self.latest_frame is not None else black

            try:
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))

                data = frame.tobytes()
                buf = Gst.Buffer.new_allocate(None, len(data), None)
                buf.fill(0, data)

                # 递增时间戳
                pts = last_pts + Gst.SECOND // self.fps
                buf.pts = pts
                buf.dts = pts
                buf.duration = Gst.SECOND // self.fps
                last_pts = pts

                ret = self.appsrc.emit('push-buffer', buf)
                self.frame_count += 1

                if self.frame_count == 1:
                    print(f"✅ 摄像头{self.camera_id}: 第一帧已推流")
                elif self.frame_count % 100 == 0:
                    print(f"📊 摄像头{self.camera_id}: 已推流 {self.frame_count} 帧")

            except Exception as e:
                print(f"⚠️ 摄像头{self.camera_id} 推流帧失败: {e}")

            time.sleep(1.0 / self.fps)

    def update_frame(self, frame):
        """主线程调用：更新要推送的帧"""
        with self.lock:
            self.latest_frame = frame

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.appsrc:
            try:
                self.appsrc.emit('end-of-stream')
            except:
                pass
            self.appsrc = None
        if self.pipeline:
            try:
                self.pipeline.set_state(Gst.State.NULL)
            except:
                pass
            self.pipeline = None
        print(f"✅ 摄像头{self.camera_id} 推流已停止")


def init_rtmp_pipelines():
    """初始化各摄像头的推流管道"""
    global rtmp_pipelines, RTMP_ENABLE

    for idx in range(len(VIDEO_PATHS)):
        camera_id = idx + 1
        rtmp_url = f"{RTMP_BASE_URL}{camera_id}"
        pipeline = GstRtmpPush(camera_id, rtmp_url,
                               width=RTMP_RESOLUTION[0],
                               height=RTMP_RESOLUTION[1],
                               fps=RTMP_FPS)
        rtmp_pipelines[camera_id] = pipeline


def push_frame_to_rtmp(frame, camera_id):
    """更新要推流的帧（推流线程会持续推送）"""
    if not RTMP_ENABLE or camera_id not in rtmp_pipelines:
        return
    pipeline = rtmp_pipelines[camera_id]
    if pipeline is None:
        return
    pipeline.update_frame(frame)  # 只更新帧，不直接推


def release_rtmp_pipelines():
    """释放所有推流管道"""
    global rtmp_pipelines
    for cam_id, pipeline in rtmp_pipelines.items():
        if pipeline is not None:
            pipeline.stop()
    rtmp_pipelines.clear()
    print("✅ 推流管道已全部释放")


# ======================== AIS 轨迹可视化（独立目录保存） ========================
def _latlon_to_xy(lat, lon, lat0, lon0):
    """经纬度 → 以参考点为中心的 XY 米制坐标"""
    import math
    x = math.radians(lon - lon0) * 6371000 * math.cos(math.radians(lat0))
    y = math.radians(lat - lat0) * 6371000
    return x, y


def save_ais_trajectory_image(track_history, output_dir=None, timestamp=None):
    """
    将 AIS 船只轨迹渲染为图像，保存到独立目录（与图像检测绘制完全分离）。
    每艘船不同颜色，含浮标标记、告警圈、航向箭头、图例。

    :param track_history: {mmsi: [(lat, lon, ts), ...]}  轨迹数据
    :param output_dir: 输出目录，默认使用 AIS_TRAJECTORY_DIR
    :param timestamp: 时间戳字符串（用于文件名），默认当前时间
    """
    if not track_history:
        print("ℹ️ AIS 轨迹为空，跳过轨迹图保存")
        return None

    if output_dir is None:
        output_dir = AIS_TRAJECTORY_DIR
    os.makedirs(output_dir, exist_ok=True)

    if timestamp is None:
        timestamp = time.strftime('%Y%m%d_%H%M%S')

    try:
        import matplotlib
        matplotlib.use('Agg')  # 非交互后端，避免与 OpenCV GUI 冲突
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        import numpy as np
    except ImportError:
        print("⚠️ matplotlib 未安装，无法生成 AIS 轨迹图")
        return None

    # ── 转换为 XY 坐标 ──
    tracks_xy = {}
    total_points = 0
    for mmsi, pts in track_history.items():
        xy_pts = []
        for lat, lon, _ in pts:
            if lat is not None and lon is not None and abs(lat) > 0.01:
                x, y = _latlon_to_xy(lat, lon, AIS_BUOY_LAT, AIS_BUOY_LON)
                xy_pts.append((x, y))
        if len(xy_pts) >= 2:
            tracks_xy[mmsi] = xy_pts
            total_points += len(xy_pts)

    if not tracks_xy:
        print("ℹ️ AIS 轨迹无有效坐标点，跳过")
        return None

    # ── 计算绘图范围 ──
    all_pts = np.array([p for pts in tracks_xy.values() for p in pts])
    max_coord = np.max(np.abs(all_pts))
    limit = max(AIS_WARNING_DISTANCE * 1.5, max_coord * 1.2)

    # ── 创建图形 ──
    fig, ax = plt.subplots(figsize=(10, 10))
    try:
        cmap = matplotlib.colormaps.get_cmap('tab10')
    except AttributeError:
        cmap = plt.cm.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, max(1, len(tracks_xy))))

    legend_elements = []

    for i, (mmsi, xy_pts) in enumerate(tracks_xy.items()):
        pts = np.array(xy_pts)
        # 去重
        _, indices = np.unique(pts, axis=0, return_index=True)
        pts = pts[np.sort(indices)]
        xs, ys = pts[:, 0], pts[:, 1]
        c = colors[i % len(colors)]

        # 散点
        ax.scatter(xs, ys, s=15, color=c, alpha=0.35, zorder=2)

        # 轨迹线
        label = f"MMSI {mmsi}"
        if len(xs) >= 4:
            try:
                from scipy.interpolate import splprep, splev
                tck, u = splprep([xs, ys], s=2)
                u_new = np.linspace(0, 1, 300)
                l_xs, l_ys = splev(u_new, tck)
                line, = ax.plot(l_xs, l_ys, color=c, linewidth=1.8, label=label, zorder=3)
            except Exception:
                line, = ax.plot(xs, ys, color=c, linewidth=1.8, linestyle='--', label=label, zorder=3)
        else:
            line, = ax.plot(xs, ys, color=c, linewidth=1.8, linestyle='--', label=label, zorder=3)
        legend_elements.append(line)

        # 航向箭头（稀疏绘制）
        arrow_len = limit * 0.04
        step = max(len(xs) // 6, 1)
        for j in range(1, len(xs), step):
            dx = xs[j] - xs[j - 1]
            dy = ys[j] - ys[j - 1]
            if abs(dx) > 1e-5 or abs(dy) > 1e-5:
                ax.arrow(xs[j - 1], ys[j - 1], dx, dy,
                         head_width=arrow_len * 0.7, head_length=arrow_len,
                         fc=c, ec=c, length_includes_head=True, zorder=4)

    # ── 浮标标记 ──
    ax.scatter(0, 0, facecolors='none', edgecolors='black', s=250,
              marker='o', linewidth=2.5, zorder=5)
    buoy_icon = Line2D([0], [0], marker='o', color='black', label='Buoy',
                       markerfacecolor='none', markersize=8,
                       markeredgewidth=1.5, linestyle='None')

    # ── 告警圈 ──
    circle = plt.Circle((0, 0), AIS_WARNING_DISTANCE, color='red', fill=False,
                        linestyle='--', linewidth=2, alpha=0.6, zorder=1)
    ax.add_patch(circle)
    zone_icon = Line2D([0], [0], color='red', linestyle='--', linewidth=1.5,
                       label=f'{int(AIS_WARNING_DISTANCE)}m Zone')

    # ── 图例 ──
    all_handles = [buoy_icon, zone_icon] + legend_elements
    ax.legend(handles=all_handles, loc='upper right', fontsize='x-small',
             ncol=2, framealpha=0.8)

    # ── 装饰 ──
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.set_title(f"AIS Track Analysis\nTime: {timestamp}  |  "
                 f"Vessels: {len(tracks_xy)}  |  Points: {total_points}", fontsize=12)
    ax.set_xlabel("East-West Distance (m)")
    ax.set_ylabel("North-South Distance (m)")

    # ── 保存 ──
    filename = f"ais_tracks_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ AIS 轨迹图已保存: {filepath}")
    return filepath


# ======================== 主程序 ========================
if __name__ == "__main__":
    # ========== 初始化 ==========
    init_recording_dirs()

    print(f"OpenCV GStreamer support: {opencv_has_gstreamer()}")
    if CHINESE_FONT_PATH and hasattr(cv2, "freetype"):
        print(f"✅ Freetype 中文就绪: {CHINESE_FONT_PATH}")
    else:
        print("⚠️ Freetype 中文不可用，将仅使用 ASCII 渲染")

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

    # ✅ 推流保活线程：在 init_rtmp_pipelines() 后立即启动
    keepalive_stop = threading.Event()
    
    def keepalive_worker():
        black = np.zeros((480, 640, 3), dtype=np.uint8)
        while not keepalive_stop.is_set():
            for cam_id in rtmp_pipelines:
                push_frame_to_rtmp(black, cam_id)
            time.sleep(1.0 / RTMP_FPS)  # 按目标帧率推
    
    keepalive_thread = threading.Thread(target=keepalive_worker, daemon=True)
    keepalive_thread.start()
    print("🔥 推流保活线程已启动（推送黑屏帧保持连接...）")

    # 告警状态变量
    alarm_collector = None             # 当前告警收集器
    prev_alarm_state = False           # 用于检测告警状态变化

    # ── AIS 融合初始化 ──
    ais_state = AISSharedState()
    ais_reader = AISReaderThread(ais_state)
    ais_reader.start()
    ais_last_trajectory_save = time.time()  # AIS 轨迹图上次保存时间

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
                        stabilized_frame, post_results, weather_info, model_names
                    )
                    # 绘制OCR结果
                    draw_frame = draw_ocr_results(draw_frame, ocr_results)
                    perf.lap('draw')

                    # RTMP推流（MPP硬编码）
                    push_frame_to_rtmp(draw_frame, camera_id)
                    perf.lap('rtmp_push')

                    # ✅ 第一次推流成功后，关闭保活线程
                    if not keepalive_stop.is_set():
                        keepalive_stop.set()
                        print("🔥 保活线程已关闭，正式推流开始")

                    # 更新帧缓存
                    frame_buffer[idx] = draw_frame

            # ======================== 2. 更新全局告警状态机（融合 AIS） ========================
            ais_has_alarm, ais_alarm_ships, ais_closest = ais_state.get_alarm_state()
            need_save_prerecord, current_alarm_state, state_changed = update_alarm_state(
                current_has_alarm, ais_has_alarm
            )

            # ======================== 3. 告警触发瞬间处理（融合 AIS 数据） ========================
            if need_save_prerecord:
                # 创建告警目录和收集器
                alarm_id = f"alarm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                alarm_root = create_alarm_record_dir_with_id(alarm_id)

                # ── 组装 AIS 数据（用于 alarm.json） ──
                combined_ais_data = {}
                if ais_closest:
                    combined_ais_data = {
                        "mmsi": ais_closest.get("mmsi"),
                        "lat": ais_closest.get("lat"),
                        "lon": ais_closest.get("lon"),
                        "distance": ais_closest.get("distance"),
                        "level": ais_closest.get("level"),
                        "ship_name": ais_closest.get("ship_name"),
                        "heading": ais_closest.get("heading"),
                        "speed": ais_closest.get("speed"),
                    }

                alarm_collector = AlarmCollector(
                    alarm_id, alarm_root, len(VIDEO_PATHS),
                    ais_data=combined_ais_data
                )

                # ── 确定告警触发来源 ──
                if current_has_alarm and ais_has_alarm:
                    alarm_collector.alarm_trigger_source = "图像检测+AIS距离"
                elif ais_has_alarm:
                    alarm_collector.alarm_trigger_source = "AIS距离"
                else:
                    alarm_collector.alarm_trigger_source = "图像检测"

                # 记录天气
                alarm_collector.set_weather(weather_label, weather_conf)

                # 提取第一个告警目标的bbox（用于JSON）
                for res in post_results:
                    x1, y1, x2, y2, conf, cls, area, is_alarm = res
                    if is_alarm:
                        alarm_collector.ship_bbox = [x1, y1, x2, y2]
                        break

                # 写入预录缓存帧
                flush_prerecord_to_disk()

                # 保存代表帧（船体框+天气）—— 直接用当前帧，不依赖预录缓存
                save_representative_frames(
                    alarm_collector, stabilized_frame, draw_frame,
                    camera_id, len(VIDEO_PATHS)
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

            # ======================== 8. AIS 轨迹图定时保存 ========================
            if AIS_ENABLE and AIS_TRAJECTORY_SAVE_INTERVAL > 0:
                now_ts = time.time()
                if now_ts - ais_last_trajectory_save >= AIS_TRAJECTORY_SAVE_INTERVAL:
                    track_hist = ais_state.get_track_history()
                    if track_hist:
                        save_ais_trajectory_image(track_hist)
                    ais_last_trajectory_save = now_ts

            perf.lap('display')
            perf.maybe_print(PROFILE_PRINT_INTERVAL)

    except Exception as e:
        print(f"程序异常：{str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # ✅ 确保保活线程关闭
        keepalive_stop.set()

        # ✅ 停止 AIS 读取线程并保存最终轨迹图
        if AIS_ENABLE and ais_reader is not None:
            # 保存最终 AIS 轨迹图
            track_hist = ais_state.get_track_history()
            if track_hist:
                print("📊 保存最终 AIS 轨迹图...")
                save_ais_trajectory_image(track_hist)
            ais_reader.stop()
            print(f"📊 AIS 累计处理报文: {ais_state.total_messages}")

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
