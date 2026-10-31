"""
相较于pipeline.py仅添加了对linux适配的处理，模型均为pt。
"""

"""
安装依赖：
# 安装系统依赖
sudo apt update && sudo apt install -y ffmpeg libopencv-dev python3-opencv
# 安装Python依赖
pip3 install ultralytics opencv-python numpy scipy matplotlib pillow paddleocr==2.7.0 paddlepaddle==2.5.2
权限配置：
# 给摄像头设备权限
sudo usermod -aG video $USER
# 重启生效
sudo reboot
模型部署：将 YOLO/PaddleOCR 模型放到/home/rk3576/models/目录下,具体路径查看代码。


paddleocr ocr -i ./test/general_ocr_002.png  --text_detection_model_name PP-OCRv5_server_det  --text_detection_model_dir PP-OCRv5_server_det_onnx  --text_recognition_model_name PP-OCRv5_server_rec  --text_recognition_model_dir PP-OCRv5_server_rec_onnx  --enable_hpi True   --device cpu

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
import glob
from datetime import datetime
import subprocess

# ======================== PIL导入（RK3576 Linux适配） ========================
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️ 未安装PIL库，中文显示可能异常，请执行：pip3 install pillow")

# ======================== 录制与推流核心配置（RK3576适配） ========================
# 录制根目录（确保RK3576有写入权限）
RECORD_ROOT = "/mnt/alarm_recordings"  # 建议挂载外部存储，避免占用系统空间
os.makedirs(RECORD_ROOT, exist_ok=True)
# 帧保存格式
FRAME_SAVE_FORMAT = "jpg"
# 视频合成参数（适配RK3576性能）
VIDEO_FPS = 25  # 降低帧率，减少算力消耗
VIDEO_CODEC = cv2.VideoWriter_fourcc(*'mp4v')
VIDEO_EXT = "mp4"
# 告警录制配置
ALARM_TRIGGERED = False  # 全局告警触发标记
RECORD_FRAMES = defaultdict(list)  # 按摄像头ID存储录制的帧路径 {camera_id: [frame_path1, frame_path2,...]}
RECORD_START_TIME = None  # 录制开始时间
ALARM_STOP_DELAY = 2  # 目标消失后延迟2秒停止录制（秒）
LAST_ALARM_TIME = None  # 最后一次检测到告警的时间
# RTSP推流配置（RK3576 ARM FFmpeg适配）
RTSP_ENABLE = True
RTSP_BASE_URL = "rtsp://192.168.1.5:8554/camera_"
RTSP_FPS = 25  # # 推流帧率（匹配检测帧率）
RTSP_RESOLUTION = (640, 480)  # 降低分辨率，提升推流稳定性
# 存储各摄像头的FFmpeg推流进程
rtsp_processes = {}
# FFmpeg推流参数（ARM架构优化，移除Windows参数）
FFMPEG_PUSH_PARAMS = [
    'ffmpeg', '-y',
    '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'bgr24',
    '-s', f"{RTSP_RESOLUTION[0]}x{RTSP_RESOLUTION[1]}", '-r', str(RTSP_FPS), '-i', '-',
    '-c:v', 'libx264', '-preset', 'fast', '-tune', 'zerolatency',  # ARM上fast比ultrafast更稳定
    '-b:v', '500k', '-f', 'rtsp', '-rtsp_transport', 'tcp',
    '-bufsize', '512k', '-timeout', '5000000',
    '-hide_banner', '-loglevel', 'error'  # 减少日志输出，降低资源占用
]

# ======================== 模型配置（RK3576 ARM适配） ========================
# RK3576模型路径（Linux绝对路径，避免相对路径问题）
CLS_MODEL_PATH = "/home/rk3576/models/yolov11n-cls/1-best.pt"
DET_DAY_MODEL_PATH = "/home/rk3576/models/yolov11n-day/5-best.pt"
DET_NIGHT_MODEL_PATH = "/home/rk3576/models/yolov11n-night/2-best.pt"

# RK3576摄像头设备路径（V4L2标准路径）
VIDEO_PATHS = [
    "/dev/video0",  # 主摄像头
    # "/dev/video1",  # 第二路摄像头（按需启用）
]

# 模型输入尺寸（RK3576算力优化，适当减小）
CLS_INPUT_SIZE = 256
DET_INPUT_SIZE = 640

# 告警阈值（保持业务逻辑不变）
CONF_THRESH = 0.4
SHIP_ALARM_THRESH = 5000
LIGHT_ALARM_THRESH = 1200
SHIP_CRITICAL_ALARM_THRESH = 8000
LIGHT_CRITICAL_ALARM_THRESH = 2000

# 工作模式（RK3576续航优化）
DEFAULT_WORK_DURATION = 120
continuous_mode = False
# 连续无告警帧数配置（用于判断恢复工作状态的帧数阈值和计数器）
NO_ALARM_FRAME_THRESH = 3   # 连续3帧无告警才恢复工作状态
no_alarm_frame_count = 0  # 连续无告警帧数计数器
# 全局计时起点
start_work_time = time.time()

# 绘制配置（RK3576中文字体适配）
FONT = cv2.FONT_HERSHEY_SIMPLEX
# RK3576 Linux中文字体路径（预装文泉驿微米黑）
CHINESE_FONT_PATH = None
font_candidates = [
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",  # RK3576主流中文字体
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
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

# ======================== 稳像相关配置（RK3576性能优化） ========================
ENABLE_STABILIZATION = True  # 稳像功能总开关
STAB_SMOOTHING_RADIUS = 30   # 平滑窗口半径（越大越稳，实时场景建议10-30）
STAB_CROP_RATIO = 0.95      # 裁剪比例（实时场景建议0.9-0.95，减少黑边）
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ======================== PaddleOCR V5 配置（RK3576 ARM适配） ========================
ENABLE_OCR = True    # OCR功能总开关
OCR_USE_GPU = False  # RK3576无NVIDIA GPU，强制CPU
OCR_MODEL_VERSION = "v5"
OCR_LANG = "ch"
OCR_CONF_THRESH = 0.5  # OCR置信度阈值（过滤低置信度结果）

# ROI图片保存配置（RK3576路径适配）
SAVE_OCR_ROI = False
ROI_SAVE_ROOT = "/mnt/ocr_roi_images"
os.makedirs(ROI_SAVE_ROOT, exist_ok=True)
ROI_SAVE_FORMAT = "jpg"
ROI_QUALITY = 95  # JPG保存质量（0-100）

# ======================== 录制相关核心函数（保持逻辑，适配路径） ========================
def init_recording_dirs():
    """初始化录制目录（RK3576权限适配）"""
    os.makedirs(RECORD_ROOT, exist_ok=True)
    for idx in range(len(VIDEO_PATHS)):
        camera_dir = os.path.join(RECORD_ROOT, f"camera_{idx+1}")
        os.makedirs(camera_dir, exist_ok=True)
        # 赋予写入权限
        os.chmod(camera_dir, 0o755)
    print(f"✅ 录制目录初始化完成：{RECORD_ROOT}")

def create_alarm_record_dir():
    """创建本次告警的录制目录"""
    global RECORD_START_TIME
    RECORD_START_TIME = datetime.now()
    time_str = RECORD_START_TIME.strftime("%Y%m%d_%H%M%S")
    alarm_root = os.path.join(RECORD_ROOT, f"alarm_{time_str}")
    os.makedirs(alarm_root, exist_ok=True)
    os.chmod(alarm_root, 0o755)
    for idx in range(len(VIDEO_PATHS)):
        camera_alarm_dir = os.path.join(alarm_root, f"camera_{idx+1}")
        os.makedirs(camera_alarm_dir, exist_ok=True)
        os.chmod(camera_alarm_dir, 0o755)
    return alarm_root

def save_frame_for_recording(frame, camera_id, alarm_root):
    """保存单帧到录制目录"""
    frame_idx = len(RECORD_FRAMES[camera_id]) + 1
    frame_filename = f"frame_{frame_idx:05d}.{FRAME_SAVE_FORMAT}"
    frame_path = os.path.join(alarm_root, f"camera_{camera_id}", frame_filename)
    # RK3576 JPG压缩优化
    cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    RECORD_FRAMES[camera_id].append(frame_path)
    return frame_path

def check_recording_status(has_alarm):
    """检查并更新录制状态"""
    global ALARM_TRIGGERED, LAST_ALARM_TIME
    if has_alarm:
        ALARM_TRIGGERED = True
        LAST_ALARM_TIME = time.time()
        if not RECORD_START_TIME:
            create_alarm_record_dir()
    else:
        if ALARM_TRIGGERED and LAST_ALARM_TIME:
            elapsed_since_last_alarm = time.time() - LAST_ALARM_TIME
            if elapsed_since_last_alarm > ALARM_STOP_DELAY:
                ALARM_TRIGGERED = False
                print(f"⚠️ 告警消失{ALARM_STOP_DELAY}秒，停止录制")

def compose_video_from_frames():
    """将保存的帧合成为视频（RK3576性能优化）"""
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
        # RK3576 VideoWriter优化
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
        # 赋予读取权限
        os.chmod(video_path, 0o644)
        print(f"✅ 摄像头{camera_id}视频合成完成：{video_path}")

# ======================== RTSP推流相关函数（RK3576 ARM适配） ========================
def init_rtsp_processes():
    """初始化各摄像头的FFmpeg推流进程（移除Windows特有参数）"""
    global rtsp_processes, RTSP_ENABLE
    # 检查RK3576上的FFmpeg
    try:
        ffmpeg_out = subprocess.check_output(["ffmpeg", "-version"], stderr=subprocess.STDOUT).decode()
        if "ffmpeg version" not in ffmpeg_out:
            print("❌ FFmpeg未安装，RTSP推流功能禁用")
            RTSP_ENABLE = False
            return
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"❌ 检查FFmpeg失败：{str(e)}，RTSP推流功能禁用")
        RTSP_ENABLE = False
        return
    # 为每个摄像头初始化推流进程
    for idx in range(len(VIDEO_PATHS)):
        camera_id = idx + 1
        rtsp_url = f"{RTSP_BASE_URL}{camera_id}"
        ffmpeg_cmd = FFMPEG_PUSH_PARAMS + [rtsp_url]
        print(f"📌 摄像头{camera_id} FFmpeg推流命令：{' '.join(ffmpeg_cmd)}")
        try:
            # 移除CREATE_NO_WINDOW（Windows特有）
            ffmpeg_proc = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            rtsp_processes[camera_id] = ffmpeg_proc
            print(f"✅ 摄像头{camera_id} RTSP推流进程初始化成功：{rtsp_url}")
        except Exception as e:
            print(f"❌ 摄像头{camera_id} RTSP推流进程启动失败：{str(e)}")
            rtsp_processes[camera_id] = None

def send_frame_to_rtsp(frame, camera_id):
    """发送单帧到RTSP流（RK3576稳定性优化）"""
    if not RTSP_ENABLE or camera_id not in rtsp_processes:
        return
    ffmpeg_proc = rtsp_processes[camera_id]
    if ffmpeg_proc is None or ffmpeg_proc.poll() is not None:
        print(f"⚠️ 摄像头{camera_id}推流进程异常，尝试重启...")
        rtsp_url = f"{RTSP_BASE_URL}{camera_id}"
        ffmpeg_cmd = FFMPEG_PUSH_PARAMS + [rtsp_url]
        try:
            ffmpeg_proc = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            rtsp_processes[camera_id] = ffmpeg_proc
            print(f"✅ 摄像头{camera_id}推流进程重启成功")
        except Exception as e:
            print(f"❌ 摄像头{camera_id}推流进程重启失败：{str(e)}")
            return
    try:
        frame_resized = cv2.resize(frame, RTSP_RESOLUTION)
        ffmpeg_proc.stdin.write(frame_resized.tobytes())
    except BrokenPipeError:
        print(f"⚠️ 摄像头{camera_id}推流管道断开，跳过当前帧")
    except Exception as e:
        print(f"⚠️ 摄像头{camera_id}帧发送失败：{str(e)}")

def release_rtsp_processes():
    """释放RTSP推流进程"""
    global rtsp_processes
    for camera_id, proc in rtsp_processes.items():
        if proc is not None and proc.poll() is None:
            proc.stdin.close()
            proc.terminate()
            proc.wait(timeout=5)  # 增加超时，避免RK3576进程僵死
    rtsp_processes.clear()
    print("✅ RTSP推流进程已释放")

# ======================== 稳像类（RK3576性能优化） ========================
class RealTimeVideoStabilizer:
    """实时逐帧视频稳像类（RK3576轻量化）"""
    def __init__(self, smoothing_radius=15, crop_ratio=0.9):
        self.smoothing_radius = smoothing_radius
        self.crop_ratio = crop_ratio
        self.prev_frame = None
        self.prev_gray = None
        self.prev_points = None
        self.trajectory = []
        self.smoothed_trajectory = []
        self.cumulative_transform = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        self.initialized = False

    def extract_features(self, frame, max_corners=100):  # 减少特征点数量
        """提取图像特征点（RK3576优化）"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        feature_params = dict(
            maxCorners=max_corners,
            qualityLevel=0.02,  # 提高质量阈值，减少特征点
            minDistance=20,
            blockSize=7
        )
        corners = cv2.goodFeaturesToTrack(gray, mask=None,** feature_params)
        return corners

    def calculate_optical_flow(self, prev_gray, curr_gray, prev_points):
        """Lucas-Kanade光流法（保持不变）"""
        lk_params = dict(
            winSize=(10, 10),  # 减小窗口，降低计算量
            maxLevel=1,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 8, 0.03)
        )
        curr_points, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_points, None, **lk_params
        )
        return curr_points, status

    def estimate_rigid_transform(self, prev_points, curr_points):
        """估计刚性变换矩阵（保持不变）"""
        transform_matrix, inliers = cv2.estimateAffinePartial2D(
            prev_points, curr_points, method=cv2.RANSAC, ransacReprojThreshold=3.0
        )
        if transform_matrix is None:
            transform_matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        return transform_matrix

    def decompose_transform(self, transform):
        """分解变换矩阵（保持不变）"""
        dx = transform[0, 2]
        dy = transform[1, 2]
        da = np.arctan2(transform[1, 0], transform[0, 0])
        return dx, dy, da

    def smooth_trajectory(self, trajectory):
        """实时轨迹平滑（RK3576简化）"""
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
        """应用变换矩阵（保持不变）"""
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
        return cropped

    def process_frame(self, curr_frame):
        """实时处理单帧（RK3576性能优化）"""
        if not self.initialized:
            self.prev_frame = curr_frame.copy()
            self.prev_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            self.prev_points = self.extract_features(self.prev_frame)
            self.initialized = True
            return curr_frame
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        if self.prev_points is None or len(self.prev_points) < 4:
            self.prev_points = self.extract_features(self.prev_frame)
            if self.prev_points is None or len(self.prev_points) < 4:
                self.prev_frame = curr_frame.copy()
                self.prev_gray = curr_gray
                return curr_frame
        curr_points, status = self.calculate_optical_flow(
            self.prev_gray, curr_gray, self.prev_points
        )
        valid_prev = self.prev_points[status == 1]
        valid_curr = curr_points[status == 1]
        if len(valid_prev) < 4:
            self.prev_points = self.extract_features(self.prev_frame)
            self.prev_frame = curr_frame.copy()
            self.prev_gray = curr_gray
            return curr_frame
        transform = self.estimate_rigid_transform(valid_prev, valid_curr)
        self.cumulative_transform = np.vstack([
            self.cumulative_transform @ np.vstack([transform, [0, 0, 1]]),
        ])[:2, :]
        dx, dy, da = self.decompose_transform(self.cumulative_transform)
        self.trajectory.append([dx, dy, da])
        self.smoothed_trajectory = self.smooth_trajectory(self.trajectory)
        if len(self.smoothed_trajectory) > 0 and len(self.trajectory) > 0:
            diff_dx = self.smoothed_trajectory[-1][0] - self.trajectory[-1][0]
            diff_dy = self.smoothed_trajectory[-1][1] - self.trajectory[-1][1]
            diff_da = self.smoothed_trajectory[-1][2] - self.trajectory[-1][2]
            comp_transform = np.array([
                [np.cos(diff_da), -np.sin(diff_da), diff_dx],
                [np.sin(diff_da), np.cos(diff_da), diff_dy]
            ], dtype=np.float32)
            stabilized_frame = self.apply_transform(curr_frame, comp_transform)
        else:
            stabilized_frame = curr_frame.copy()
        self.prev_frame = curr_frame.copy()
        self.prev_gray = curr_gray
        self.prev_points = self.extract_features(curr_frame)
        return stabilized_frame

    def reset(self):
        """重置稳像器状态"""
        self.prev_frame = None
        self.prev_gray = None
        self.prev_points = None
        self.trajectory = []
        self.smoothed_trajectory = []
        self.cumulative_transform = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        self.initialized = False

# ======================== ROI保存辅助函数（保持不变）========================
def create_camera_dir(root_dir, camera_id):
    camera_dir = os.path.join(root_dir, f"camera_{camera_id}")
    os.makedirs(camera_dir, exist_ok=True)
    os.chmod(camera_dir, 0o755)
    return camera_dir

def generate_roi_filename(camera_id, x1, y1, x2, y2):
    timestamp = int(time.time() * 1000)
    filename = f"roi_camera{camera_id}_{timestamp}_x{x1}y{y1}x{x2}y{y2}.{FRAME_SAVE_FORMAT}"
    return filename

# ======================== PaddleOCR V5 初始化与处理函数（RK3576 ARM适配） ========================
def init_paddle_ocr_v5():
    """初始化PaddleOCR V5（RK3576 ARM CPU适配）"""
    try:
        from paddleocr import PaddleOCR
        # RK3576 ARM轻量化配置
        ocr = PaddleOCR(
            text_detection_model_dir="/home/rk3576/models/paddleocr_v5/PP-OCRv5_server_det",
            text_recognition_model_dir="/home/rk3576/models/paddleocr_v5/PP-OCRv5_server_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            ocr_version="PP-OCRv5",
            device="cpu",  # 强制CPU，RK3576无NVIDIA GPU
            cpu_threads=4  # 利用RK3576多核优势
        )
        print("✅ PaddleOCR V5 初始化成功（RK3576 ARM适配）")
        return ocr
    except ImportError:
        print("❌ 未安装PaddleOCR，请执行：pip3 install paddleocr>=2.7.0 paddlepaddle==2.5.2")
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
    """判断文本是否为有效文本（保持不变）"""
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
    """绘制支持中文的文本（RK3576适配）"""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    try:
        if CHINESE_FONT_PATH and PIL_AVAILABLE:
            font_height = max(10, int(font_size * FONT_MULTIPALE))
            font = ImageFont.truetype(CHINESE_FONT_PATH, font_height)
        else:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    pil_color = (color[2], color[1], color[0])
    draw.text(pos, text, font=font, fill=pil_color, stroke_width=0.3)
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img_bgr

def get_text_size(text, font_size=0.8):
    """计算文本尺寸（支持中文）"""
    if not PIL_AVAILABLE or not CHINESE_FONT_PATH:
        (w, h), _ = cv2.getTextSize(text, FONT, font_size, FONT_THICKNESS)
        return (w, h)
    try:
        font_height = max(10, int(font_size * FONT_MULTIPALE))
        font = ImageFont.truetype(CHINESE_FONT_PATH, font_height)
        dummy_img = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(dummy_img)
        try:
            left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        except AttributeError:
            left, top, right, bottom = font.getbbox(text)
        w = right - left
        h = bottom - top
        return (w + 1, h + 1)
    except:
        (w, h), _ = cv2.getTextSize(text, FONT, font_size, FONT_THICKNESS)
        return (w, h)

def process_ship_ocr(frame, post_results, weather_label, camera_id):
    """OCR处理（保持逻辑，适配RK3576路径）"""
    ocr_results = []
    if not ENABLE_OCR or ocr_engine is None or weather_label in ["night", "夜晚"]:
        return ocr_results, frame
    alarm_ship_rois = []
    for res in post_results:
        if len(res) < 9:
            continue
        x1, y1, x2, y2, conf, cls, area, is_alarm, is_critical_alarm = res
        if is_alarm:
            alarm_ship_rois.append(res)
    if len(alarm_ship_rois) == 0:
        return ocr_results, frame
    rec_texts = []
    rec_scores = []
    rec_polys = []
    for roi_res in alarm_ship_rois:
        if len(roi_res) < 9:
            continue
        x1, y1, x2, y2, conf, cls, area, is_alarm, is_critical_alarm = roi_res
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
                os.chmod(save_path, 0o644)
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
                continue
            if not is_valid_text(text):
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
                    cv2.rectangle(frame, (text_x1, text_y1), 
                                (text_x2, text_y2), OCR_BOX_COLOR, BOX_THICKNESS//2)
            except Exception as e:
                print(f"⚠️ OCR绘制异常：{str(e)}")
    return ocr_results, frame

# ======================== 绘制圆角矩形（保持不变） ========================
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

# ======================== 加载模型（RK3576 ARM适配） ========================
# 移除CUDA，使用CPU（RK3576无NVIDIA GPU）
cls_model = YOLO(CLS_MODEL_PATH)
cls_model.to('cpu')

det_day_model = YOLO(DET_DAY_MODEL_PATH)
det_day_model.to('cpu')

det_night_model = YOLO(DET_NIGHT_MODEL_PATH)
det_night_model.to('cpu')

# ======================== 核心函数（RK3576性能优化） ========================
def get_weather_class(frame):
    """场景分类（移除half=True，CPU不支持）"""
    results = cls_model.predict(
        source=frame,          
        imgsz=CLS_INPUT_SIZE,  
        device='cpu',          # RK3576使用CPU
        half=False,            # CPU不支持半精度
        verbose=False,         
        stream=False,          
        batch=1,               
        workers=0              
    )
    result = results[0]
    top1_idx = result.probs.top1
    top1_conf = result.probs.top1conf.item()
    cls_label = result.names[top1_idx]  
    print(f"分类结果：{cls_label} | 置信度：{top1_conf:.3f}")
    return cls_label, top1_conf

def detect_target(frame, weather_cls):
    """目标检测（移除half=True，CPU不支持）"""
    det_model = det_night_model if weather_cls == "night" else det_day_model
    results = det_model.predict(
        source=frame,
        imgsz=DET_INPUT_SIZE,
        device='cpu',
        half=False,  # CPU不支持半精度
        verbose=False,
        stream=False,
        batch=1,
        workers=0
    )
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
    """后处理（保持逻辑不变）"""
    h, w = orig_frame_shape
    post_results = []
    has_alarm = False
    critical_alarm_signal = False
    for (x1n, y1n, x2n, y2n, conf, cls) in det_results:
        if conf < CONF_THRESH:
            continue
        x1 = int(x1n * w)
        y1 = int(y1n * h)
        x2 = int(x2n * w)
        y2 = int(y2n * h)
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        area = width * height
        if weather_cls == "night":
            is_alarm = area >= LIGHT_ALARM_THRESH
            is_critical_alarm = area >= LIGHT_CRITICAL_ALARM_THRESH
        else:
            is_alarm = area >= SHIP_ALARM_THRESH
            is_critical_alarm = area >= SHIP_CRITICAL_ALARM_THRESH
        if is_alarm:
            has_alarm = True
        if is_critical_alarm:
            critical_alarm_signal = True
        post_results.append((x1, y1, x2, y2, conf, cls, area, is_alarm, is_critical_alarm))
    if critical_alarm_signal:
        print(f"⚠️ 触发严重告警信号！当前天气类型：{weather_cls}")
    return post_results, has_alarm

def draw_custom_boxes(frame, post_results, weather_info, model_names):
    """绘制函数（RK3576性能优化）"""
    h, w = frame.shape[:2]
    weather_label, weather_conf = weather_info
    # 绘制场景信息
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
    # 绘制检测框
    for (x1, y1, x2, y2, conf, cls, area, is_alarm, is_critical_alarm) in post_results:
        box_color = ALARM_COLOR if is_alarm else NORMAL_COLOR
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, BOX_THICKNESS)
        cls_name = model_names.get(int(cls), "unknown")
        alarm_text = "CRITICAL_ALARM" if is_critical_alarm else "ALARM" if is_alarm else "NORMAL"
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
    """检查工作模式（保持不变）"""
    global continuous_mode, no_alarm_frame_count, NO_ALARM_FRAME_THRESH
    if has_alarm:
        print("发现告警目标！！！")
        no_alarm_frame_count = 0
        continuous_mode = True
    else:
        no_alarm_frame_count += 1
        print(f"连续无告警帧数：{no_alarm_frame_count}/{NO_ALARM_FRAME_THRESH}")
        if no_alarm_frame_count >= NO_ALARM_FRAME_THRESH:
            continuous_mode = False
            print("连续3帧无告警目标，恢复工作状态")
    return continuous_mode

def check_work_duration():
    """检查工作时长（保持不变）"""
    if continuous_mode:
        return True
    elapsed_time = time.time() - start_work_time
    return elapsed_time < DEFAULT_WORK_DURATION

# ======================== 主程序（RK3576适配） ========================
if __name__ == "__main__":
    # 初始化录制目录
    init_recording_dirs()
    
    # 安装PIL（RK3576使用pip3）
    if not PIL_AVAILABLE:
        print("📦 正在安装PIL库...")
        os.system("pip3 install pillow -q")
        try:
            from PIL import Image, ImageDraw, ImageFont
            PIL_AVAILABLE = True
            print("✅ PIL库安装成功")
        except:
            print("❌ PIL库安装失败，中文显示可能异常")
    
    # 初始化视频捕获（RK3576 V4L2适配）
    caps = []
    stabilizers = []
    for path in VIDEO_PATHS:
        cap = cv2.VideoCapture(path, cv2.CAP_V4L2)  # 强制V4L2后端，适配RK3576
        if not cap.isOpened():
            print(f"警告：无法打开视频 {path}")
            caps.append(None)
            stabilizers.append(None)
        else:
            # RK3576摄像头参数优化
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 15)  # 降低帧率，提升稳定性
            caps.append(cap)
            if ENABLE_STABILIZATION:
                stabilizer = RealTimeVideoStabilizer(
                    smoothing_radius=STAB_SMOOTHING_RADIUS,
                    crop_ratio=STAB_CROP_RATIO
                )
                stabilizers.append(stabilizer)
            else:
                stabilizers.append(None)
    
    # 初始化RTSP推流进程
    if RTSP_ENABLE:
        init_rtsp_processes()
    
    # 主循环（RK3576异常处理优化）
    try:
        while True:
            if not check_work_duration():
                print(f"工作时长已达{DEFAULT_WORK_DURATION}秒，退出程序")
                break
            
            global_has_alarm = False
            
            for idx, (cap, stabilizer) in enumerate(zip(caps, stabilizers)):
                camera_id = idx + 1
                if cap is None:
                    continue
                
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    if stabilizer is not None:
                        stabilizer.reset()
                    continue
                
                # 稳像处理
                stabilized_frame = frame
                if ENABLE_STABILIZATION and stabilizer is not None:
                    stabilized_frame = stabilizer.process_frame(frame)
                
                # 场景分类
                weather_label, weather_conf = get_weather_class(stabilized_frame)
                weather_info = (weather_label, weather_conf)
                
                # 目标检测
                det_results = detect_target(stabilized_frame, weather_label)
                
                # 后处理
                orig_h, orig_w = stabilized_frame.shape[:2]
                post_results, has_alarm = postprocess_detections(det_results, (orig_h, orig_w), weather_label)
                
                if has_alarm:
                    global_has_alarm = True
                
                # 更新工作模式
                check_work_mode(has_alarm)
                
                # OCR处理
                ocr_results, stabilized_frame = process_ship_ocr(
                    stabilized_frame.copy(), post_results, weather_label, camera_id
                )
                if ocr_results:
                    print(f"❗相机{camera_id} OCR识别结果：{[item['text'] for item in ocr_results]}")
                
                # 获取类别名
                model_names = det_night_model.names if weather_label == "night" else det_day_model.names
                
                # 绘制
                frame_drawn = draw_custom_boxes(stabilized_frame, post_results, weather_info, model_names)
                
                # RTSP推流
                send_frame_to_rtsp(frame_drawn, camera_id)
                
                # 录制逻辑
                check_recording_status(global_has_alarm)
                if ALARM_TRIGGERED and RECORD_START_TIME:
                    alarm_root = os.path.join(RECORD_ROOT, f"alarm_{RECORD_START_TIME.strftime('%Y%m%d_%H%M%S')}")
                    save_frame_for_recording(frame_drawn, camera_id, alarm_root)
                
                # 显示（RK3576小屏适配）
                cv2.imshow(f"Camera {camera_id} (RK3576)", cv2.resize(frame_drawn, (960, 540)))  # 减小显示尺寸
            
            # RK3576按键响应优化
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("用户手动退出程序")
                break
    
    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"程序异常：{str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # 合成视频
        compose_video_from_frames()
        # 释放RTSP进程
        release_rtsp_processes()
        # 释放视频资源
        for cap in caps:
            if cap is not None:
                cap.release()
        cv2.destroyAllWindows()
        print("✅ 程序正常退出，资源已释放（RK3576适配版）")