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
import subprocess  # 确保导入subprocess

# ======================== PIL导入（可视化兼容处理） ========================
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️ 未安装PIL库，中文显示可能异常，请执行：pip install pillow")

# ======================== 录制与推流核心配置 ========================
# 录制根目录
RECORD_ROOT = "./alarm_recordings"
# 帧保存格式
FRAME_SAVE_FORMAT = "jpg"
# 视频合成参数
VIDEO_FPS = 25  # 视频帧率
VIDEO_CODEC = cv2.VideoWriter_fourcc(*'mp4v')  # MP4编码
VIDEO_EXT = "mp4"
# 告警录制配置
ALARM_TRIGGERED = False  # 全局告警触发标记
RECORD_FRAMES = defaultdict(list)  # 按摄像头ID存储录制的帧路径 {camera_id: [frame_path1, frame_path2,...]}
RECORD_START_TIME = None  # 录制开始时间
ALARM_STOP_DELAY = 2  # 目标消失后延迟2秒停止录制（秒）
LAST_ALARM_TIME = None  # 最后一次检测到告警的时间
# RTSP推流配置（测试验证的稳定方案）
RTSP_ENABLE = True
RTSP_BASE_URL = "rtsp://192.168.1.5:8554/camera_"  # RTSP服务器地址（需提前启动rtsp-simple-server）
RTSP_FPS = 25  # 推流帧率（匹配检测帧率）
RTSP_RESOLUTION = (640, 480)  # 推流分辨率（可根据实际调整）
# 存储各摄像头的FFmpeg推流进程
rtsp_processes = {}  
# FFmpeg推流核心参数（测试验证的稳定配置）
FFMPEG_PUSH_PARAMS = [
    'ffmpeg', '-y',
    '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'bgr24',
    '-s', f"{RTSP_RESOLUTION[0]}x{RTSP_RESOLUTION[1]}", '-r', str(RTSP_FPS), '-i', '-',
    '-c:v', 'libx264', '-preset', 'ultrafast', '-tune', 'zerolatency',
    '-b:v', '800k', '-f', 'rtsp', '-rtsp_transport', 'tcp',
    '-bufsize', '1024k', '-timeout', '5000000'
]

# ======================== 原有配置（保留） ========================
# 统一模型路径
CLS_MODEL_PATH = "./models/yolov11n-cls/1-best.pt"  # 场景分类模型
DET_DAY_MODEL_PATH = "./models/yolov11n-day/5-best.pt"  # 白天船舶检测模型
DET_NIGHT_MODEL_PATH = "./models/yolov11n-night/2-best.pt"  # 夜晚灯光检测模型

# 4路视频路径（建议扩展为多摄像头）
VIDEO_PATHS = [
    0,  
]

# 模型输入尺寸
CLS_INPUT_SIZE = 256  
DET_INPUT_SIZE = 640

# 告警阈值
CONF_THRESH = 0.4
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

# ======================== 录制相关核心函数 ========================
def init_recording_dirs():
    """初始化录制目录结构"""
    # 创建根目录
    os.makedirs(RECORD_ROOT, exist_ok=True)
    # 为每个摄像头创建子目录
    for idx in range(len(VIDEO_PATHS)):
        camera_dir = os.path.join(RECORD_ROOT, f"camera_{idx+1}")
        os.makedirs(camera_dir, exist_ok=True)
    print(f"✅ 录制目录初始化完成：{RECORD_ROOT}")

def create_alarm_record_dir():
    """创建本次告警的录制目录（按时间戳命名）"""
    global RECORD_START_TIME
    RECORD_START_TIME = datetime.now()
    time_str = RECORD_START_TIME.strftime("%Y%m%d_%H%M%S")
    alarm_root = os.path.join(RECORD_ROOT, f"alarm_{time_str}")
    os.makedirs(alarm_root, exist_ok=True)
    # 为每个摄像头创建告警子目录
    for idx in range(len(VIDEO_PATHS)):
        camera_alarm_dir = os.path.join(alarm_root, f"camera_{idx+1}")
        os.makedirs(camera_alarm_dir, exist_ok=True)
    return alarm_root

def save_frame_for_recording(frame, camera_id, alarm_root):
    """保存单帧到录制目录"""
    # 生成帧文件名（按顺序编号）
    frame_idx = len(RECORD_FRAMES[camera_id]) + 1
    frame_filename = f"frame_{frame_idx:05d}.{FRAME_SAVE_FORMAT}"
    frame_path = os.path.join(alarm_root, f"camera_{camera_id}", frame_filename)
    # 保存帧
    cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    # 添加到帧列表
    RECORD_FRAMES[camera_id].append(frame_path)
    return frame_path

def check_recording_status(has_alarm):
    """检查并更新录制状态"""
    global ALARM_TRIGGERED, LAST_ALARM_TIME
    
    # 如果检测到告警，更新状态
    if has_alarm:
        ALARM_TRIGGERED = True
        LAST_ALARM_TIME = time.time()
        if not RECORD_START_TIME:
            create_alarm_record_dir()  # 首次触发告警时创建目录
    else:
        # 无告警时，检查是否需要停止录制
        if ALARM_TRIGGERED and LAST_ALARM_TIME:
            elapsed_since_last_alarm = time.time() - LAST_ALARM_TIME
            if elapsed_since_last_alarm > ALARM_STOP_DELAY:
                ALARM_TRIGGERED = False
                print(f"⚠️ 告警消失{ALARM_STOP_DELAY}秒，停止录制")

def compose_video_from_frames():
    """将保存的帧合成为视频"""
    print("\n📹 开始合成视频...")
    for camera_id, frame_paths in RECORD_FRAMES.items():
        if not frame_paths:
            continue
        
        # 获取帧尺寸（从第一帧读取）
        sample_frame = cv2.imread(frame_paths[0])
        if sample_frame is None:
            print(f"❌ 摄像头{camera_id}无有效帧，跳过合成")
            continue
        h, w = sample_frame.shape[:2]
        
        # 生成视频保存路径
        time_str = RECORD_START_TIME.strftime("%Y%m%d_%H%M%S") if RECORD_START_TIME else "unknown"
        video_path = os.path.join(RECORD_ROOT, f"camera_{camera_id}_alarm_{time_str}.{VIDEO_EXT}")
        
        # 创建视频写入器
        video_writer = cv2.VideoWriter(
            video_path,
            VIDEO_CODEC,
            VIDEO_FPS,
            (w, h)
        )
        
        if not video_writer.isOpened():
            print(f"❌ 摄像头{camera_id}视频写入器初始化失败")
            continue
        
        # 按顺序写入帧
        for frame_path in sorted(frame_paths):
            frame = cv2.imread(frame_path)
            if frame is not None:
                video_writer.write(frame)
        
        # 释放资源
        video_writer.release()
        print(f"✅ 摄像头{camera_id}视频合成完成：{video_path}")

# ======================== RTSP推流相关函数（替换为测试验证的版本） ========================
def init_rtsp_processes():
    """初始化各摄像头的FFmpeg推流进程（测试验证的稳定方式）"""
    global rtsp_processes, RTSP_ENABLE
    # 检查FFmpeg是否可用
    try:
        ffmpeg_out = os.popen("ffmpeg -version").read()
        if "ffmpeg version" not in ffmpeg_out:
            print("❌ FFmpeg未配置到环境变量，RTSP推流功能禁用")
            RTSP_ENABLE = False
            return
    except Exception as e:
        print(f"❌ 检查FFmpeg失败：{str(e)}，RTSP推流功能禁用")
        RTSP_ENABLE = False
        return
    
    # 为每个摄像头初始化推流进程
    for idx in range(len(VIDEO_PATHS)):
        camera_id = idx + 1
        rtsp_url = f"{RTSP_BASE_URL}{camera_id}"
        # 拼接完整的FFmpeg命令
        ffmpeg_cmd = FFMPEG_PUSH_PARAMS + [rtsp_url]
        print(f"📌 摄像头{camera_id} FFmpeg推流命令：{' '.join(ffmpeg_cmd)}")
        
        try:
            # 启动FFmpeg推流进程
            ffmpeg_proc = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            rtsp_processes[camera_id] = ffmpeg_proc
            print(f"✅ 摄像头{camera_id} RTSP推流进程初始化成功：{rtsp_url}")
        except Exception as e:
            print(f"❌ 摄像头{camera_id} RTSP推流进程启动失败：{str(e)}")
            rtsp_processes[camera_id] = None

def send_frame_to_rtsp(frame, camera_id):
    """发送单帧到RTSP流（测试验证的稳定方式）"""
    if not RTSP_ENABLE or camera_id not in rtsp_processes:
        return
    
    ffmpeg_proc = rtsp_processes[camera_id]
    if ffmpeg_proc is None or ffmpeg_proc.poll() is not None:
        print(f"⚠️ 摄像头{camera_id}推流进程异常，尝试重启...")
        # 重启推流进程
        rtsp_url = f"{RTSP_BASE_URL}{camera_id}"
        ffmpeg_cmd = FFMPEG_PUSH_PARAMS + [rtsp_url]
        try:
            ffmpeg_proc = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            rtsp_processes[camera_id] = ffmpeg_proc
            print(f"✅ 摄像头{camera_id}推流进程重启成功")
        except Exception as e:
            print(f"❌ 摄像头{camera_id}推流进程重启失败：{str(e)}")
            return
    
    # 调整帧尺寸并发送
    try:
        frame_resized = cv2.resize(frame, RTSP_RESOLUTION)
        ffmpeg_proc.stdin.write(frame_resized.tobytes())
    except Exception as e:
        print(f"⚠️ 摄像头{camera_id}帧发送失败：{str(e)}")

def release_rtsp_processes():
    """释放RTSP推流进程"""
    global rtsp_processes
    for camera_id, proc in rtsp_processes.items():
        if proc is not None and proc.poll() is None:
            proc.stdin.close()
            proc.terminate()
            proc.wait()
    rtsp_processes.clear()
    print("✅ RTSP推流进程已释放")

# ======================== 稳像类（保留） ========================
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

# ======================== ROI保存辅助函数（保留）========================
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
    filename = f"roi_camera{camera_id}_{timestamp}_x{x1}y{y1}x{x2}y{y2}.{FRAME_SAVE_FORMAT}"
    return filename

# ======================== PaddleOCR V5 初始化与处理函数（保留） ========================
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
            device="gpu" if OCR_USE_GPU else "cpu"
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

# ======================== 绘制圆角矩形（保留） ========================
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

# ======================== 加载模型（保留） ========================
# 加载分类模型
cls_model = YOLO(CLS_MODEL_PATH)
cls_model.to('cuda')

# 加载检测模型
det_day_model = YOLO(DET_DAY_MODEL_PATH)
det_night_model = YOLO(DET_NIGHT_MODEL_PATH)
det_day_model.to('cuda')
det_night_model.to('cuda')

# ======================== 核心函数（保留） ========================
def get_weather_class(frame):
    """场景分类：改用YOLO predict方法，自动预处理"""
    results = cls_model.predict(
        source=frame,          
        imgsz=CLS_INPUT_SIZE,  
        device='cuda',            
        half=True,             
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
    """目标检测：改用predict方法，自动预处理"""
    det_model = det_night_model if weather_cls == "night" else det_day_model
    
    results = det_model.predict(
        source=frame,
        imgsz=DET_INPUT_SIZE,
        device='cuda',
        half=True,
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
    h, w = orig_frame_shape
    post_results = []
    has_alarm = False

    for (x1n, y1n, x2n, y2n, conf, cls) in det_results:
        # 低置信度直接跳过
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

# ======================== 主程序（集成测试通过的推流逻辑） ========================
if __name__ == "__main__":
    # 初始化录制目录
    init_recording_dirs()
    
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
    
    # 初始化RTSP推流进程（替换原cv2.VideoWriter方式）
    if RTSP_ENABLE:
        init_rtsp_processes()
    
    # 主循环
    try:
        while True:
            if not check_work_duration():
                print(f"工作时长已达{DEFAULT_WORK_DURATION}秒，退出程序")
                break
            
            # 标记是否有任意摄像头检测到告警
            global_has_alarm = False
            
            for idx, (cap, stabilizer) in enumerate(zip(caps, stabilizers)):
                camera_id = idx + 1
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
                
                # 更新全局告警状态
                if has_alarm:
                    global_has_alarm = True
                
                # 4. 更新工作模式
                check_work_mode(has_alarm)
                
                # 5. 仅对白天告警船只ROI做PaddleOCR V5识别（传入相机编号idx+1）
                ocr_results, stabilized_frame = process_ship_ocr(
                    stabilized_frame.copy(), post_results, weather_label, camera_id
                )
                # 打印OCR结果
                if ocr_results:
                    print(f"❗相机{camera_id} OCR识别结果：{[item['text'] for item in ocr_results]}")
                
                # 6. 获取类别名
                model_names = det_night_model.names if weather_label == "night" else det_day_model.names
                
                # 7. 绘制（使用优化后的圆角样式）
                frame_drawn = draw_custom_boxes(stabilized_frame, post_results, weather_info, model_names)
                
                # ========== 集成测试通过的RTSP推流逻辑 ==========
                send_frame_to_rtsp(frame_drawn, camera_id)
                
                # ========== 录制逻辑 ==========
                # 检查并更新录制状态
                check_recording_status(global_has_alarm)
                # 如果触发录制，保存当前帧
                if ALARM_TRIGGERED and RECORD_START_TIME:
                    alarm_root = os.path.join(RECORD_ROOT, f"alarm_{RECORD_START_TIME.strftime('%Y%m%d_%H%M%S')}")
                    save_frame_for_recording(frame_drawn, camera_id, alarm_root)
                
                # 显示（1344×760 匹配原视频比例，自动适配稳像后的尺寸）
                cv2.imshow(f"Camera {camera_id} (Stabilized+OCR+Record+RTSP)", cv2.resize(frame_drawn, (1344, 760)))
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("用户手动退出程序")
                break
    
    except Exception as e:
        print(f"程序异常：{str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # 合成视频
        compose_video_from_frames()
        # 释放RTSP推流进程（替换原释放推流器的函数）
        release_rtsp_processes()
        # 释放视频捕获资源
        for cap in caps:
            if cap is not None:
                cap.release()
        cv2.destroyAllWindows()
        print("✅ 程序正常退出，资源已释放")