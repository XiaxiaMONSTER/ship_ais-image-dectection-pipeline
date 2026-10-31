#!/usr/bin/env python3
"""
双码流间歇录制工具 - 参照全流程脚本风格
- 主码流 + 子码流同时录制，分开保存
- 工作模式：录制5分钟 → 休息5分钟 → 循环
- 磁盘剩余4G时自动停止
- 使用摄像头原始码率，不重新编码
"""

import cv2
import time
import os
import threading
import signal
import sys
from datetime import datetime

# ======================== 配置参数 ========================
# 摄像头RTSP地址
RTSP_MAIN = "rtsp://admin:ADMIN123@169.254.243.102:554/stream1"      # 主码流
RTSP_SUB = "rtsp://admin:ADMIN123@169.254.243.102:554/live/0/SUB"    # 子码流

# 主码流参数（3200x1800）
MAIN_WIDTH = 3200
MAIN_HEIGHT = 1800

# 子码流参数（1280x720）
SUB_WIDTH = 1280
SUB_HEIGHT = 720

# 录制参数
RECORD_DURATION = 300   # 录制时长（秒），5分钟
REST_DURATION = 300     # 休息时长（秒），5分钟

# 保存路径
SAVE_ROOT = "./recordings"

# 磁盘容量配置（剩余空间低于此值停止录制）
MIN_FREE_SPACE_GB = 4

# GStreamer配置（使用摄像头原始码率，不重新编码）
USE_GSTREAMER = True

# ======================== 工具函数 ========================
def get_disk_free_space(path):
    """获取磁盘剩余空间（GB）"""
    try:
        stat = os.statvfs(path)
        free_bytes = stat.f_bavail * stat.f_frsize
        return free_bytes / (1024 ** 3)
    except Exception as e:
        print(f"⚠️ 磁盘空间检查失败: {e}")
        return float('inf')

def build_gstreamer_pipeline(rtsp_url):
    """构建GStreamer管道 - 使用摄像头原始码率，不重新编码"""
    return (
        f'rtspsrc location="{rtsp_url}" latency=100 protocols=tcp ! '
        f'rtph265depay ! h265parse ! mppvideodec ! '
        f'videoconvert ! video/x-raw,format=BGR ! '
        f'appsink max-buffers=1 drop=true sync=false'
    )

def open_camera(rtsp_url):
    """打开摄像头（优先使用GStreamer硬解）"""
    if USE_GSTREAMER:
        try:
            pipeline = build_gstreamer_pipeline(rtsp_url)
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                print(f"✅ 使用GStreamer硬解: {rtsp_url}")
                return cap
        except Exception as e:
            print(f"⚠️ GStreamer启动失败，降级到OpenCV: {e}")
    
    cap = cv2.VideoCapture(rtsp_url)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print(f"✅ 使用OpenCV: {rtsp_url}")
    return cap

def generate_filename(stream_type, timestamp=None):
    """生成文件名"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stream_type}_{timestamp}.mp4"

# ======================== 视频录制器类 ========================
class DualStreamRecorder:
    def __init__(self):
        self.running = False
        self.recording = False
        self.main_cap = None
        self.sub_cap = None
        self.main_writer = None
        self.sub_writer = None
        self.main_file = None
        self.sub_file = None
        self.record_start_time = None
        self.cycle_start_time = None
        self.current_cycle = 0
        self.total_frames = 0
        
        # 获取摄像头原始帧率
        self.main_fps = 20
        self.sub_fps = 20
        
        # 创建保存目录
        self.main_dir = os.path.join(SAVE_ROOT, "main_stream")
        self.sub_dir = os.path.join(SAVE_ROOT, "sub_stream")
        os.makedirs(self.main_dir, exist_ok=True)
        os.makedirs(self.sub_dir, exist_ok=True)
        
        # 注册信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """处理Ctrl+C信号"""
        print("\n\n收到停止信号，正在停止录制...")
        self.stop()
        sys.exit(0)
    
    def _get_camera_fps(self, cap):
        """获取摄像头实际帧率"""
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps > 0 and fps < 60:
                return int(fps)
        except:
            pass
        return 20  # 默认20fps
    
    def _create_writer(self, filepath, width, height, fps):
        """创建视频写入器 - 使用摄像头原始帧率"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"无法创建视频文件: {filepath}")
        return writer
    
    def _record_cycle(self):
        """单次录制循环（录制5分钟）"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.main_file = os.path.join(self.main_dir, generate_filename("main", timestamp))
        self.sub_file = os.path.join(self.sub_dir, generate_filename("sub", timestamp))
        
        print(f"\n🎬 [循环 {self.current_cycle}] 开始录制...")
        print(f"  主码流: {os.path.basename(self.main_file)}")
        print(f"  子码流: {os.path.basename(self.sub_file)}")
        
        # 获取实际帧率
        self.main_fps = self._get_camera_fps(self.main_cap)
        self.sub_fps = self._get_camera_fps(self.sub_cap)
        print(f"  主码流帧率: {self.main_fps}fps, 子码流帧率: {self.sub_fps}fps")
        
        self.main_writer = self._create_writer(self.main_file, MAIN_WIDTH, MAIN_HEIGHT, self.main_fps)
        self.sub_writer = self._create_writer(self.sub_file, SUB_WIDTH, SUB_HEIGHT, self.sub_fps)
        
        self.recording = True
        self.record_start_time = time.time()
        
        frame_count = 0
        main_fail_count = 0
        sub_fail_count = 0
        
        # 记录实际写入帧数
        main_frame_count = 0
        sub_frame_count = 0
        
        while self.recording:
            # 检查磁盘空间
            free_gb = get_disk_free_space(SAVE_ROOT)
            if free_gb < MIN_FREE_SPACE_GB:
                print(f"\n⚠️ 磁盘空间不足 ({free_gb:.1f}GB < {MIN_FREE_SPACE_GB}GB)")
                print("   停止录制...")
                self.recording = False
                break
            
            # 检查录制时长
            elapsed = time.time() - self.record_start_time
            if elapsed >= RECORD_DURATION:
                print(f"\n✅ 录制完成 ({elapsed:.0f}秒)")
                break
            
            # 读取主码流
            ret_main, frame_main = self.main_cap.read()
            if not ret_main or frame_main is None:
                main_fail_count += 1
                if main_fail_count > 30:
                    print("⚠️ 主码流读取失败，尝试重连...")
                    self._reconnect_main()
                    main_fail_count = 0
                continue
            main_fail_count = 0
            
            # 读取子码流
            ret_sub, frame_sub = self.sub_cap.read()
            if not ret_sub or frame_sub is None:
                sub_fail_count += 1
                if sub_fail_count > 30:
                    print("⚠️ 子码流读取失败，尝试重连...")
                    self._reconnect_sub()
                    sub_fail_count = 0
                continue
            sub_fail_count = 0
            
            # 调整尺寸（如果摄像头输出尺寸不一致）
            if frame_main.shape[:2] != (MAIN_HEIGHT, MAIN_WIDTH):
                frame_main = cv2.resize(frame_main, (MAIN_WIDTH, MAIN_HEIGHT))
            if frame_sub.shape[:2] != (SUB_HEIGHT, SUB_WIDTH):
                frame_sub = cv2.resize(frame_sub, (SUB_WIDTH, SUB_HEIGHT))
            
            # 写入文件
            self.main_writer.write(frame_main)
            self.sub_writer.write(frame_sub)
            
            main_frame_count += 1
            sub_frame_count += 1
            frame_count += 1
            self.total_frames += 1
            
            # 每100帧显示进度
            if frame_count % 100 == 0:
                remaining = RECORD_DURATION - elapsed
                free_gb = get_disk_free_space(SAVE_ROOT)
                print(f"\r  录制进度: {elapsed:.0f}/{RECORD_DURATION}s | "
                      f"帧数: {frame_count} | 剩余: {remaining:.0f}s | "
                      f"剩余空间: {free_gb:.1f}GB", end="")
        
        # 关闭写入器
        self._close_writers()
        self.recording = False
        
        # 显示文件大小和实际帧率
        if os.path.exists(self.main_file):
            size_mb = os.path.getsize(self.main_file) / (1024 * 1024)
            actual_fps = main_frame_count / RECORD_DURATION
            print(f"\n  主码流: {size_mb:.1f}MB, 实际帧率: {actual_fps:.1f}fps")
        if os.path.exists(self.sub_file):
            size_mb = os.path.getsize(self.sub_file) / (1024 * 1024)
            actual_fps = sub_frame_count / RECORD_DURATION
            print(f"  子码流: {size_mb:.1f}MB, 实际帧率: {actual_fps:.1f}fps")
    
    def _close_writers(self):
        """关闭视频写入器"""
        if self.main_writer:
            self.main_writer.release()
            self.main_writer = None
        if self.sub_writer:
            self.sub_writer.release()
            self.sub_writer = None
    
    def _reconnect_main(self):
        """重连主码流"""
        if self.main_cap:
            self.main_cap.release()
        self.main_cap = open_camera(RTSP_MAIN)
        if not self.main_cap.isOpened():
            print("❌ 主码流重连失败，等待重试...")
            time.sleep(2)
            self.main_cap = open_camera(RTSP_MAIN)
    
    def _reconnect_sub(self):
        """重连子码流"""
        if self.sub_cap:
            self.sub_cap.release()
        self.sub_cap = open_camera(RTSP_SUB)
        if not self.sub_cap.isOpened():
            print("❌ 子码流重连失败，等待重试...")
            time.sleep(2)
            self.sub_cap = open_camera(RTSP_SUB)
    
    def _status_display(self):
        """状态显示线程"""
        while self.running:
            free_gb = get_disk_free_space(SAVE_ROOT)
            if self.recording:
                elapsed = time.time() - self.record_start_time if self.record_start_time else 0
                remaining = RECORD_DURATION - elapsed
                status = f"🎬 录制中 {elapsed:.0f}s/{RECORD_DURATION}s | 剩余: {remaining:.0f}s"
            else:
                if self.cycle_start_time:
                    elapsed = time.time() - self.cycle_start_time
                    if elapsed < RECORD_DURATION:
                        rest_remaining = RECORD_DURATION - elapsed
                        status = f"⏳ 准备录制 | {rest_remaining:.0f}s后开始"
                    else:
                        rest_remaining = REST_DURATION - (elapsed - RECORD_DURATION)
                        status = f"💤 休息中 | 下次录制: {rest_remaining:.0f}s后"
                else:
                    status = "⏳ 等待启动..."
            
            print(f"\r{status} | 剩余空间: {free_gb:.1f}GB | 总帧数: {self.total_frames}", end="")
            time.sleep(1)
    
    def run(self):
        """主运行循环"""
        print("=" * 70)
        print("  双码流间歇录制工具")
        print("=" * 70)
        print(f"主码流: {RTSP_MAIN}")
        print(f"  分辨率: {MAIN_WIDTH}x{MAIN_HEIGHT}")
        print(f"子码流: {RTSP_SUB}")
        print(f"  分辨率: {SUB_WIDTH}x{SUB_HEIGHT}")
        print(f"录制模式: {RECORD_DURATION}s 录制 → {REST_DURATION}s 休息")
        print(f"保存目录: {SAVE_ROOT}")
        print(f"停止阈值: {MIN_FREE_SPACE_GB}GB")
        print("=" * 70)
        print("按 Ctrl+C 停止录制\n")
        
        # 显示当前磁盘状态
        free_gb = get_disk_free_space(SAVE_ROOT)
        print(f"📊 当前剩余空间: {free_gb:.1f}GB")
        print(f"📊 可用录制空间: {free_gb - MIN_FREE_SPACE_GB:.1f}GB")
        print(f"📊 停止阈值: {MIN_FREE_SPACE_GB}GB\n")
        
        # 打开摄像头
        print("正在连接摄像头...")
        self.main_cap = open_camera(RTSP_MAIN)
        self.sub_cap = open_camera(RTSP_SUB)
        
        if not self.main_cap.isOpened():
            print("❌ 主码流连接失败，程序退出")
            return
        if not self.sub_cap.isOpened():
            print("❌ 子码流连接失败，程序退出")
            return
        
        # 获取摄像头实际参数
        main_fps = self._get_camera_fps(self.main_cap)
        sub_fps = self._get_camera_fps(self.sub_cap)
        print(f"✅ 主码流帧率: {main_fps}fps")
        print(f"✅ 子码流帧率: {sub_fps}fps\n")
        
        self.running = True
        self.current_cycle = 0
        
        # 启动状态显示线程
        status_thread = threading.Thread(target=self._status_display, daemon=True)
        status_thread.start()
        
        while self.running:
            # 检查磁盘空间
            free_gb = get_disk_free_space(SAVE_ROOT)
            if free_gb < MIN_FREE_SPACE_GB:
                print(f"\n⚠️ 磁盘空间不足 ({free_gb:.1f}GB < {MIN_FREE_SPACE_GB}GB)")
                print("   停止录制...")
                break
            
            self.current_cycle += 1
            self.cycle_start_time = time.time()
            
            # 录制阶段
            self._record_cycle()
            
            if not self.running:
                break
            
            # 休息阶段
            print(f"\n💤 [循环 {self.current_cycle}] 进入休息阶段 ({REST_DURATION}秒)...")
            rest_start = time.time()
            while self.running and (time.time() - rest_start) < REST_DURATION:
                free_gb = get_disk_free_space(SAVE_ROOT)
                if free_gb < MIN_FREE_SPACE_GB:
                    print(f"\n⚠️ 磁盘空间不足 ({free_gb:.1f}GB < {MIN_FREE_SPACE_GB}GB)")
                    print("   停止录制...")
                    self.running = False
                    break
                time.sleep(1)
        
        self._cleanup()
    
    def _cleanup(self):
        """清理资源"""
        self.running = False
        self.recording = False
        self._close_writers()
        
        if self.main_cap:
            self.main_cap.release()
            self.main_cap = None
        if self.sub_cap:
            self.sub_cap.release()
            self.sub_cap = None
        
        print("\n✅ 录制已停止")
        print(f"   主码流目录: {self.main_dir}")
        print(f"   子码流目录: {self.sub_dir}")
        print(f"   总录制帧数: {self.total_frames}")
    
    def stop(self):
        """停止录制"""
        self.running = False
        self.recording = False
        time.sleep(0.5)
        self._cleanup()

# ======================== 主程序 ========================
def main():
    recorder = DualStreamRecorder()
    try:
        recorder.run()
    except KeyboardInterrupt:
        print("\n\n用户中断")
        recorder.stop()
    except Exception as e:
        print(f"\n❌ 程序异常: {e}")
        import traceback
        traceback.print_exc()
        recorder.stop()

if __name__ == "__main__":
    main()