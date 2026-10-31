"""
基于 23-2 脚本核心读取逻辑，复刻的极简纯录像脚本。
修改：不建文件夹，直接保存到桌面；剥离所有检测；保持原视频流属性。
"""

import os
import sys
import time
import mmap
import shutil
import struct
import subprocess
import tempfile
import threading
import argparse
from pathlib import Path

# ======================== 配置区 ========================
VIDEO_PATHS = [
    "rtsp://admin:ADMIN123@169.254.243.102:554/live/0/SUB",
]

# 如果原流是 25 帧，这里填 25 即可保持原速。如果不知道，填 25 是最通用的。
OUTPUT_FPS = 25  

# 直接保存到桌面（不建文件夹）
if os.name == 'nt':
    SAVE_DIR = Path(os.path.expanduser("~/Desktop"))
else:
    SAVE_DIR = Path("/root/Desktop")  # 开发板默认 root 桌面

# ======================== 内置独立采集进程 (复刻原逻辑) ========================
HEADER = struct.Struct('!QIIII')  # monotonic_ns, height, width, channels, bytes
CAPACITY = 32 * 1024 * 1024

def gst_available(cv2):
    return any('GStreamer:' in line and line.split(':', 1)[1].strip().startswith('YES')
               for line in cv2.getBuildInformation().splitlines())

def open_capture(cv2, path, use_gstreamer=True):
    rtsp = str(path).lower().startswith(('rtsp://', 'rtsps://'))
    if rtsp and use_gstreamer and gst_available(cv2):
        quoted = str(path).replace('\\', '\\\\').replace('"', '\\"')
        pipeline = (
            'rtspsrc location="%s" latency=200 protocols=tcp tcp-timeout=10000000 ! '
            'rtph265depay ! h265parse ! mppvideodec ! videoconvert ! '
            'video/x-raw,format=BGR ! appsink max-buffers=1 drop=true sync=false'
        ) % quoted
        print('[capture worker] opening backend=GStreamer/MPP', flush=True)
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            return cap
        cap.release()
        print('[capture worker] GStreamer open failed; trying FFmpeg', flush=True)

    params = []
    if hasattr(cv2, 'CAP_PROP_OPEN_TIMEOUT_MSEC') and hasattr(cv2, 'CAP_PROP_READ_TIMEOUT_MSEC'):
        params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000]
    try:
        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG, params)
    except (TypeError, cv2.error):
        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    print('[capture worker] opening backend=FFmpeg', flush=True)
    return cap

def stop_process(proc, grace=2.0):
    if proc is None:
        return
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            proc.kill()
    proc.wait(timeout=3)

class ProcessVideoCapture:
    def __init__(self, path, camera_id, use_gstreamer=True, frame_timeout=5):
        self.path, self.camera_id = path, camera_id
        self.use_gstreamer, self.frame_timeout = use_gstreamer, frame_timeout
        self.latest_frame, self.latest_ts = None, 0.0
        self.generation = 0
        self.lock = threading.Lock()
        self._stop = threading.Event()
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._supervise, daemon=True, name=f'Camera-{self.camera_id}')
        self.thread.start()
        return self

    def _buffer(self):
        folder = tempfile.gettempdir()
        if Path('/dev/shm').is_dir() and shutil.disk_usage('/dev/shm').free > CAPACITY + 16 * 1024**2:
            folder = '/dev/shm'
        f = tempfile.NamedTemporaryFile(prefix=f'camera_{self.camera_id}_', suffix='.mmap', dir=folder)
        try:
            os.posix_fallocate(f.fileno(), 0, HEADER.size + CAPACITY)
            view = mmap.mmap(f.fileno(), HEADER.size + CAPACITY)
        except BaseException:
            f.close()
            raise
        return f, view

    def _invalidate(self):
        with self.lock:
            self.latest_frame, self.latest_ts = None, 0.0

    def _supervise(self):
        import fcntl
        import numpy as np
        delay = 2.0
        while not self._stop.is_set():
            f, view, proc = None, None, None
            received_at = None
            has_frame = False
            began = time.monotonic()
            self._invalidate()
            try:
                f, view = self._buffer()
                env = os.environ.copy()
                env['CAMERA_RTSP_URL'] = str(self.path)
                env['CAMERA_PARENT_PID'] = str(os.getpid())
                cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker',
                       '--buffer', f.name, '--camera', str(self.camera_id),
                       '--gst', str(int(self.use_gstreamer))]
                proc = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL,
                                        close_fds=True, start_new_session=True)
                sequence = 0
                while not self._stop.wait(0.01):
                    if proc.poll() is not None:
                        raise RuntimeError(f'capture child exited {proc.returncode}')
                    frame = None
                    try:
                        fcntl.flock(f.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
                    except BlockingIOError:
                        pass
                    else:
                        try:
                            stamp, h, w, c, size = HEADER.unpack(view[:HEADER.size])
                            if stamp and stamp != sequence:
                                if c != 3 or size != h * w * c or not 0 < size <= CAPACITY:
                                    raise RuntimeError('invalid frame header')
                                frame = np.frombuffer(view, dtype=np.uint8, count=size,
                                                      offset=HEADER.size).reshape(h, w, c).copy()
                                sequence = stamp
                                received_at = stamp / 1e9
                        finally:
                            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    
                    if frame is not None:
                        with self.lock:
                            self.latest_frame, self.latest_ts = frame, received_at
                            if not has_frame:
                                self.generation += 1
                        has_frame = True
                    
                    now = time.monotonic()
                    if received_at is None and now - began > 45:
                        raise TimeoutError('open/first-frame timeout')
                    if received_at is not None and now - received_at > 20:
                        raise TimeoutError(f'read stalled for {now - received_at:.1f}s')
            except Exception as exc:
                print(f'[camera {self.camera_id}] {exc}; restart in {delay:.1f}s', flush=True)
            finally:
                self._invalidate()
                try:
                    if proc is not None: stop_process(proc)
                finally:
                    try:
                        if view is not None: view.close()
                    finally:
                        if f is not None: f.close()
            if self._stop.wait(delay):
                break

    def read(self):
        with self.lock:
            if self.latest_frame is None or time.monotonic() - self.latest_ts > self.frame_timeout:
                return False, None
            return True, self.latest_frame.copy()

    def release(self):
        self._stop.set()
        if self.thread:
            self.thread.join(timeout=7)

def worker(args):
    import faulthandler
    import fcntl
    import signal
    import ctypes
    faulthandler.enable(all_threads=True)
    expected_parent = int(os.environ['CAMERA_PARENT_PID'])
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'unable to arm parent-death signal')
    if os.getppid() != expected_parent:
        return

    import cv2
    import numpy as np
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    path = os.environ['CAMERA_RTSP_URL']
    cap = None
    
    lifecycle = Path(tempfile.gettempdir()) / f'rk_camera_lifecycle_{os.getuid()}.lock'
    try:
        with lifecycle.open('a+b') as gate:
            fcntl.flock(gate, fcntl.LOCK_EX)
            if stopped: return
            cap = open_capture(cv2, path, use_gstreamer=bool(args.gst))
            fcntl.flock(gate, fcntl.LOCK_UN)
        
        with open(args.buffer, 'r+b', buffering=0) as f:
            with mmap.mmap(f.fileno(), HEADER.size + CAPACITY) as view:
                while not stopped:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        raise RuntimeError('stream read failed')
                    
                    frame = np.ascontiguousarray(frame, dtype=np.uint8)
                    h, w, c = frame.shape
                    if c != 3 or frame.nbytes > CAPACITY:
                        raise RuntimeError('frame exceeds 32MiB capacity')
                    
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    try:
                        view[HEADER.size:HEADER.size + frame.nbytes] = frame.tobytes()
                        view[:HEADER.size] = HEADER.pack(time.monotonic_ns(), h, w, c, frame.nbytes)
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    finally:
        if cap is not None:
            with lifecycle.open('a+b') as gate:
                fcntl.flock(gate, fcntl.LOCK_EX)
                cap.release()

if __name__ == '__main__' and '--worker' in sys.argv:
    p = argparse.ArgumentParser()
    p.add_argument('--worker', action='store_true', required=True)
    p.add_argument('--buffer', required=True)
    p.add_argument('--camera', type=int, required=True)
    p.add_argument('--gst', type=int, default=1)
    try:
        worker(p.parse_args())
    except Exception as exc:
        print(f'[capture worker] {exc}', flush=True)
        sys.exit(1)
    sys.exit(0)


# ======================== 父进程纯录像逻辑 ========================
if __name__ == '__main__':
    import cv2
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)

    caps = []
    for i, path in enumerate(VIDEO_PATHS):
        cap = ProcessVideoCapture(path, camera_id=i+1, use_gstreamer=True).start()
        caps.append(cap)

    writers = [None] * len(caps)
    # 使用 mp4v 编码
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 

    try:
        print(f"⏳ 等待视频流接入，准备保存至桌面...")
        while True:
            for i, cap in enumerate(caps):
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                
                # 如果是第一帧，根据原始分辨率初始化 VideoWriter
                if writers[i] is None:
                    h, w = frame.shape[:2]
                    
                    # 直接命名在桌面根目录
                    filename = SAVE_DIR / f"camera_{i+1}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
                    writer = cv2.VideoWriter(str(filename), fourcc, OUTPUT_FPS, (w, h))
                    
                    if not writer.isOpened():
                        print(f"❌ 无法创建视频文件: {filename}")
                        continue
                    writers[i] = writer
                    print(f"✅ [Camera {i+1}] 开始录像: {filename} | 原始分辨率: {w}x{h} | 写入FPS: {OUTPUT_FPS}")
                
                # 直接写入原视频帧，无任何处理
                writers[i].write(frame)
            
            # 短暂休眠防 CPU 100%
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n🛑 接收到退出信号，正在结束录像...")
    finally:
        for cap in caps:
            cap.release()
        for i, writer in enumerate(writers):
            if writer is not None:
                writer.release()
                print(f"[Camera {i+1}] 视频已保存并关闭。")
        print("👋 程序退出。")