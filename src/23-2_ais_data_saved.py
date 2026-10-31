"""

2026-10-10 修改：全景600秒保存，媒体1800秒发送，告警另行立即唤醒；
send_service/{panoramas,rtmp_recordings}/YYYYMMDD；RTSP采集进程隔离及看门狗；
采集与上传已合并进本脚本；默认工作120秒，0秒为永久运行。仅需本脚本及编译发送程序。
基于19-2_ais_data_saved.py，
19-4 更新说明：
  1. 告警视频合成编码由 mp4v 改为 H.264（avc1→H264→mp4v 自动回退，编码器全失败时保留全部帧）
  2. 视频合成成功后清理JPEG帧序列：默认保留20%（每5张留1张），由 FRAME_CLEANUP_KEEP_RATIO 控制
  3. 白天船舶检测置信度阈值提高为 0.5（仅白天船模，夜晚灯光模型保持 0.45）
20-2 更新说明：
  1. 添加命令行参数解析（argparse），支持 --AIS_MODE / --AIS_BUOY_LAT / --AIS_BUOY_LON 等参数
  2. 修复本地视频结束时 stabilized_frame 未定义导致崩溃的问题
  3. 告警录制改用绘制后帧（带检测框+OCR结果），不再保存原始帧用于合成视频
  4. 录制帧添加节流控制（RECORD_SAVE_INTERVAL），降低合成视频帧率
  5. 原图改为独立保存，间隔 RAW_SAVE_INTERVAL（默认10s）保存一张原始帧
  6. 优化稳像类，修复部分场景下稳像失败的问题
23-2 更新说明：
  1. 浮标经纬度改由发送状态/告警信息的同一 RS485 串口接收
  2. RS485 改为常驻双向连接，统一负责 GPS 读取与 STATE/ALARM 发送，避免重复打开串口冲突
单 RS232 更新说明：
  1. AIS 不再直连接收机，改为接收主控通过 RS232 转发的 AIVDM 报文
  2. 浮标经纬度、AIS、STATE 和 ALARM 全部共用一个全双工 RS232 串口
  3. 串口由 SerialTransport 唯一持有，禁止 AIS 线程重复打开同一设备
  4. AIS 距离告警会启动当前在线摄像头的告警录像
  5. STATE 中 cam 改为实时在线摄像头数，并增加 SD 卡已用率 sd
  6. 待发送全景图改为 WebP，只调整质量而不改变分辨率
  7. 待发送告警视频改为 H.265/HEVC，优先使用 GStreamer mpph265enc
  8. STATE 在线摄像头字段改为 cameras_<数量>，例如 cameras_4
  9. alarm_recordings 在 /home 分区可用空间低于预留比例时，循环删除最旧完整告警归档
  10. GPS 最后一次有效坐标在运行期间永不超时，无效定位不再清空已有坐标
  11. GPS 最后有效坐标原子保存到本地文件，程序重启后继续使用
  12. 增加 RS232 字节/分行统计与 GPS 校验、格式、有效坐标限频诊断日志
  13. RS232 接收统计和 GPS 解析诊断日志默认关闭，需要时可通过启动参数开启
  14. AIS 轨迹图增加独立开关，默认关闭；告警、定时和退出保存统一受该开关控制
  15. OCR、代表帧、录像帧、船体裁剪图和 AIS 轨迹图统一改为 WebP
  16. 图像和 AIS 保持独立告警；两者重叠时仅按状态变化追加融合通知，AIS active 保证先发送
  17. alarm_recordings 本地归档增加独立开关并默认关闭；关闭时录像帧转入临时目录，不影响发送视频

yolo_rknn，OCR为ppocrv4_rknn版本的pipeline
# 类别名称（顺序与文件夹名一致）
names:
  0: cloudy
  1: foggy
  2: night
  3: rainy
  4: sunny

"""

import os
# Set native worker limits before importing OpenCV/numpy; subprocesses also set them.
for _thread_var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_thread_var] = '1'
import faulthandler
faulthandler.enable(all_threads=True)

# ======================== 内置独立采集进程 ========================
"""Linux camera capture isolation. Parent never owns a cv2.VideoCapture.

The subprocess is started with exec, not fork: RKNN/Gst state is not inherited.
Raw BGR frames cross an mmap protected with a process file lock. No JPEG loss.
Run only through ProcessVideoCapture; requires the board's existing cv2/numpy.
"""
import argparse
import mmap
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time

HEADER = struct.Struct('!QIIII')  # monotonic_ns, height, width, channels, bytes
CAPACITY = 32 * 1024 * 1024


def gst_available(cv2):
    return any('GStreamer:' in line and line.split(':', 1)[1].strip().startswith('YES')
               for line in cv2.getBuildInformation().splitlines())


def open_capture(cv2, path, use_gstreamer=True):
    """One selected backend per attempt. A failed GST open must not probe others."""
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
        if not cap.isOpened():
            cap.release()
            raise RuntimeError('GStreamer open failed; no backend fallback; supervisor will retry')
        return cap
    params = []
    if hasattr(cv2, 'CAP_PROP_OPEN_TIMEOUT_MSEC') and hasattr(cv2, 'CAP_PROP_READ_TIMEOUT_MSEC'):
        params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000,
                  cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000]
    print('[capture worker] opening backend=FFmpeg (explicit)', flush=True)
    try:
        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG, params)
    except (TypeError, cv2.error):
        # Compatibility with old Python bindings, staying on the same backend.
        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError('FFmpeg open failed; supervisor will retry')
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
    def __init__(self, path, camera_id, use_gstreamer=True, frame_timeout=5,
                 stall_timeout=20, open_timeout=45, reconnect_base=2,
                 reconnect_max=60, prerecord_callback=None, max_fps=15):
        if os.name != 'posix':
            raise RuntimeError('ProcessVideoCapture requires board Linux')
        self.path, self.camera_id = path, camera_id
        self.use_gstreamer, self.frame_timeout = use_gstreamer, frame_timeout
        self.stall_timeout, self.open_timeout = stall_timeout, open_timeout
        self.reconnect_base, self.reconnect_max = reconnect_base, reconnect_max
        self.prerecord_callback, self.max_fps = prerecord_callback, max_fps
        self.latest_frame, self.latest_ts = None, 0.0
        self.generation = 0
        self.lock = threading.Lock()
        self._stop = threading.Event()
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._supervise, daemon=True,
                                       name='Camera-%s-supervisor' % self.camera_id)
        self.thread.start()
        return self

    def _buffer(self):
        # Reserve space up front: avoid SIGBUS from an exhausted tmpfs.
        folder = tempfile.gettempdir()
        if Path('/dev/shm').is_dir() and shutil.disk_usage('/dev/shm').free > CAPACITY + 16 * 1024**2:
            folder = '/dev/shm'
        else:
            print('[camera %s] /dev/shm unavailable/full; using temporary mmap' % self.camera_id, flush=True)
        f = tempfile.NamedTemporaryFile(prefix='camera_%s_' % self.camera_id,
                                        suffix='.mmap', dir=folder)
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
        delay = self.reconnect_base
        while not self._stop.is_set():
            f, view, proc = None, None, None
            received_at = None
            has_frame = False
            began = time.monotonic()
            self._invalidate()
            try:
                f, view = self._buffer()
                env = os.environ.copy()
                for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
                    env[key] = '1'
                env['CAMERA_RTSP_URL'] = str(self.path)  # credentials absent from argv/log
                env['CAMERA_PARENT_PID'] = str(os.getpid())
                cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker',
                       '--buffer', f.name, '--camera', str(self.camera_id),
                       '--gst', str(int(self.use_gstreamer)), '--max-fps', str(self.max_fps)]
                proc = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL,
                                        close_fds=True, start_new_session=True)
                sequence = 0
                while not self._stop.wait(0.02):
                    if proc.poll() is not None:
                        raise RuntimeError('capture child exited %s' % proc.returncode)
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
                            if self.generation == 0 or not has_frame:
                                self.generation += 1
                        if not has_frame:
                            print('[camera %s] stream recovered, generation=%s' %
                                  (self.camera_id, self.generation), flush=True)
                        has_frame = True
                        if self.prerecord_callback:
                            self.prerecord_callback(self.camera_id, frame)
                    now = time.monotonic()
                    if received_at is None and now - began > self.open_timeout:
                        raise TimeoutError('open/first-frame timeout')
                    if received_at is not None and now - received_at > self.stall_timeout:
                        raise TimeoutError('read stalled for %.1fs' % (now - received_at))
                    if received_at is not None and now - began >= 30:
                        delay = self.reconnect_base
            except Exception as exc:
                print('[camera %s] %s; restart in %.1fs' %
                      (self.camera_id, exc, delay), flush=True)
            finally:
                self._invalidate()
                try:
                    if proc is not None:
                        stop_process(proc)
                finally:
                    try:
                        if view is not None:
                            view.close()
                    finally:
                        if f is not None:
                            f.close()
            if self._stop.wait(delay + (self.camera_id % 5) * 0.25):
                break
            delay = min(self.reconnect_max, delay * 2)

    def read(self):
        with self.lock:
            if self.latest_frame is None or time.monotonic() - self.latest_ts > self.frame_timeout:
                return False, None
            return True, self.latest_frame.copy()

    def is_alive(self):
        with self.lock:
            return self.latest_frame is not None and time.monotonic() - self.latest_ts <= self.frame_timeout

    def release(self):
        self._stop.set()
        if self.thread:
            self.thread.join(timeout=7)
            if self.thread.is_alive():
                print('[camera %s] supervisor did not stop within 7s' % self.camera_id, flush=True)


def worker(args):
    import faulthandler
    import fcntl
    import signal
    import ctypes
    faulthandler.enable(all_threads=True)
    # Linux parent-death signal prevents orphaned decoders after a main crash.
    expected_parent = int(os.environ['CAMERA_PARENT_PID'])
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
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
    # All cameras use the same per-user lifecycle lock, not the inference lock.
    lifecycle = Path(tempfile.gettempdir()) / ('rk_camera_lifecycle_%s.lock' % os.getuid())
    try:
        with lifecycle.open('a+b') as gate:
            fcntl.flock(gate, fcntl.LOCK_EX)
            if stopped:
                return
            cap = open_capture(cv2, path, use_gstreamer=bool(args.gst))
            fcntl.flock(gate, fcntl.LOCK_UN)
        with open(args.buffer, 'r+b', buffering=0) as f:
            with mmap.mmap(f.fileno(), HEADER.size + CAPACITY) as view:
                last_publish = 0.0
                while not stopped:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        raise RuntimeError('stream read failed')
                    now = time.monotonic()
                    if args.max_fps > 0 and now - last_publish < 1 / args.max_fps:
                        continue
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
                    last_publish = now
    finally:
        if cap is not None:
            with lifecycle.open('a+b') as gate:
                fcntl.flock(gate, fcntl.LOCK_EX)
                cap.release()  # only this process/thread ever owns the capture


if __name__ == '__main__' and '--worker' in sys.argv:
    p = argparse.ArgumentParser()
    p.add_argument('--worker', action='store_true', required=True)
    p.add_argument('--buffer', required=True)
    p.add_argument('--camera', type=int, required=True)
    p.add_argument('--gst', type=int, default=1)
    p.add_argument('--max-fps', type=float, default=15)
    try:
        worker(p.parse_args())
    except Exception as exc:
        print('[capture worker] %s' % exc, flush=True)
        sys.exit(1)

    sys.exit(0)  # worker must not fall through to RKNN/main initialization

# ======================== 主程序依赖：仅父进程加载 ========================
import cv2
cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)
import sys
import argparse
import numpy as np
import time
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import os
import shutil
import re
from datetime import datetime
import glob
import threading
import collections
import json
import math


# ======================== 内置存储与发送：开始 ========================
from dataclasses import dataclass
from datetime import timezone
from functools import wraps
from pathlib import Path
import hashlib
import stat
import subprocess
import tempfile

# SD 卡识别：每次启动按设备类型、CID 和真实挂载表绑定卡。
class SdUnavailable(RuntimeError):
    pass


def _read(path):
    return Path(path).read_text(encoding="utf-8").strip()


def _device_id(path):
    number = Path(path).stat().st_dev
    return "%d:%d" % (os.major(number), os.minor(number))


def _unescape(value):
    # mountinfo 用八进制转义空格、Tab、换行、反斜线。
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), value)


@dataclass(frozen=True)
class Mount:
    mount_id: str
    device: str                  # major:minor，不能用 /dev 名称猜挂载点
    root: str
    point: Path
    writable: bool


def read_mounts(path="/proc/self/mountinfo"):
    result = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        left, right = line.split(" - ", 1)
        a, b = left.split(), right.split()
        result.append(Mount(a[0], a[2], _unescape(a[3]), Path(_unescape(a[4])),
                            "rw" in a[5].split(",") and "rw" in b[2].split(",")))
    return result


@dataclass(frozen=True)
class Card:
    disk: str
    cid: str
    volume: str
    device: str


def discover_card(sys_class="/sys/class/block", mounts=()):
    """接受唯一 SD 卡、唯一分区（或无分区的整卡文件系统）。歧义时拒绝猜测。"""
    base = Path(sys_class)
    disks = []
    for entry in sorted(base.glob("mmcblk*")):
        if not re.fullmatch(r"mmcblk\d+", entry.name):
            continue
        try:
            if _read(entry / "device/type") == "SD":
                disks.append(entry)
        except FileNotFoundError:
            continue
    if len(disks) != 1:
        raise SdUnavailable("需要且只能有一张原生卡槽 SD 卡；当前检测到 %d 张" % len(disks))
    disk = disks[0]
    parts = sorted(p for p in base.glob(disk.name + "p*")
                   if re.fullmatch(re.escape(disk.name) + r"p\d+", p.name))
    all_devices = {_read(p / "dev") for p in [disk] + parts}
    system_points = {"/", "/boot", "/boot/efi", "/usr", "/var", "/home", "/opt", "/etc"}
    if any(m.device in all_devices and m.point.as_posix() in system_points for m in mounts):
        raise SdUnavailable("这张 SD 卡承载系统目录，拒绝作为待发送数据卡")
    if len(parts) > 1:
        raise SdUnavailable("SD 卡含多个分区，请使用单一数据分区；脚本不会自动格式化或猜选")
    volume = parts[0] if parts else disk
    cid = _read(disk / "device/cid")
    if not re.fullmatch(r"[0-9a-fA-F]{32}", cid):
        raise SdUnavailable("无法读取有效 SD 卡 CID，不能安全绑定当前卡")
    return Card(disk.name, cid.lower(), volume.name, _read(volume / "dev"))


def select_mount(card, mounts):
    matches = [m for m in mounts if m.device == card.device]
    if not matches:
        return None
    if len(matches) != 1 or matches[0].root != "/":
        raise SdUnavailable("SD 分区存在多个挂载点或子目录绑定挂载，无法唯一选择")
    mount = matches[0]
    if not mount.writable:
        raise SdUnavailable("SD 卡以只读方式挂载，请先处理文件系统/挂载问题")
    return mount


class SdStorage:
    def __init__(self, card, mount, reserve_bytes=256 * 1024 * 1024,
                 sys_class="/sys/class/block", mountinfo="/proc/self/mountinfo"):
        self.card, self.mount = card, mount
        self.sys_class, self.mountinfo = Path(sys_class), mountinfo
        self.reserve_bytes = reserve_bytes
        self.send_root = mount.point / "send_service"
        self.state_dir = mount.point / ".stream_upload_state"
        self._invalid = False

    def assert_ready(self):
        """重新检查挂载实例及 CID；一旦失效，本对象永久停止使用，需重启重建。"""
        if self._invalid:
            raise SdUnavailable("SD 卡绑定已失效，请安全换卡后重启程序")
        try:
            now = discover_card(self.sys_class, read_mounts(self.mountinfo))
            mounted = select_mount(now, read_mounts(self.mountinfo))
            if now != self.card or mounted != self.mount:
                raise SdUnavailable("SD 卡或挂载点已变化")
            # 可检测挂载点被另一个文件系统覆盖、挂载路径被软链接替换。
            if self.mount.point.resolve() != self.mount.point:
                raise SdUnavailable("挂载路径包含软链接")
            st = self.mount.point.stat()
            if _device_id(self.mount.point) != self.card.device:
                raise SdUnavailable("挂载点已不属于当前 SD 卡")
            for path in (self.send_root, self.state_dir):
                if path.is_symlink():
                    raise SdUnavailable("数据/状态目录不能是软链接：%s" % path)
                if path.exists() and (not path.is_dir() or path.stat().st_dev != st.st_dev
                                      or os.path.ismount(path)):
                    raise SdUnavailable("数据/状态目录不在当前 SD 文件系统：%s" % path)
        except (OSError, ValueError, SdUnavailable) as exc:
            self._invalid = True
            raise SdUnavailable("SD 卡不可用：%s" % exc) from exc

    def require_space(self, extra_bytes=0):
        self.assert_ready()
        free = shutil.disk_usage(self.mount.point).free
        if free < self.reserve_bytes + extra_bytes:
            raise SdUnavailable("SD 卡空间不足：剩余 %.1f MiB；请停止程序并换卡" % (free / 2**20))

    def prepare(self):
        self.require_space()
        # 只有确认真实挂载存在之后才创建数据目录。
        for path in (self.send_root, self.state_dir):
            self.assert_ready()
            path.mkdir(exist_ok=True)
        try:
            with tempfile.NamedTemporaryFile(prefix="write_probe_", dir=self.state_dir) as f:
                f.write(b"sd-write-check\n")
                f.flush()
                os.fsync(f.fileno())
        except OSError as exc:
            raise SdUnavailable("SD 卡不可写（权限/文件系统）：%s" % exc) from exc
        self.assert_ready()

    def day_dir(self, subdir):
        if subdir not in ("panoramas", "rtmp_recordings"):
            raise ValueError("send_service 只允许 panoramas 和 rtmp_recordings")
        self.require_space()
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        current = self.send_root
        for name in (subdir, day):
            current = current / name
            self.assert_ready()
            if current.is_symlink():
                raise SdUnavailable("发送子目录不能是软链接：%s" % current)
            current.mkdir(exist_ok=True)
            if current.stat().st_dev != self.mount.point.stat().st_dev or os.path.ismount(current):
                raise SdUnavailable("发送子目录被其他挂载覆盖：%s" % current)
        return str(current)


def open_sd_storage(auto_mount=True, wait_seconds=15, reserve_bytes=256 * 1024 * 1024):
    """启动入口。已挂载直接采用真实路径；未挂载可请 udisksctl 自动挂载。"""
    if os.name != "posix" or not Path("/sys/class/block").is_dir():
        raise SdUnavailable("需要开发板 Linux 的 /sys 和 /proc，不能在 Windows 上识别 SD 卡")
    deadline = time.monotonic() + wait_seconds
    attempted = set()
    mount_error = "请检查自动挂载服务及权限"
    last_error = "未找到 SD 卡"
    while True:
        try:
            mounts = read_mounts()
            card = discover_card(mounts=mounts)
            mount = select_mount(card, mounts)
            if mount is None:
                key = (card.cid, card.volume)
                if auto_mount and key not in attempted:
                    attempted.add(key)
                    executable = shutil.which("udisksctl")
                    if not executable:
                        mount_error = "没有 udisksctl；请配置系统自动挂载或先手动挂载"
                        raise SdUnavailable("已找到 /dev/%s，但尚未挂载：%s" % (card.volume, mount_error))
                    result = subprocess.run([executable, "mount", "--block-device", "/dev/" + card.volume,
                                             "--no-user-interaction"], capture_output=True, text=True, timeout=10)
                    # 不解析本地化命令输出；重新从内核挂载表取真实目录。
                    mount = select_mount(card, read_mounts())
                    if mount is None:
                        mount_error = "自动挂载失败（权限、文件系统支持或卡未格式化）：" + result.stderr.strip()
                        raise SdUnavailable(mount_error)
                if mount is None:
                    raise SdUnavailable("SD 卡 /dev/%s 尚未挂载：%s" % (card.volume, mount_error))
            storage = SdStorage(card, mount, reserve_bytes)
            storage.prepare()
            print("[SD] 设备=/dev/%s，挂载点=%s" % (card.volume, mount.point))
            print("[SD] 待发送目录=%s；上传记录=%s" % (storage.send_root, storage.state_dir))
            return storage
        except (OSError, subprocess.SubprocessError, SdUnavailable) as exc:
            last_error = str(exc)
            if time.monotonic() >= deadline:
                raise SdUnavailable(last_error) from exc
            time.sleep(min(1, max(0, deadline - time.monotonic())))


# eMMC 当天媒体与 SD 历史备份，预留实际容量的 5%。
MEDIA_LOCK = threading.RLock()


def media_write(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        with MEDIA_LOCK:
            return func(*args, **kwargs)
    return wrapped


def _read_mount_points():
    """Use the kernel mount table, not ordinary-file st_dev heuristics."""
    if not sys.platform.startswith('linux'):
        return set()
    points = set()
    with open('/proc/self/mountinfo', encoding='utf-8', errors='surrogateescape') as f:
        for line in f:
            fields = line.split()
            if len(fields) < 6 or ' - ' not in line:
                raise OSError('Invalid /proc/self/mountinfo entry')
            point = re.sub(r'\\([0-7]{3})',
                           lambda m: chr(int(m.group(1), 8)), fields[4])
            points.add(os.path.normpath(point))
    return points


def _unsafe_media_entry(path, directory_device, mount_points):
    """OverlayFS regular files may legitimately use a different st_dev."""
    path = Path(path)
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode):
        return True
    if os.path.normpath(os.path.abspath(path)) in mount_points:
        return True
    if stat.S_ISDIR(info.st_mode):
        return info.st_dev != directory_device or os.path.ismount(path)
    # Reject sockets/devices/FIFOs; regular files may belong to overlay upper/lower.
    return not stat.S_ISREG(info.st_mode)


def utc_day():
    return datetime.now(timezone.utc).strftime('%Y%m%d')


def sync_directory(path):
    if os.name == 'posix':
        fd = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def date_dirs(root):
    result = []
    for path in root.iterdir():
        if not re.fullmatch(r'\d{8}', path.name):
            continue
        try:
            datetime.strptime(path.name, '%Y%m%d')
        except ValueError:
            continue
        if path.is_symlink() or os.path.ismount(path):
            raise OSError('Refusing linked/mounted day: %s' % path)
        if path.is_dir():
            result.append(path)
    return sorted(result)


SD_BACKUP_ENABLE = True  # 默认开启；命令行 --SD_BACKUP_ENABLE 0 可关闭


class MediaStorage:
    def __init__(self, root, sd=None, reserve_ratio=0.05, backup_enabled=False):
        self.root = Path(root)
        self.sd = sd
        self.reserve_ratio = reserve_ratio
        self.backup_enabled = backup_enabled
        self._stop = threading.Event()
        self._thread = None
        self._copied = {}
        self.uploader = None

    def prepare(self):
        if self.root.is_symlink():
            raise OSError('eMMC media root cannot be a symlink')
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.backup_enabled:
            print('[SD] 备份功能已关闭；媒体仅保存在 eMMC')
            return
        # 启用备份时才识别/挂载 SD；绝不回退到 eMMC 上的同名目录。
        if self.sd is None:
            self.sd = open_sd_storage(reserve_bytes=0)
        if self.root.stat().st_dev == self.sd.mount.point.stat().st_dev:
            raise OSError('Media source must be on eMMC, not the SD filesystem')
        legacy = [p for p in self.root.iterdir()
                  if re.fullmatch(r'\d{4}-\d{2}-\d{2}', p.name)]
        if legacy:
            print('[media] Old YYYY-MM-DD archives retained; sender scans both layouts')
        self.sync_once()

    def day_dir(self, subdir, day=None):
        if subdir not in ('panoramas', 'rtmp_recordings'):
            raise ValueError(subdir)
        day = day or utc_day()
        datetime.strptime(day, '%Y%m%d')
        current = self.root
        device = current.stat().st_dev
        for part in (subdir, day):
            current = current / part
            if current.is_symlink() or os.path.ismount(current):
                raise OSError('Unsafe media directory: %s' % current)
            current.mkdir(exist_ok=True)
            if current.stat().st_dev != device:
                raise OSError('Media directory changed filesystem')
        return str(current)

    def _checked_dir(self, path):
        self.sd.assert_ready()
        if path.is_symlink() or os.path.ismount(path):
            raise SdUnavailable('Unsafe SD directory: %s' % path)
        path.mkdir(exist_ok=True)
        if path.stat().st_dev != self.sd.mount.point.stat().st_dev:
            raise SdUnavailable('Directory left SD filesystem: %s' % path)

    def _space(self, extra, protected):
        while True:
            self.sd.assert_ready()
            usage = shutil.disk_usage(self.sd.mount.point)
            if usage.free >= int(usage.total * self.reserve_ratio) + extra:
                return
            candidates = []
            for category in ('panoramas', 'rtmp_recordings'):
                folder = self.sd.send_root / category
                if folder.exists():
                    candidates.extend(p for p in date_dirs(folder)
                                      if p.name < utc_day() and p.name != protected)
            if not candidates:
                raise SdUnavailable('SD reserve 5% cannot be maintained; no older day to delete')
            oldest = min(p.name for p in candidates)
            for archive in (p for p in candidates if p.name == oldest):
                self._validate_tree(archive)
                self.sd.assert_ready()
                shutil.rmtree(archive)
                print('[SD] Removed oldest archive: %s' % archive)

    @staticmethod
    def _validate_tree(root):
        device = root.stat().st_dev
        mount_points = _read_mount_points()
        for folder, dirs, files in os.walk(root, followlinks=False):
            for name in dirs + files:
                path = Path(folder) / name
                if _unsafe_media_entry(path, device, mount_points):
                    raise OSError('Unsafe archive entry: %s' % path)

    def _copy(self, source, target, day):
        self.sd.assert_ready()
        if target.is_symlink():
            raise SdUnavailable('Linked archive file: %s' % target)
        st = source.stat()
        signature = (st.st_size, st.st_mtime_ns)
        if self._copied.get(str(target)) == signature and target.is_file() and target.stat().st_size == st.st_size:
            return
        # Compare contents: FAT timestamps have different precision from eMMC.
        if target.is_file() and target.stat().st_size == source.stat().st_size:
            with source.open('rb') as a, target.open('rb') as b:
                while True:
                    chunk = a.read(1024 * 1024)
                    if chunk != b.read(1024 * 1024):
                        break
                    if not chunk:
                        self._copied[str(target)] = signature
                        return
        self._space(source.stat().st_size, day)
        tmp = target.with_name(target.name + '.backup.part')
        self.sd.assert_ready()
        if tmp.is_symlink():
            raise SdUnavailable('Linked temporary file: %s' % tmp)
        with source.open('rb') as src, tmp.open('wb') as dst:
            shutil.copyfileobj(src, dst)
            dst.flush()
            os.fsync(dst.fileno())
        self.sd.assert_ready()
        os.replace(tmp, target)
        sync_directory(target.parent)
        self._copied[str(target)] = signature

    def sync_once(self):
        with MEDIA_LOCK:
            self.sd.assert_ready()
            for category, suffixes in (('panoramas', ('.jpg', '.webp')),
                                       ('rtmp_recordings', ('.mp4',))):
                folder = self.root / category
                if not folder.exists():
                    continue
                self._validate_tree(folder)
                dest_root = self.sd.send_root / category
                self._checked_dir(dest_root)
                for day in date_dirs(folder):
                    dest = dest_root / day.name
                    self._checked_dir(dest)
                    files = []
                    for path in sorted(day.iterdir()):
                        if path.suffix.lower() in suffixes and re.fullmatch(
                                r'camera_[1-9]\d*_\d{8}_\d{6}(?:_\d+)?', path.stem):
                            self._copy(path, dest / path.name, day.name)
                            files.append(path)
                    # Preserve unsent previous-day data on eMMC. Backup alone is
                    # not upload success. Never silently lose the retry source.
                    if day.name < utc_day() and self.uploader is not None and files:
                        other = [p for p in day.iterdir() if p not in files]
                        if not other and all(self.uploader.is_sent(p) for p in files):
                            self.sd.assert_ready()
                            sync_directory(dest)
                            sync_directory(dest_root)
                            shutil.rmtree(day)
                            self._copied = {k: v for k, v in self._copied.items()
                                            if dest not in Path(k).parents}
            self._space(0, utc_day())

    def start(self):
        if not self.backup_enabled:
            return
        self._thread = threading.Thread(target=self._run, name='SD-Backup', daemon=False)
        self._thread.start()

    def _run(self):
        while not self._stop.wait(30):
            try:
                self.sync_once()
            except Exception as exc:
                print('[SD] Backup failed; eMMC retained, retry in 30s: %s' % exc)

    def stop(self):
        if not self.backup_enabled:
            return
        self._stop.set()
        if self._thread:
            self._thread.join()
        try:
            self.sync_once()
        except Exception as exc:
            print('[SD] Final backup failed; eMMC retained: %s' % exc)


# 发送器从 eMMC 创建快照；每30分钟或告警触发，SD保留备份。
class MediaUploader:
    def __init__(self, root, binary, ip, port=7878, state_dir=None,
                 retry_seconds=30, timeout=600, settle_seconds=2, period_seconds=1800):
        # Source and snapshots are on eMMC; SD is backup only.
        self.root = Path(os.path.abspath(root))
        self._source_identity = None
        self.binary = Path(binary).resolve()
        self.ip, self.port = ip, port
        self.state_dir = Path(state_dir or Path(__file__).parent / '.stream_upload_state_23').resolve()
        if self.state_dir == self.root or self.root in self.state_dir.parents:
            raise ValueError('Upload state must be outside send_service')
        self.retry_seconds, self.timeout = retry_seconds, timeout
        if min(retry_seconds, timeout, period_seconds) <= 0 or settle_seconds < 0:
            raise ValueError('Invalid uploader intervals')
        self.settle_seconds = settle_seconds
        self.period_seconds = period_seconds
        self._active = threading.Event()
        self._queue_lock = threading.Lock()
        self._alarm_paths = set()
        self._sent_lock = threading.RLock()
        self.sent = {}
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        self._lock_file = None
        self._pending_lock = threading.RLock()
        self._pending_paths = set()
        self._next_reconcile = 0.0
        self.reconcile_seconds = 3600

    def start(self):
        self._ensure_binary_executable()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._lock_file = open(self.state_dir / 'worker.lock', 'a+b')
        try:
            if os.name == 'nt':
                import msvcrt
                self._lock_file.seek(0)
                self._lock_file.write(b'0')
                self._lock_file.flush()
                self._lock_file.seek(0)
                msvcrt.locking(self._lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._load_state()
        except Exception:
            self._lock_file.close()
            self._lock_file = None
            raise
        self._thread = threading.Thread(target=self._run, name='Media-Uploader', daemon=False)
        self._thread.start()

    def _ensure_binary_executable(self):
        if not self.binary.is_file():
            raise RuntimeError(f'Sender missing: {self.binary}')
        if os.access(self.binary, os.X_OK):
            return
        current_mode = self.binary.stat().st_mode
        executable_mode = current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
        os.chmod(self.binary, executable_mode)
        if not os.access(self.binary, os.X_OK):
            raise RuntimeError(f'Sender not executable after chmod: {self.binary}')

    def _load_state(self):
        path = self.state_dir / 'sent.json'
        if path.exists():
            # Fail closed on corrupt state; do not silently re-send everything.
            self.sent = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(self.sent, dict):
                raise ValueError('Invalid sent.json')
        alarm_state = self.state_dir / 'alarm_pending.json'
        if alarm_state.exists():
            paths = json.loads(alarm_state.read_text(encoding='utf-8'))
            if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
                raise ValueError('Invalid alarm_pending.json')
            with self._queue_lock:
                self._alarm_paths.update(paths)

    def _save_state(self):
        target = self.state_dir / 'sent.json'
        tmp = target.with_suffix('.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(self.sent, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)

    @staticmethod
    def _signature(path):
        s = path.stat()
        return [s.st_size, s.st_mtime_ns]

    def _scan_pending(self):
        """All unconfirmed days, new layout plus retained pre-upgrade archives."""
        result = []
        if not self.root.is_dir():
            return result
        identity = self._check_source()
        mount_points = _read_mount_points()
        directory_device = self.root.stat().st_dev
        folders = []
        for category, suffix in (('panoramas', r'(?:jpg|webp)'), ('rtmp_recordings', 'mp4')):
            base = self.root / category
            if base.is_dir() and not base.is_symlink() and not os.path.ismount(base):
                folders.extend((day, suffix) for day in date_dirs(base))
        for day in sorted(self.root.iterdir()):
            if day.is_dir() and not day.is_symlink() and not os.path.ismount(day) and re.fullmatch(
                    r'\d{4}-\d{2}-\d{2}', day.name):
                for category, suffix in (('panoramas', r'(?:jpg|webp)'), ('rtmp_recordings', 'mp4')):
                    folders.append((day / category, suffix))
        for folder, suffix in folders:
            if not folder.is_dir() or folder.is_symlink() or os.path.ismount(folder):
                continue
            for path in sorted(folder.iterdir()):
                try:
                    if _unsafe_media_entry(path, directory_device, mount_points) or not path.is_file():
                        continue
                except OSError:
                    continue
                if not re.fullmatch(r'camera_[1-9]\d*_\d{8}_\d{6}(?:_\d+)?\.' + suffix, path.name):
                    continue
                try:
                    sig = self._signature(path)
                except OSError:
                    continue
                key = identity + '/' + path.relative_to(self.root).as_posix()
                if sig[0] > 0 and self.sent.get(key) != sig:
                    result.append((path, key, sig))
        return sorted(result, key=lambda x: (x[0].name, x[1]))

    def register(self, path):
        """File publication registers work without triggering the periodic timer."""
        path = Path(os.path.abspath(path))
        relative = path.relative_to(self.root)
        remote = self.remote_path(path)
        if (len(remote.parts) != 3 or remote.parts[0] not in
                ('panoramas', 'rtmp_recordings') or
                not re.fullmatch(r'\d{8}', remote.parts[1]) or
                not re.fullmatch(r'camera_[1-9]\d*_\d{8}_\d{6}(?:_\d+)?\.(?:jpg|webp|mp4)', remote.name)):
            raise ValueError('Unsupported upload media path: %s' % relative)
        with self._pending_lock:
            self._pending_paths.add(relative.as_posix())

    def pending(self):
        """Startup/hourly recovery scan; regular batches stat only queued files."""
        if not self.root.is_dir():
            return []
        identity = self._check_source()
        mount_points = _read_mount_points()
        directory_device = self.root.stat().st_dev
        # Hold this lock across recovery so concurrent publication cannot be lost.
        with self._pending_lock:
            if time.monotonic() >= self._next_reconcile:
                self._pending_paths.update(
                    path.relative_to(self.root).as_posix()
                    for path, key, sig in self._scan_pending())
                self._next_reconcile = time.monotonic() + self.reconcile_seconds
            result = []
            for relative in list(self._pending_paths):
                path = self.root / relative
                try:
                    if _unsafe_media_entry(path, directory_device, mount_points):
                        raise OSError('Unsafe upload file')
                    for parent in path.parents:
                        if parent == self.root:
                            break
                        if _unsafe_media_entry(parent, directory_device, mount_points):
                            raise OSError('Unsafe upload directory')
                    sig = self._signature(path)
                except OSError:
                    self._pending_paths.discard(relative)
                    continue
                key = identity + '/' + relative
                with self._sent_lock:
                    sent = self.sent.get(key) == sig
                if sent:
                    self._pending_paths.discard(relative)
                elif sig[0] > 0 and time.time_ns() - sig[1] >= self.settle_seconds * 1e9:
                    result.append((path, key, sig))
            return sorted(result, key=lambda item: (item[0].name, item[1]))

    def is_sent(self, path):
        key = self._check_source() + '/' + path.relative_to(self.root).as_posix()
        with self._sent_lock:
            return self.sent.get(key) == self._signature(path)

    def _check_source(self):
        st = self.root.stat()
        identity = hashlib.sha256(
            f'{self.root.resolve()}:{st.st_dev}:{st.st_ino}'.encode()
        ).hexdigest()
        if self._source_identity is None:
            self._source_identity = identity
        elif identity != self._source_identity:
            raise RuntimeError('eMMC source directory changed; restart before uploading')
        return identity

    def notify(self, path=None):
        """Persist alarm JPEG eligibility; videos are intrinsically alarm media.

        Periodic pictures wait for the timer even when an alarm happens early.
        """
        if path is not None:
            self.register(path)
            relative = Path(path).relative_to(self.root).as_posix()
            with self._queue_lock:
                self._alarm_paths.add(relative)
                self.state_dir.mkdir(parents=True, exist_ok=True)
                target = self.state_dir / 'alarm_pending.json'
                tmp = target.with_suffix('.tmp')
                with tmp.open('w', encoding='utf-8') as f:
                    json.dump(sorted(self._alarm_paths), f)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, target)
        self._wake.set()

    def remote_path(self, path):
        relative = path.relative_to(self.root)
        parts = relative.parts
        if len(parts) == 3 and re.fullmatch(r'\d{4}-\d{2}-\d{2}', parts[0]):
            return Path(parts[1]) / parts[0].replace('-', '') / parts[2]
        return relative

    def _send_snapshot(self, snapshot):
        # Snapshot preserves media_type/YYYYMMDD/file relative to send_service.
        # Explicit path, recursive scan: existing binary needs no --date-dir.
        cmd = [str(self.binary), self.ip, str(snapshot), '--port', str(self.port),
               '--period', '0', '--last-n', '0', '--no-date-dir',
               '--suffix', '.jpg,.webp,.mp4', '--cameras', '0',
               '--incremental', '--sent-file', str(snapshot / '.stream_sender.sent')]
        with open(self.state_dir / 'sender.log', 'ab') as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + self.timeout
            try:
                while proc.poll() is None:
                    if self._stop.wait(0.2) or time.monotonic() >= deadline:
                        raise TimeoutError('Sender interrupted or timed out; batch remains pending')
                if proc.returncode != 0:
                    raise RuntimeError(f'Sender exited with {proc.returncode}; see sender.log')
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()

    def upload_once(self, alarm_only=False):
        """Keep an active batch and its binary journal until durable confirmation."""
        batch = self.state_dir / 'active_batch'
        manifest_path = self.state_dir / 'active_batch.json'
        identity = self._check_source()
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            if manifest['identity'] != identity:
                raise RuntimeError('Active batch belongs to another source; retained for inspection')
            records = manifest['records']
            for record in records:
                relative = Path(record['remote'])
                if relative.is_absolute() or '..' in relative.parts or len(relative.parts) != 3:
                    raise ValueError('Invalid active batch manifest')
                target = batch / relative
                if not target.is_file() or target.is_symlink() or self._signature(target) != record['snapshot_sig']:
                    raise RuntimeError('Active snapshot changed; originals retained')
        else:
            items = self.pending()
            if alarm_only:
                with self._queue_lock:
                    marked = self._alarm_paths.copy()
                items = [item for item in items if item[0].suffix == '.mp4' or
                         item[0].relative_to(self.root).as_posix() in marked]
            if not items:
                return False
            # A folder without a manifest is an incomplete pre-transmission build.
            if batch.exists():
                MediaStorage._validate_tree(batch)
                shutil.rmtree(batch)
            batch.mkdir(parents=True)
            records = []
            for path, key, sig in items[:64]:
                self._check_source()
                if self._stop.is_set():
                    return False
                if sig[0] > 0xFFFFFFFF:
                    raise ValueError('File exceeds protocol size limit: %s' % path)
                remote = self.remote_path(path)
                target = batch / remote
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    if self._signature(path) != sig:
                        continue
                    # Copy isolates retries from later source-file modifications.
                    shutil.copy2(path, target)
                    if self._signature(path) != sig or target.stat().st_size != sig[0]:
                        target.unlink(missing_ok=True)
                        continue
                    with target.open('r+b') as f:
                        os.fsync(f.fileno())
                    records.append({'key': key, 'sig': sig,
                                    'remote': remote.as_posix(),
                                    'snapshot_sig': self._signature(target)})
                except OSError:
                    target.unlink(missing_ok=True)
                    continue
            if not records:
                return False
            for folder, dirs, files in os.walk(batch, topdown=False):
                sync_directory(folder)
            manifest = {'identity': identity, 'records': records}
            tmp = manifest_path.with_suffix('.tmp')
            with tmp.open('w', encoding='utf-8') as f:
                json.dump(manifest, f, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, manifest_path)
            sync_directory(self.state_dir)
        self._send_snapshot(batch)
        self._check_source()
        journal = batch / '.stream_sender.sent'
        # New deployment specifies one absolute successfully sent path per line.
        confirmed = set(journal.read_text(encoding='utf-8').splitlines()) if journal.is_file() else set()
        expected = {str((batch / record['remote']).resolve()) for record in records}
        if not expected.issubset(confirmed):
            raise RuntimeError('Sender did not confirm all batch files; batch retained')
        with self._sent_lock:
            previous = self.sent.copy()
            self.sent.update({record['key']: record['sig'] for record in records})
            try:
                self._save_state()
                sync_directory(self.state_dir)
            except Exception:
                self.sent = previous
                raise
        # Forget the manifest only after sent.json has been committed.
        manifest_path.unlink()
        sync_directory(self.state_dir)
        MediaStorage._validate_tree(batch)
        shutil.rmtree(batch)
        print('[upload] Confirmed %s media files; originals retained' % len(records))
        return True

    def _run(self):
        next_due = time.monotonic() + self.period_seconds
        try:
            while not self._stop.is_set():
                wait = max(0, next_due - time.monotonic())
                self._wake.wait(wait)
                if self._stop.is_set():
                    break
                alarm = self._wake.is_set()
                self._wake.clear()  # later alarms remain set for the next pass
                now = time.monotonic()
                due = now >= next_due
                if not (alarm or due):
                    continue
                if due:
                    next_due += (int((now - next_due) // self.period_seconds) + 1) * self.period_seconds
                print('[upload] Trigger=%s' % ('alarm' if alarm else '30-minute timer'))
                self._active.set()
                try:
                    if self._stop.wait(self.settle_seconds):
                        break
                    while not self._stop.is_set():
                        try:
                            if not self.upload_once(alarm_only=not due):
                                break
                        except Exception as exc:
                            print('[upload] %s; retry in %ss' % (exc, self.retry_seconds))
                            if self._stop.wait(self.retry_seconds):
                                break
                finally:
                    self._active.clear()
        finally:
            if self._lock_file:
                self._lock_file.close()
                self._lock_file = None

    def stop(self, grace_seconds=15):
        """Drain already triggered work; exit does not trigger periodic data."""
        if self._thread is None:
            return
        deadline = time.monotonic() + max(0, grace_seconds)
        while (self._wake.is_set() or self._active.is_set()) and time.monotonic() < deadline:
            if not self._thread.is_alive():
                break
            time.sleep(0.05)
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            print('[upload] Worker is still finishing local IO; unconfirmed data retained')
# ======================== 内置存储与发送：结束 ========================


# ======================== GPS 定位与配置坐标回退 ========================
class GPSPosition:
    def __init__(self, latitude, longitude, timeout=30.0, enabled=True,
                 state_file=None, persist_interval=5.0,
                 debug=False, debug_interval=5.0):
        self.fallback = (latitude, longitude)
        # timeout 仅保留用于兼容旧启动参数；最后一次有效定位不再过期。
        self.timeout = timeout
        self.enabled = enabled
        self._fix = None
        self._received = 0.0
        self._lock = threading.Lock()
        self.state_file = Path(state_file).expanduser().absolute() if state_file else None
        self.persist_interval = max(0.1, float(persist_interval))
        self._last_persist_attempt = 0.0
        self._last_persist_error = None
        self.debug = bool(debug)
        self.debug_interval = max(0.5, float(debug_interval))
        self._gps_rx_count = 0
        self._gps_valid_count = 0
        self._gps_invalid_count = 0
        self._last_debug_time = 0.0
        self._load_last_fix()

    def _debug_result(self, result, sentence, fix=None, detail='', force=False):
        """限频输出GPS解析结果，避免主控高频发送时刷屏。"""
        if not self.debug:
            return
        now = time.monotonic()
        if not force and now - self._last_debug_time < self.debug_interval:
            return
        self._last_debug_time = now
        position = '' if fix is None else f' lat={fix[0]:.6f}, lon={fix[1]:.6f}'
        suffix = '' if not detail else f' ({detail})'
        print(f"[GPS RX] {result}: total={self._gps_rx_count}, "
              f"valid={self._gps_valid_count}, invalid={self._gps_invalid_count}"
              f"{position}{suffix}; raw={sentence!r}")

    @staticmethod
    def _valid_decimal_fix(latitude, longitude):
        return (math.isfinite(latitude) and math.isfinite(longitude)
                and -90 <= latitude <= 90 and -180 <= longitude <= 180)

    def _load_last_fix(self):
        if not self.enabled or self.state_file is None or not self.state_file.exists():
            return
        try:
            if self.state_file.is_symlink() or not self.state_file.is_file():
                raise OSError("定位状态路径不能是软链接或非普通文件")
            data = json.loads(self.state_file.read_text(encoding='utf-8'))
            latitude = float(data['latitude'])
            longitude = float(data['longitude'])
            if not self._valid_decimal_fix(latitude, longitude):
                raise ValueError("经纬度超出有效范围")
            self._fix = (latitude, longitude)
            self._received = time.monotonic()
            if self.debug:
                print(f"✅ 已加载上次有效GPS: lat={latitude:.6f}, lon={longitude:.6f} "
                      f"({self.state_file})")
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            print(f"⚠️ GPS本地状态文件无效，等待新定位: {exc}")

    def _persist_locked(self, force=False):
        if not self.enabled or self.state_file is None or self._fix is None:
            return False
        now = time.monotonic()
        if not force and now - self._last_persist_attempt < self.persist_interval:
            return False
        self._last_persist_attempt = now
        tmp = self.state_file.with_name(self.state_file.name + '.tmp')
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            if self.state_file.is_symlink() or tmp.is_symlink():
                raise OSError("GPS状态文件或临时文件不能是软链接")
            latitude, longitude = self._fix
            data = {
                'latitude': latitude,
                'longitude': longitude,
                'saved_at_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
            }
            with open(tmp, 'w', encoding='utf-8') as handle:
                json.dump(data, handle, ensure_ascii=True, separators=(',', ':'))
                handle.write('\n')
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.state_file)
            sync_directory(self.state_file.parent)
            self._last_persist_error = None
            return True
        except OSError as exc:
            message = str(exc)
            if message != self._last_persist_error:
                print(f"⚠️ GPS坐标持久化失败，内存坐标仍有效: {exc}")
                self._last_persist_error = message
            return False
        finally:
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass

    def flush(self):
        """程序正常退出时强制保存当前最新GPS。"""
        with self._lock:
            return self._persist_locked(force=True)

    @staticmethod
    def _coordinate(value, hemisphere, latitude):
        directions = ('N', 'S') if latitude else ('E', 'W')
        if hemisphere not in directions:
            raise ValueError('Invalid hemisphere')
        raw = float(value)
        if not math.isfinite(raw) or raw < 0:
            raise ValueError('Invalid coordinate')
        degrees, minutes = divmod(raw, 100)
        result = degrees + minutes / 60
        if minutes >= 60 or result > (90 if latitude else 180):
            raise ValueError('Coordinate out of range')
        return -result if hemisphere in ('S', 'W') else result

    def feed(self, sentence):
        """解析 POSITION 或标准 RMC/GGA/GLL；识别到定位报文时返回 True。"""
        if not sentence.startswith('$'):
            return False
        fields = sentence.split('*', 1)[0].split(',')
        kind = fields[0][-3:]
        is_position = fields[0] == '$POSITION'
        is_nmea_gps = len(fields[0]) == 6 and kind in ('RMC', 'GGA', 'GLL')
        if not is_position and not is_nmea_gps:
            return False
        self._gps_rx_count += 1
        if not self.enabled:
            self._debug_result('已识别但GPS功能已关闭', sentence)
            return True
        try:
            parts = sentence[1:].strip().split('*')
            if len(parts) != 2:
                self._gps_invalid_count += 1
                self._debug_result('拒绝', sentence, detail='缺少或包含多个校验和分隔符')
                return True
            payload, checksum = parts
            actual = 0
            for char in payload:
                actual ^= ord(char)
            if len(checksum) != 2 or actual != int(checksum, 16):
                self._gps_invalid_count += 1
                self._debug_result(
                    '拒绝', sentence,
                    detail=f'校验失败 expected={actual:02X}, received={checksum}',
                )
                return True
            if is_position:
                # $POSITION,<latitude>,<longitude>,<A|V>*CS\r\n
                if len(fields) != 4 or fields[3] not in ('A', 'V'):
                    self._gps_invalid_count += 1
                    self._debug_result('拒绝', sentence, detail='POSITION字段数或状态错误')
                    return True
                if fields[3] == 'V':
                    # 无效定位不覆盖最后一次有效坐标。
                    self._gps_invalid_count += 1
                    self._debug_result('忽略', sentence, detail='POSITION状态为V，保留上次有效坐标')
                    return True
                latitude, longitude = float(fields[1]), float(fields[2])
                if (not math.isfinite(latitude) or not math.isfinite(longitude)
                        or not -90 <= latitude <= 90 or not -180 <= longitude <= 180):
                    self._gps_invalid_count += 1
                    self._debug_result('拒绝', sentence, detail='经纬度超出有效范围')
                    return True
                fix = (latitude, longitude)
                valid = True
            elif kind == 'RMC':
                valid = fields[2] == 'A' and (len(fields) <= 12 or fields[12] not in ('N', 'E', 'S'))
                lat, ns, lon, ew = fields[3:7]
            elif kind == 'GGA':
                valid = int(fields[6]) in (1, 2, 3, 4, 5)
                lat, ns, lon, ew = fields[2:6]
            else:
                valid = fields[6] == 'A' and (len(fields) <= 7 or fields[7] not in ('N', 'E', 'S'))
                lat, ns, lon, ew = fields[1:5]
            if not valid:
                # 无效 RMC/GGA/GLL 不覆盖最后一次有效坐标。
                self._gps_invalid_count += 1
                self._debug_result('忽略', sentence, detail=f'{kind}定位状态无效')
                return True
            if not is_position:
                fix = (self._coordinate(lat, ns, True), self._coordinate(lon, ew, False))
            with self._lock:
                self._fix, self._received = fix, time.monotonic()
                persisted = self._persist_locked()
            self._gps_valid_count += 1
            detail = '已更新，已写本地文件' if persisted else '已更新，文件写入节流中'
            self._debug_result('有效', sentence, fix=fix, detail=detail)
        except (ValueError, IndexError) as exc:
            self._gps_invalid_count += 1
            self._debug_result('拒绝', sentence, detail=f'数值/字段解析失败: {exc}')
        return True

    def position(self):
        with self._lock:
            # 只要曾收到有效 GPS，之后始终使用最后一次有效坐标。
            if self.enabled and self._fix is not None:
                return (*self._fix, 'GPS')
            return (*self.fallback, 'configured')


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
CLS_MODEL_PATH = "/root/Desktop/rknn_model_zoo_and_models/yolov11n-cls/cls_1-best_255_fp.rknn"        # 场景分类模型
DET_DAY_MODEL_PATH = "/root/Desktop/rknn_model_zoo_and_models/yolov11n-day/day_5-best_255_i8.rknn"    # 白天船舶检测模型
DET_NIGHT_MODEL_PATH = "/root/Desktop/rknn_model_zoo_and_models/yolov11n-night/night_2-best_255_i8.rknn"  # 夜晚灯光检测模型

# 4路视频路径 - 修改为海康网络摄像头RTSP流
VIDEO_PATHS = [
    # 海康网络摄像头RTSP流格式：rtsp://用户名:密码@IP地址:端口/流路径
    "rtsp://admin:ADMIN123@169.254.241.100:554/stream1",  # 夜通航网络摄像头1
    #"rtsp://admin:ADMIN123@169.254.241.100:554/live/0/SUB",  # 子码流（推荐，占用资源少）
    
    #"rtsp://admin:xiayuxuan113@169.254.242.101:554/stream1",  # 海康网络摄像头2
    #"rtsp://admin:xiayuxuan113@169.254.242.101:554/Streaming/Channels/102",  # 子码流
    
    
    # "rtsp://admin:ADMIN123@169.254.242.101:554/stream1",  # 夜通航网络摄像头2
    #"rtsp://admin:ADMIN123@169.254.242.101:554/live/0/SUB",  # 子码流
    "rtsp://admin:ADMIN123@169.254.243.102:554/stream1",  # 夜通航网络摄像头3
    #"rtsp://admin:ADMIN123@169.254.243.102:554/live/0/SUB",  # 子码流
    # "rtsp://admin:ADMIN123@169.254.244.103:554/stream1",  # 夜通航网络摄像头4
    #"rtsp://admin:ADMIN123@169.254.244.103:554/live/0/SUB",  # 子码流
    #"/root/Desktop/pipeline/test/多云-近处小船（P船号）.mp4",     #多云    
    #"/root/Desktop/pipeline/test/晴天短视频.mp4",  #晴天    
    #"/root/Desktop/pipeline/test/雾天.mp4",  #雾天  正常
    #"/root/Desktop/pipeline/test/夜晚-带月亮.mp4",   #夜晚  不正常，识别成foggy
]

# ======================== 多摄像头采集优化配置 ========================
USE_GSTREAMER = True
FRAME_TIMEOUT = 5.0
CLS_INTERVAL = 5  # 每隔N帧做一次分类，其余帧复用结果
CAPTURE_RECONNECT_INTERVAL = 2.0  # 退避从 2 秒增加到 60 秒

# 视频输入帧率控制（0=全速处理, >0=限制每秒处理帧数）
# 无告警时用较低帧率节省 CPU，告警时自动切换为告警帧率
TARGET_FPS_IDLE = 0      # 无告警时帧率
TARGET_FPS_ALARM = 0     # 告警时帧率（0=全速，也可设为20固定值）
# 模型输入尺寸
CLS_INPUT_SIZE = 256  
DET_INPUT_SIZE = 640

# 告警像素占比阈值（基准分辨率 1920×1280，适配16:9任意分辨率）
SHIP_ALARM_THRESH = 0.00185    # 船舶告警：5000 / (1920×1280) = 0.002
LIGHT_ALARM_THRESH = 0.0004  # 灯光告警：1200 / (1920×1280) = 0.0005

# 工作模式
DEFAULT_WORK_DURATION = 0  # 沿用本次原件默认永久运行；--WORK_DURATION 120=工作120秒，0=永久运行
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
DET_MODEL_PATH = "/root/Desktop/rknn_model_zoo_and_models/PPOCR_rknn/det/ppocrv4_det.rknn"
REC_MODEL_PATH = "/root/Desktop/rknn_model_zoo_and_models/PPOCR_rknn/rec/ppocrv4_rec.rknn"
# 配套字典路径
DICT_PATH = "/root/Desktop/rknn_model_zoo_and_models/PPOCR_rknn/ppocr_keys_v1.txt"

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
ROI_SAVE_ROOT = "/root/Desktop/pipeline/ocr_roi_images"  # ROI保存根目录
ROI_SAVE_FORMAT = "webp"  # OCR ROI 统一保存为 WebP
ROI_QUALITY = 85  # WebP 保存质量（0-100）

# ======================== 录制与推流核心配置 ========================
# 本地告警归档开关。关闭后不创建、不写入 RECORD_ROOT；视频合成帧使用临时目录。
ALARM_ARCHIVE_ENABLE = False
# 本地告警归档根目录
RECORD_ROOT = "/root/Desktop/pipeline/alarm_recordings"
# 归档关闭时的当次进程临时工作目录，只用于合成待发送视频，退出时清理。
ALARM_WORK_ROOT = os.path.join(tempfile.gettempdir(), f"pipeline_alarm_work_{os.getpid()}")
FAILED_ALARM_WORK_DIRS = set()  # failed encoder evidence must survive exit cleanup
# /home 在开发板上挂载于 /dev/mmcblk0p8。默认至少保留该分区 10% 可用空间。
ALARM_STORAGE_RESERVE_PERCENT = 10.0
# 发送服务根目录（全景图和发送版视频，按 UTC 天归档）
SEND_ROOT = "/root/Desktop/pipeline/send_service"
media_storage = MediaStorage(SEND_ROOT, backup_enabled=SD_BACKUP_ENABLE)
SEND_IMG_MAX_KB = 100       # 发送全景图目标大小（KB）；JPEG 优先保留分辨率，超限时缩小以满足100KB
SEND_VIDEO_MAX_MB = 5       # 发送视频大小上限（MB）
PANORAMA_FORMAT = "jpg"    # send_service/panoramas/YYYYMMDD/*.jpg
# 帧保存格式
FRAME_SAVE_FORMAT = "webp"
LOCAL_WEBP_QUALITY = 85     # 本地告警图像、录像帧和轨迹图 WebP 质量
# 视频合成参数
VIDEO_FPS = 10  # 视频帧率
# OpenCV H.265 候选：GStreamer MPP 失败后依次尝试；不回退到 H.264/mp4v。
VIDEO_CODEC_CANDIDATES = (
    ('H265', cv2.VideoWriter_fourcc(*'H265')),
    ('HEVC', cv2.VideoWriter_fourcc(*'HEVC')),
    ('hvc1', cv2.VideoWriter_fourcc(*'hvc1')),
    ('hev1', cv2.VideoWriter_fourcc(*'hev1')),
)
VIDEO_EXT = "mp4"
# 视频合成成功后保留的帧比例（默认每10张留1张作抽帧证据，其余删除释放磁盘；设为 0 = 全部删除）
FRAME_CLEANUP_KEEP_RATIO = 0.1
RECORD_SAVE_INTERVAL = 1.0 / VIDEO_FPS  # 录制帧保存间隔（秒），由 VIDEO_FPS 决定，保证视频实时播放
RAW_SAVE_INTERVAL = 10.0     # 原图保存间隔（秒），告警期间每路摄像头独立计时
# 告警录制配置
# ======================== 全局告警状态机配置 ========================
ALARM_STATE = False                # 任意一路录像中，仅用于工作时长判断
CAMERA_RECORD_STATES = {}
RECORD_DISTANCE = 50.0
PHOTO_DISTANCES = (200.0, 150.0, 100.0, 50.0)
ALARM_LAST_TRIGGER_TIME = None     # 最后一次任意一路触发告警的时间
ALARM_STOP_DELAY = 5.0             # 告警停止延迟（最后一路告警消失后，继续录5秒才停）


# ======================== 预录帧缓存配置 ========================
ENABLE_PRERECORD = False            # 开启预录功能（解决告警前几秒丢失）
PRERECORD_SECONDS = 3.0            # 预录时长（秒），建议3-5秒
PRERECORD_FPS = 10                 # 计算需要缓存的最大帧数（假设10fps）
# 每路摄像头的预录帧缓存：{camera_id: deque([(timestamp, frame), ...])}
prerecord_buffers = defaultdict(collections.deque)
PRERECORD_LOCK = threading.Lock()
_prerecord_last = {}

# 原有变量（兼容 create_alarm_record_dir 等函数）
RECORD_FRAMES = defaultdict(list)
RECORD_START_TIME = None

# ======================== 显示配置 ========================
ENABLE_DISPLAY = True

# ======================== AIS 融合配置 ========================
# AIS（船舶自动识别系统）通过串口/网络接收周围船只的实时位置广播
# 与本系统的图像检测融合，实现"图像检测 OR AIS距离<阈值"的联合告警
AIS_ENABLE = True                       # AIS 集成总开关（False=完全不加载AIS模块）
AIS_READ_INTERVAL = 0.2                 # AIS 读取尝试间隔（秒），可配置。AIS 数据更新频率本身较低（几秒一次），无需高频读取
AIS_DATA_SOURCE = 'SERIAL'             # 数据源：'FILE'（本地文件测试）, 'SERIAL'（串口）, 'NETWORK'（TCP）
SERIAL_ENABLE = True                    # 主控 RS232 收发总开关
SERIAL_RX_DEBUG = False                # 串口接收统计默认关闭，排查时可通过参数开启
SERIAL_RX_DEBUG_INTERVAL = 5.0         # 串口接收统计打印间隔（秒）
AIS_SERIAL_PORT = '/dev/ttyS8'         # 与主控通信的唯一 RS232 串口（收 AIS/定位，发状态/告警）
#AIS_SERIAL_BAUDRATE = 115200           # 与主控约定的波特率（默认 115200, 8-N-1）
AIS_SERIAL_BAUDRATE = 9600           # 与主控约定的波特率（默认 115200, 8-N-1）
AIS_SERIAL_BYTESIZE = 8
AIS_SERIAL_PARITY = 'N'
AIS_SERIAL_STOPBITS = 1
AIS_NETWORK_HOST = '192.168.1.10'      # 网络数据源主机
AIS_NETWORK_PORT = 8001                # 网络数据源端口
AIS_INPUT_FILE = '2点_ais.txt'          # FILE模式下的输入文件名（位于项目根目录）
AIS_BUOY_LAT = 36.333056  		# 浮标（本船/观测点）纬度
AIS_BUOY_LON = 120.716111 		# 浮标（本船/观测点）经度
GPS_ENABLE = True  # 从同一 RS232 接收主控转发的 POSITION/RMC/GGA/GLL
GPS_TIMEOUT = 30.0  # 兼容参数；最后一次有效 GPS 坐标不再因超时失效
GPS_LAST_POSITION_FILE = "/root/Desktop/pipeline/.last_gps_position.json"
GPS_PERSIST_INTERVAL = 5.0  # 内存每报实时更新；文件最多每5秒写一次，降低eMMC磨损
GPS_RX_DEBUG = True                    # GPS信息检查
GPS_RX_DEBUG_INTERVAL = 5.0
# 导入模块时不读写文件；主程序解析命令行参数后再启用持久化。
gps_position = GPSPosition(AIS_BUOY_LAT, AIS_BUOY_LON, GPS_TIMEOUT)


def get_buoy_position():
    return gps_position.position()[:2]


AIS_WARNING_DISTANCE = 200.0           # AIS 告警距离阈值（米），船只进入此范围触发告警
AIS_TRAJECTORY_ENABLE = False           # AIS 轨迹图独立开关，默认关闭以降低 CPU 和存储占用
AIS_MAX_PLOT_DISTANCE = 500.0          # 轨迹图最大绘制距离（米），只绘制此范围内的记录，防止远船把警戒圈缩得太小
AIS_ALARM_COOLDOWN = 300               # AIS 同一船只告警冷却时间（秒），5分钟内不重复报

# ======================== 单目测距配置 ========================
# 纯图像视觉单目测距：固定标准船体长度，靠检测框像素宽度 + 相机视场角估算距离
CAM_HFOV = 93.1        # 相机水平视场角（度）
CAM_VFOV = 49.6        # 相机垂直视场角（度）
SHIP_LENGTH = 15.0     # 固定标准船体长度（米）

# ======================== 主控 RS232 告警与状态发送 ========================
# 不再单独配置告警串口；发送与 AIS/定位接收共用 AIS_SERIAL_PORT。

# ======================== main.rs 发送程序配置 ========================
# main.rs 编译后的 stream_sender 二进制，与脚本同目录，定时/告警由唯一线程调用它单次发送
STREAM_SENDER_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stream_sender_linux_arm64")     # 二进制文件名（与脚本同目录）
STREAM_SENDER_IP = "192.168.1.51"       # F4 主控 IP（请按实际配置）
STREAM_SENDER_PORT = 7878               # F4 主控 stream_recv 端口

# 全景图片周期保存间隔（每路摄像头独立计时；告警期间也照常保存）
PANORAMA_SAVE_INTERVAL = 600  # 每 10 分钟；告警图另算，不重置周期
UPLOAD_PERIOD_SECONDS = 1800  # 每 30 分钟；使用同一个编译发送程序
AIS_TARGET_TIMEOUT = 120.0  # 本机接收后超过此秒数，目标不再参与活动告警
UPLOAD_RETRY_SECONDS = 30
UPLOAD_TIMEOUT = 600
UPLOAD_EXIT_GRACE = 15
upload_worker = None

# ======================== RS232 状态信息配置 ========================
# 开发板状态信息通过共用 RS232 周期发送给 F4 主控
DEVICE_ID = "receiver_01"          # 设备ID
STATE_SEND_INTERVAL = 30           # 状态信息发送间隔（秒）
serial_transport = None            # 程序启动后创建的唯一 RS232 双向连接
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
PROFILE_OUTPUT_PATH = os.path.join(AIS_DATA_ROOT, "perf_19_log.txt")  # 统计数据输出文件

# ======================== 发送服务保存辅助函数（额外留档） ========================
def _send_day_dir(subdir, day=None):
    """媒体先保存到 eMMC，当天数据由后台复制到 SD。"""
    return media_storage.day_dir(subdir, day=day)


def _new_send_path(subdir, camera_id, suffix):
    """Called under MEDIA_LOCK; wait for a real UTC second on name collision."""
    deadline = time.monotonic() + 3.0
    while True:
        stamp = _utc_file_stamp()
        folder = _send_day_dir(subdir, day=stamp[:8])
        path = os.path.join(folder, f"camera_{camera_id}_{stamp}.{suffix}")
        if not any(os.path.lexists(p) for p in (path, path + '.part', path + '.part.mp4')):
            return path
        if time.monotonic() >= deadline:
            raise FileExistsError('UTC filename collision; existing file preserved: %s' % path)
        time.sleep(0.05)


def _utc_file_stamp():
    """UTC 时间戳字符串，用于文件命名，如 20260901_082003"""
    return time.strftime('%Y%m%d_%H%M%S', time.gmtime())


def _save_image_limited(img, dest_path, max_kb):
    """将待发送全景图编码为 JPEG，写完后原子发布。

    优先降低 JPEG 质量并保留分辨率；最低质量仍超限时才缩小画面，
    确保发布的发送图不超过 max_kb KiB。
    """
    if img is None or max_kb <= 0:
        return None
    selected = None
    selected_quality = None
    for quality in (90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 1):
        ok, buf = cv2.imencode(
            '.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        if not ok:
            continue
        if selected is None or len(buf) < len(selected):
            selected, selected_quality = buf, quality
        if len(buf) <= max_kb * 1024:
            selected, selected_quality = buf, quality
            break
    if selected is None:
        raise IOError(f"OpenCV JPEG 编码失败: {dest_path}")
    if len(selected) > max_kb * 1024:
        # Diagram specifies 100KB. Resize only if quality=1 cannot meet it.
        reduced = img
        for attempt in range(20):
            h, w = reduced.shape[:2]
            reduced = cv2.resize(reduced, (max(2, int(w * 0.8)), max(2, int(h * 0.8))),
                                 interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode('.jpg', reduced, [cv2.IMWRITE_JPEG_QUALITY, 20])
            if ok and len(buf) <= max_kb * 1024:
                selected = buf
                print(f'[panorama] Reduced resolution to {reduced.shape[1]}x{reduced.shape[0]} '
                      f'to meet {max_kb}KB limit')
                break
        if len(selected) > max_kb * 1024:
            raise IOError(f'JPEG无法达到{max_kb}KB上限；未发布文件')
    tmp = dest_path + '.part'
    try:
        with open(tmp, 'wb') as f:
            f.write(selected.tobytes())
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, dest_path)
        worker = globals().get('upload_worker')
        if worker is not None:
            worker.register(dest_path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return dest_path


def _save_webp_image(img, dest_path, quality=LOCAL_WEBP_QUALITY):
    """将 OpenCV BGR 图像保存为 WebP；不改变原始分辨率。"""
    if img is None:
        return None
    quality = max(0, min(100, int(quality)))
    if not cv2.imwrite(dest_path, img, [cv2.IMWRITE_WEBP_QUALITY, quality]):
        raise IOError(f"WebP 图像保存失败: {dest_path}")
    return dest_path


def _compose_send_video(frame_paths, dest_path, fps, max_mb):
    """生成宽度不超过 640 的发送视频，验证后原子发布；失败保留全部原帧。"""
    if not frame_paths:
        return None
    sample = cv2.imread(frame_paths[0])
    if sample is None:
        return None
    h, w = sample.shape[:2]
    scale = min(640.0 / w, 1.0)
    sw, sh = max(2, int(w * scale) // 2 * 2), max(2, int(h * scale) // 2 * 2)
    tmp = dest_path + '.part.mp4'  # 不匹配上传文件命名规则
    writer = None
    check = None
    try:
        writer = open_record_writer(tmp, fps, (sw, sh))
        if writer is None:
            return None
        for fp in sorted(frame_paths):
            frame = cv2.imread(fp)
            if frame is None:
                raise IOError(f"录制帧不可读: {fp}")
            writer.write(cv2.resize(frame, (sw, sh)))
        writer.release()
        writer = None
        if not os.path.isfile(tmp) or os.path.getsize(tmp) == 0:
            raise IOError("合成视频为空")
        check = cv2.VideoCapture(tmp)
        decoded = 0
        while True:
            ok, frame = check.read()
            if not ok:
                break
            decoded += 1
        check.release()
        check = None
        if decoded != len(frame_paths):
            raise IOError(f"视频帧验证失败: {decoded}/{len(frame_paths)}")
        size_mb = os.path.getsize(tmp) / (1024.0 * 1024.0)
        if size_mb > max_mb:
            print(f"⚠️ 视频 {size_mb:.1f}MB 超过参考值 {max_mb}MB；上传超时由独立线程管理")
        os.replace(tmp, dest_path)
        worker = globals().get('upload_worker')
        if worker is not None:
            worker.register(dest_path)
        return dest_path
    except Exception as e:
        print(f"⚠️ 视频生成失败，保留原帧: {e}")
        return None
    finally:
        if writer is not None:
            writer.release()
        if check is not None:
            check.release()
        if os.path.exists(tmp):
            os.remove(tmp)


# ── 中文 → 英文映射（用于串口报文） ──
_ALARM_TYPE_EN = {"船舶入侵告警": "ship_intrusion"}
_ALARM_LEVEL_EN = {"一级": "level_1", "二级": "level_2", "三级": "level_3"}
_TRIGGER_SOURCE_EN = {"图像检测": "image_detection", "AIS距离": "AIS_distance", "图像检测+AIS距离": "image_AIS"}


def estimate_distance_by_bbox(bbox_width_px, image_width_px):
    """单目测距：根据检测框像素宽度 + 固定船长，估算船只到摄像头直线距离（米）。
    原理：相似三角形 D = L * f / w，焦距 f = (W/2) / tan(HFOV/2)。
    分辨率自适应：image_width_px 传入当前帧实际宽度，动态计算焦距。"""
    if bbox_width_px is None or bbox_width_px <= 0 or image_width_px <= 0:
        return None
    focal = (image_width_px / 2.0) / np.tan(np.radians(CAM_HFOV / 2.0))
    return SHIP_LENGTH * focal / bbox_width_px


def _to_ascii(text):
    """将文本转换为 ASCII（中文优先转拼音，否则 Unicode 转义），用于 RS232 传输。"""
    text = str(text or "")
    if not text:
        return ""
    try:
        from pypinyin import lazy_pinyin
        return "".join(lazy_pinyin(text))
    except ImportError:
        pass
    out = []
    for ch in text:
        if ord(ch) < 128:
            out.append(ch)
        else:
            out.append(f"\\u{ord(ch):04x}")
    return "".join(out)


def _nmea_checksum(body):
    """NMEA 校验和：$ 与 * 之间字符的异或，返回两位大写十六进制。"""
    c = 0
    for ch in body:
        c ^= ord(ch)
    return f"{c:02X}"


class SerialTransport:
    """唯一 RS232 连接：接收主控转发的 AIS/定位，并线程安全发送状态/告警。"""
    def __init__(self, line_handler=None):
        self._serial = None
        self._serial_lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._last_error = None
        self._line_handler = line_handler

    def set_line_handler(self, line_handler):
        """设置 AIS 行处理回调；串口所有权始终保留在本类。"""
        self._line_handler = line_handler

    def start(self):
        if not SERIAL_ENABLE or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="RS232-AIS-GPS-State")
        self._thread.start()

    def _open(self):
        import serial
        return serial.Serial(
            AIS_SERIAL_PORT, AIS_SERIAL_BAUDRATE,
            bytesize=AIS_SERIAL_BYTESIZE,
            parity=AIS_SERIAL_PARITY,
            stopbits=AIS_SERIAL_STOPBITS,
            timeout=0.2
        )

    def _dispatch_line(self, line):
        """按报文类型分流：定位交给 GPSPosition，AIVDM 交给 AISReaderThread。"""
        #print(f"[RS232 RX] {line!r}")   #打印接收的232信息
        line = line.strip()
        if not line:
            return

        # 允许主控在报文前附加时间戳或其他前缀。
        gps_start = line.find('$')
        if gps_start >= 0 and gps_position.feed(line[gps_start:]):
            return

        if not (AIS_ENABLE and AIS_DATA_SOURCE == 'SERIAL' and self._line_handler):
            return
        ais_start = line.find('!AIVDM')
        if ais_start < 0:
            return
        timestamp_match = re.match(r'^\[(\d+)\]', line)
        ais_line = line[ais_start:]
        if timestamp_match:
            ais_line = f"[{timestamp_match.group(1)}]{ais_line}"
        self._line_handler(ais_line)

    def _run(self):
        while not self._stop.is_set():
            try:
                with self._serial_lock:
                    self._serial = self._open()
                print(f"✅ 主控 RS232 双向串口已连接: {AIS_SERIAL_PORT} @ {AIS_SERIAL_BAUDRATE}bps")
                self._last_error = None
                pending = b''
                rx_bytes = 0
                rx_lines = 0
                last_rx_report = time.monotonic()
                while not self._stop.is_set():
                    with self._serial_lock:
                        data = self._serial.read(self._serial.in_waiting or 1)
                    if data:
                        rx_bytes += len(data)
                        pending += data
                        while b'\n' in pending:
                            raw, pending = pending.split(b'\n', 1)
                            rx_lines += 1
                            sentence = raw.decode('ascii', errors='ignore').strip()
                            self._dispatch_line(sentence)
                    now = time.monotonic()
                    if (SERIAL_RX_DEBUG
                            and now - last_rx_report >= SERIAL_RX_DEBUG_INTERVAL):
                        print(f"[RS232 RX] {SERIAL_RX_DEBUG_INTERVAL:.1f}s统计: "
                              f"bytes={rx_bytes}, lines={rx_lines}, pending={len(pending)}")
                        rx_bytes = 0
                        rx_lines = 0
                        last_rx_report = now
                    if len(pending) > 8192:
                        print(f"⚠️ [RS232 RX] 超过8192字节未收到换行符，已丢弃缓存")
                        pending = b''
            except Exception as exc:
                if self._last_error != str(exc):
                    print(f"⚠️ 主控 RS232 串口错误: {exc}，5秒后重试...")
                    self._last_error = str(exc)
                self._close_serial()
                self._stop.wait(5)
        self._close_serial()

    def send(self, sentence):
        if not SERIAL_ENABLE:
            return False
        try:
            with self._serial_lock:
                if self._serial is None or not self._serial.is_open:
                    raise RuntimeError("主控 RS232 串口尚未连接")
                self._serial.write(sentence.encode('ascii', 'replace'))
                self._serial.flush()
            return True
        except Exception as exc:
            print(f"⚠️ RS232 发送失败: {exc}")
            return False

    def _close_serial(self):
        with self._serial_lock:
            if self._serial is not None:
                try:
                    self._serial.close()
                except Exception:
                    pass
                self._serial = None

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._close_serial()


def _send_serial_sentence(body):
    """构建 $body*checksum 报文并通过共用 RS232 发送。"""
    if not SERIAL_ENABLE:
        return False
    sentence = f"${body}*{_nmea_checksum(body)}\r\n"
    if serial_transport is not None and serial_transport.send(sentence):
        print(f"📡 RS232 发送: {sentence.strip()}")
        return True
    return False


def send_alarm_signal(collector, status="active"):
    """通过共用 RS232 发送完整告警信息（14 字段 + 校验和）。
    status: 'active'（告警触发）或 'ended'（告警结束）。"""
    ais = collector.ais_data or {}
    alarm_id = collector.alarm_id
    alarm_type = _ALARM_TYPE_EN.get(collector.alarm_type, collector.alarm_type)
    alarm_level = _ALARM_LEVEL_EN.get(collector.alarm_level, collector.alarm_level)
    alarm_time = collector.start_time.strftime("%Y-%m-%d %H:%M:%S")
    trigger_source = _TRIGGER_SOURCE_EN.get(collector.alarm_trigger_source, collector.alarm_trigger_source)
    buoy_pos = (collector.buoy_pos or "").replace("°", "").replace(" ", "")
    mmsi = ais.get("mmsi", "0")
    ship_name = _to_ascii(ais.get("ship_name", ""))
    dist = ais.get("distance")
    if dist is None:
        dist = collector.image_distance
    distance = f"{dist}m" if dist is not None else "0m"
    target_lat = ais.get("lat", "")
    target_lon = ais.get("lon", "")
    alarm_status = status
    weather = collector.weather_label or "unknown"

    body = (f"ALARM,{alarm_id},{alarm_type},{alarm_level},{alarm_time},{trigger_source},"
            f"{buoy_pos},{mmsi},{ship_name},{distance},{target_lat},{target_lon},"
            f"{alarm_status},{weather}")
    return _send_serial_sentence(body)


# CPU 采样缓存（用于计算 CPU 使用率）
_last_cpu_idle = 0
_last_cpu_total = 0


def _get_system_status():
    """读取 CPU/内存/温度/磁盘使用率（返回百分比字符串，不含前缀）。"""
    global _last_cpu_idle, _last_cpu_total
    cpu, mem, temp, disk = "0.0", "0.0", "0.0", "0.0"
    try:
        with open('/proc/meminfo') as f:
            mi = {}
            for line in f:
                p = line.split()
                if len(p) >= 2:
                    mi[p[0].rstrip(':')] = int(p[1])
        total = mi.get('MemTotal', 1)
        avail = mi.get('MemAvailable', 0)
        mem = f"{100.0 * (1 - avail / total):.1f}"
    except Exception:
        pass
    try:
        with open('/sys/class/thermal/thermal_zone0/temp') as f:
            temp = f"{int(f.read()) / 1000.0:.1f}"
    except Exception:
        pass
    try:
        st = os.statvfs('/')
        disk = f"{100.0 * (1 - st.f_bavail / max(st.f_blocks, 1)):.1f}"
    except Exception:
        pass
    try:
        with open('/proc/stat') as f:
            parts = f.readline().split()
        idle = int(parts[4]) + int(parts[5])
        total = sum(int(x) for x in parts[1:])
        if _last_cpu_total > 0:
            d_idle = idle - _last_cpu_idle
            d_total = total - _last_cpu_total
            if d_total > 0:
                cpu = f"{100.0 * (1 - d_idle / d_total):.1f}"
        _last_cpu_idle = idle
        _last_cpu_total = total
    except Exception:
        pass
    return cpu, mem, temp, disk


def _get_sd_usage():
    """返回 SD 卡已用百分比（不含前缀）。

    SD 备份未启用、尚未初始化或卡已离线时返回 0.0，保持 STATE
    报文字段始终为可解析的数值格式。
    """
    try:
        storage = globals().get('media_storage')
        sd = getattr(storage, 'sd', None)
        if sd is None:
            return "0.0"
        sd.assert_ready()
        usage = shutil.disk_usage(sd.mount.point)
        return f"{100.0 * (1 - usage.free / max(usage.total, 1)):.1f}"
    except Exception:
        return "0.0"


def send_state_signal(camera_online_count):
    """通过共用 RS232 发送设备状态信息（9 字段 + 校验和）。"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    cpu, mem, temp, disk = _get_system_status()
    sd = _get_sd_usage()
    body = (f"STATE,{DEVICE_ID},{timestamp},"
            f"cpu{cpu},mem{mem},temp{temp},disk{disk},"
            f"cameras_{camera_online_count},sd{sd}")
    return _send_serial_sentence(body)


def invoke_stream_sender(path=None):
    """唤醒唯一上传线程；只发送两个媒体子目录，不发送告警/状态信息。"""
    if upload_worker is not None:
        upload_worker.notify(path)


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
        return any('GStreamer:' in line and line.split(':', 1)[1].strip().startswith('YES')
                   for line in cv2.getBuildInformation().splitlines())
    except Exception:
        return False




def _cache_prerecord(camera_id, frame):
    if not ENABLE_PRERECORD:
        return
    now = time.monotonic()
    with PRERECORD_LOCK:
        if now - _prerecord_last.get(camera_id, 0) < 1.0 / max(1, PRERECORD_FPS):
            return
        _prerecord_last[camera_id] = now
        buffer = prerecord_buffers[camera_id]
        buffer.append((now, frame.copy()))
        while buffer and (now - buffer[0][0] > PRERECORD_SECONDS or
                          len(buffer) > max(1, int(PRERECORD_SECONDS * PRERECORD_FPS))):
            buffer.popleft()


class AsyncVideoCapture(ProcessVideoCapture):
    def __init__(self, path, camera_id):
        super().__init__(path, camera_id, use_gstreamer=USE_GSTREAMER,
                         frame_timeout=FRAME_TIMEOUT,
                         reconnect_base=CAPTURE_RECONNECT_INTERVAL,
                         prerecord_callback=_cache_prerecord)


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
        self.last_transform = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        
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
        if transform is None:
            stabilized_frame = curr_frame.copy()
        else:
            # 直接使用“逆运动补偿”进行稳像：
            # 估计出当前帧相对上一帧的运动后，把当前帧按逆变换校正，避免把误差逐帧叠加成形变。
            alpha = 0.25
            smoothed_transform = np.array([
                [self.last_transform[0, 0] * (1.0 - alpha) + transform[0, 0] * alpha,
                 self.last_transform[0, 1] * (1.0 - alpha) + transform[0, 1] * alpha,
                 self.last_transform[0, 2] * (1.0 - alpha) + transform[0, 2] * alpha],
                [self.last_transform[1, 0] * (1.0 - alpha) + transform[1, 0] * alpha,
                 self.last_transform[1, 1] * (1.0 - alpha) + transform[1, 1] * alpha,
                 self.last_transform[1, 2] * (1.0 - alpha) + transform[1, 2] * alpha]
            ], dtype=np.float32)

            inv_transform = cv2.invertAffineTransform(smoothed_transform)
            stabilized_frame = self.apply_transform(curr_frame, inv_transform)
            self.last_transform = smoothed_transform

            # 保留历史信息，便于调试和后续扩展
            dx, dy, da = self.decompose_transform(smoothed_transform)
            self.trajectory.append([dx, dy, da])
            self.smoothed_trajectory = self.smooth_trajectory(self.trajectory)
        
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
    格式：roi_cameraID_时间戳_x1y1x2y2.webp
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
    def __init__(self, alarm_id, root_dir, camera_count, buoy_pos=None, ais_data=None):
        if buoy_pos is None:
            latitude, longitude = get_buoy_position()
            buoy_pos = f"{longitude}°, {latitude}°"
        self.alarm_id = alarm_id
        self.root_dir = root_dir               # 告警根目录（如 alarm_20260506_083045，仅存 raw + 合成视频帧）
        self.send_dir = None                   # 本地告警目录（RECORD_ROOT/alarm_id，不参与文件上传）
        self.camera_count = camera_count
        self.buoy_pos = buoy_pos               # 浮标位置（可从配置读取）
        self.alarm_level = "三级"               # 告警等级（默认最低等级，有AIS距离时动态调整）
        self.alarm_type = "船舶入侵告警"
        self.start_time = datetime.utcnow()  # UTC时间
        self.end_time = None
        self.weather_label = "unknown"
        self.weather_conf = 0.0
        self.ship_bbox = []                   # 告警船舶边界框（图像检测时填充，AIS单独触发时为空）
        self.ocr_text = None                    # 最佳舷号文本
        self.ship_image_paths = {}              # cam_id -> 船号图片路径
        self.bbox_image_path = None             # 船体框图像路径（代表帧）
        self.weather_image_path = None          # 天气图像路径
        self.video_paths = {}                   # cam_id -> 视频文件路径
        self.detection_json_path = None         # 检测结果JSON路径（可选）
        self.ais_data = ais_data or {}          # AIS 融合数据：{mmsi, lat, lon, distance, level, ship_name}（融合告警时填充）
        self.image_distance = None              # 图像单目测距估算的距离（米），纯图像检测时填充
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


class SerialAlarmManager:
    """图像/AIS 独立告警 + 重叠期融合通知；只在状态变化时发送。"""
    EVENT_LABELS = {
        "image": "图像告警",
        "ais": "AIS 告警",
        "fusion": "融合告警",
    }

    def __init__(self):
        self.image_states = {}
        self.image_distances = {}
        # 仅保存三个轻量状态记录，不创建额外线程，也不周期重复发送告警。
        self.events = {
            event_type: {
                "collector": None,
                "active_sent": False,
                "next_retry": 0.0,
            }
            for event_type in self.EVENT_LABELS
        }
        self.pending_ended = []
        self.pending_ended_next_retry = 0.0

    def _update_image_state(self, camera_id, detected, distance, now):
        active, last_trigger = self.image_states.get(camera_id, (False, now))
        if detected:
            active, last_trigger = True, now
            if distance is not None:
                self.image_distances[camera_id] = distance
        elif active and now - last_trigger > ALARM_STOP_DELAY:
            active = False
            self.image_distances.pop(camera_id, None)
        self.image_states[camera_id] = (active, last_trigger)

    @staticmethod
    def _ais_data(ais_closest):
        if not ais_closest:
            return {}
        return {
            "mmsi": ais_closest.get("mmsi"),
            "lat": ais_closest.get("lat"),
            "lon": ais_closest.get("lon"),
            "distance": ais_closest.get("distance"),
            "level": ais_closest.get("level"),
            "ship_name": ais_closest.get("ship_name"),
            "heading": ais_closest.get("heading"),
            "speed": ais_closest.get("speed"),
        }

    def _refresh_collector(self, collector, event_type, ais_closest,
                           weather_label, weather_conf):
        include_ais = event_type in ("ais", "fusion")
        ais_data = self._ais_data(ais_closest) if include_ais else {}
        image_distance = min(self.image_distances.values()) if self.image_distances else None

        collector.ais_data = ais_data
        collector.image_distance = (
            round(image_distance, 2)
            if event_type in ("image", "fusion") and image_distance is not None
            else None
        )
        if event_type == "fusion":
            collector.alarm_trigger_source = "图像检测+AIS距离"
        elif event_type == "ais":
            collector.alarm_trigger_source = "AIS距离"
        else:
            collector.alarm_trigger_source = "图像检测"

        level_map = {1: "一级", 2: "二级", 3: "三级"}
        ais_level = ais_data.get("level")
        if ais_level in level_map:
            collector.alarm_level = level_map[ais_level]
        elif collector.image_distance is not None:
            if image_distance <= 100:
                collector.alarm_level = "一级"
            elif image_distance <= 150:
                collector.alarm_level = "二级"
            else:
                collector.alarm_level = "三级"
        collector.set_weather(weather_label, weather_conf)

    def _start_or_refresh_event(self, event_type, should_be_active, ais_closest,
                                weather_label, weather_conf, now,
                                allow_active_send=True):
        event = self.events[event_type]
        collector = event["collector"]
        label = self.EVENT_LABELS[event_type]

        if should_be_active:
            if collector is None:
                alarm_id = (
                    f"alarm_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
                    f"_{event_type}"
                )
                collector = AlarmCollector(
                    alarm_id, _alarm_storage_root(), len(VIDEO_PATHS),
                    ais_data=self._ais_data(ais_closest)
                    if event_type in ("ais", "fusion") else {}
                )
                event["collector"] = collector
                event["active_sent"] = False
                event["next_retry"] = 0.0
                print(f"🚨 {label}已触发，准备发送 ALARM active")
                if event_type in ('image', 'ais'):
                    pending_photos.update(range(1, len(VIDEO_PATHS) + 1))
                    invoke_stream_sender()

            self._refresh_collector(
                collector, event_type, ais_closest, weather_label, weather_conf
            )
            if (allow_active_send and not event["active_sent"]
                    and now >= event["next_retry"]):
                event["active_sent"] = send_alarm_signal(collector, "active")
                if not event["active_sent"]:
                    event["next_retry"] = now + 1.0
            return event["active_sent"]

        if collector is not None:
            collector.end_time = datetime.utcnow()
            if event["active_sent"] and not send_alarm_signal(collector, "ended"):
                self.pending_ended.append(collector)
                self.pending_ended_next_retry = now + 1.0
            event["collector"] = None
            event["active_sent"] = False
            event["next_retry"] = 0.0
            print(f"✅ {label}已结束，已处理 ALARM ended")
        return False

    def _retry_pending_ended(self, now):
        if not self.pending_ended or now < self.pending_ended_next_retry:
            return
        remaining = []
        for collector in self.pending_ended:
            if not send_alarm_signal(collector, "ended"):
                remaining.append(collector)
        self.pending_ended = remaining
        if remaining:
            self.pending_ended_next_retry = now + 1.0
        else:
            self.pending_ended_next_retry = 0.0

    def update(self, camera_id, image_detected, image_distance,
               ais_has_alarm, ais_closest, weather_label, weather_conf):
        """分别维护图像、AIS 和融合告警；AIS active 保证先于融合 active。"""
        now = time.monotonic()
        self._retry_pending_ended(now)
        self._update_image_state(camera_id, image_detected, image_distance, now)
        image_active = any(active for active, _ in self.image_states.values())

        # 独立事件先发送：图像告警不携带 AIS，AIS 告警不依赖图像结束。
        image_sent = self._start_or_refresh_event(
            "image", image_active, ais_closest,
            weather_label, weather_conf, now
        )
        ais_sent = self._start_or_refresh_event(
            "ais", ais_has_alarm, ais_closest,
            weather_label, weather_conf, now
        )

        # 两类告警同时活动时另建融合事件。为保证接收端顺序明确，
        # 只有图像和 AIS 的独立 active 都已成功发送后，才发送融合 active。
        fusion_active = image_active and ais_has_alarm
        self._start_or_refresh_event(
            "fusion", fusion_active, ais_closest,
            weather_label, weather_conf, now,
            allow_active_send=(image_sent and ais_sent)
        )

    def stop(self):
        """退出前结束仍活动的串口告警。"""
        for event in self.events.values():
            collector = event["collector"]
            if collector is not None and event["active_sent"]:
                send_alarm_signal(collector, "ended")
            event["collector"] = None
            event["active_sent"] = False
        for collector in self.pending_ended:
            send_alarm_signal(collector, "ended")
        self.pending_ended.clear()


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
        """由 AIS 读取线程调用：更新单艘船的信息。
        位置信息每次全量更新（每条 position report 都有）；
        船名/航向/航速仅在新值非 None 时覆盖，避免被无该字段的报文冲掉旧数据。"""
        with self.lock:
            old = self.ships.get(mmsi, {})
            self.ships[mmsi] = {
                'lat': lat, 'lon': lon, 'distance': distance,
                'in_zone': in_zone, 'alarm_level': level,
                'last_seen': timestamp,
                'received_at': time.monotonic(),
                'heading': heading if heading is not None else old.get('heading'),
                'speed': speed if speed is not None else old.get('speed'),
                'ship_name': ship_name if ship_name is not None else old.get('ship_name'),
            }
            # 轨迹历史（限制长度，防止内存溢出）
            if mmsi not in self.track_history:
                self.track_history[mmsi] = []
            self.track_history[mmsi].append((lat, lon, timestamp))
            if len(self.track_history[mmsi]) > 10000:
                self.track_history[mmsi] = self.track_history[mmsi][-5000:]

    def update_alarm_status(self):
        """由 AIS 读取线程调用：重新计算告警状态"""
        latitude, longitude = get_buoy_position()
        with self.lock:
            # 本机 GPS 更新或超时回退时，即使船只未发新报文也重新测距。
            for info in self.ships.values():
                info['distance'] = _ais_calc_distance(latitude, longitude, info['lat'], info['lon'])
                info['alarm_level'] = AISReaderThread._get_alarm_level(info['distance'])
                info['in_zone'] = info['alarm_level'] is not None
            self.alarm_ships = [
                (mmsi, info['distance'], info['alarm_level'])
                for mmsi, info in self.ships.items()
                if info['in_zone'] and time.monotonic() - info['received_at'] <= AIS_TARGET_TIMEOUT
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
        self.update_alarm_status()  # 即使串口不再来报文也会使过期目标失效
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
    - 支持三种数据源：FILE（本地测试）、SERIAL（主控转发）、NETWORK（TCP）
    - SERIAL 模式不打开设备，由 SerialTransport 将 AIVDM 行分发到本类
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
        # 告警日志按日合并，延迟创建（有告警数据时才打开，避免空文件）
        os.makedirs(AIS_ALARM_DIR, exist_ok=True)
        self._alarm_log_file = None
        self._alarm_log_date = None   # 当前告警文件日期，跨天自动切换
        if AIS_DATA_SOURCE == 'SERIAL':
            print(f"✅ AIS 解析器已就绪（数据源=主控 RS232 转发, 告警距离<{AIS_WARNING_DISTANCE}m）")
        else:
            self.thread = threading.Thread(target=self._run, daemon=True, name="AIS-Reader")
            self.thread.start()
            print(f"✅ AIS 读取线程已启动（数据源={AIS_DATA_SOURCE}, 读取间隔={AIS_READ_INTERVAL}s, 告警距离<{AIS_WARNING_DISTANCE}m）")

    def _run(self):
        """根据配置选择数据源"""
        if AIS_DATA_SOURCE == 'FILE':
            self._run_file()
        elif AIS_DATA_SOURCE == 'SERIAL':
            return  # 串口由 SerialTransport 唯一持有
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
        """兼容入口：单串口架构下禁止在 AIS 线程重复打开串口。"""
        return

    def feed_serial_line(self, line):
        """接收 SerialTransport 分发的主控 AIVDM 报文。"""
        if self.running and AIS_DATA_SOURCE == 'SERIAL':
            self._process_line(line)

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

    def _ensure_alarm_log(self):
        """延迟创建告警日志文件（按日合并，避免每次启动产生空文件）。"""
        today = time.strftime('%Y%m%d')
        if self._alarm_log_file is not None and self._alarm_log_date == today:
            return  # 已有当天的文件
        # 跨天或首次打开：关闭旧文件，创建新文件
        if self._alarm_log_file is not None:
            self._alarm_log_file.close()
        os.makedirs(AIS_ALARM_DIR, exist_ok=True)
        filename = os.path.join(AIS_ALARM_DIR, f"ais_alarm_{today}.jsonl")
        self._alarm_log_file = open(filename, 'a', encoding='utf-8')
        self._alarm_log_date = today

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

        if lat is None or lon is None or mmsi is None or not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return

        # 使用 Haversine 公式计算与浮标的距离
        buoy_lat, buoy_lon = get_buoy_position()
        distance = _ais_calc_distance(buoy_lat, buoy_lon, lat, lon)

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

        # 告警时按需写入告警日志（延迟创建，按日合并）
        if in_zone:
            self._ensure_alarm_log()
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
    if not ALARM_ARCHIVE_ENABLE:
        return None
    send_dir = collector.send_dir or collector.root_dir
    ais = collector.ais_data
    # 从 AIS 数据或默认值填充目标船只信息
    mmsi = ais.get("mmsi", "unknown") if ais else "unknown"
    ship_name = ais.get("ship_name", "unknown") if ais else "unknown"
    dist_val = ais.get("distance") if ais else None
    if dist_val is None:
        dist_val = collector.image_distance
    distance_str = f"{dist_val}m" if dist_val is not None else "unknown"
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
            f"camera_{i+1}": (os.path.relpath(v, send_dir) if v else "")
            for i in range(collector.camera_count)
            for v in [collector.video_paths.get(i+1, "")]
        },
        "ship_images": {
            f"camera_{i}": (os.path.relpath(p, send_dir) if p else "")
            for i, p in collector.ship_image_paths.items()
        },
        "bbox_image": os.path.relpath(collector.bbox_image_path, send_dir) if collector.bbox_image_path else "",
        "weather_image": os.path.relpath(collector.weather_image_path, send_dir) if collector.weather_image_path else ""
    }
    json_path = os.path.join(send_dir, "alarm.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ 告警JSON已保存：{json_path}")
    return json_path


def save_ship_roi_image(frame, ship_bbox, ocr_text, alarm_root, cam_id):
    """裁剪船舶区域并保存为船号图像"""
    if not ALARM_ARCHIVE_ENABLE:
        return None
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

    os.makedirs(alarm_root, exist_ok=True)
    filename = f"ship_camera{cam_id}.webp"
    path = os.path.join(alarm_root, filename)
    _save_webp_image(crop, path)
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
                _save_webp_image(ship_roi, save_path, ROI_QUALITY)
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

# ======================== RKNN Lite 模型封装 ========================
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

    def __init__(self, model_path, model_type="det", input_size=640, names=None, conf_thresh=None):
        """
        :param names: 自定义类别映射字典，如 {0:"ship"}
        :param conf_thresh: 实例级置信度阈值（白天/夜晚模型可不同），None 时用类默认 CONF_THRESH
        """
        self.model_path = model_path
        self.model_type = model_type
        self.input_size = input_size
        self.conf_thresh = conf_thresh if conf_thresh is not None else self.CONF_THRESH
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
            self.conf_thresh, self.IOU_THRESH
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



# ======================== 加载 RKNN 模型 ========================
cls_model = YOLOv11RKNN(
    CLS_MODEL_PATH, model_type="cls", input_size=CLS_INPUT_SIZE,
    names={0: "cloudy", 1: "foggy", 2: "night", 3: "rainy", 4: "sunny"}
)
det_day_model = YOLOv11RKNN(
    DET_DAY_MODEL_PATH, model_type="det", input_size=DET_INPUT_SIZE,
    names={0: "ship"}, conf_thresh=0.2  # 白天船舶检测置信度阈值
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
        total_area = w * h
        area_ratio = area / total_area if total_area > 0 else 0.0

        if weather_cls == "night":
            is_alarm = area_ratio >= LIGHT_ALARM_THRESH
        else:
            is_alarm = area_ratio >= SHIP_ALARM_THRESH
        
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
        distance = estimate_distance_by_bbox(x2 - x1, w)
        distance_text = f"{distance:.1f}m" if distance is not None else "N/A"
        label_text = (f"{cls_name}({conf:.2f})| Area:{area}| "
                      f"Distance:{distance_text}| {alarm_text}")
        
        label_w, label_h = get_text_size(label_text, FONT_SCALE)
        label_height = label_h + 2 * PADDING
        label_width = label_w + 2 * PADDING
        
        label_tl_y = y1 - label_height - LABEL_OFFSET
        if label_tl_y < 0:
            label_tl_y = y2 + LABEL_OFFSET
        
        label_tl_y = max(0, min(label_tl_y, h - label_height - 5))
        label_tl_x = max(0, min(x1, w - label_width - 5))
        
        label_br_x = label_tl_x + label_width
        #label_br_y = label_tl_y + label_height - 2 * PADDING
        label_br_y = label_tl_y + label_height

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
    :param raw_frame: 触发告警那一路的原始稳像帧（无绘制，用于 weather.webp）
    :param draw_frame: 触发告警那一路的已绘制帧（带检测框，用于 bbox_frame.webp）
    :param trigger_cam_id: 触发告警的摄像头编号 (1-based)
    :param camera_count: 摄像头总数
    """
    # 仅本地归档开启时保存：weather.webp + bbox_frame.webp
    if not ALARM_ARCHIVE_ENABLE or raw_frame is None or draw_frame is None:
        return  # AIS 独立告警允许没有摄像头画面
    send_dir = collector.send_dir or collector.root_dir
    os.makedirs(send_dir, exist_ok=True)

    weather_path = os.path.join(send_dir, "weather.webp")
    _save_webp_image(raw_frame, weather_path)
    collector.set_weather_image(weather_path)
    print(f"✅ 天气代表帧已保存: {weather_path}")

    bbox_path = os.path.join(send_dir, "bbox_frame.webp")
    _save_webp_image(draw_frame, bbox_path)
    collector.set_bbox_image(bbox_path)
    print(f"✅ 船体框代表帧已保存: {bbox_path}")


def check_work_mode(has_alarm):
    global continuous_mode
    continuous_mode = bool(has_alarm)
    return continuous_mode


def check_work_duration():
    if continuous_mode or ALARM_STATE:
        return True
    return DEFAULT_WORK_DURATION <= 0 or time.monotonic() - start_work_time < DEFAULT_WORK_DURATION


# ======================== 录制相关核心函数 ========================
def _alarm_storage_root():
    """返回当前告警工作目录；本地归档关闭时绝不使用 RECORD_ROOT。"""
    return RECORD_ROOT if ALARM_ARCHIVE_ENABLE else ALARM_WORK_ROOT


def _record_storage_mount(path):
    """返回 path 所在文件系统的最近挂载点。"""
    current = Path(path).resolve()
    while current.parent != current and not os.path.ismount(current):
        current = current.parent
    return current


def _is_safe_completed_alarm_dir(path, root):
    """只允许删除 RECORD_ROOT 下已完成且不跨文件系统的告警目录。"""
    try:
        if path.parent != root or path.is_symlink() or os.path.ismount(path) or not path.is_dir():
            return False
        if not re.fullmatch(r'alarm_\d{8}_\d{6}(?:_\d{6})?(?:_camera_\d+)?', path.name):
            return False
        marker = path / 'alarm.json'
        if not marker.is_file() or marker.is_symlink():
            return False
        root_device = root.stat().st_dev
        mount_points = _read_mount_points()
        for folder, dirs, files in os.walk(path, followlinks=False):
            for name in dirs + files:
                item = Path(folder) / name
                if _unsafe_media_entry(item, root_device, mount_points):
                    return False
        return True
    except OSError:
        return False


def cleanup_alarm_recordings():
    """空间不足时按修改时间从旧到新删除已完成告警归档。

    未生成 alarm.json 的目录视为正在录像或未完成现场，绝不删除。
    返回本次删除的目录数。
    """
    if not ALARM_ARCHIVE_ENABLE:
        return 0
    root = Path(RECORD_ROOT).resolve()
    if not root.is_dir():
        return 0
    usage = shutil.disk_usage(root)
    reserve_bytes = int(usage.total * ALARM_STORAGE_RESERVE_PERCENT / 100.0)
    if usage.free >= reserve_bytes:
        return 0

    candidates = []
    for path in root.iterdir():
        if _is_safe_completed_alarm_dir(path, root):
            try:
                candidates.append((path.stat().st_mtime_ns, path))
            except OSError:
                continue
    candidates.sort(key=lambda item: (item[0], item[1].name))

    removed = 0
    for _, path in candidates:
        usage = shutil.disk_usage(root)
        if usage.free >= reserve_bytes:
            break
        # 删除前再次校验，避免扫描后目录状态发生变化。
        if not _is_safe_completed_alarm_dir(path, root):
            continue
        shutil.rmtree(path)
        removed += 1
        print(f"[告警存储] 空间不足，已删除最旧归档: {path}")

    usage = shutil.disk_usage(root)
    if usage.free < reserve_bytes:
        print(f"⚠️ [告警存储] 仍低于预留线 "
              f"{ALARM_STORAGE_RESERVE_PERCENT:.1f}%；没有更多可安全删除的完整归档")
    return removed


def init_recording_dirs():
    if ALARM_ARCHIVE_ENABLE:
        os.makedirs(RECORD_ROOT, exist_ok=True)
        mount = _record_storage_mount(RECORD_ROOT)
        usage = shutil.disk_usage(RECORD_ROOT)
        print(f"✅ 本地告警归档已开启：{RECORD_ROOT}")
        print(f"[告警存储] 挂载点={mount}，已用={100.0 * usage.used / max(usage.total, 1):.1f}%，"
              f"预留空间={ALARM_STORAGE_RESERVE_PERCENT:.1f}%")
        cleanup_alarm_recordings()
        return
    os.makedirs(ALARM_WORK_ROOT, exist_ok=True)
    print(f"ℹ️ 本地告警归档已关闭；视频合成临时目录：{ALARM_WORK_ROOT}")


def cleanup_alarm_work_dir(path):
    """归档关闭时安全删除一个已结束事件的临时工作目录。"""
    if ALARM_ARCHIVE_ENABLE or not path:
        return
    try:
        work_root = Path(ALARM_WORK_ROOT).resolve()
        target = Path(path).resolve()
        if str(target) in FAILED_ALARM_WORK_DIRS:
            print(f'[alarm] 视频合成失败，保留临时原帧供检查：{target}')
            return
        if (target.parent != work_root or target.is_symlink()
                or not target.name.startswith("alarm_") or not target.is_dir()):
            return
        shutil.rmtree(target)
        print(f"🧹 已清理告警临时素材：{target}")
    except OSError as exc:
        print(f"⚠️ 告警临时素材清理失败: {exc}")


def cleanup_alarm_work_root():
    """程序退出时清理本进程专用的告警临时根目录。"""
    if ALARM_ARCHIVE_ENABLE:
        return
    try:
        target = Path(ALARM_WORK_ROOT).resolve()
        temp_root = Path(tempfile.gettempdir()).resolve()
        if any(Path(p).parent == target and Path(p).is_dir() for p in FAILED_ALARM_WORK_DIRS):
            print(f'[alarm] 未完成视频的原帧保留于 {target}；不会在退出时删除')
            return
        if (target.parent == temp_root and target.name.startswith("pipeline_alarm_work_")
                and target.is_dir() and not target.is_symlink()):
            shutil.rmtree(target)
    except OSError as exc:
        print(f"⚠️ 告警临时目录清理失败: {exc}")

def create_alarm_record_dir():
    global RECORD_START_TIME
    cleanup_alarm_recordings()
    RECORD_START_TIME = datetime.now()
    time_str = RECORD_START_TIME.strftime("%Y%m%d_%H%M%S")
    alarm_root = os.path.join(_alarm_storage_root(), f"alarm_{time_str}")
    os.makedirs(alarm_root, exist_ok=True)
    for idx in range(len(VIDEO_PATHS)):
        camera_alarm_dir = os.path.join(alarm_root, f"camera_{idx+1}")
        os.makedirs(camera_alarm_dir, exist_ok=True)
    return alarm_root

# ======================== 全局告警状态机更新函数（融合 AIS） ========================
def update_alarm_state(camera_id, should_record):
    """各路独立启动/延迟停止，支持图像近距离或 AIS 告警触发。"""
    global ALARM_STATE
    now = time.monotonic()
    active, last_trigger = CAMERA_RECORD_STATES.get(camera_id, (False, now))
    previous = active
    if should_record:
        active, last_trigger = True, now
    elif active and now - last_trigger > ALARM_STOP_DELAY:
        active = False
    CAMERA_RECORD_STATES[camera_id] = (active, last_trigger)
    ALARM_STATE = any(state[0] for state in CAMERA_RECORD_STATES.values())
    return active and not previous, active, active != previous


class ApproachPhotos:
    """每个目标一次接近只触发每个距离档一次；离开或超时后重置。"""
    def __init__(self):
        self.targets = {}

    def update(self, distances, now):
        triggered = set()
        for key in list(self.targets):
            if now - self.targets[key][1] > AIS_TARGET_TIMEOUT:
                del self.targets[key]
        for key, distance in distances.items():
            if distance is None or not np.isfinite(distance) or distance < 0:
                continue
            if distance > PHOTO_DISTANCES[0]:
                self.targets.pop(key, None)
                continue
            seen, _ = self.targets.get(key, (set(), now))
            crossed = {d for d in PHOTO_DISTANCES if distance <= d} - seen
            # 距离跳档时只拍当前真实画面，不伪造已错过距离的照片。
            if crossed:
                triggered.add(min(crossed))
            self.targets[key] = (seen | crossed, now)
        return triggered


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
        alarm_root = os.path.join(
            _alarm_storage_root(),
            f"alarm_{RECORD_START_TIME.strftime('%Y%m%d_%H%M%S')}"
        )

    print(f"💾 写入预录缓存（最近{PRERECORD_SECONDS}秒）...")
    # 遍历所有摄像头，写入预录帧
    for cid in range(1, len(VIDEO_PATHS)+1):
        if cid in prerecord_buffers:
            count = 0
            with PRERECORD_LOCK:
                snapshot = list(prerecord_buffers[cid])
            for (ts, prerec_frame) in snapshot:
                save_frame_for_recording(prerec_frame, cid, alarm_root)
                count += 1
            if count > 0:
                print(f"  摄像头{cid}: 写入{count}帧预录数据")

def save_frame_for_recording(frame, camera_id, alarm_root):
    """保存绘制后帧（带检测框+OCR），用于合成告警视频"""
    frame_idx = len(RECORD_FRAMES[camera_id]) + 1
    frame_filename = f"frame_{frame_idx:05d}.{FRAME_SAVE_FORMAT}"
    frame_path = os.path.join(alarm_root, f"camera_{camera_id}", frame_filename)
    _save_webp_image(frame, frame_path)
    RECORD_FRAMES[camera_id].append(frame_path)
    return frame_path


def save_raw_frame_for_recording(frame, camera_id, alarm_root):
    """保存原始帧（无绘制），按时间戳命名，不参与视频合成"""
    if not ALARM_ARCHIVE_ENABLE:
        return None
    ts = datetime.now().strftime('%H%M%S_%f')[:9]  # HHMMSS_ms
    raw_filename = f"raw_{ts}.{FRAME_SAVE_FORMAT}"
    raw_dir = os.path.join(alarm_root, f"camera_{camera_id}", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    raw_path = os.path.join(raw_dir, raw_filename)
    _save_webp_image(frame, raw_path)
    return raw_path


def _open_mpp_h265_writer(video_path, fps, size):
    """使用 RK3576 GStreamer MPP 硬编 H.265，不可用时返回 None。"""
    if not opencv_has_gstreamer():
        return None
    width, height = size
    fps_num = max(1, int(round(float(fps) * 1000)))
    safe_path = str(video_path).replace('\\', '\\\\').replace('"', '\\"')
    pipeline = (
        'appsrc is-live=false format=time ! '
        f'video/x-raw,format=BGR,width={width},height={height},framerate={fps_num}/1000 ! '
        'videoconvert ! video/x-raw,format=NV12 ! '
        'mpph265enc ! h265parse ! mp4mux ! '
        f'filesink location="{safe_path}" sync=false'
    )
    try:
        writer = cv2.VideoWriter(
            pipeline, cv2.CAP_GSTREAMER, 0, float(fps), size, True
        )
        if writer.isOpened():
            print("✅ 视频编码器: RK3576 MPP H.265 (mpph265enc)")
            return writer
        writer.release()
    except Exception as exc:
        print(f"⚠️ MPP H.265 编码器打开失败: {exc}")
    return None


def open_record_writer(video_path, fps, size):
    """优先尝试 MPP H.265，再尝试 OpenCV HEVC FourCC；全部失败返回 None。"""
    writer = _open_mpp_h265_writer(video_path, fps, size)
    if writer is not None:
        return writer
    for codec_name, fourcc in VIDEO_CODEC_CANDIDATES:
        writer = cv2.VideoWriter(video_path, fourcc, fps, size)
        if writer.isOpened():
            print(f"✅ 视频编码器: OpenCV {codec_name} (H.265/HEVC)")
            return writer
        writer.release()
    return None


def cleanup_recorded_frames(camera_id):
    """视频合成成功后清理 WebP 帧序列：按 FRAME_CLEANUP_KEEP_RATIO 抽帧保留，其余删除。
    仅做列表切片 + os.remove（元数据操作），CPU开销可忽略；raw原图和代表帧不受影响。"""
    frame_paths = sorted(RECORD_FRAMES.get(camera_id, []))
    if not frame_paths:
        return 0, 0
    if FRAME_CLEANUP_KEEP_RATIO <= 0:
        keep = set()
    else:
        step = max(1, round(1.0 / FRAME_CLEANUP_KEEP_RATIO))
        keep = set(frame_paths[::step])
    removed = 0
    for fp in frame_paths:
        if fp in keep:
            continue
        try:
            os.remove(fp)
            removed += 1
        except OSError:
            pass
    return removed, len(keep)


@media_write
def compose_alarm_videos(collector, only_camera=None):
    print("\n📹 合成告警视频...")
    # 只保存一份发送版视频到发送服务目录（UTC 命名 + 缩放控 5MB），不再额外存原始分辨率视频
    # Date is frozen per file so a UTC midnight cannot mismatch its directory.
    for camera_id, frame_paths in RECORD_FRAMES.items():
        if only_camera is not None and camera_id != only_camera:
            continue
        if not frame_paths:
            continue
        try:
            send_video_path = _new_send_path('rtmp_recordings', camera_id, 'mp4')
        except FileExistsError as exc:
            FAILED_ALARM_WORK_DIRS.add(str(Path(collector.root_dir).resolve()))
            print(f'[video] {exc}; raw frames retained')
            continue
        saved = _compose_send_video(frame_paths, send_video_path, VIDEO_FPS, SEND_VIDEO_MAX_MB)
        if saved is None:
            if not ALARM_ARCHIVE_ENABLE:
                FAILED_ALARM_WORK_DIRS.add(str(Path(collector.root_dir).resolve()))
            # 编码失败时保留完整原帧，禁止后续事件/退出清理覆盖这一决定
            print(f"  ⚠️ 摄像头{camera_id} 视频编码器打开失败，保留全部帧")
            continue
        collector.add_video(camera_id, saved)
        removed, kept = cleanup_recorded_frames(camera_id)
        print(f"  摄像头{camera_id} -> {saved}（帧清理：删{removed}张/留{kept}张）")

def create_alarm_record_dir_with_id(alarm_id):
    cleanup_alarm_recordings()
    alarm_root = os.path.join(_alarm_storage_root(), alarm_id)
    for idx in range(len(VIDEO_PATHS)):
        os.makedirs(os.path.join(alarm_root, f"camera_{idx+1}"), exist_ok=True)
    return alarm_root

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
    if not AIS_TRAJECTORY_ENABLE:
        return None

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
    buoy_lat, buoy_lon = get_buoy_position()
    tracks_xy = {}
    total_points = 0
    for mmsi, pts in track_history.items():
        xy_pts = []
        for lat, lon, _ in pts:
            if lat is not None and lon is not None and abs(lat) > 0.01:
                x, y = _latlon_to_xy(lat, lon, buoy_lat, buoy_lon)
                # 只保留在最大绘制距离内的点，过滤掉过远的记录
                if (x * x + y * y) <= AIS_MAX_PLOT_DISTANCE * AIS_MAX_PLOT_DISTANCE:
                    xy_pts.append((x, y))
        if len(xy_pts) >= 1:
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
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
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
    filename = f"ais_tracks_{timestamp}.webp"
    filepath = os.path.join(output_dir, filename)
    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    _save_webp_image(bgr, filepath)
    plt.close(fig)
    print(f"✅ AIS 轨迹图已保存: {filepath}")
    return filepath


@media_write
def save_panorama(frame, camera_id, alarm=False):
    """保存 JPG；只有告警图立即唤醒上传，周期图等待半小时定时器。"""
    if frame is None:
        return
    try:
        send_path = _new_send_path('panoramas', camera_id, PANORAMA_FORMAT)
        if _save_image_limited(frame, send_path, SEND_IMG_MAX_KB):
            if alarm:
                invoke_stream_sender(send_path)
            return True
    except Exception as e:
        print(f"⚠️ 全景图保存失败: {e}")
    return False


# ======================== 主程序 ========================
if __name__ == "__main__":
    # ======================== 命令行参数解析 ========================
    parser = argparse.ArgumentParser(
        description='Ship Detection Pipeline — 船舶检测管线',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python 19-2_ais_data_saved.py --AIS_MODE SERIAL --AIS_BUOY_LAT 19.564408 --AIS_BUOY_LON 110.822627
  python 19-2_ais_data_saved.py --AIS_MODE FILE --AIS_WARNING_DISTANCE 150.0
  python 19-2_ais_data_saved.py --DISPLAY 0
        """)
    # ── AIS 参数 ──
    ais_grp = parser.add_argument_group('AIS 船舶自动识别系统')
    ais_grp.add_argument('--AIS_ENABLE', type=int, default=AIS_ENABLE,
                         help=f'AIS 集成总开关（0=禁用, 1=启用），默认 {AIS_ENABLE}')
    ais_grp.add_argument('--AIS_MODE', type=str, default=AIS_DATA_SOURCE,
                         choices=['FILE', 'SERIAL', 'NETWORK'],
                         help=f'数据源类型，默认 {AIS_DATA_SOURCE}')
    ais_grp.add_argument('--SERIAL_PORT', '--AIS_SERIAL_PORT', '--ALARM_SERIAL_PORT',
                         dest='SERIAL_PORT', type=str, default=AIS_SERIAL_PORT,
                         help=f'与主控通信的唯一 RS232 设备；旧参数名仍兼容，默认 {AIS_SERIAL_PORT}')
    ais_grp.add_argument('--SERIAL_BAUDRATE', '--AIS_BAUDRATE', '--ALARM_SERIAL_BAUDRATE',
                         dest='SERIAL_BAUDRATE', type=int, default=AIS_SERIAL_BAUDRATE,
                         help=f'共用 RS232 波特率；旧参数名仍兼容，默认 {AIS_SERIAL_BAUDRATE}')
    ais_grp.add_argument('--AIS_NETWORK_HOST', type=str, default=AIS_NETWORK_HOST,
                         help=f'TCP 数据源主机，默认 {AIS_NETWORK_HOST}')
    ais_grp.add_argument('--AIS_NETWORK_PORT', type=int, default=AIS_NETWORK_PORT,
                         help=f'TCP 数据源端口，默认 {AIS_NETWORK_PORT}')
    ais_grp.add_argument('--AIS_INPUT_FILE', type=str, default=AIS_INPUT_FILE,
                         help=f'FILE 模式输入文件名，默认 {AIS_INPUT_FILE}')
    ais_grp.add_argument('--AIS_BUOY_LAT', type=float, default=AIS_BUOY_LAT,
                         help=f'本船/浮标纬度，默认 {AIS_BUOY_LAT}')
    ais_grp.add_argument('--AIS_BUOY_LON', type=float, default=AIS_BUOY_LON,
                         help=f'本船/浮标经度，默认 {AIS_BUOY_LON}')
    ais_grp.add_argument('--AIS_WARNING_DISTANCE', type=float, default=AIS_WARNING_DISTANCE,
                         help=f'告警距离阈值（米），默认 {AIS_WARNING_DISTANCE}')
    ais_grp.add_argument('--AIS_COOLDOWN', type=int, default=AIS_ALARM_COOLDOWN,
                         help=f'同一船只冷却时间（秒），默认 {AIS_ALARM_COOLDOWN}')
    ais_grp.add_argument('--AIS_TRAJECTORY_ENABLE', type=int, choices=[0, 1],
                         default=int(AIS_TRAJECTORY_ENABLE),
                         help=f'AIS轨迹图开关（0=关闭，1=开启），默认 {int(AIS_TRAJECTORY_ENABLE)}')
    ais_grp.add_argument('--AIS_MAX_PLOT_DISTANCE', type=float, default=AIS_MAX_PLOT_DISTANCE,
                         help=f'轨迹图最大绘制距离（米），默认 {AIS_MAX_PLOT_DISTANCE}')
    # ── 共用 RS232 状态与告警信号 ──
    alarm_grp = parser.add_argument_group('主控 RS232 状态与告警')
    alarm_grp.add_argument('--SERIAL_ENABLE', '--ALARM_SERIAL_ENABLE',
                           dest='SERIAL_ENABLE', type=int, choices=[0, 1],
                           default=int(SERIAL_ENABLE),
                           help=f'主控 RS232 收发开关（0=禁用, 1=启用），默认 {int(SERIAL_ENABLE)}')
    alarm_grp.add_argument('--DEVICE_ID', type=str, default=DEVICE_ID,
                           help=f'设备ID，默认 {DEVICE_ID}')
    alarm_grp.add_argument('--STATE_SEND_INTERVAL', type=int, default=STATE_SEND_INTERVAL,
                           help=f'状态信息发送间隔（秒），默认 {STATE_SEND_INTERVAL}')
    alarm_grp.add_argument('--SERIAL_RX_DEBUG', type=int, choices=[0, 1],
                           default=int(SERIAL_RX_DEBUG),
                           help=f'串口接收统计日志（0=关闭，1=开启），默认 {int(SERIAL_RX_DEBUG)}')
    alarm_grp.add_argument('--SERIAL_RX_DEBUG_INTERVAL', type=float,
                           default=SERIAL_RX_DEBUG_INTERVAL,
                           help=f'串口接收统计间隔（秒），默认 {SERIAL_RX_DEBUG_INTERVAL}')
    # ── main.rs 发送程序 ──
    sender_grp = parser.add_argument_group('main.rs 发送程序')
    sender_grp.add_argument('--STREAM_SENDER_BIN', type=str, default=STREAM_SENDER_BIN,
                            help=f'stream_sender 二进制名，默认 {STREAM_SENDER_BIN}')
    sender_grp.add_argument('--STREAM_SENDER_IP', type=str, default=STREAM_SENDER_IP,
                            help=f'F4 主控 IP，默认 {STREAM_SENDER_IP}')
    sender_grp.add_argument('--STREAM_SENDER_PORT', type=int, default=STREAM_SENDER_PORT,
                            help=f'F4 主控端口，默认 {STREAM_SENDER_PORT}')
    sender_grp.add_argument('--UPLOAD_PERIOD_SECONDS', type=int, default=UPLOAD_PERIOD_SECONDS,
                            help='媒体发送周期（秒），默认1800；告警立即发送')
    sender_grp.add_argument('--PANORAMA_SAVE_INTERVAL', type=int, default=PANORAMA_SAVE_INTERVAL,
                            help=f'全景图保存间隔（秒），默认 {PANORAMA_SAVE_INTERVAL}')
    # ── 显示与性能 ──
    disp_grp = parser.add_argument_group('显示与性能')
    disp_grp.add_argument('--DISPLAY', type=int, default=ENABLE_DISPLAY,
                          help=f'显示开关（0=禁用, 1=启用），默认 {ENABLE_DISPLAY}')
    disp_grp.add_argument('--STABILIZATION', type=int, default=ENABLE_STABILIZATION,
                          help=f'稳像开关（0=禁用, 1=启用），默认 {ENABLE_STABILIZATION}')
    disp_grp.add_argument('--PRERECORD', type=int, default=ENABLE_PRERECORD,
                          help=f'预录开关（0=禁用, 1=启用），默认 {ENABLE_PRERECORD}')
    disp_grp.add_argument('--PROFILE', type=int, default=PROFILE_ENABLE,
                          help=f'性能统计开关（0=禁用, 1=启用），默认 {PROFILE_ENABLE}')
    disp_grp.add_argument('--WORK_DURATION', type=int, default=DEFAULT_WORK_DURATION,
                          help=f'工作时间（秒，0=持续运行），默认 {DEFAULT_WORK_DURATION}')
    # ── 解析 ──
    parser.add_argument('--GPS_ENABLE', type=int, choices=[0, 1], default=1,
                        help='从主控共用 RS232 读取 POSITION/NMEA 定位，默认开启；无定位时回退配置坐标')
    parser.add_argument('--GPS_TIMEOUT', type=float, default=GPS_TIMEOUT,
                        help='GPS旧版兼容参数；当前版本始终保留最后一次有效坐标')
    parser.add_argument('--GPS_LAST_POSITION_FILE', type=str,
                        default=GPS_LAST_POSITION_FILE,
                        help=f'最后一次有效GPS坐标持久化文件，默认 {GPS_LAST_POSITION_FILE}')
    parser.add_argument('--GPS_PERSIST_INTERVAL', type=float,
                        default=GPS_PERSIST_INTERVAL,
                        help=f'GPS坐标文件最小写入间隔（秒），默认 {GPS_PERSIST_INTERVAL}')
    parser.add_argument('--GPS_RX_DEBUG', type=int, choices=[0, 1],
                        default=int(GPS_RX_DEBUG),
                        help=f'GPS解析诊断日志（0=关闭，1=开启），默认 {int(GPS_RX_DEBUG)}')
    parser.add_argument('--GPS_RX_DEBUG_INTERVAL', type=float,
                        default=GPS_RX_DEBUG_INTERVAL,
                        help=f'GPS有效/无效统计间隔（秒），默认 {GPS_RX_DEBUG_INTERVAL}')
    parser.add_argument('--SD_BACKUP_ENABLE', type=int, choices=[0, 1],
                        default=int(SD_BACKUP_ENABLE),
                        help='SD卡备份开关（0=关闭，1=开启），默认 1；关闭时不检测或挂载SD卡')
    parser.add_argument('--ALARM_ARCHIVE_ENABLE', type=int, choices=[0, 1],
                        default=int(ALARM_ARCHIVE_ENABLE),
                        help=f'alarm_recordings本地归档开关（0=关闭，1=开启），'
                             f'默认 {int(ALARM_ARCHIVE_ENABLE)}')
    parser.add_argument('--ALARM_STORAGE_RESERVE_PERCENT', type=float,
                        default=ALARM_STORAGE_RESERVE_PERCENT,
                        help=f'/home 分区为 alarm_recordings 循环清理预留的空间百分比，'
                             f'默认 {ALARM_STORAGE_RESERVE_PERCENT:.1f}')
    args = parser.parse_args()
    if args.PANORAMA_SAVE_INTERVAL <= 0 or args.UPLOAD_PERIOD_SECONDS <= 0:
        parser.error('全景保存和上传周期必须大于0')
    if args.WORK_DURATION < 0:
        parser.error('--WORK_DURATION 必须大于等于0，0为持续运行')
    if args.UPLOAD_PERIOD_SECONDS < 3 * args.PANORAMA_SAVE_INTERVAL:
        parser.error('上传周期须至少为保存周期的3倍，以提供每路3张周期图')
    if not 0 < args.ALARM_STORAGE_RESERVE_PERCENT < 100:
        parser.error('--ALARM_STORAGE_RESERVE_PERCENT 必须大于 0 且小于 100')
    ALARM_STORAGE_RESERVE_PERCENT = args.ALARM_STORAGE_RESERVE_PERCENT
    ALARM_ARCHIVE_ENABLE = bool(args.ALARM_ARCHIVE_ENABLE)
    SD_BACKUP_ENABLE = bool(args.SD_BACKUP_ENABLE)
    media_storage.backup_enabled = SD_BACKUP_ENABLE
    media_storage.prepare()
    if args.GPS_TIMEOUT <= 0:
        parser.error('--GPS_TIMEOUT 必须大于 0')
    if args.GPS_PERSIST_INTERVAL <= 0:
        parser.error('--GPS_PERSIST_INTERVAL 必须大于 0')
    if args.GPS_RX_DEBUG_INTERVAL <= 0 or args.SERIAL_RX_DEBUG_INTERVAL <= 0:
        parser.error('GPS/串口诊断日志间隔必须大于 0')
    if not (-90 <= args.AIS_BUOY_LAT <= 90 and -180 <= args.AIS_BUOY_LON <= 180):
        parser.error('回退经纬度超出有效范围')

    # ── 应用参数到全局配置 ──
    AIS_ENABLE = bool(args.AIS_ENABLE)
    AIS_DATA_SOURCE = args.AIS_MODE
    AIS_SERIAL_PORT = args.SERIAL_PORT
    AIS_SERIAL_BAUDRATE = args.SERIAL_BAUDRATE
    AIS_NETWORK_HOST = args.AIS_NETWORK_HOST
    AIS_NETWORK_PORT = args.AIS_NETWORK_PORT
    AIS_INPUT_FILE = args.AIS_INPUT_FILE
    AIS_BUOY_LAT = args.AIS_BUOY_LAT
    AIS_BUOY_LON = args.AIS_BUOY_LON
    GPS_ENABLE = bool(args.GPS_ENABLE)
    GPS_TIMEOUT = args.GPS_TIMEOUT
    GPS_LAST_POSITION_FILE = args.GPS_LAST_POSITION_FILE
    GPS_PERSIST_INTERVAL = args.GPS_PERSIST_INTERVAL
    GPS_RX_DEBUG = bool(args.GPS_RX_DEBUG)
    GPS_RX_DEBUG_INTERVAL = args.GPS_RX_DEBUG_INTERVAL
    gps_position = GPSPosition(
        AIS_BUOY_LAT, AIS_BUOY_LON, GPS_TIMEOUT, GPS_ENABLE,
        state_file=GPS_LAST_POSITION_FILE,
        persist_interval=GPS_PERSIST_INTERVAL,
        debug=GPS_RX_DEBUG,
        debug_interval=GPS_RX_DEBUG_INTERVAL,
    )
    AIS_WARNING_DISTANCE = args.AIS_WARNING_DISTANCE
    AIS_TRAJECTORY_ENABLE = bool(args.AIS_TRAJECTORY_ENABLE)
    AIS_MAX_PLOT_DISTANCE = args.AIS_MAX_PLOT_DISTANCE
    AIS_ALARM_COOLDOWN = args.AIS_COOLDOWN
    SERIAL_ENABLE = bool(args.SERIAL_ENABLE)
    SERIAL_RX_DEBUG = bool(args.SERIAL_RX_DEBUG)
    SERIAL_RX_DEBUG_INTERVAL = args.SERIAL_RX_DEBUG_INTERVAL
    DEVICE_ID = args.DEVICE_ID
    STATE_SEND_INTERVAL = args.STATE_SEND_INTERVAL
    STREAM_SENDER_BIN = args.STREAM_SENDER_BIN
    if not os.path.isabs(STREAM_SENDER_BIN):
        STREAM_SENDER_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), STREAM_SENDER_BIN)
    STREAM_SENDER_IP = args.STREAM_SENDER_IP
    STREAM_SENDER_PORT = args.STREAM_SENDER_PORT
    PANORAMA_SAVE_INTERVAL = args.PANORAMA_SAVE_INTERVAL
    UPLOAD_PERIOD_SECONDS = args.UPLOAD_PERIOD_SECONDS
    ENABLE_DISPLAY = bool(args.DISPLAY)
    ENABLE_STABILIZATION = bool(args.STABILIZATION)
    ENABLE_PRERECORD = bool(args.PRERECORD)
    PROFILE_ENABLE = bool(args.PROFILE)
    DEFAULT_WORK_DURATION = args.WORK_DURATION

    # ── 打印当前运行配置 ──
    print(f"📋 运行配置: AIS_MODE={AIS_DATA_SOURCE}  SERIAL={AIS_SERIAL_PORT}@{AIS_SERIAL_BAUDRATE}  "
          f"BUOY=({AIS_BUOY_LAT},{AIS_BUOY_LON})  "
          f"WARNING={AIS_WARNING_DISTANCE}m  AIS_TRAJECTORY={AIS_TRAJECTORY_ENABLE}  "
          f"DISPLAY={ENABLE_DISPLAY}  "
          f"SD_BACKUP={SD_BACKUP_ENABLE}  ALARM_ARCHIVE={ALARM_ARCHIVE_ENABLE}  "
          f"RX_DEBUG={SERIAL_RX_DEBUG}/{GPS_RX_DEBUG}  "
          f"PANORAMA={PANORAMA_SAVE_INTERVAL}s UPLOAD={UPLOAD_PERIOD_SECONDS}s "
          f"WORK_DURATION={DEFAULT_WORK_DURATION}s (0=持续)")

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
    camera_alarm_flags = {}
    alarm_collectors = {}
    approach_photos = ApproachPhotos()
    pending_photos = set()

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

    # Startup first frame + 10/20 minute frames provide three images in first batch.
    # Advance fixed monotonic deadlines; alarms do not postpone periodic captures.
    next_panorama_time = defaultdict(time.monotonic)
    camera_generations = defaultdict(int)

    # ========== 告警录制节流计时（每路摄像头独立） ==========
    last_record_save = defaultdict(float)  # camera_id → 上次保存绘制帧的时间戳
    last_raw_save = defaultdict(float)     # camera_id → 上次保存原图的时间戳

    # ========== 帧率控制 ==========
    last_process_time = time.perf_counter()

    # ========== RS232 状态信息定时发送 ==========
    last_state_send = time.time()

    # 告警状态变量
    alarm_collector = None             # 当前告警收集器
    prev_alarm_state = False           # 用于检测告警状态变化
    serial_alarm_manager = SerialAlarmManager()

    # ── AIS 融合初始化 ──
    ais_state = AISSharedState()
    ais_reader = AISReaderThread(ais_state)
    ais_reader.start()
    serial_transport = SerialTransport(ais_reader.feed_serial_line)
    serial_transport.start()
    ais_last_trajectory_save = time.time()  # AIS 轨迹图上次保存时间

    perf = PerfTimer(enabled=PROFILE_ENABLE, output_path=PROFILE_OUTPUT_PATH)
    start_work_time = time.monotonic()  # 模型/摄像头初始化不计入采集时长
    upload_worker = MediaUploader(
        SEND_ROOT, STREAM_SENDER_BIN, STREAM_SENDER_IP, STREAM_SENDER_PORT,
        state_dir="/root/Desktop/pipeline/.stream_upload_state_23",
        retry_seconds=UPLOAD_RETRY_SECONDS, timeout=UPLOAD_TIMEOUT,
        period_seconds=UPLOAD_PERIOD_SECONDS,
    )
    media_storage.uploader = upload_worker

    try:
        media_storage.start()
        upload_worker.start()
        while True:
            perf.tick()

            # ======================== 1. 单路轮询检测 ========================
            idx = current_cam_idx
            cap = caps[idx]
            stabilizer = stabilizers[idx]
            camera_id = idx + 1
            current_has_alarm = False

            stabilized_frame = None
            draw_frame = None
            if cap is not None and not cap.is_alive():
                frame_buffer.pop(idx, None)  # do not display a disconnected stream as live
            post_results = []
            ocr_results = []
            weather_label, weather_conf = "unknown", 0.0
            if cap is not None:
                ret, frame = cap.read()
                perf.lap('capture')
                if ret and frame is not None:
                    if camera_generations[camera_id] != cap.generation:
                        camera_generations[camera_id] = cap.generation
                        if stabilizer is not None:
                            stabilizer.reset()
                        frame_buffer.pop(idx, None)
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
                    # 由下方各路图像与 AIS 的融合状态统一决定工作模式

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

                    # 更新帧缓存
                    frame_buffer[idx] = draw_frame


            # ======================== 2. 更新全局告警状态机（融合 AIS） ========================
            ais_has_alarm, ais_alarm_ships, ais_closest = ais_state.get_alarm_state()
            distances = {
                ('ais', mmsi): info['distance']
                for mmsi, info in ais_state.get_all_ships().items()
                if time.monotonic() - info['received_at'] <= AIS_TARGET_TIMEOUT
            }
            image_distances = []
            if stabilized_frame is not None:
                image_distances = [
                    estimate_distance_by_bbox(res[2] - res[0], stabilized_frame.shape[1])
                    for res in post_results
                ]
                image_distances = [d for d in image_distances if d is not None]
            if image_distances:
                distances[('camera', camera_id)] = min(image_distances)
            if approach_photos.update(distances, time.monotonic()):
                pending_photos.update(range(1, len(caps) + 1))

            # 每路用本次新采集画面完成距离拍照，断流时保留待拍标记。
            has_near_target = any(d < RECORD_DISTANCE for d in image_distances)
            image_alarm_distance = min(image_distances) if image_distances else None

            # 串口告警与 50 米录像条件解耦：图像检测告警或 AIS 告警均可触发。
            serial_alarm_manager.update(
                camera_id=camera_id,
                image_detected=current_has_alarm,
                image_distance=image_alarm_distance,
                ais_has_alarm=ais_has_alarm,
                ais_closest=ais_closest,
                weather_label=weather_label,
                weather_conf=weather_conf,
            )

            # 图像目标进入 50m，或 AIS 告警激活时，启动当前在线摄像头录像。
            should_record = has_near_target or (
                ais_has_alarm and stabilized_frame is not None
            )
            need_save_prerecord, current_alarm_state, state_changed = update_alarm_state(
                camera_id, should_record
            )
            alarm_collector = alarm_collectors.get(camera_id)
            continuous_mode = check_work_mode(ALARM_STATE or ais_has_alarm or any(
                now_distance <= PHOTO_DISTANCES[0] for now_distance in image_distances
            ))
            now = time.monotonic()
            if stabilized_frame is not None:
                # Keep the 10-minute cycle even during a prolonged alarm.
                if now >= next_panorama_time[camera_id]:
                    if save_panorama(stabilized_frame, camera_id):
                        next_panorama_time[camera_id] += (
                            int((now - next_panorama_time[camera_id]) // PANORAMA_SAVE_INTERVAL) + 1
                        ) * PANORAMA_SAVE_INTERVAL
                # Alarm/distance image is additional, with a separate filename.
                if camera_id in pending_photos:
                    if save_panorama(stabilized_frame, camera_id, alarm=True):
                        pending_photos.discard(camera_id)

            # ======================== RS232 状态信息定时发送 ========================
            if time.time() - last_state_send >= STATE_SEND_INTERVAL:
                camera_online_count = sum(
                    1 for camera in caps
                    if camera is not None and camera.is_alive()
                )
                send_state_signal(camera_online_count)
                last_state_send = time.time()

            # ======================== 3. 告警触发瞬间处理（融合 AIS 数据） ========================
            if need_save_prerecord:
                # 创建告警目录和收集器
                alarm_id = f"alarm_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}_camera_{camera_id}"
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
                alarm_collectors[camera_id] = alarm_collector
                # 开启归档时写 alarm_recordings；关闭时仅使用进程临时目录合成发送视频。
                alarm_collector.send_dir = alarm_root
                os.makedirs(alarm_collector.send_dir, exist_ok=True)

                # ── 提取第一个告警目标的 bbox，并用它估算图像距离 ──
                for res in post_results:
                    x1, y1, x2, y2, conf, cls, area, is_alarm = res
                    target_distance = estimate_distance_by_bbox(
                        x2 - x1, stabilized_frame.shape[1]
                    ) if stabilized_frame is not None else None
                    if target_distance is not None and target_distance < RECORD_DISTANCE:
                        alarm_collector.ship_bbox = [x1, y1, x2, y2]
                        break

                image_distance = None
                if alarm_collector.ship_bbox:
                    bx1, by1, bx2, by2 = alarm_collector.ship_bbox
                    frame_w = stabilized_frame.shape[1] if stabilized_frame is not None else 0
                    image_distance = estimate_distance_by_bbox(bx2 - bx1, frame_w)
                    if image_distance is not None:
                        print(f"📏 图像估算距离: {image_distance:.1f}m（框宽 {bx2 - bx1}px）")

                # 保存图像估算距离（供 alarm.json 和 RS232 报文在 AIS 缺失时使用）
                alarm_collector.image_distance = round(image_distance, 2) if image_distance is not None else None

                # ── 从 AIS 数据确定告警等级（AIS 缺失时用图像距离） ──
                level_map = {1: "一级", 2: "二级", 3: "三级"}
                ais_level = combined_ais_data.get("level")
                if ais_level in level_map:
                    alarm_collector.alarm_level = level_map[ais_level]
                elif image_distance is not None:
                    if image_distance <= 100:
                        alarm_collector.alarm_level = "一级"
                    elif image_distance <= 150:
                        alarm_collector.alarm_level = "二级"
                    else:
                        alarm_collector.alarm_level = "三级"

                # ── 确定告警触发来源 ──
                if has_near_target and ais_has_alarm:
                    alarm_collector.alarm_trigger_source = "图像检测+AIS距离"
                elif ais_has_alarm:
                    alarm_collector.alarm_trigger_source = "AIS距离"
                else:
                    alarm_collector.alarm_trigger_source = "图像检测"

                # 记录天气（提前设置，告警信号需要 weather 字段）
                alarm_collector.set_weather(weather_label, weather_conf)

                # 写入预录缓存帧
                # 仅从进入 50m 的当前帧开始录制，不写入远距离预录帧。

                # 保存代表帧（船体框+天气）—— 直接用当前帧，不依赖预录缓存
                save_representative_frames(
                    alarm_collector, stabilized_frame, draw_frame,
                    camera_id, len(VIDEO_PATHS)
                )

            # ======================== 4. 告警持续期间录制与OCR记录 ========================
            if current_alarm_state:
                if alarm_collector is not None:
                    # ── 持续更新 AIS 数据：始终记录当前距离最近的船 ──
                    if ais_closest:
                        alarm_collector.ais_data = {
                            "mmsi": ais_closest.get("mmsi"),
                            "lat": ais_closest.get("lat"),
                            "lon": ais_closest.get("lon"),
                            "distance": ais_closest.get("distance"),
                            "level": ais_closest.get("level"),
                            "ship_name": ais_closest.get("ship_name"),
                            "heading": ais_closest.get("heading"),
                            "speed": ais_closest.get("speed"),
                        }

                    alarm_root = alarm_collector.root_dir
                    now = time.time()

                    # ── 保存绘制后帧（带检测框+OCR），用于合成视频，按 RECORD_SAVE_INTERVAL 节流 ──
                    if (draw_frame is not None
                            and now - last_record_save[camera_id] >= RECORD_SAVE_INTERVAL):
                        save_frame_for_recording(draw_frame, camera_id, alarm_root)
                        last_record_save[camera_id] = now
                    if (ALARM_ARCHIVE_ENABLE and stabilized_frame is not None
                            and now - last_raw_save[camera_id] >= RAW_SAVE_INTERVAL):
                        save_raw_frame_for_recording(stabilized_frame, camera_id, alarm_root)
                        last_raw_save[camera_id] = now

                    # 处理当前检测路的OCR船号
                    if ALARM_ARCHIVE_ENABLE and ocr_results:
                        best_ocr = max(ocr_results, key=lambda x: x['conf'])
                        alarm_collector.set_ocr(best_ocr['text'])
                        # 保存船号图像（首次或高置信度更新）
                        if (camera_id not in alarm_collector.ship_image_paths or
                            best_ocr['conf'] > 0.7):
                            ship_path = save_ship_roi_image(
                                stabilized_frame, best_ocr['ship_bbox'],
                                best_ocr['text'], alarm_collector.send_dir, camera_id
                            )
                            if ship_path:
                                alarm_collector.add_ship_image(camera_id, ship_path)

            # ======================== 5. 告警结束处理 ========================
            if state_changed and not current_alarm_state and alarm_collector is not None:
                # ── 2. 保存告警时刻 AIS 轨迹图 ──
                ais_trajectory_path = None
                if ALARM_ARCHIVE_ENABLE and AIS_ENABLE and AIS_TRAJECTORY_ENABLE:
                    track_hist = ais_state.get_track_history()
                    if track_hist:
                        print("📊 保存告警时刻 AIS 轨迹图...")
                        ais_trajectory_path = save_ais_trajectory_image(track_hist, output_dir=alarm_collector.send_dir)
                        if ais_trajectory_path:
                            fixed = os.path.join(alarm_collector.send_dir, 'ais_trajectory.webp')
                            if ais_trajectory_path != fixed:
                                shutil.move(ais_trajectory_path, fixed)
                            ais_trajectory_path = fixed

                # ── 3. 合成发送视频；本地归档开启时才生成 alarm.json ──
                compose_alarm_videos(alarm_collector, camera_id)
                alarm_collector.end_time = datetime.utcnow()
                if ALARM_ARCHIVE_ENABLE:
                    generate_alarm_json(alarm_collector)
                    cleanup_alarm_recordings()
                else:
                    cleanup_alarm_work_dir(alarm_collector.root_dir)

                # ── 4. 视频保存完成后，唤醒媒体上传线程 ──
                invoke_stream_sender()

                # 清空录制缓存和状态
                RECORD_FRAMES.pop(camera_id, None)
                with PRERECORD_LOCK:
                    prerecord_buffers.pop(camera_id, None)
                alarm_collectors.pop(camera_id, None)
                alarm_collector = None
                if ALARM_ARCHIVE_ENABLE:
                    print("✅ 告警事件归档完成")
                else:
                    print("✅ 告警视频处理完成（本地归档关闭）")

            # 融合告警更新和归档完成后才检查工作期限。
            if not check_work_duration():
                print(f"工作时长已达{DEFAULT_WORK_DURATION}秒且无活动告警，退出程序")
                break

            # ======================== 6. 切换到下一路摄像头 ========================
            current_cam_idx = (current_cam_idx + 1) % len(caps)

            # ======================== 7. 画面拼接显示 ========================
            if ENABLE_DISPLAY:
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

                # 拼接后若画面过大，等比缩放到适合屏幕的最大尺寸（避免放不下）
                MAX_DISPLAY_W, MAX_DISPLAY_H = 1280, 720
                ch, cw = canvas.shape[:2]
                if cw > MAX_DISPLAY_W or ch > MAX_DISPLAY_H:
                    scale = min(MAX_DISPLAY_W / cw, MAX_DISPLAY_H / ch)
                    canvas = cv2.resize(canvas, (int(cw * scale), int(ch * scale)))

                cv2.imshow("Ship Detection - ALL Cameras", canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):
                    print("用户手动退出程序")
                    break
            else:
                # 关闭显示时跳过拼接和 imshow，仅短暂 sleep 避免空转
                time.sleep(0.005)

            # ======================== 8. AIS 轨迹图定时保存 ========================
            if AIS_ENABLE and AIS_TRAJECTORY_ENABLE and AIS_TRAJECTORY_SAVE_INTERVAL > 0:
                now_ts = time.time()
                if now_ts - ais_last_trajectory_save >= AIS_TRAJECTORY_SAVE_INTERVAL:
                    track_hist = ais_state.get_track_history()
                    if track_hist:
                        save_ais_trajectory_image(track_hist)
                    ais_last_trajectory_save = now_ts

            perf.lap('display')
            perf.maybe_print(PROFILE_PRINT_INTERVAL)

            # ── 帧率控制：无告警时慢速，告警时全速 ──
            effective_fps = TARGET_FPS_ALARM if ALARM_STATE else TARGET_FPS_IDLE
            if effective_fps > 0:
                target_interval = 1.0 / effective_fps
                elapsed = time.perf_counter() - last_process_time
                if elapsed < target_interval:
                    time.sleep(target_interval - elapsed)
                last_process_time = time.perf_counter()

    except KeyboardInterrupt:
        print('[exit] 用户中断（Ctrl+C）')
    except Exception as e:
        print(f"程序异常：{str(e)}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        try:
            if AIS_ENABLE and ais_reader is not None:
                ais_reader.stop()
                if AIS_TRAJECTORY_ENABLE:
                    track_hist = ais_state.get_track_history()
                    if track_hist:
                        save_ais_trajectory_image(track_hist)
            for camera_id, alarm_collector in list(alarm_collectors.items()):
                print(f"正在处理摄像头{camera_id}未完成的告警...")
                compose_alarm_videos(alarm_collector, camera_id)
                alarm_collector.end_time = datetime.utcnow()
                if ALARM_ARCHIVE_ENABLE:
                    generate_alarm_json(alarm_collector)
                    cleanup_alarm_recordings()
                else:
                    cleanup_alarm_work_dir(alarm_collector.root_dir)
                RECORD_FRAMES.pop(camera_id, None)
                invoke_stream_sender()  # completed alarm media remains an immediate trigger
            cleanup_alarm_work_root()
        finally:
            try:
                for cap in caps:
                    if cap is not None:
                        cap.release()
                cv2.destroyAllWindows()
            finally:
                try:
                    media_storage.stop()
                finally:
                    try:
                        if upload_worker is not None:
                            upload_worker.stop(grace_seconds=UPLOAD_EXIT_GRACE)
                    finally:
                        if serial_alarm_manager is not None:
                            serial_alarm_manager.stop()
                        if serial_transport is not None:
                            serial_transport.stop()
                        if gps_position is not None:
                            gps_position.flush()
                print("资源已释放；未完成上传的媒体将在下次启动时重试")
