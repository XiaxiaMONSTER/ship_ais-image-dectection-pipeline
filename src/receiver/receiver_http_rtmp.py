#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
多路RTMP接收程序 + HTTP 接收服务 — 支持分段录制 + 全景图片接收 + 告警数据接收

功能：
- 同时接收1-4路RTMP流（告警期间推流）
- 空闲超时自动分段保存视频（分周期隔断）
- HTTP 服务接收全景图片（POST /panorama）和告警数据（POST /alarm）
- 重连策略：
    - 连续3次无法获得有效帧（无论连接失败或无帧），之后固定2分钟重试间隔
    - 成功收到第一帧后，重置计数，恢复短间隔（2秒起步）
- 支持手动重置连接（按 r 键）
"""

import cv2
import time
import os
import threading
import json
import numpy as np
import http.server
import cgi
import io
from collections import deque
from datetime import datetime

# ======================== RTMP 配置 ========================
RTMP_BASE_URL = "rtmp://192.168.1.100:1935/camera_"
CAMERA_COUNT = 1

DISPLAY_W, DISPLAY_H = 640, 480
VIDEO_FPS = 20

RECORD_VIDEO = True
RECORD_SPLIT_IDLE_SEC = 5
OUTPUT_DIR = "rtmp_recordings"

# ======================== HTTP 接收服务配置 ========================
HTTP_ENABLE = True               # 是否启动 HTTP 接收服务
HTTP_LISTEN_PORT = 8080          # 监听端口
HTTP_PANORAMA_DIR = "panoramas"  # 全景图片保存根目录
HTTP_ALARM_DIR = "alarms"        # 告警数据保存根目录

# 重连参数
NO_FRAME_TIMEOUT = 5.0                  # 连接后等待第一帧的超时（秒）
MAX_CONSECUTIVE_FAILURES = 3            # 连续多少次失败（连接失败或无帧）后进入长间隔
LONG_RECONNECT_DELAY = 120              # 长间隔（秒），推流端长期离线时使用
INITIAL_RECONNECT_DELAY = 2             # 初始重连间隔（秒）
MAX_SHORT_DELAY = 30                    # 短间隔退避最大值（秒）

def print_log(level, msg):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{timestamp}] [{level}] {msg}")

def build_camera_streams(camera_count):
    streams = {}
    for cam_id in range(1, camera_count + 1):
        streams[cam_id] = {
            "url": f"{RTMP_BASE_URL}{cam_id}",
            "cap": None,
            "frame": None,
            "has_valid_frame": False,
            "frame_count": 0,
            "writer": None,
            "output_path": None,
            "reconnect_delay": INITIAL_RECONNECT_DELAY,
            "consecutive_failures": 0,      # 统一失败计数（连接失败或无帧）
            "last_frame_time": 0,
            "error_count": 0,
            "stop_flag": False,
        }
    return streams

def capture_worker(cam_id, info, stop_event):
    while not stop_event.is_set() and not info["stop_flag"]:
        # ---------- 连接管理 ----------
        if info["cap"] is None:
            # 根据失败计数决定本次重连等待时长
            if info["consecutive_failures"] >= MAX_CONSECUTIVE_FAILURES:
                delay = LONG_RECONNECT_DELAY
                print_log("INFO", f"Cam{cam_id}: 已连续{info['consecutive_failures']}次失败，使用长间隔 {delay}秒")
            else:
                delay = info["reconnect_delay"]
                print_log("INFO", f"Cam{cam_id}: 正在连接（第{info['consecutive_failures']+1}次尝试），{delay:.0f}秒后重试")
            
            # 实际等待
            for _ in range(int(delay)):
                if stop_event.is_set() or info["stop_flag"]:
                    return
                time.sleep(1)

            print_log("INFO", f"Cam{cam_id}: 正在连接 {info['url']} ...")
            cap = cv2.VideoCapture(info["url"], cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            try:
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000)
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000)
            except:
                pass

            # ---------- 连接结果处理（统一失败计数） ----------
            if not cap.isOpened():
                cap.release()
                info["consecutive_failures"] += 1
                # 指数退避（仅用于短间隔，长间隔时忽略）
                if info["consecutive_failures"] < MAX_CONSECUTIVE_FAILURES:
                    info["reconnect_delay"] = min(info["reconnect_delay"] * 1.5, MAX_SHORT_DELAY)
                print_log("WARNING", f"Cam{cam_id}: 连接失败（累计{info['consecutive_failures']}次）")
                continue

            # 连接成功，尝试读取第一帧（带超时）
            info["cap"] = cap
            connect_time = time.time()
            first_frame_ok = False
            while time.time() - connect_time < NO_FRAME_TIMEOUT:
                if stop_event.is_set() or info["stop_flag"]:
                    cap.release()
                    return
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    first_frame_ok = True
                    break
                time.sleep(0.05)

            if not first_frame_ok:
                cap.release()
                info["cap"] = None
                info["consecutive_failures"] += 1
                if info["consecutive_failures"] < MAX_CONSECUTIVE_FAILURES:
                    info["reconnect_delay"] = min(info["reconnect_delay"] * 1.5, MAX_SHORT_DELAY)
                print_log("WARNING", f"Cam{cam_id}: {NO_FRAME_TIMEOUT}秒内无帧（累计{info['consecutive_failures']}次）")
                continue

            # ---------- 成功收到第一帧，重置所有计数 ----------
            info["consecutive_failures"] = 0
            info["has_valid_frame"] = True
            info["reconnect_delay"] = INITIAL_RECONNECT_DELAY
            info["error_count"] = 0
            info["last_frame_time"] = time.time()

            # 重连成功后，创建新录制文件（writer此时应该已经是None）
            if RECORD_VIDEO:
                # 确保旧的writer已经被关闭
                if info["writer"] is not None:
                    try:
                        info["writer"].release()
                        print_log("INFO", f"Cam{cam_id}: 关闭残留的录制文件")
                    except:
                        pass
                    info["writer"] = None
                
                # 创建新录制文件
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                filename = f"camera_{cam_id}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
                info["output_path"] = os.path.join(OUTPUT_DIR, filename)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(info["output_path"], fourcc, VIDEO_FPS, (DISPLAY_W, DISPLAY_H))
                if writer.isOpened():
                    info["writer"] = writer
                    print_log("INFO", f"Cam{cam_id}: 开始新录制 → {info['output_path']}")
                else:
                    print_log("WARNING", f"Cam{cam_id}: 录制初始化失败")

            print_log("INFO", f"Cam{cam_id}: 连接成功，开始接收")

        # ---------- 正常帧读取循环 ----------
        while not stop_event.is_set() and not info["stop_flag"] and info["cap"] is not None:
            ret, frame = info["cap"].read()
            now = time.time()

            if not ret or frame is None or frame.size == 0:
                info["error_count"] += 1
                if info["error_count"] <= 3:
                    print_log("WARNING", f"Cam{cam_id}: 读取失败（连续{info['error_count']}次）")

                # 关键修改：检测到断流后，立即关闭当前录制文件
                if RECORD_VIDEO and info["writer"] is not None:
                    # 只要连续失败3次，就认为推流已结束，立即关闭文件
                    if info["error_count"] >= 3:
                        try:
                            info["writer"].release()
                            print_log("INFO", f"Cam{cam_id}: 推流断开，立即关闭录制文件 → {info['output_path']}")
                        except Exception as e:
                            print_log("WARNING", f"Cam{cam_id}: 关闭录制文件失败: {e}")
                        info["writer"] = None

                # 连续失败5次，触发重连
                if info["error_count"] >= 5:
                    print_log("WARNING", f"Cam{cam_id}: 连续读取失败{info['error_count']}次，主动重连")
                    try:
                        info["cap"].release()
                    except:
                        pass
                    info["cap"] = None
                    info["has_valid_frame"] = False
                    info["error_count"] = 0
                    # 增加统一失败计数
                    info["consecutive_failures"] += 1
                    if info["consecutive_failures"] < MAX_CONSECUTIVE_FAILURES:
                        info["reconnect_delay"] = min(info["reconnect_delay"] * 1.5, MAX_SHORT_DELAY)
                    break
                time.sleep(0.05)
                continue

            # 读取成功，重置错误计数
            info["error_count"] = 0

            # 图像预处理
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # 花屏检测
            if frame.shape[2] == 3:
                b_mean, g_mean, r_mean = np.mean(frame[:,:,0]), np.mean(frame[:,:,1]), np.mean(frame[:,:,2])
                if (g_mean > 200 and b_mean < 30 and r_mean < 30) or (r_mean > 200 and g_mean < 30 and b_mean < 30):
                    print_log("WARNING", f"Cam{cam_id}: 花屏帧跳过")
                    continue

            # 缩放
            if frame.shape[1] != DISPLAY_W or frame.shape[0] != DISPLAY_H:
                frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))

            # 更新共享帧
            info["frame"] = frame.copy()
            info["frame_count"] += 1
            idle_gap = now - info["last_frame_time"] if info["last_frame_time"] > 0 else 0
            info["last_frame_time"] = now

            # ---------- 分段录制 ----------
            if RECORD_VIDEO:
                # 检查是否需要分段（空闲超时后恢复）
                if idle_gap > RECORD_SPLIT_IDLE_SEC and info["writer"] is not None:
                    try:
                        info["writer"].release()
                        print_log("INFO", f"Cam{cam_id}: 空闲{idle_gap:.0f}秒后恢复，关闭旧文件")
                    except:
                        pass
                    info["writer"] = None

                # 如果没有writer，创建新文件
                if info["writer"] is None:
                    os.makedirs(OUTPUT_DIR, exist_ok=True)
                    filename = f"camera_{cam_id}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
                    info["output_path"] = os.path.join(OUTPUT_DIR, filename)
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(info["output_path"], fourcc, VIDEO_FPS, (DISPLAY_W, DISPLAY_H))
                    if writer.isOpened():
                        info["writer"] = writer
                        print_log("INFO", f"Cam{cam_id}: 开始新录制 → {info['output_path']}")
                    else:
                        print_log("WARNING", f"Cam{cam_id}: 录制初始化失败")

                # 写入帧
                if info["writer"] is not None:
                    try:
                        info["writer"].write(frame)
                    except Exception as e:
                        print_log("WARNING", f"Cam{cam_id}: 写入失败: {e}")

            time.sleep(0.001)

    # 线程退出清理
    if info["cap"] is not None:
        try:
            info["cap"].release()
        except:
            pass
    if info["writer"] is not None:
        try:
            info["writer"].release()
            print_log("INFO", f"Cam{cam_id}: 最终保存 → {info['output_path']}")
        except:
            pass
    print_log("INFO", f"Cam{cam_id}: 采集线程退出")

def start_all_captures(streams, stop_event):
    threads = []
    for cam_id, info in streams.items():
        t = threading.Thread(target=capture_worker, args=(cam_id, info, stop_event), daemon=False)
        t.start()
        threads.append(t)
    return threads

def stitch_frames(frames, w, h):
    rendered = []
    for f in frames:
        if f is None:
            rendered.append(np.zeros((h, w, 3), dtype=np.uint8))
        else:
            rendered.append(f)
    n = len(rendered)
    if n == 1:
        return rendered[0]
    elif n == 2:
        return np.hstack(rendered)
    elif n == 3:
        top = np.hstack(rendered[:2])
        bottom = np.hstack([rendered[2], np.zeros((h, w, 3), dtype=np.uint8)])
        return np.vstack([top, bottom])
    else:
        row1 = np.hstack(rendered[:2])
        row2 = np.hstack(rendered[2:4])
        return np.vstack([row1, row2])

# ======================== HTTP 请求处理器 ========================
class _ReceiverHTTPHandler(http.server.BaseHTTPRequestHandler):
    """处理 POST /panorama 和 POST /alarm，接收来自 RK3576 的上传数据。"""

    # 类变量：由外部设置保存目录
    panorama_dir = HTTP_PANORAMA_DIR
    alarm_dir = HTTP_ALARM_DIR

    def log_message(self, format, *args):
        """重定向日志到统一 print_log 格式。"""
        print_log("HTTP", format % args)

    def do_POST(self):
        if self.path == '/panorama' or self.path.startswith('/panorama'):
            self._handle_panorama()
        elif self.path == '/alarm' or self.path.startswith('/alarm'):
            self._handle_alarm()
        else:
            self.send_error(404, "Not Found")
            self.end_headers()

    def _parse_multipart(self):
        """
        解析 multipart/form-data 请求体。
        使用 cgi.FieldStorage（标准库，兼容 Python 3.x）。
        """
        content_type = self.headers.get('Content-Type', '')
        if 'multipart/form-data' not in content_type:
            return None

        # 构造环境变量供 cgi.FieldStorage 使用
        env = os.environ.copy()
        env['REQUEST_METHOD'] = 'POST'
        env['CONTENT_TYPE'] = content_type
        env['CONTENT_LENGTH'] = self.headers.get('Content-Length', '0')

        # 从请求体读取
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)

        fp = io.BytesIO(body)
        form = cgi.FieldStorage(
            fp=fp,
            environ=env,
            keep_blank_values=True
        )
        return form

    def _handle_panorama(self):
        """POST /panorama — 接收全景图片 (camera_id, timestamp, image)"""
        try:
            form = self._parse_multipart()
            if form is None:
                self.send_error(400, "Bad Request: expected multipart/form-data")
                return

            camera_id = form.getfirst('camera_id', 'unknown')
            timestamp = form.getfirst('timestamp', datetime.now().strftime('%Y%m%d_%H%M%S'))
            image_field = form.getfirst('image')

            if image_field is None:
                self.send_error(400, "Bad Request: missing 'image' field")
                return

            # 保存 JPEG 图片
            ts_safe = timestamp.replace(':', '').replace('T', '_')[:15]
            cam_dir = os.path.join(self.panorama_dir, f"camera_{camera_id}")
            os.makedirs(cam_dir, exist_ok=True)
            filename = f"{ts_safe}.jpg"
            filepath = os.path.join(cam_dir, filename)

            if hasattr(image_field, 'file'):
                image_data = image_field.file.read()
            else:
                image_data = image_field

            with open(filepath, 'wb') as f:
                f.write(image_data)

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            resp = json.dumps({"status": "ok", "file": filepath})
            self.wfile.write(resp.encode('utf-8'))
            print_log("INFO", f"全景图片已保存: {filepath}")

        except Exception as e:
            print_log("ERROR", f"全景接收异常: {e}")
            self.send_error(500, str(e))

    def _handle_alarm(self):
        """POST /alarm — 接收告警数据 (alarm_json + ship_image_*)"""
        try:
            form = self._parse_multipart()
            if form is None:
                self.send_error(400, "Bad Request: expected multipart/form-data")
                return

            # 解析 alarm_json
            alarm_json_field = form.getfirst('alarm_json')
            if alarm_json_field is None:
                self.send_error(400, "Bad Request: missing 'alarm_json' field")
                return

            if hasattr(alarm_json_field, 'file'):
                alarm_data = json.loads(alarm_json_field.file.read().decode('utf-8'))
            else:
                alarm_data = json.loads(alarm_json_field.decode('utf-8'))

            alarm_id = alarm_data.get('alarm_id', f"alarm_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

            # 创建告警目录
            alarm_path = os.path.join(self.alarm_dir, alarm_id)
            os.makedirs(alarm_path, exist_ok=True)

            # 保存 alarm.json
            json_path = os.path.join(alarm_path, 'alarm.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(alarm_data, f, indent=2, ensure_ascii=False)

            # 保存附带的船号图片（ship_image_1, ship_image_2, ...）
            saved_images = []
            for key in list(form.keys()):
                if key.startswith('ship_image_'):
                    field = form.getfirst(key)
                    if field:
                        cam_id = key.replace('ship_image_', '')
                        img_name = f"ship_camera{cam_id}.jpg"
                        img_path = os.path.join(alarm_path, img_name)
                        if hasattr(field, 'file'):
                            img_data = field.file.read()
                        else:
                            img_data = field
                        with open(img_path, 'wb') as f_img:
                            f_img.write(img_data)
                        saved_images.append(img_path)

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            resp = json.dumps({
                "status": "ok",
                "alarm_id": alarm_id,
                "alarm_dir": alarm_path,
                "json_path": json_path,
                "saved_images": saved_images
            })
            self.wfile.write(resp.encode('utf-8'))
            print_log("WARN", f"!!! 告警数据已接收: {alarm_path} (图片{len(saved_images)}张)")

        except Exception as e:
            print_log("ERROR", f"告警接收异常: {e}")
            import traceback
            traceback.print_exc()
            self.send_error(500, str(e))


def start_http_server(port=8080):
    """启动 HTTP 接收服务（阻塞式，需在独立线程中调用）。"""
    if not HTTP_ENABLE:
        print_log("INFO", "HTTP 接收服务已禁用 (HTTP_ENABLE=False)")
        return None

    server = http.server.HTTPServer(('0.0.0.0', port), _ReceiverHTTPHandler)
    print_log("INFO", f"HTTP 接收服务已启动 → http://0.0.0.0:{port}")
    print_log("INFO", f"  全景上传: POST /panorama")
    print_log("INFO", f"  告警上传: POST /alarm")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print_log("INFO", "HTTP 接收服务已停止")


def main():
    print_log("INFO", "===== 多路RTMP接收程序启动（RTMP + HTTP 接收） =====")
    streams = build_camera_streams(CAMERA_COUNT)
    stop_event = threading.Event()
    threads = start_all_captures(streams, stop_event)

    # 启动 HTTP 接收服务（后台线程）
    http_thread = None
    if HTTP_ENABLE:
        # 确保保存目录存在
        os.makedirs(HTTP_PANORAMA_DIR, exist_ok=True)
        os.makedirs(HTTP_ALARM_DIR, exist_ok=True)
        http_thread = threading.Thread(
            target=start_http_server,
            args=(HTTP_LISTEN_PORT,),
            daemon=True,
            name="HTTP-Receiver"
        )
        http_thread.start()
        time.sleep(0.5)  # 给 HTTP 服务一点启动时间

    fps_window = deque(maxlen=30)
    last_time = time.time()
    last_status = time.time()

    print_log("INFO", "按 q 退出，按 r 手动重置所有连接")

    try:
        while True:
            frames = [streams[cam_id]["frame"] for cam_id in sorted(streams.keys())]
            canvas = stitch_frames(frames, DISPLAY_W, DISPLAY_H)

            now = time.time()
            fps_window.append(1.0 / max(now - last_time, 0.001))
            last_time = now
            if len(fps_window) >= 30:
                fps_val = sum(fps_window) / len(fps_window)
                cv2.putText(canvas, f"FPS: {fps_val:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            if now - last_status > 1.0:
                last_status = now
                y = 60
                # HTTP 服务状态
                if HTTP_ENABLE:
                    http_label = f"HTTP: 0.0.0.0:{HTTP_LISTEN_PORT} (panorama + alarm)"
                    cv2.putText(canvas, http_label, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
                    y += 22
                for cam_id in sorted(streams.keys()):
                    info = streams[cam_id]
                    status = "OK" if info["has_valid_frame"] else ("Conn" if info["cap"] else "Off")
                    color = (0,255,0) if info["has_valid_frame"] else (0,165,255)
                    cv2.putText(canvas, f"Cam{cam_id}: {status}  frames:{info['frame_count']}",
                                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    y += 20

            cv2.imshow("Multi-RTMP Receiver", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                print_log("INFO", "手动重置所有连接")
                for info in streams.values():
                    if info["cap"]:
                        try:
                            info["cap"].release()
                        except:
                            pass
                    info["cap"] = None
                    info["has_valid_frame"] = False
                    info["consecutive_failures"] = 0
                    info["reconnect_delay"] = INITIAL_RECONNECT_DELAY
                    info["error_count"] = 0
                    if info["writer"]:
                        try:
                            info["writer"].release()
                        except:
                            pass
                        info["writer"] = None
    except KeyboardInterrupt:
        print_log("INFO", "收到中断信号")
    finally:
        stop_event.set()
        for info in streams.values():
            info["stop_flag"] = True
        for t in threads:
            t.join(timeout=3)
        cv2.destroyAllWindows()
        print_log("INFO", "程序已退出，所有资源已释放")

if __name__ == "__main__":
    main()