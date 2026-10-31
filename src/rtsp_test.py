import cv2
import subprocess
import os
import sys
import time

# ===================== 核心配置 =====================
CAMERA_ID = 0  # USB摄像头编号
SENDER_IP = "192.168.1.5"  # 发送端局域网IP
#RTSP_URL = f"rtsp://127.0.0.1:8554/test_camera"  
RTSP_URL = f"rtsp://192.168.1.5:8554/test_camera"  
CAMERA_RESOLUTION = "640x480"
FPS = 25

# ===================== 调试配置（关键：不闪退） =====================
DEBUG_MODE = True  # 调试模式：强制停留、显示所有日志
DELAY_AFTER_ERROR = 15  # 错误后停留15秒

# ===================== 工具函数（添加详细日志） =====================
def print_log(level, msg):
    """带时间戳的日志输出"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{timestamp}] [{level}] {msg}")

def check_admin():
    """检查是否管理员权限（仅日志，不强制退出）"""
    try:
        import ctypes
        is_admin = ctypes.windll.shell32.IsUserAnAdmin()
    except:
        is_admin = False
    if not is_admin:
        print_log("WARNING", "当前未以管理员身份运行！可能导致摄像头/网络权限不足")
    else:
        print_log("INFO", "✅ 已获取管理员权限")
    return is_admin

def check_rtsp_server():
    """检查RTSP服务器（仅日志，不强制退出）"""
    try:
        proc_check = os.popen("tasklist | findstr rtsp-simple-server").read()
        if "rtsp-simple-server.exe" in proc_check:
            print_log("INFO", "✅ 检测到rtsp-simple-server正在运行")
            return True
        else:
            print_log("ERROR", "❌ 未检测到rtsp-simple-server进程！请先启动该程序")
            return False
    except Exception as e:
        print_log("ERROR", f"检查RTSP服务器失败：{str(e)}")
        return False

def check_ffmpeg():
    """检查FFmpeg是否可用"""
    try:
        ffmpeg_out = os.popen("ffmpeg -version").read()
        if "ffmpeg version" in ffmpeg_out:
            print_log("INFO", "✅ FFmpeg配置成功")
            return True
        else:
            print_log("ERROR", "❌ FFmpeg未配置到环境变量！")
            return False
    except Exception as e:
        print_log("ERROR", f"检查FFmpeg失败：{str(e)}")
        return False

# ===================== 主程序（添加异常捕获） =====================
if __name__ == "__main__":
    print_log("INFO", "===== 启动USB摄像头RTSP推流程序 =====")
    
    # 1. 前置检查（仅日志，不退出）
    is_admin = check_admin()
    is_rtsp_running = check_rtsp_server()
    is_ffmpeg_ok = check_ffmpeg()
    
    # 2. 打开摄像头（异常捕获）
    cap = None
    try:
        print_log("INFO", f"尝试打开摄像头 {CAMERA_ID}（DSHOW后端）")
        cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(CAMERA_RESOLUTION.split('x')[0]))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(CAMERA_RESOLUTION.split('x')[1]))
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FPS, FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if cap.isOpened():
            print_log("INFO", f"✅ 摄像头打开成功，分辨率：{CAMERA_RESOLUTION}")
        else:
            print_log("ERROR", "❌ 摄像头已创建对象，但无法打开！")
    except Exception as e:
        print_log("ERROR", f"打开摄像头失败：{str(e)}")
    
    # 3. 启动FFmpeg推流（异常捕获）
    ffmpeg_proc = None
    if cap and cap.isOpened() and is_ffmpeg_ok:
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', CAMERA_RESOLUTION, '-r', str(FPS), '-i', '-',
            '-c:v', 'libx264', '-preset', 'ultrafast', '-tune', 'zerolatency',
            '-b:v', '800k', '-f', 'rtsp', '-rtsp_transport', 'tcp',
            RTSP_URL
        ]
        print_log("INFO", f"FFmpeg命令：{' '.join(ffmpeg_cmd)}")
        
        try:
            print_log("INFO", "尝试启动FFmpeg推流进程...")
            ffmpeg_proc = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,  # 显示FFmpeg输出（调试）
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            print_log("INFO", "✅ FFmpeg推流进程启动成功")
        except Exception as e:
            print_log("ERROR", f"启动FFmpeg失败：{str(e)}")
    
    # 4. 推流循环（简化，仅调试）
    if cap and cap.isOpened() and ffmpeg_proc:
        print_log("INFO", "✅ 推流中... 按q退出（窗口需有焦点）")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print_log("WARNING", "摄像头画面读取中断，重试...")
                    time.sleep(0.1)
                    continue
                ffmpeg_proc.stdin.write(frame.tobytes())
                cv2.imshow("Push Preview", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except Exception as e:
            print_log("ERROR", f"推流循环出错：{str(e)}")
    
    # 5. 清理资源（强制延迟）
    print_log("INFO", "开始清理资源...")
    if ffmpeg_proc:
        ffmpeg_proc.stdin.close()
        ffmpeg_proc.terminate()
    if cap:
        cap.release()
        cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW).release()
    cv2.destroyAllWindows()
    
    # 调试模式：强制停留
    if DEBUG_MODE:
        print_log("INFO", f"程序执行完毕，{DELAY_AFTER_ERROR}秒后关闭窗口...")
        time.sleep(DELAY_AFTER_ERROR)
    print_log("INFO", "===== 程序退出 =====")