import cv2
import time
import os

# 核心：纯合法RTSP地址（无任何拼接）
RTSP_URL = "rtsp://192.168.1.5:8554/test_camera"
# 关键：低版本OpenCV强制RTSP TCP传输（仅环境变量，不修改URL）
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# 1. 打开RTSP流（仅核心参数，无高版本API）
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
# 仅保留通用参数（所有OpenCV 4.x支持）
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 关闭缓冲，低延迟
cap.set(cv2.CAP_PROP_FPS, 25)        # 匹配推流帧率

if not cap.isOpened():
    print("❌ 无法连接RTSP流！排查：")
    print(f"  1. 推流程序是否运行（final_push.py）")
    print(f"  2. rtsp-simple-server是否以管理员身份启动")
    print(f"  3. 推流地址是否为：{RTSP_URL}")
    exit()

# 2. 接收主循环
print(f"✅ 成功连接RTSP流：{RTSP_URL}")
print("✅ 接收画面中... 按q退出（点击接收窗口后按q）")
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        time.sleep(0.1)
        continue
    # 显示接收画面
    cv2.imshow("RTSP Receive Preview", frame)
    frame_count += 1
    if frame_count % 50 == 0:
        print(f"📊 已接收 {frame_count} 帧（接收正常）")
    # 退出逻辑
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 3. 清理资源
cap.release()
cv2.destroyAllWindows()
print("✅ 接收结束，资源已释放")