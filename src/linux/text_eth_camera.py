import cv2
import sys

def test_network_camera(camera_url):
    """
    测试网络摄像头的核心函数
    :param camera_url: 网络摄像头的访问地址
    """
    # 打开网络摄像头流
    cap = cv2.VideoCapture(camera_url)
    
    # 检查是否成功连接摄像头
    if not cap.isOpened():
        print(f"❌ 无法连接到网络摄像头: {camera_url}")
        print("请检查：1. 摄像头IP/端口是否正确 2. 用户名密码是否正确 3. eth0网卡是否能访问摄像头")
        sys.exit(1)
    
    print("✅ 成功连接到网络摄像头！")
    print("📌 操作说明：")
    print("   - 按 'q' 键退出")
    print("   - 按 's' 键保存当前帧为 screenshot.jpg")
    
    # 循环读取摄像头帧并显示
    while True:
        # 读取一帧画面
        ret, frame = cap.read()
        
        # 检查帧是否读取成功
        if not ret:
            print("⚠️  无法读取摄像头画面，可能是连接中断")
            break
        
        # 显示画面（窗口标题为 "Network Camera Test"）
        cv2.imshow("Network Camera Test (eth0)", frame)
        
        # 键盘事件监听（等待1ms）
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # 按 q 退出
            break
        elif key == ord('s'):
            # 按 s 保存截图
            cv2.imwrite("screenshot.jpg", frame)
            print("📸 截图已保存为 screenshot.jpg")
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("🔌 已断开摄像头连接，测试结束")

if __name__ == "__main__":
    # ===================== 核心配置 =====================
    # 替换为你的网络摄像头实际地址！
    # 示例1（RTSP）：rtsp://admin:123456@192.168.1.100:554/stream1
    # 示例2（HTTP）：http://192.168.1.100:8080/stream
    CAMERA_URL = "rtsp://admin:xiayuxuan113@192.168.1.64:554/stream1"
    
    # 启动测试
    test_network_camera(CAMERA_URL)
