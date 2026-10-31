import cv2
import sys

def detect_available_cameras(max_test=5):
    """检测可用的摄像头编号（适配Windows后端）"""
    available_cameras = []
    # 测试不同后端（优先DSHOW，兼容大部分摄像头）
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_VFW]
    
    for idx in range(max_test):
        for backend in backends:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available_cameras.append((idx, backend))
                    cap.release()
                    break  # 找到可用后端就停止测试该摄像头
            cap.release()
    
    # 去重，只保留摄像头编号
    return list({cam[0] for cam in available_cameras})

def show_camera_feed(camera_id):
    """展示指定编号摄像头的实时画面（使用兼容后端）"""
    # 优先使用CAP_DSHOW后端（解决MSMF兼容问题）
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    
    # 兜底：如果DSHOW失败，尝试MSMF
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_id, cv2.CAP_MSMF)
    
    if not cap.isOpened():
        print(f"❌ 无法打开摄像头 {camera_id}（所有后端均失败）")
        return

    # 关闭自动曝光/自动白平衡（可选，提升画面稳定性）
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25=关闭自动曝光
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)           # 0=关闭自动白平衡
    
    cv2.namedWindow(f"Camera {camera_id} - Press 'q' to quit", cv2.WINDOW_NORMAL)
    print(f"✅ 成功打开摄像头 {camera_id}，按 'q' 退出画面")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  摄像头帧读取失败，退出")
            break
        
        cv2.imshow(f"Camera {camera_id} - Press 'q' to quit", frame)
        
        # 按q退出（注意：waitKey不能省略，否则窗口卡死）
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 关键：释放摄像头（DSHOW后端必须显式释放，否则会占用）
    cap.release()
    cv2.destroyAllWindows()
    print(f"📌 摄像头 {camera_id} 已关闭")

if __name__ == "__main__":
    print("🔍 正在检测可用摄像头（适配Windows后端）...")
    available = detect_available_cameras()
    
    if not available:
        print("❌ 未检测到任何可用摄像头")
        sys.exit(1)
    
    print(f"✅ 检测到可用摄像头编号：{available}")
    
    # 选择测试摄像头
    if len(available) == 1:
        camera_id = available[0]
    else:
        camera_id = int(input(f"请输入要测试的摄像头编号（可选：{available}）："))
        if camera_id not in available:
            print(f"❌ 编号 {camera_id} 不是可用摄像头")
            sys.exit(1)
    
    show_camera_feed(camera_id)