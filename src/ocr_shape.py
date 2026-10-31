import cv2
import time
import os

def init_paddle_ocr_v5(text_det_limit_side_len=960):
    """
    初始化PaddleOCR V5（无参数错误+无警告版）
    :param text_det_limit_side_len: 检测模型限制边长（测试核心参数）
    :return: 初始化后的OCR模型
    """
    try:
        from paddleocr import PaddleOCR
        
        # 你的模型路径（完全保持不变）
        det_model_dir = r"P:\.vscode\Ship_Detection\pipeline\models\paddleocr_v5\PP-OCRv5_server_det"
        rec_model_dir = r"P:\.vscode\Ship_Detection\pipeline\models\paddleocr_v5\PP-OCRv5_server_rec"
        
        # 初始化OCR（删除所有无效/报错参数，仅保留合法参数）
        ocr = PaddleOCR(
            # 模型路径（必填）
            text_detection_model_dir=det_model_dir,
            text_recognition_model_dir=rec_model_dir,
            # 关闭无用功能（轻量化）
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            # 运行设备
            device="gpu",
            # ===================== 测试核心：输入尺寸参数（新版合法） =====================
            text_det_limit_side_len=text_det_limit_side_len,
            text_det_limit_type="max",
        )
        print(f"✅ PaddleOCR V5 初始化成功 | 检测输入尺寸限制: {text_det_limit_side_len}")
        return ocr
    except Exception as e:
        print(f"❌ 初始化失败: {str(e)}")
        return None

def test_ocr_input_size(img_path, test_size_list: list):
    """
    测试不同输入尺寸对PaddleOCRv5的影响
    """
    # 校验图片
    if not os.path.exists(img_path):
        print(f"❌ 图片不存在: {img_path}")
        return
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ 图片读取失败")
        return

    print("="*80)
    print(f"📷 测试图片: {img_path} | 原始尺寸: {img.shape[1]}x{img.shape[0]}")
    print(f"📏 待测试尺寸列表: {test_size_list}")
    print("="*80)

    # 遍历测试所有尺寸
    for size in test_size_list:
        print(f"\n------ 测试检测边长: {size} ------")
        ocr_model = init_paddle_ocr_v5(text_det_limit_side_len=size)
        if ocr_model is None:
            continue

        # 推理计时
        start_time = time.time()
        result = ocr_model.predict(img)
        cost_time = round((time.time() - start_time) * 1000, 2)

        # 结果解析
        if not result or not result[0]:
            print(f"🔍 未检测到文字 | 耗时: {cost_time}ms")
            continue

        text_boxes = result[0]
        detect_num = len(text_boxes)
        rec_texts = [box[1][0] for box in text_boxes]

        print(f"🔍 检测到文本数量: {detect_num}")
        print(f"📝 识别文本: {rec_texts}")
        print(f"⚡ 推理耗时: {cost_time} ms")
        print(f"✅ 尺寸 {size} 测试完成")

if __name__ == "__main__":
    # 测试配置（你的图片路径）
    TEST_IMG_PATH = r"P:\.vscode\Ship_Detection\pipeline\test\ocr1.jpg"
    TEST_SIZE_LIST = [640, 800, 960, 1024, 1280, 1536]
    
    # 执行测试
    test_ocr_input_size(TEST_IMG_PATH, TEST_SIZE_LIST)