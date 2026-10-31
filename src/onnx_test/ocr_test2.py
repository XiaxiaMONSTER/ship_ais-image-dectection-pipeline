import cv2
import numpy as np
import onnxruntime as ort
import time
import re
from PIL import Image, ImageDraw, ImageFont
import os

# ======================== 配置项（适配你的环境） ========================
# ONNX模型路径（请替换为你的实际ONNX模型路径）
DET_ONNX_PATH = r"P:\.vscode\Ship_Detection\pipeline\models\paddleocr_v5\PP-OCRv5_server_det_onnx\inference.onnx"
REC_ONNX_PATH = r"P:\.vscode\Ship_Detection\pipeline\models\paddleocr_v5\PP-OCRv5_server_rec_onnx\inference.onnx"
# 识别字典路径（PaddleOCR默认字典，不存在则用内置）
REC_CHAR_DICT_PATH = "./ppocr/utils/ppocr_keys_v1.txt"

# OCR配置（新手测试建议降低阈值）
OCR_DET_THRESH = 0.3  # 检测置信度阈值（降低！）
OCR_REC_THRESH = 0.3  # 识别置信度阈值（降低！）
DET_INPUT_SIZE = (640, 640)  # 检测模型输入尺寸（需和转换ONNX时一致）
REC_INPUT_SIZE = (32, 320)   # 识别模型输入尺寸（需和转换ONNX时一致）
ENABLE_GPU = False  # 先禁用GPU，排除GPU兼容性问题！
TEXT_MIN_LENGTH = 1  # 降低文本最小长度要求

# 可视化配置
FONT_SCALE = 1
FONT_MULTIPALE = 20
PADDING = 2
CORNER_RADIUS = 8
OCR_TEXT_COLOR = (255, 255, 255)  # 文字颜色
OCR_BG_COLOR = (0, 0, 0)          # 背景色
OCR_BOX_COLOR = (255, 165, 0)     # 检测框颜色

# 中文字体路径（自动适配系统）
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
PIL_AVAILABLE = True if CHINESE_FONT_PATH else False

# ======================== 工具函数（修复核心逻辑） ========================
def resize_img(img, input_size=(640, 640)):
    """PaddleOCR标准图像缩放（保持比例+填充）"""
    img_h, img_w = img.shape[:2]
    max_len = max(img_h, img_w)
    scale = input_size[0] / max_len
    new_h = int(img_h * scale)
    new_w = int(img_w * scale)
    resized_img = cv2.resize(img, (new_w, new_h))
    
    # 填充为正方形（和PaddleOCR一致）
    pad_img = np.zeros((input_size[0], input_size[1], 3), dtype=np.float32)
    pad_img[:new_h, :new_w, :] = resized_img
    return pad_img, scale, (new_w, new_h)

def det_postprocess(pred, scale, ori_shape, input_size, thresh=0.3):
    """修复PaddleOCR V5检测模型后处理"""
    # 打印输入维度（调试关键）
    print(f"[调试] det_postprocess输入pred shape: {pred.shape}")
    
    # 核心：维度校验，不符合预期直接返回空列表
    if len(pred.shape) != 4 or pred.shape[1] != 8:
        print(f"错误：检测模型输出维度异常！预期(1,8,H,W)，实际{pred.shape}")
        return []
    
    # 原后处理逻辑（略）
    pred = pred[0]
    heatmap = pred[0:2, :, :]  # 置信度热力图
    offset = pred[2:8, :, :]   # 坐标偏移量
    
    # 1. 提取置信度>阈值的像素点
    score = heatmap[1, :, :]  # 文本区域置信度
    coords = np.where(score > thresh)
    if len(coords[0]) == 0:
        return []
    
    # 2. 解析坐标（还原到原图尺寸）
    h, w = score.shape
    boxes = []
    for y, x in zip(coords[0], coords[1]):
        # 偏移量解析（修正坐标还原逻辑）
        dx1 = offset[0, y, x]
        dy1 = offset[1, y, x]
        dx2 = offset[2, y, x]
        dy2 = offset[3, y, x]
        dx3 = offset[4, y, x]
        dy3 = offset[5, y, x]
        dx4 = offset[6, y, x]
        dy4 = offset[7, y, x]
        
        # 修正坐标还原公式（适配PaddleOCR V5缩放逻辑）
        x_base = x * 4
        y_base = y * 4
        x1 = int((x_base + dx1) * scale / (input_size[0] / ori_shape[1]))
        y1 = int((y_base + dy1) * scale / (input_size[1] / ori_shape[0]))
        x2 = int((x_base + dx2) * scale / (input_size[0] / ori_shape[1]))
        y2 = int((y_base + dy2) * scale / (input_size[1] / ori_shape[0]))
        x3 = int((x_base + dx3) * scale / (input_size[0] / ori_shape[1]))
        y3 = int((y_base + dy3) * scale / (input_size[1] / ori_shape[0]))
        x4 = int((x_base + dx4) * scale / (input_size[0] / ori_shape[1]))
        y4 = int((y_base + dy4) * scale / (input_size[1] / ori_shape[0]))
        
        # 转换为矩形框 + 过滤小框
        min_x = min(x1, x2, x3, x4)
        min_y = min(y1, y2, y3, y4)
        max_x = max(x1, x2, x3, x4)
        max_y = max(y1, y2, y3, y4)
        if (max_x - min_x) < 5 or (max_y - min_y) < 5:
            continue
        
        boxes.append([min_x, min_y, max_x, max_y, score[y, x]])
    return boxes

# ======================== ONNX Runtime 初始化 ========================
class ONNXPaddleOCR:
    def __init__(self, det_onnx_path, rec_onnx_path, char_dict_path):
        # 初始化ONNX Runtime会话（先禁用GPU）
        providers = ['CPUExecutionProvider']
        if ENABLE_GPU and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        print(f"使用推理后端: {providers}")
        
        # 加载检测模型 + 打印输入输出信息
        self.det_session = ort.InferenceSession(det_onnx_path, providers=providers)
        self.det_input_name = self.det_session.get_inputs()[0].name
        det_input_shape = self.det_session.get_inputs()[0].shape
        det_outputs = self.det_session.get_outputs()
        print(f"检测模型输入 - 名称: {self.det_input_name}, 预期shape: {det_input_shape}")
        for i, out in enumerate(det_outputs):
            print(f"检测模型输出{i} - 名称: {out.name}, 预期shape: {out.shape}")
        
        # 加载识别模型 + 打印输入输出信息
        self.rec_session = ort.InferenceSession(rec_onnx_path, providers=providers)
        self.rec_input_name = self.rec_session.get_inputs()[0].name
        rec_input_shape = self.rec_session.get_inputs()[0].shape
        rec_outputs = self.rec_session.get_outputs()
        print(f"识别模型输入 - 名称: {self.rec_input_name}, 预期shape: {rec_input_shape}")
        for i, out in enumerate(rec_outputs):
            print(f"识别模型输出{i} - 名称: {out.name}, 预期shape: {out.shape}")
        
        # 加载识别字典
        self.char_dict = self._load_char_dict(char_dict_path)

    def _load_char_dict(self, dict_path):
        """加载识别字符字典（适配PaddleOCR默认格式）"""
        if os.path.exists(dict_path):
            with open(dict_path, 'r', encoding='utf-8') as f:
                char_list = [line.strip() for line in f.readlines()]
            print(f"加载字典成功，共{len(char_list)}个字符")
            return char_list
        else:
            print(f"未找到字典文件 {dict_path}，使用内置中文字典")
            # PaddleOCR默认字典（精简版）
            return ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
                    '一', '二', '三', '四', '五', '六', '七', '八', '九', '十',
                    '百', '千', '万', '亿', '零', '壹', '贰', '叁', '肆', '伍',
                    '陆', '柒', '捌', '玖', '拾', '佰', '仟', '船', '号', '渔', '货', '客']

    def det_preprocess(self, img):
        """检测模型预处理（和PaddleOCR完全一致）"""
        # BGR转RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 缩放+填充
        pad_img, scale, new_size = resize_img(img_rgb, DET_INPUT_SIZE)
        # 归一化
        pad_img = pad_img / 255.0
        # 转换为CHW格式
        pad_img = np.transpose(pad_img, (2, 0, 1)).astype(np.float32)
        # 增加batch维度
        pad_img = np.expand_dims(pad_img, axis=0)
        return pad_img, scale, img.shape[:2]

    def rec_preprocess(self, img_roi):
        """识别模型预处理（修复：保留3通道，不灰度化）"""
        # BGR转RGB
        img_rgb = cv2.cvtColor(img_roi, cv2.COLOR_BGR2RGB)
        # 缩放（保持高度为32，等比例缩放宽度）
        h, w = img_rgb.shape[:2]
        scale = h / REC_INPUT_SIZE[0]
        new_w = int(w / scale)
        if new_w > REC_INPUT_SIZE[1]:
            new_w = REC_INPUT_SIZE[1]
        img_resized = cv2.resize(img_rgb, (new_w, REC_INPUT_SIZE[0]))
        
        # 填充到320宽度
        pad_img = np.ones((REC_INPUT_SIZE[0], REC_INPUT_SIZE[1], 3), dtype=np.float32) * 255
        pad_img[:, :new_w, :] = img_resized
        
        # 归一化
        pad_img = pad_img / 255.0
        # CHW格式
        pad_img = np.transpose(pad_img, (2, 0, 1)).astype(np.float32)
        # 增加batch维度
        pad_img = np.expand_dims(pad_img, axis=0)
        return pad_img

    def rec_postprocess(self, pred):
        """识别模型后处理（修复解码逻辑）"""
        # pred shape: (1, 37, 80) -> (时间步, 字符数)
        pred = pred[0]
        # 取每个时间步的最大概率字符
        pred_idx = np.argmax(pred, axis=1)
        # 解码（去重+跳过空白符）
        text = ''
        prev_idx = -1
        conf_sum = 0.0
        valid_count = 0
        for idx in pred_idx:
            if idx == 0 or idx == prev_idx:  # 空白符或重复字符
                continue
            if idx < len(self.char_dict):
                text += self.char_dict[idx]
                conf_sum += np.max(pred[np.where(pred_idx == idx)])
                valid_count += 1
            prev_idx = idx
        # 计算平均置信度
        conf = conf_sum / valid_count if valid_count > 0 else 0.0
        return text, conf

    def is_valid_text(self, text):
        """放宽文本过滤条件"""
        if not text or len(text) < TEXT_MIN_LENGTH:
            return False
        # 只过滤纯标点/空白
        if re.match(r'^[\s\W_]+$', text):
            return False
        return True

    def predict(self, img):
        """执行OCR识别（修复核心逻辑）"""
        # 1. 检测模型推理
        det_input, scale, ori_shape = self.det_preprocess(img)
        print(f"[调试] 检测输入shape: {det_input.shape}")
        
        try:
            det_pred = self.det_session.run(None, {self.det_input_name: det_input})
        except Exception as e:
            print(f"检测模型推理失败: {e}")
            return []
        print(f"[调试] 检测输出数量: {len(det_pred)}, 第一个输出shape: {det_pred[0].shape}")
        
        # 检测后处理（捕获异常）
        try:
            boxes = det_postprocess(det_pred[0], scale, ori_shape, DET_INPUT_SIZE, OCR_DET_THRESH)
        except Exception as e:
            print(f"检测后处理失败: {e}")
            import traceback
            traceback.print_exc()  # 打印详细错误堆栈
            return []
        print(f"检测到{len(boxes)}个文本框")
        
        # 2. 识别模型推理（原逻辑，略）
        results = []
        for box in boxes:
            x1, y1, x2, y2, det_conf = box
            # 裁剪ROI（防止越界）
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img.shape[1]-1, x2)
            y2 = min(img.shape[0]-1, y2)
            roi = img[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            
            # 识别推理（添加异常捕获）
            try:
                rec_input = self.rec_preprocess(roi)
                rec_pred = self.rec_session.run(None, {self.rec_input_name: rec_input})
                text, rec_conf = self.rec_postprocess(rec_pred[0])
            except Exception as e:
                print(f"识别失败（框{box}）: {e}")
                continue
            
            # 过滤低置信度
            if rec_conf >= OCR_REC_THRESH and self.is_valid_text(text):
                results.append({
                    "text": text,
                    "conf": rec_conf,
                    "bbox": (x1, y1, x2, y2)
                })
        return results

# ======================== 可视化函数（保留） ========================
def draw_rounded_rectangle(img, top_left, bottom_right, color, thickness=-1, radius=10):
    x1, y1 = top_left
    x2, y2 = bottom_right
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = max(x1+1, x2), max(y1+1, y2)
    radius = min(radius, (x2-x1)//2, (y2-y1)//2)
    
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

def draw_chinese_text(img, text, pos, font_size=0.8, color=(255,255,255)):
    if not PIL_AVAILABLE:
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_size, color, 1)
        return img
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    
    font_height = max(12, int(font_size * FONT_MULTIPALE))
    font = ImageFont.truetype(CHINESE_FONT_PATH, font_height)
    pil_color = (color[2], color[1], color[0])
    
    draw.text(pos, text, font=font, fill=pil_color, stroke_width=0.5)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def get_text_size(text, font_size=0.8):
    if not PIL_AVAILABLE:
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_size, 1)
        return (w, h)
    
    font_height = max(12, int(font_size * FONT_MULTIPALE))
    font = ImageFont.truetype(CHINESE_FONT_PATH, font_height)
    dummy_img = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy_img)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return (right - left + 2, bottom - top + 2)

def draw_ocr_results(img, ocr_results):
    h, w = img.shape[:2]
    for res in ocr_results:
        text = res["text"]
        conf = res["conf"]
        x1, y1, x2, y2 = res["bbox"]
        
        # 绘制检测框
        cv2.rectangle(img, (x1, y1), (x2, y2), OCR_BOX_COLOR, 2)
        
        # 绘制文本标签
        display_text = f"{text}({conf:.2f})"
        text_w, text_h = get_text_size(display_text, FONT_SCALE)
        
        # 计算标签位置
        label_tl_x = x1
        label_tl_y = max(0, y1 - text_h - 2 * PADDING - 5)
        label_br_x = min(w, label_tl_x + text_w + 2 * PADDING)
        label_br_y = label_tl_y + text_h + 2 * PADDING
        
        # 绘制圆角背景
        img = draw_rounded_rectangle(img, (label_tl_x, label_tl_y), 
                                    (label_br_x, label_br_y), OCR_BG_COLOR, -1, CORNER_RADIUS)
        
        # 绘制文本
        text_pos = (label_tl_x + PADDING, label_tl_y + PADDING)
        img = draw_chinese_text(img, display_text, text_pos, FONT_SCALE, OCR_TEXT_COLOR)
    return img

# ======================== 测试主函数（修复路径问题） ========================
def test_ocr_onnx(image_path):
    """测试ONNX版本OCR"""
    # 检查图片是否存在
    if not os.path.exists(image_path):
        print(f"错误：图片路径不存在 -> {image_path}")
        return
    
    # 初始化ONNX OCR
    print("初始化ONNX OCR引擎...")
    try:
        ocr = ONNXPaddleOCR(
            det_onnx_path=DET_ONNX_PATH,
            rec_onnx_path=REC_ONNX_PATH,
            char_dict_path=REC_CHAR_DICT_PATH
        )
    except Exception as e:
        print(f"初始化失败：{e}")
        return
    
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return
    
    # 执行OCR
    print("执行OCR识别...")
    start_time = time.time()
    try:
        ocr_results = ocr.predict(img)
    except Exception as e:
        print(f"推理失败：{e}")
        return
    end_time = time.time()
    
    # 打印结果
    print(f"识别耗时: {end_time - start_time:.3f}秒")
    print(f"最终识别到{len(ocr_results)}个有效文本：")
    for i, res in enumerate(ocr_results):
        print(f"  {i+1}. 文本: {res['text']}, 置信度: {res['conf']:.3f}, 位置: {res['bbox']}")
    
    # 绘制结果并显示
    img_draw = draw_ocr_results(img.copy(), ocr_results)
    cv2.imshow("OCR Result (ONNX)", img_draw)
    
    # 创建output文件夹（修复路径问题）
    output_dir = "./output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存结果
    save_path = os.path.join(output_dir, "ocr_onnx_result.jpg")
    cv2.imwrite(save_path, img_draw)
    print(f"结果已保存到: {save_path}")
    
    # 等待按键关闭窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ======================== 运行测试 ========================
if __name__ == "__main__":
    # 替换为你的测试图片路径
    TEST_IMAGE_PATH = r"P:\.vscode\Ship_Detection\pipeline\test\ocr1.jpg"
    test_ocr_onnx(TEST_IMAGE_PATH)