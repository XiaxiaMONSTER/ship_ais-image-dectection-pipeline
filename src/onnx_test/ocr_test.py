
import onnxruntime as ort
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Tuple, List

"""
如遇IR版本兼容问题，请重新安装依赖：
pip install onnx==1.16.0 onnxruntime-gpu==1.19.2 opencv-python matplotlib
"""

# ===================== 配置项（可根据需求调整） =====================
MODEL_PATH = "./PP-OCRv5_server_det_onnx/inference.onnx"  # ONNX模型路径
TEST_IMAGE_PATH = "./test/ocr1.jpg"  # 测试图片路径（替换为你的图片路径）
OUTPUT_IMAGE_PATH = "./output/det_result.jpg"  # 检测结果保存路径
CONF_THRESH = 0.1  # 置信度阈值，过滤低置信度框
VISUALIZATION_COLOR = (0, 255, 0)  # 检测框颜色（BGR）
VISUALIZATION_THICKNESS = 2  # 检测框线宽
# 手动指定模型输入尺寸（PP-OCRv5默认常用尺寸，可根据你的模型调整）
INPUT_HEIGHT = 1024
INPUT_WIDTH = 1024
USE_GPU = True  # 是否使用GPU（无GPU则设为False）
# ====================================================================


class PPOCRv5Detector:
    """PP-OCRv5文本检测ONNX模型推理类"""

    def __init__(self, model_path: str, input_h: int, input_w: int, use_gpu: bool = True):
        """
        初始化推理器
        :param model_path: ONNX模型路径
        :param input_h: 模型输入高度（手动指定，解决动态维度问题）
        :param input_w: 模型输入宽度（手动指定，解决动态维度问题）
        :param use_gpu: 是否使用GPU推理（需安装onnxruntime-gpu且有可用GPU）
        """
        self.model_path = model_path
        self.input_h = input_h  # 手动指定固定输入高度
        self.input_w = input_w  # 手动指定固定输入宽度
        self.use_gpu = use_gpu
        self.session = self._load_model()
        print(f"✅ ONNX模型加载成功！手动指定输入形状：(1, 3, {self.input_h}, {self.input_w})")
        print("输入节点：", [input.name for input in self.session.get_inputs()])
        print("输出节点：", [output.name for output in self.session.get_outputs()])

    def _load_model(self) -> ort.InferenceSession:
        """加载ONNX模型，修复provider_options长度不匹配问题"""
        try:
            # 修复：根据是否使用GPU配置providers和provider_options
            if self.use_gpu:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                # 仅给CUDA指定provider_options，CPU无需配置
                provider_options = [{"device_id": 0}, {}]  # 长度与providers一致
            else:
                providers = ["CPUExecutionProvider"]
                provider_options = None  # CPU无需provider_options

            session = ort.InferenceSession(
                self.model_path,
                providers=providers,
                provider_options=provider_options if self.use_gpu else None
            )
            return session
        except Exception as e:
            print(f"⚠️ GPU推理初始化失败，自动回退到CPU：{str(e)}")
            # 兜底：强制使用CPU
            session = ort.InferenceSession(
                self.model_path,
                providers=["CPUExecutionProvider"]
            )
            self.use_gpu = False
            return session

    def preprocess(self, image_path: str) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        图像预处理（适配PP-OCRv5输入要求，使用手动指定的固定尺寸）
        :param image_path: 图片路径
        :return: 预处理后的张量、原始图片尺寸 (h, w)
        """
        # 1. 读取图片
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"无法读取图片：{image_path}")
        orig_h, orig_w = img.shape[:2]
        print(f"📷 原始图片尺寸：{orig_w}x{orig_h}")

        # 2. 预处理：缩放（使用手动指定的固定尺寸）、归一化、转置
        img_resized = cv2.resize(img, (self.input_w, self.input_h))  # 修复：使用整数尺寸
        img_norm = img_resized / 255.0  # 归一化到0-1
        img_trans = np.transpose(img_norm, (2, 0, 1))  # HWC -> CHW
        img_tensor = np.expand_dims(img_trans, axis=0).astype(np.float32)  # 增加batch维度

        return img_tensor, (orig_h, orig_w)

    def infer(self, image_tensor: np.ndarray) -> np.ndarray:
        """
        模型推理
        :param image_tensor: 预处理后的输入张量
        :return: 模型输出（文本检测结果）
        """
        try:
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: image_tensor})
            return outputs[0]  # PP-OCRv5检测模型主输出为第一个结果
        except Exception as e:
            raise RuntimeError(f"推理失败：{str(e)}")

    def postprocess(self, output: np.ndarray, orig_size: Tuple[int, int]) -> List[np.ndarray]:
        """
        后处理：解析检测框，缩放到原始图片尺寸，过滤低置信度框
        :param output: 模型输出
        :param orig_size: 原始图片尺寸 (h, w)
        :return: 过滤后的检测框列表（每个框为4个点的坐标）
        """
        orig_h, orig_w = orig_size
        # 模型输出下采样倍数（PP-OCRv5检测模型通常下采样4倍，可根据模型调整）
        downsample_ratio = 4
        output_h = self.input_h // downsample_ratio
        output_w = self.input_w // downsample_ratio

        # PP-OCRv5输出格式：(1, 8, output_h, output_w)，8对应4个点的(x,y)坐标
        output = output.squeeze(0)  # 去掉batch维度 (8, output_h, output_w)
        output = output.transpose(1, 2, 0)  # (output_h, output_w, 8)

        # 解析检测框
        boxes = []
        for y in range(output_h):
            for x in range(output_w):
                # 获取当前位置的置信度（最后一个通道为置信度）
                conf = output[y, x, -1]
                if conf < CONF_THRESH:
                    continue

                # 获取4个点的坐标（缩放回输入尺寸）
                points = output[y, x, :8].reshape(4, 2)
                points[:, 0] = points[:, 0] * (self.input_w / output_w)
                points[:, 1] = points[:, 1] * (self.input_h / output_h)

                # 再缩放到原始图片尺寸
                points[:, 0] = points[:, 0] * (orig_w / self.input_w)
                points[:, 1] = points[:, 1] * (orig_h / self.input_h)

                boxes.append(points.astype(np.int32))

        return boxes

    def visualize(self, image_path: str, boxes: List[np.ndarray], save_path: str) -> None:
        """
        可视化检测结果：绘制检测框，显示并保存图片
        :param image_path: 原始图片路径
        :param boxes: 检测框列表
        :param save_path: 结果保存路径
        """
        # 1. 读取图片
        img = cv2.imread(image_path)
        # 2. 绘制每个检测框（四边形）
        for box in boxes:
            cv2.polylines(img, [box], isClosed=True, color=VISUALIZATION_COLOR, thickness=VISUALIZATION_THICKNESS)
        # 3. 转换颜色空间（OpenCV BGR → Matplotlib RGB）
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 4. 显示图片
        plt.figure(figsize=(12, 8))
        plt.imshow(img_rgb)
        plt.axis("off")
        plt.title(f"PP-OCRv5 Text Detection (GPU: {self.use_gpu}, Boxes: {len(boxes)})")
        plt.tight_layout()
        plt.show()

        # 5. 保存图片
        cv2.imwrite(save_path, img)
        print(f"📸 检测结果已保存至：{save_path}")


def main():
    """主函数：完整推理+可视化流程"""
    # 1. 检查文件是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 模型文件不存在：{MODEL_PATH}")
        return
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"❌ 测试图片不存在：{TEST_IMAGE_PATH}")
        return

    # 2. 初始化检测器（传入手动指定的输入尺寸）
    try:
        detector = PPOCRv5Detector(
            model_path=MODEL_PATH,
            input_h=INPUT_HEIGHT,
            input_w=INPUT_WIDTH,
            use_gpu=USE_GPU
        )
    except RuntimeError as e:
        print(f"❌ 检测器初始化失败：{e}")
        return

    # 3. 预处理
    try:
        img_tensor, orig_size = detector.preprocess(TEST_IMAGE_PATH)
    except FileNotFoundError as e:
        print(f"❌ 预处理失败：{e}")
        return

    # 4. 推理
    try:
        output = detector.infer(img_tensor)
    except RuntimeError as e:
        print(f"❌ 推理失败：{e}")
        return

    # 5. 后处理
    boxes = detector.postprocess(output, orig_size)
    print(f"🔍 检测到文本框数量：{len(boxes)}")

    # 6. 可视化
    detector.visualize(TEST_IMAGE_PATH, boxes, OUTPUT_IMAGE_PATH)


if __name__ == "__main__":
    main()