"""
RKNN YOLO 模型对比脚本
支持导入 1 个或 2 个 RKNN 检测模型，对同一视频/图片进行检测，
结果以左右拼接对比方式展示，两侧支持自定义中文标题。

用法示例:
  # 单模型（左=原图, 右=检测结果）
  python model_compare.py -m day.rknn -v test.mp4 --title "白天模型"

  # 双模型对比（带保存）
  python model_compare.py -m /home/Tronlong/Desktop/rknn_model_zoo_and_models/yolov11n-day/day_4-best_255_fp.rknn /home/Tronlong/Desktop/rknn_model_zoo_and_models/yolov11n-day/day_5-best_255_fp.rknn -v /home/Tronlong/Desktop/pipeline/test/手机比例_暴风雨中航行的船只视频.mp4 \
      --title "无极端天气优化" "含极端天气优化" -o /home/Tronlong/Desktop/pipeline/output/compare_result.mp4

  # 图片对比
  python model_compare.py -m a.rknn b.rknn -i photo.jpg \
      --title "模型A" "模型B"

参考: src/16_cv2_data_saved.py 中 YOLO_RKNN / YOLOv11RKNN 的处理逻辑
"""

import cv2
import numpy as np
import os
import sys
import time
import argparse

# ======================== Freetype 混合渲染（零拷贝中文绘制） ========================
_ft_handle = None
_CHINESE_FONT_PATH = None


def _init_font():
    global _CHINESE_FONT_PATH, _ft_handle
    font_candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ]
    for fp in font_candidates:
        if os.path.exists(fp):
            _CHINESE_FONT_PATH = fp
            break
    if _CHINESE_FONT_PATH and hasattr(cv2, "freetype"):
        _ft_handle = cv2.freetype.createFreeType2()
        _ft_handle.loadFontData(fontFileName=_CHINESE_FONT_PATH, id=0)
        print(f"✅ Freetype 中文就绪: {_CHINESE_FONT_PATH}")
    else:
        print("⚠️ Freetype 中文不可用，将仅使用 ASCII 渲染")


def _get_ft():
    global _ft_handle, _CHINESE_FONT_PATH
    if _ft_handle is None:
        _init_font()
    return _ft_handle


def _needs_freetype(c):
    cp = ord(c)
    return not (0x20 <= cp <= 0x7E)


def _put_mixed_text(img, text, org, font_height, color, thickness):
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


def draw_chinese_text(img, text, pos, font_size=20, color=(255, 255, 255), thickness=2):
    """绘制支持中文的文本（原地修改）"""
    x, y_top = pos
    y_baseline = y_top + font_size
    _put_mixed_text(img, text, (x, y_baseline), font_size, color, thickness)
    return img


def get_text_size(text, font_size=20, thickness=2):
    """计算混合文本尺寸"""
    return _get_mixed_text_size(text, font_size, thickness)


# ======================== 绘制工具 ========================
def draw_rounded_rectangle(img, top_left, bottom_right, color, thickness=-1, radius=10):
    x1, y1 = max(0, int(top_left[0])), max(0, int(top_left[1]))
    x2, y2 = max(x1 + 1, int(bottom_right[0])), max(y1 + 1, int(bottom_right[1]))
    radius = min(radius, max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2))

    if thickness == -1:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    else:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)

    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)
    return img


def draw_title_bar(img, title, color=(40, 40, 40)):
    """在图像顶部绘制标题栏（支持中文）"""
    h, w = img.shape[:2]
    font_height = 24
    tw, th = get_text_size(title, font_height)

    bar_h = th + 20
    bar_h = min(bar_h, h // 6)

    # 半透明背景
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), color, -1)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)

    # 居中文字
    text_x = max(10, (w - tw) // 2)
    text_y = (bar_h - th) // 2
    draw_chinese_text(img, title, (text_x, text_y), font_size=font_height,
                       color=(255, 255, 255), thickness=2)
    return img


def draw_detections(img, detections, names, colors=None):
    """在图像上绘制检测框和标签"""
    if colors is None:
        colors = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
        ]

    h, w = img.shape[:2]
    font_size = 18
    box_thickness = 2

    for det in detections:
        x1n, y1n, x2n, y2n, conf, cls_id = det

        x1 = int(x1n * w)
        y1 = int(y1n * h)
        x2 = int(x2n * w)
        y2 = int(y2n * h)

        cls_int = int(cls_id)
        color = colors[cls_int % len(colors)]
        cls_name = names.get(cls_int, f"cls{cls_int}")

        # 检测框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, box_thickness)

        # 标签
        label = f"{cls_name} {conf:.2f}"
        lw, lh = get_text_size(label, font_size)
        padding = 4

        label_y1 = y1 - lh - padding * 2 - 6
        if label_y1 < 0:
            label_y1 = y2 + 6

        draw_rounded_rectangle(
            img,
            (x1, label_y1),
            (x1 + lw + padding * 2, label_y1 + lh + padding * 2),
            color, thickness=-1, radius=6
        )
        draw_chinese_text(img, label, (x1 + padding, label_y1 + padding),
                           font_size=font_size, color=(255, 255, 255), thickness=2)

    return img


# ======================== RKNN Lite 封装 ========================
class YOLO_RKNN:
    """RKNN 模型底层封装"""

    def __init__(self, model_path):
        from rknnlite.api import RKNNLite
        self.rknn_lite = RKNNLite()
        ret = self.rknn_lite.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f"RKNN 加载失败: {model_path}")
        ret = self.rknn_lite.init_runtime()
        if ret != 0:
            raise RuntimeError(f"RKNN 初始化失败: {model_path}")
        print(f"✅ RKNN 模型就绪: {model_path}")

    def run(self, inputs, data_format=None):
        return self.rknn_lite.inference(inputs=inputs, data_format=data_format)


# ======================== YOLOv11 RKNN 检测模型 ========================
class YOLOv11RKNN:
    """YOLOv11 RKNN 检测模型（与 16_cv2_data_saved.py 一致）"""

    CONF_THRESH = 0.45
    IOU_THRESH = 0.25

    def __init__(self, model_path, input_size=640, names=None):
        self.model_path = model_path
        self.input_size = input_size
        self.rknn = YOLO_RKNN(model_path)

        if names is not None:
            self.names = names
        else:
            self.names = {0: "ship"}

        print(f"✅ YOLOv11 RKNN 检测模型: {model_path}, 尺寸={input_size}, 类别={self.names}")

    def _letterbox(self, img, new_size):
        h0, w0 = img.shape[:2]
        r = min(new_size / h0, new_size / w0)
        new_w = int(round(w0 * r))
        new_h = int(round(h0 * r))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        dw = (new_size - new_w) / 2
        dh = (new_size - new_h) / 2
        top = int(round(dh - 0.1))
        bottom = int(round(dh + 0.1))
        left = int(round(dw - 0.1))
        right = int(round(dw + 0.1))

        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        return padded, r, (dw, dh)

    def _preprocess(self, img):
        """检测预处理：RGB 顺序，NHWC 格式，uint8"""
        padded, r, (dw, dh) = self._letterbox(img, self.input_size)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        blob = np.expand_dims(rgb, axis=0)
        return blob, r, (dw, dh)

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
    def _box_process(position, grid_h, grid_w, input_size=640):
        col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
        col = col.reshape(1, 1, grid_h, grid_w)
        row = row.reshape(1, 1, grid_h, grid_w)
        grid = np.concatenate((col, row), axis=1)

        stride = np.array([input_size // grid_h, input_size // grid_w]).reshape(1, 2, 1, 1)

        position = YOLOv11RKNN._dfl_numpy(position)
        box_xy = grid + 0.5 - position[:, 0:2, :, :]
        box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
        xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)
        return xyxy

    def _postprocess(self, outputs, orig_shape, r, dw, dh):
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
        num_classes = len(self.names)

        for (h, w), tensors in layers.items():
            box_t = score_t = obj_t = None

            for t in tensors:
                c = t.shape[1]
                if c == 64:
                    box_t = t
                elif c == num_classes:
                    score_t = t
                elif c == 1:
                    obj_t = t

            if box_t is None or score_t is None:
                continue

            s = score_t
            if obj_t is not None:
                s = s * obj_t

            s = s.transpose(0, 2, 3, 1).reshape(-1, num_classes)
            b = self._box_process(box_t, h, w, self.input_size).transpose(0, 2, 3, 1).reshape(-1, 4)

            max_scores = np.max(s, axis=1)
            max_class_ids = np.argmax(s, axis=1)

            idx = max_scores > 0.05
            all_boxes.append(b[idx])
            all_scores.append(max_scores[idx])
            all_class_ids.append(max_class_ids[idx])

        if not all_boxes:
            return []

        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        class_ids = np.concatenate(all_class_ids, axis=0)

        # NMS
        nms_boxes = []
        for box in boxes:
            nms_boxes.append([box[0], box[1], box[2] - box[0], box[3] - box[1]])

        indices = cv2.dnn.NMSBoxes(
            nms_boxes, scores.tolist(),
            self.CONF_THRESH, self.IOU_THRESH
        )

        if len(indices) == 0:
            return []

        if isinstance(indices, np.ndarray):
            indices = indices.flatten()

        final_boxes = boxes[indices]
        final_scores = scores[indices]
        final_cls = class_ids[indices]

        orig_h, orig_w = orig_shape
        results = []
        for box, score, cls_id in zip(final_boxes, final_scores, final_cls):
            x1 = (box[0] - dw) / r
            y1 = (box[1] - dh) / r
            x2 = (box[2] - dw) / r
            y2 = (box[3] - dh) / r

            x1n = np.clip(x1 / orig_w, 0.0, 1.0)
            y1n = np.clip(y1 / orig_h, 0.0, 1.0)
            x2n = np.clip(x2 / orig_w, 0.0, 1.0)
            y2n = np.clip(y2 / orig_h, 0.0, 1.0)

            results.append((x1n, y1n, x2n, y2n, float(score), float(cls_id)))

        return results

    def predict(self, frame):
        """对单帧进行检测，返回 [(x1n, y1n, x2n, y2n, conf, cls_id), ...]"""
        shape = frame.shape[:2]
        blob, r, (dw, dh) = self._preprocess(frame)
        out = self.rknn.run([blob], data_format='nhwc')
        return self._postprocess(out, shape, r, dw, dh)


# ======================== 模型对比器 ========================
class ModelComparator:
    """双模型对比引擎"""

    def __init__(self, model_paths, names_list=None, input_size=640,
                 titles=None, conf_thresh=None, iou_thresh=None):
        """
        :param model_paths: RKNN 模型路径列表 (1 或 2 个)
        :param names_list: 每个模型的类别名 dict 列表，如 [{0:"ship"}, {0:"light"}]
        :param input_size: 模型输入尺寸
        :param titles: 两侧标题 [left_title, right_title]
        :param conf_thresh: 覆盖置信度阈值
        :param iou_thresh: 覆盖 IOU 阈值
        """
        self.num_models = len(model_paths)
        if self.num_models < 1 or self.num_models > 2:
            raise ValueError("仅支持 1 个或 2 个模型")

        self.models = []
        for i, mp in enumerate(model_paths):
            names = names_list[i] if names_list and i < len(names_list) else None
            model = YOLOv11RKNN(mp, input_size=input_size, names=names)
            if conf_thresh is not None:
                model.CONF_THRESH = conf_thresh
            if iou_thresh is not None:
                model.IOU_THRESH = iou_thresh
            self.models.append(model)

        # 标题
        if titles is None:
            titles = [f"模型 {i + 1}" for i in range(self.num_models)]
        while len(titles) < 2:
            titles.append(f"模型 {len(titles) + 1}")
        self.titles = titles[:2]

    def process_frame(self, frame):
        """处理一帧，返回 (left_img, right_img)"""
        orig = frame.copy()

        if self.num_models == 1:
            # 单模型：左侧 = 原图（带标题），右侧 = 检测结果（带标题）
            dets = self.models[0].predict(frame)
            left = orig.copy()
            right = orig.copy()
            draw_detections(right, dets, self.models[0].names)
            draw_title_bar(left, self.titles[0])
            draw_title_bar(right, self.titles[1])
        else:
            # 双模型：左右各显示一个模型的检测结果
            dets_a = self.models[0].predict(frame)
            dets_b = self.models[1].predict(frame)

            left = orig.copy()
            right = orig.copy()

            draw_detections(left, dets_a, self.models[0].names)
            draw_detections(right, dets_b, self.models[1].names)

            draw_title_bar(left, self.titles[0])
            draw_title_bar(right, self.titles[1])

        return left, right

    def make_canvas(self, left, right, target_height=720):
        """将左右图像拼接成画布，等比缩放"""
        lh, lw = left.shape[:2]
        rh, rw = right.shape[:2]

        # 统一高度
        h = min(lh, rh)
        scale = target_height / h

        new_lw = int(lw * scale)
        new_lh = int(lh * scale)
        new_rw = int(rw * scale)
        new_rh = int(rh * scale)

        left_resized = cv2.resize(left, (new_lw, new_lh))
        right_resized = cv2.resize(right, (new_rw, new_rh))

        # 拼接
        canvas = np.hstack([left_resized, right_resized])

        # 中间分割线
        mid_x = new_lw
        cv2.line(canvas, (mid_x, 0), (mid_x, new_lh), (255, 255, 255), 3)

        # 底部状态栏
        bar_h = 36
        bar = np.zeros((bar_h, canvas.shape[1], 3), dtype=np.uint8)
        bar[:] = (40, 40, 40)

        # 左侧模型信息
        info_l = f"Model A: {self.models[0].names}"
        draw_chinese_text(bar, info_l, (10, 8), font_size=18,
                           color=(200, 200, 200), thickness=1)

        # 右侧模型信息
        if self.num_models >= 2:
            info_r = f"Model B: {self.models[1].names}"
            tw, _ = get_text_size(info_r, 18)
            draw_chinese_text(bar, info_r, (canvas.shape[1] - tw - 10, 8),
                               font_size=18, color=(200, 200, 200), thickness=1)
        else:
            info_r = "Left: Original  |  Right: Detection"
            tw, _ = get_text_size(info_r, 18)
            draw_chinese_text(bar, info_r, (canvas.shape[1] - tw - 10, 8),
                               font_size=18, color=(200, 200, 200), thickness=1)

        canvas = np.vstack([canvas, bar])
        return canvas

    def run_image(self, image_path, output_path=None, display=True):
        """处理单张图片"""
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"❌ 无法读取图片: {image_path}")
            return

        print(f"📷 图片尺寸: {frame.shape[1]}x{frame.shape[0]}")

        t0 = time.perf_counter()
        left, right = self.process_frame(frame)
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"⏱️ 推理耗时: {elapsed:.1f}ms")

        canvas = self.make_canvas(left, right)

        if output_path:
            cv2.imwrite(output_path, canvas)
            print(f"💾 结果已保存: {output_path}")

        if display:
            cv2.imshow("Model Comparison", canvas)
            print("按任意键关闭...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return canvas

    def run_video(self, video_path, output_path=None, display=True, skip_frames=0):
        """处理视频文件"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ 无法打开视频: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"🎬 视频: {w}x{h}, {fps:.1f}fps, {total_frames} 帧")

        # 准备输出
        writer = None
        if output_path:
            # 先读一帧测 canvas 尺寸
            ret, frame = cap.read()
            if ret:
                left, right = self.process_frame(frame)
                canvas = self.make_canvas(left, right)
                ch, cw = canvas.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_path, fourcc, fps, (cw, ch))
                print(f"📹 输出视频: {output_path} ({cw}x{ch})")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frame_idx = 0
        total_elapsed = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if skip_frames > 0 and frame_idx % (skip_frames + 1) != 1:
                continue

            t0 = time.perf_counter()
            left, right = self.process_frame(frame)
            elapsed = (time.perf_counter() - t0) * 1000
            total_elapsed += elapsed

            canvas = self.make_canvas(left, right)

            if writer:
                writer.write(canvas)

            if display:
                cv2.imshow("Model Comparison", canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):
                    print("用户退出")
                    break
                elif key == ord(' '):
                    print("⏸️ 暂停，按任意键继续...")
                    cv2.waitKey(0)

            if frame_idx % 50 == 0:
                avg = total_elapsed / frame_idx
                print(f"  帧 {frame_idx}/{total_frames} | 平均 {avg:.1f}ms/frame | "
                      f"≈{1000/avg:.1f} FPS" if avg > 0 else f"  帧 {frame_idx}")

        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()

        avg = total_elapsed / max(frame_idx, 1)
        print(f"\n✅ 完成 {frame_idx} 帧 | 平均耗时: {avg:.1f}ms | "
              f"≈{1000/avg:.1f} FPS" if avg > 0 else f"\n✅ 完成 {frame_idx} 帧")

    def run_rtsp(self, rtsp_url, display=True):
        """处理 RTSP 实时流"""
        print(f"🔗 连接 RTSP: {rtsp_url}")
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            # 尝试 GStreamer
            pipeline = (
                f'rtspsrc location="{rtsp_url}" latency=100 protocols=tcp ! '
                f'rtph265depay ! h265parse ! mppvideodec ! '
                f'videoconvert ! video/x-raw,format=BGR ! '
                f'appsink max-buffers=1 drop=true sync=false'
            )
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if not cap.isOpened():
                print(f"❌ 无法连接 RTSP: {rtsp_url}")
                return

        frame_idx = 0
        total_elapsed = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame_idx += 1
            t0 = time.perf_counter()
            left, right = self.process_frame(frame)
            elapsed = (time.perf_counter() - t0) * 1000
            total_elapsed += elapsed

            canvas = self.make_canvas(left, right)

            if display:
                cv2.imshow("Model Comparison - RTSP", canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):
                    print("用户退出")
                    break

            if frame_idx % 100 == 0:
                avg = total_elapsed / frame_idx
                print(f"  帧 {frame_idx} | 平均 {avg:.1f}ms | "
                      f"≈{1000/avg:.1f} FPS" if avg > 0 else f"  帧 {frame_idx}")

        cap.release()
        cv2.destroyAllWindows()


# ======================== 命令行入口 ========================
def main():
    parser = argparse.ArgumentParser(
        description="RKNN YOLO 模型对比 — 1~2 个模型对同一视频/图片做检测，左右拼接对比",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单模型：左=原图，右=检测
  python model_compare.py -m day.rknn -v test.mp4 --title "白天模型" "检测结果"

  # 双模型对比
  python model_compare.py -m day.rknn night.rknn -v test.mp4 --title "白天模型" "夜晚模型"

  # 图片对比
  python model_compare.py -m a.rknn b.rknn -i photo.jpg -o result.jpg

  # RTSP 实时流
  python model_compare.py -m day.rknn night.rknn --rtsp rtsp://admin:password@192.168.1.100:554/stream1
        """)

    parser.add_argument("-m", "--models", nargs="+", required=True,
                        help="RKNN 模型路径 (1个或2个)")
    parser.add_argument("-v", "--video", default=None,
                        help="输入视频文件路径")
    parser.add_argument("-i", "--image", default=None,
                        help="输入图片文件路径")
    parser.add_argument("--rtsp", default=None,
                        help="RTSP 流地址")
    parser.add_argument("-o", "--output", default=None,
                        help="输出文件路径 (视频用 .mp4, 图片用 .jpg/.png)")
    parser.add_argument("--titles", "--title", nargs="+", default=None,
                        dest="titles",
                        help="两侧标题 (1个模型时: [左侧标题] [右侧标题]; "
                             "2个模型时: [模型A标题] [模型B标题], 支持中文)")
    parser.add_argument("-s", "--input-size", type=int, default=640,
                        help="模型输入尺寸 (默认 640)")
    parser.add_argument("--names", nargs="+", default=None,
                        help="每个模型的类别名列表, 如: '0:ship,1:boat' '0:light' ")
    parser.add_argument("--conf", type=float, default=None,
                        help="覆盖置信度阈值")
    parser.add_argument("--iou", type=float, default=None,
                        help="覆盖 IOU 阈值")
    parser.add_argument("--no-display", action="store_true",
                        help="不显示窗口 (仅保存文件)")
    parser.add_argument("--skip", type=int, default=0,
                        help="跳帧间隔 (视频加速, 0=不跳帧)")

    args = parser.parse_args()

    # 校验模型数量
    if len(args.models) > 2:
        print("❌ 最多支持 2 个模型")
        sys.exit(1)

    # 校验输入源
    inputs = [args.video, args.image, args.rtsp]
    if sum(1 for x in inputs if x is not None) != 1:
        print("❌ 请指定恰好一种输入: --video / --image / --rtsp")
        sys.exit(1)

    # 解析类别名
    names_list = None
    if args.names:
        names_list = []
        for ns in args.names:
            d = {}
            for pair in ns.split(","):
                k, v = pair.split(":")
                d[int(k)] = v
            names_list.append(d)

    # 解析标题
    titles = args.titles

    # 如果单模型且未指定标题，第二个默认为 "Detection"
    if len(args.models) == 1 and titles is None:
        titles = ["Original", "Detection"]
    if len(args.models) == 1 and titles and len(titles) == 1:
        titles = [titles[0], "Detection"]

    # 初始化
    print("=" * 60)
    print("🚀 RKNN YOLO 模型对比")
    print("=" * 60)
    comparator = ModelComparator(
        model_paths=args.models,
        names_list=names_list,
        input_size=args.input_size,
        titles=titles,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
    )
    print(f"📋 标题: {comparator.titles}")
    print("=" * 60)

    display = not args.no_display

    try:
        if args.image:
            comparator.run_image(args.image, output_path=args.output, display=display)
        elif args.video:
            comparator.run_video(args.video, output_path=args.output,
                                  display=display, skip_frames=args.skip)
        elif args.rtsp:
            comparator.run_rtsp(args.rtsp, display=display)
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
