"""
YOLOv11 RKNN FP vs INT8 模型性能对比脚本

用法:
  python benchmark_fp_vs_int8.py
  python benchmark_fp_vs_int8.py --test_image /path/to/test.jpg
  python benchmark_fp_vs_int8.py --fp_model xxx.rknn --int8_model xxx.rknn --warmup 30 --iterations 200
"""

import cv2
import numpy as np
import time
import os
import sys
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ======================== RKNN 推理封装（同11_rtmp_data_saved.py） ========================
from rknnlite.api import RKNNLite


class YOLO_RKNN:
    def __init__(self, model_path):
        self.model_path = model_path
        self.rknn_lite = RKNNLite()
        ret = self.rknn_lite.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f'RKNN 加载失败: {model_path}')
        ret = self.rknn_lite.init_runtime()
        if ret != 0:
            raise RuntimeError(f'RKNN 初始化失败: {model_path}')

    def run(self, inputs, data_format=None):
        return self.rknn_lite.inference(inputs=inputs, data_format=data_format)


# ======================== 辅助类 ========================
class _ClsProbs:
    def __init__(self, top1, top1conf):
        self.top1 = top1
        self.top1conf = top1conf


class _ClsResult:
    def __init__(self, top1_idx, top1_conf, names):
        self.probs = _ClsProbs(top1_idx, top1_conf)
        self.names = names


class _Boxes:
    def __init__(self, dets):
        self._dets = dets

    def __iter__(self):
        return iter(self._dets)

    def __len__(self):
        return len(self._dets)


class _DetResult:
    def __init__(self, dets, names):
        self.boxes = _Boxes(dets) if dets else None
        self.names = names


# ======================== YOLOv11 RKNN 推理类 ========================
class YOLOv11RKNN:
    CONF_THRESH = 0.45
    IOU_THRESH = 0.25

    def __init__(self, model_path, model_type="det", input_size=640, names=None):
        self.model_path = model_path
        self.model_type = model_type
        self.input_size = input_size
        self.rknn = YOLO_RKNN(model_path)
        self.names = names if names is not None else {0: "ship"}

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

    def _preprocess_det(self, img):
        padded, r, (dw, dh) = self._letterbox(img, self.input_size)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        blob = np.expand_dims(rgb, axis=0)
        return blob, r, (dw, dh)

    # ============== 后处理（必要部分） ==============
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
        box_xy = grid + 0.5 - position[:, 0:2, :, :]
        box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
        xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)
        return xyxy

    def _postprocess_det(self, outputs, orig_shape, r, dw, dh):
        if outputs is None or len(outputs) == 0:
            return []
        orig_h, orig_w = orig_shape
        layers = {}
        for out in outputs:
            h, w = out.shape[2:4]
            if (h, w) not in layers:
                layers[(h, w)] = []
            layers[(h, w)].append(out)

        all_boxes = []
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
            b = self._box_process(box_t, h, w).transpose(0, 2, 3, 1).reshape(-1, 4)

            max_scores = np.max(s, axis=1)
            max_class_ids = np.argmax(s, axis=1)

            keep = max_scores > self.CONF_THRESH
            if not np.any(keep):
                continue
            b = b[keep]
            max_scores = max_scores[keep]
            max_class_ids = max_class_ids[keep]

            for i in range(len(b)):
                x1 = (b[i][0] - dw) / r
                y1 = (b[i][1] - dh) / r
                x2 = (b[i][2] - dw) / r
                y2 = (b[i][3] - dh) / r
                x1n = np.clip(x1 / orig_w, 0.0, 1.0)
                y1n = np.clip(y1 / orig_h, 0.0, 1.0)
                x2n = np.clip(x2 / orig_w, 0.0, 1.0)
                y2n = np.clip(y2 / orig_h, 0.0, 1.0)
                all_boxes.append((x1n, y1n, x2n, y2n, float(max_scores[i]),
                                  float(max_class_ids[i])))
        return all_boxes

    def preprocess(self, frame):
        """独立预处理，预处理一次供纯NPU推理基准测试复用"""
        return self._preprocess_det(frame)

    def predict_raw(self, frame):
        """返回推理原始输出（不含后处理），用于纯耗时对比"""
        blob, _, _ = self._preprocess_det(frame)
        out = self.rknn.run([blob], data_format='nhwc')
        return out

    def predict_raw_from_blob(self, blob):
        """直接对已预处理的 blob 推理（纯NPU耗时）"""
        return self.rknn.run([blob], data_format='nhwc')

    def predict(self, frame):
        """完整推理 + 后处理"""
        shape = frame.shape[:2]
        blob, r, (dw, dh) = self._preprocess_det(frame)
        out = self.rknn.run([blob], data_format='nhwc')
        dets = self._postprocess_det(out, shape, r, dw, dh)
        return _DetResult(dets, self.names)


# ======================== 基准测试核心 ========================
def load_test_image(path=None):
    """加载测试图像，若未指定则生成模拟图"""
    if path and os.path.exists(path):
        img = cv2.imread(path)
        if img is not None:
            print(f"  测试图像: {path} ({img.shape[1]}x{img.shape[0]})")
            return img
    # 生成随机模拟图（模拟海上场景）
    print("  未指定测试图像，生成模拟海上图像 (640x640)")
    img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    # 加一点蓝色调模拟海天背景
    img[:, :, 0] = img[:, :, 0] * 0.5 + 100
    return img


def benchmark_model(model_path, test_frame, warmup=30, iterations=200, label=""):
    """对单个模型执行性能测试"""
    size_mb = os.path.getsize(model_path) / (1024 * 1024)

    print(f"\n{'='*60}")
    print(f"  模型: {label}")
    print(f"  路径: {model_path}")
    print(f"  文件大小: {size_mb:.2f} MB")
    print(f"{'='*60}")

    # 1. 测量加载时间
    t0 = time.time()
    model = YOLOv11RKNN(model_path, model_type="det", input_size=640, names={0: "ship"})
    load_time = time.time() - t0
    print(f"  加载时间: {load_time:.3f}s")

    # 2. 预计算一次预处理结果（纯NPU耗时测试复用）
    blob, _, _ = model.preprocess(test_frame)

    # 3. Warmup（用预处理的blob）
    print(f"  Warmup {warmup} 次推理...")
    for i in range(warmup):
        _ = model.predict_raw_from_blob(blob)

    # 4. 纯NPU推理耗时（预处理完成后，仅模型推理）
    print(f"  运行 {iterations} 次纯NPU推理（不含预处理+后处理）...")
    npu_times = []
    for i in range(iterations):
        t0 = time.time()
        _ = model.predict_raw_from_blob(blob)
        npu_times.append((time.time() - t0) * 1000)

    npu_mean = np.mean(npu_times)
    npu_median = np.median(npu_times)
    npu_min = np.min(npu_times)
    npu_max = np.max(npu_times)
    npu_std = np.std(npu_times)
    npu_fps = 1000.0 / npu_mean

    print(f"\n  ┌─ 纯NPU推理耗时 (ms) ─────────────────────────┐")
    print(f"  │  均值: {npu_mean:.3f} ms   标准差: {npu_std:.3f} ms  │")
    print(f"  │  中位数: {npu_median:.3f} ms  最小值: {npu_min:.3f} ms  │")
    print(f"  │  最大值: {npu_max:.3f} ms                           │")
    print(f"  │  NPU FPS: {npu_fps:.1f}                             │")
    print(f"  └────────────────────────────────────────────────────┘")

    # 5. 完整流水线耗时（预处理 + NPU推理 + 后处理）
    print(f"  运行 {iterations} 次完整流水线（预处理+NPU+后处理）...")
    full_times = []
    det_counts = []
    for i in range(iterations):
        t0 = time.time()
        result = model.predict(test_frame)
        t = (time.time() - t0) * 1000
        full_times.append(t)
        if result.boxes is not None:
            det_counts.append(len(result.boxes))
        else:
            det_counts.append(0)

    full_mean = np.mean(full_times)
    full_median = np.median(full_times)
    full_fps = 1000.0 / full_mean
    avg_dets = np.mean(det_counts)

    print(f"\n  ┌─ 完整流水线耗时 (ms) ────────────────────────┐")
    print(f"  │  均值: {full_mean:.3f} ms                      │")
    print(f"  │  中位数: {full_median:.3f} ms                  │")
    print(f"  │  流水线 FPS: {full_fps:.1f}                    │")
    print(f"  │  平均检测数: {avg_dets:.1f} 个目标              │")
    print(f"  └────────────────────────────────────────────────┘")

    return {
        "label": label,
        "size_mb": size_mb,
        "load_time": load_time,
        "npu_mean_ms": npu_mean,
        "npu_median_ms": npu_median,
        "npu_min_ms": npu_min,
        "npu_max_ms": npu_max,
        "npu_std_ms": npu_std,
        "npu_fps": npu_fps,
        "full_mean_ms": full_mean,
        "full_median_ms": full_median,
        "full_fps": full_fps,
        "avg_detections": avg_dets,
    }

def print_comparison(results):
    """打印FP vs INT8对比表格"""
    if len(results) < 2:
        return

    r1, r2 = results[0], results[1]
    if r1["npu_mean_ms"] > 0:
        ratio = r2["npu_mean_ms"] / r1["npu_mean_ms"]
    else:
        ratio = 0
    fps_ratio = r2["npu_fps"] / r1["npu_fps"] if r1["npu_fps"] > 0 else 0

    print(f"\n{'='*65}")
    print(f"  📊 FP vs INT8 性能对比总表")
    print(f"{'='*65}")
    print(f"  {'指标':<23} {'FP':<14} {'INT8':<14} {'倍率':<10}")
    print(f"  {'-'*23} {'-'*14} {'-'*14} {'-'*10}")
    print(f"  {'文件大小 (MB)':<23} {r1['size_mb']:<14.2f} {r2['size_mb']:<14.2f} "
          f"{r2['size_mb']/r1['size_mb']:<10.2f}x")
    print(f"  {'模型加载时间 (s)':<23} {r1['load_time']:<14.3f} {r2['load_time']:<14.3f} "
          f"{'':<10}")
    print(f"  {'纯NPU推理均值 (ms)':<23} {r1['npu_mean_ms']:<14.3f} {r2['npu_mean_ms']:<14.3f} "
          f"{ratio:<10.2f}x")
    print(f"  {'纯NPU推理中位数 (ms)':<23} {r1['npu_median_ms']:<14.3f} {r2['npu_median_ms']:<14.3f} "
          f"{r2['npu_median_ms']/r1['npu_median_ms']:<10.2f}x")
    print(f"  {'纯NPU最小值 (ms)':<23} {r1['npu_min_ms']:<14.3f} {r2['npu_min_ms']:<14.3f} "
          f"{'':<10}")
    print(f"  {'纯NPU最大值 (ms)':<23} {r1['npu_max_ms']:<14.3f} {r2['npu_max_ms']:<14.3f} "
          f"{'':<10}")
    print(f"  {'纯NPU标准差 (ms)':<23} {r1['npu_std_ms']:<14.3f} {r2['npu_std_ms']:<14.3f} "
          f"{'':<10}")
    print(f"  {'纯NPU FPS':<23} {r1['npu_fps']:<14.1f} {r2['npu_fps']:<14.1f} "
          f"{fps_ratio:<10.2f}x")
    print(f"  {'完整流水线 FPS':<23} {r1['full_fps']:<14.1f} {r2['full_fps']:<14.1f} "
          f"{r2['full_fps']/r1['full_fps']:<10.2f}x")
    print(f"  {'平均检测数':<23} {r1['avg_detections']:<14.1f} {r2['avg_detections']:<14.1f} "
          f"{'':<10}")
    print(f"{'='*65}")

    speedup = (r1["npu_mean_ms"] / r2["npu_mean_ms"]) - 1
    print(f"\n  ▶ INT8 相比 FP NPU推理加速: {speedup*100:+.1f}%")
    print(f"  ▶ INT8 文件大小: {r2['size_mb']:.1f}MB (FP的{r2['size_mb']/r1['size_mb']*100:.0f}%)")
    print()


def main():
    parser = argparse.ArgumentParser(description="YOLOv11 RKNN FP vs INT8 性能对比")
    parser.add_argument("--test_image", type=str, default="/home/Tronlong/Desktop/pipeline/test/近处小船-字号15.jpg",
                        help="测试图像路径（默认生成模拟图）")
    parser.add_argument("--fp_model", type=str,
                        default="/home/Tronlong/Desktop/rknn_model_zoo_and_models/"
                                "yolov11n-day/day_5-best_255_fp.rknn",
                        help="FP 模型路径")
    parser.add_argument("--int8_model", type=str,
                        default="/home/Tronlong/Desktop/rknn_model_zoo_and_models/"
                                "yolov11n-day/day_5-best_255_i8.rknn",
                        help="INT8 模型路径")
    parser.add_argument("--cls_fp", type=str,
                        default="/home/Tronlong/Desktop/rknn_model_zoo_and_models/"
                                "yolov11n-cls/cls_1-best_255_fp.rknn",
                        help="分类 FP 模型路径（可选）")
    parser.add_argument("--cls_int8", type=str,
                        default="/home/Tronlong/Desktop/rknn_model_zoo_and_models/"
                                "yolov11n-cls/cls_1-best_255_i8.rknn",
                        help="分类 INT8 模型路径（可选）")
    parser.add_argument("--warmup", type=int, default=20,
                        help="预热推理次数")
    parser.add_argument("--iterations", type=int, default=100,
                        help="基准测试推理次数")
    args = parser.parse_args()

    print("=" * 65)
    print("  YOLOv11 RKNN — FP vs INT8 模型性能对比基准")
    print(f"  Warmup: {args.warmup} 次 | 测试: {args.iterations} 次")
    print("=" * 65)

    # 加载测试图像
    test_frame = load_test_image(args.test_image)

    results = []

    # 检测模型对比
    for model_path, label in [(args.fp_model, "FP32/FP16 (检测)"),
                               (args.int8_model, "INT8 (检测)")]:
        if not os.path.exists(model_path):
            print(f"\n⚠️  模型不存在，跳过: {model_path}")
            continue
        ret = benchmark_model(model_path, test_frame,
                              warmup=args.warmup,
                              iterations=args.iterations,
                              label=label)
        results.append(ret)

    if len(results) >= 2:
        print_comparison(results)
    else:
        print("\n⚠️  对比需要两个检测模型 (FP + INT8)")

    # 分类模型对比（可选）
    cls_results = []
    for model_path, label in [(args.cls_fp, "FP32/FP16 (分类)"),
                               (args.cls_int8, "INT8 (分类)")]:
        if not os.path.exists(model_path):
            continue
        print(f"\n{'='*60}")
        print(f"  分类模型: {label}")
        print(f"  路径: {model_path}")
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"  文件大小: {size_mb:.2f} MB")

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from rknnlite.api import RKNNLite as ClsRKNNLite
        rknn = ClsRKNNLite()
        t0 = time.time()
        rknn.load_rknn(model_path)
        rknn.init_runtime()
        load_time = time.time() - t0

        cls_frame = cv2.resize(test_frame, (256, 256))
        rgb = cv2.cvtColor(cls_frame, cv2.COLOR_BGR2RGB)
        chw = rgb.transpose(2, 0, 1).astype(np.uint8)
        blob = np.expand_dims(chw, axis=0)

        # warmup
        for _ in range(args.warmup):
            rknn.inference(inputs=[blob])

        times = []
        for _ in range(args.iterations):
            t0 = time.time()
            rknn.inference(inputs=[blob])
            times.append((time.time() - t0) * 1000)

        mean_ms = np.mean(times)
        print(f"  加载时间: {load_time:.3f}s")
        print(f"  推理均值: {mean_ms:.3f} ms")
        print(f"  分类 FPS: {1000.0/mean_ms:.1f}")
        cls_results.append({"label": label, "mean_ms": mean_ms})

        rknn.release()
        del rknn

    if len(cls_results) == 2:
        r1, r2 = cls_results[0], cls_results[1]
        print(f"\n  ┌─ 分类模型对比 ─────────────────────────────┐")
        print(f"  │  {r1['label']:<20} {r1['mean_ms']:.3f} ms              │")
        print(f"  │  {r2['label']:<20} {r2['mean_ms']:.3f} ms              │")
        print(f"  │  INT8 加速比: {r1['mean_ms']/r2['mean_ms']:.2f}x          │")
        print(f"  └────────────────────────────────────────────────┘")

    print("\n✅ 基准测试完成\n")


if __name__ == "__main__":
    main()
