# Ship Detection Pipeline

## 硬件环境

- **主控板卡**: RK3576 开发板
- **摄像头**: 4 个摄像头分别连接开发板的 4 个不同网口
  - 密码: `ADMIN123`
  - 编码格式: H265
- **AIS 接收设备**: 通过串口连接开发板
- **远程 PC（控制中心）**: 与开发板通过网线直连

## 项目结构

- `src/` — 主程序代码（pipeline、RTSP 拉流、OCR 等）
- `AIS/` — AIS 数据接收与解析模块
- `models/` — YOLO 模型权重文件
- `ultralytics/` — Ultralytics YOLO 相关依赖
- `test/` — 测试脚本
- `output/` — 输出结果
- `alarm_recordings/` — 告警录像与截图
- `ocr_roi_images/` — OCR 区域图像

## 编码规范

- 在编写或修改代码时，始终遵循 Karpathy 编程指导原则：避免过度复杂化、做精准修改、明确可验证的成功标准、优先使用现有工具而非重复造轮子。
