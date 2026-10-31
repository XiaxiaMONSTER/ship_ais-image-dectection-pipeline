from paddleocr import PaddleOCR

ocr = PaddleOCR(
    text_detection_model_dir=r"P:\.vscode\Ship_Detection\pipeline\models\paddleocr_v5\PP-OCRv5_server_det",
    text_recognition_model_dir=r"P:\.vscode\Ship_Detection\pipeline\models\paddleocr_v5\PP-OCRv5_server_rec",
    use_doc_orientation_classify=False, # 通过 use_doc_orientation_classify 参数指定不使用文档方向分类模型
    use_doc_unwarping=False, # 通过 use_doc_unwarping 参数指定不使用文本图像矫正模型
    use_textline_orientation=False, # 通过 use_textline_orientation 参数指定不使用文本行方向分类模型
    #lang="en", #使用英文模型
    ocr_version="PP-OCRv5",
    device="gpu"
)
print("✅ PaddleOCR V5 初始化成功")

result = ocr.predict(r"P:\.vscode\Ship_Detection\pipeline\test\roi_camera2_1768484406049_x58y656x573y934.jpg")
for res in result:
    res.print()
    res.save_to_img("./output/")
    res.save_to_json("./output/res.json")