import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# 全局变量，用于存储加载的模型，避免重复加载
model = None

def load_yolo_model(weights_path: str):
    """
    加载YOLOv8模型。
    自动检测是PyTorch .pt文件还是ONNX .onnx文件。
    """
    global model
    if model is None:
        print(f"Loading model from {weights_path}...")
        # device='cpu' 确保在CPU上运行
        model = YOLO(weights_path) # YOLO类会自动处理.pt和.onnx文件
        print("Model loaded successfully.")
    return model

def run_segmentation(image_data: np.ndarray, weights_path: str):
    """
    对输入的NumPy图像数据运行YOLOv8实例分割。
    Args:
        image_data (np.ndarray): 输入图像的NumPy数组 (BGR格式)。
        weights_path (str): YOLOv8模型权重文件的路径 (.pt 或 .onnx)。
    Returns:
        np.ndarray: 带有分割结果的图像 (BGR格式), 如果没有检测到则返回原始图像。
    """
    model = load_yolo_model(weights_path)

    # YOLOv8模型期望RGB格式的图片
    # image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)

    # 运行推理，指定device='cpu'
    # 注意：ultralytics库的YOLO类在进行推理时，其内部会自动将图片转换为合适的格式（例如PIL Image或内部tensor），
    # 并自动处理BGR/RGB转换，所以直接传入np.ndarray通常是OK的。
    results = model(image_data, device='cpu', conf=0.25, iou=0.7) # conf和iou可调整

    # 绘制结果到图像上
    # results.plot() 方法会返回一个带有BBOX和Masks的NumPy图像
    annotated_frame = results[0].plot() # results[0] 是第一个图片的推理结果

    # 如果需要，可以将annotated_frame从RGB转换为BGR（如果之后要用OpenCV显示）
    # annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

    return annotated_frame

if __name__ == '__main__':
    # 简单的本地测试
    # 请替换为你的模型路径和测试图片路径
    test_weights_path = 'weights/my_yolov8_seg_model.pt' # 或 .pt
    test_image_path = 'testimg/SEM-251119-003-012(before).jpg'

    # 加载图片
    img = cv2.imread(test_image_path)
    if img is None:
        print(f"Error: Could not load image from {test_image_path}")
    else:
        print(f"Running inference on {test_image_path}...")
        result_img = run_segmentation(img, test_weights_path)
        cv2.imshow("Original Image", img)
        cv2.imshow("Segmentation Result", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite("segmented_output.jpg", result_img)
        print("Test complete. Result saved to segmented_output.jpg")

