from ultralytics import YOLO
if __name__ == '__main__':
    # Load a model
    # model = YOLO('yolov8n.pt')  # 加载yolov8n
    # model = YOLO('runs/detect/train22/weights/best.pt')  # 也可以加载你自己的模型
    model = YOLO('runs/detect/train27/weights/best.pt')  # 也可以加载你自己的模型
    # model = YOLO('runs1/EfficientNet+BiFPN/weights/best.pt')  # 也可以加载你自己的模型

    # Validate the model
    metrics = model.val(iou=0.5, batch=32, data='data_yaml/NEU-DET_origin.yaml')
    metrics.box.map    # 查看目标检测 map50-95 的性能
    metrics.box.map50  # 查看目标检测 map50 的性能
    metrics.box.map75  # 查看目标检测 map75 的性能
    metrics.box.maps   # 返回一个列表包含每一个类别的 map50-95
