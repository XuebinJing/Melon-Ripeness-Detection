from ultralytics import YOLO
# 加载模型

#model = YOLO("weights/yolov8n.pt")  # 加载预训练模型
model = YOLO("ultralytics/cfg/models/v8/yolov8.yaml")


# Use the model

results = model.train(data="dataset/melon.yaml", epochs=100, batch=16)  # 训练模型
