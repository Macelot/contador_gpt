from ultralytics import YOLO

# Carregar modelo YOLOv8 pré-treinado
model = YOLO('yolov8n.pt')

# Treinar o modelo com seus dados
model.train(data='data.yaml', epochs=50, imgsz=640)