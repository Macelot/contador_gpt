import cv2
from ultralytics import YOLO

# Carregar o modelo treinado
model = YOLO('runs/detect/train/weights/best.pt')

def detect_vehicles(image, model):
    results = model(image)
    return results

def draw_bboxes(image, results):
    for result in results:
        for bbox in result.boxes:
            x_min, y_min, x_max, y_max = map(int, bbox.xyxy[0])
            confidence = bbox.conf[0]
            class_id = int(bbox.cls[0])
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            cv2.putText(image, f'{class_id} {confidence:.2f}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow('Detecção de Veículos', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Extração de frames (se necessário)
extract_frames('pedagio.mp4', 'frames_pedagio')

# Detecção e visualização de veículos
image = cv2.imread('frames_pedagio/frame_0.jpg')
results = detect_vehicles(image, model)
draw_bboxes(image, results)
