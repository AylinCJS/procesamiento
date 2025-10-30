from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')  # o tu modelo entrenado .pt
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated_frame = results[0].plot()

    cv2.imshow('Detección YOLOv8', annotated_frame)
    if cv2.waitKey(1) & 0xFF == 27:  # tecla ESC
        break

cap.release()
cv2.destroyAllWindows()
