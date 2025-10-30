from ultralytics import YOLO
import cv2

# Carga tu modelo entrenado
model = YOLO("C:/Users/citla/Desktop/procesamiento/procesamiento/runs/detect/train3/weights/best.pt")

# Abre la cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detecta objetos en el frame
    results = model(frame)
    
    # Dibuja resultados
    annotated_frame = results[0].plot()
    
    # Muestra la imagen
    cv2.imshow("YOLOv8 Detection", annotated_frame)
    
    # Presiona ESC para salir
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
