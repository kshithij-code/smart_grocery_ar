import cv2
import time
from ultralytics import YOLO

# Load trained model
model = YOLO("oreo_model_v2/oreo_yolov8n_v2/weights/best.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# Nutrition info
nutrition_text = """OREO 🍪 DETECTED
Calories: 483
Protein: 4.4g
Sugar: 38.8g
Fat: 19.1g"""
nutrition_lines = nutrition_text.split("\n")

# Font and styling
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.6
font_color = (255, 255, 255)
bg_color = (0, 0, 0)
thickness = 1

# Detection flag and timer
last_detected_time = 0
cooldown = 2  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run model prediction
    results = model.predict(source=frame, conf=0.6, verbose=False)

    detected = False

    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = result.names[cls_id]

            # Only accept confident, meaningful boxes
            if label.lower() == "oreo" and conf > 0.75:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)

                if area > 5000:
                    detected = True
                    last_detected_time = time.time()
                    break

    # Only show if recently detected
    if time.time() - last_detected_time < cooldown:
        x, y = 20, 40
        for i, line in enumerate(nutrition_lines):
            (w, h), _ = cv2.getTextSize(line, font, font_scale, thickness)
            cv2.rectangle(frame, (x - 10, y + i * 30 - 25), (x + w + 10, y + i * 30), bg_color, -1)
            cv2.putText(frame, line, (x, y + i * 30 - 10), font, font_scale, font_color, thickness)

    cv2.imshow("Oreo Nutrition Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
