import cv2
from ultralytics import YOLO

# Load the trained model
model = YOLO("oreo_model_final/oreo_yolov8n_final2/weights/best.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# Nutrition label to display
nutrition_info = [
    "OREO DETECTED",
    "Calories: 160",
    "Fat: 7g",
    "Carbs: 25g",
    "Sugar: 14g",
    "Protein: 1g"
]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model(frame, verbose=False)[0]

    # Check if Oreo is detected
    oreo_detected = any(box.cls == 0 for box in results.boxes)

    # Draw results
    annotated_frame = results.plot()

    # Only show nutrition if Oreo is detected
    if oreo_detected:
        y_offset = 30
        for line in nutrition_info:
            cv2.putText(annotated_frame, line, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25q

    cv2.imshow("Oreo Detector", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
