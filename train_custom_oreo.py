from ultralytics import YOLO

# Load YOLOv8 model (nano version)
model = YOLO("yolov8n.pt")  # You can change to yolov8s.pt if you want higher accuracy

# Train the model using the updated dataset
model.train(
    data="oreo_Detector_Dataset/data.yaml",  # ✅ correct path based on your folder name
    epochs=30,
    imgsz=640,
    name="oreo_yolov8n_final",
    project="oreo_model_final"
)
