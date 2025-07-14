from ultralytics import YOLO

# Define paths
DATA_PATH = "./three items.v1i.yolov5pytorch/data.yaml"
MODEL_PATH = "yolov8n.pt"  # Or yolov8s.pt/yolov8m.pt depending on resources

# Create and train the model
model = YOLO(MODEL_PATH)

# Train the model
model.train(
    data=DATA_PATH,
    epochs=50,
    imgsz=640,
    batch=16,       # You can adjust this depending on your GPU/CPU
    project="./runs",
    name="three_items_train",
    exist_ok=True   # Overwrites existing run folder if exists
)

# After training, best model will be saved at:
# U:/smart_grocery_ar/runs/three_items_train/weights/best.pt
