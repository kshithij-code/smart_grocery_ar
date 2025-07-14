from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io
import base64
import cv2
import numpy as np
import tempfile
import os

app = Flask(__name__)
CORS(app)
model = YOLO("../runs/three_items_train/weights/best.pt")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.json['image']
        header, encoded = data.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        results = model.predict(image, conf=0.4)
        names = model.names

        detections = []
        image_np = np.array(image)
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = names[cls_id]
                confidence = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image_np, f"{label} {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                detections.append({'label': label, 'confidence': confidence})

        _, buffer = cv2.imencode(".jpg", image_np)
        encoded_image = base64.b64encode(buffer).decode("utf-8")

        return jsonify({'success': True, 'detections': detections, 'image': f"data:image/jpeg;base64,{encoded_image}"})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)