from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from flask_cors import CORS  # Import CORS
import cv2
from ultralytics import YOLO
import base64
import io
from PIL import Image
from waitress import serve
import numpy as np
from collections import Counter  # To handle duplicate boxes

app = Flask(__name__)

# Enable CORS for all origins
CORS(app)

socketio = SocketIO(app, cors_allowed_origins="*")  # Allow all origins

# Global variable to store the current model
current_model_path = 'yolov11n_modelLumpySkinwith2class_old.pt'
model = YOLO(current_model_path)
confidence_thresholds = {
    "Normal Skin Cows": 0,  
    "LSD Cows": 0.9,   
    "Cow": 0,  
    "Lump": 0.9 
}

# Helper function to convert image to base64 for web display
def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Helper function to downscale image
def resize_image(img, width=160):
    # Calculate the aspect ratio
    aspect_ratio = float(img.shape[1]) / float(img.shape[0])
    new_height = int(width / aspect_ratio)
    resized_img = cv2.resize(img, (width, new_height))
    return resized_img

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('change_model')
def handle_change_model(data):
    global model, current_model_path, confidence_thresholds
    new_model_path = data.get('model_path')
    try:
        # Load the new model
        model = YOLO(new_model_path)
        current_model_path = new_model_path

        if "yolov11n_modelLumpySkinwith2class_old.pt" in new_model_path.lower():
            confidence_thresholds = {
                "Normal Skin Cows": 0,
                "LSD Cows": 0.9
            }
        elif "yolov11n_modelLumpySkinwith2classdeeper.pt" in new_model_path.lower():
            confidence_thresholds = {
                "Cow": 0,
                "Lump": 0.9
            }
        else:
            confidence_thresholds = {}  # Default empty if the model is unknown

        emit('model_changed', {'success': True, 'model': new_model_path})

    except Exception as e:
        emit('model_changed', {'success': False, 'error': str(e)})

@socketio.on('stream')
def handle_stream(data):
    global confidence_thresholds
    # Decode the received image data
    img_data = base64.b64decode(data['image'])
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Store the original dimensions for rescaling the bounding boxes later
    orig_height, orig_width = img.shape[:2]

    # Downscale the image for faster processing
    img_resized = resize_image(img, width=640)  # Resize to 160px width

    # Perform YOLO detection
    results = model(img_resized)
    boxes = results[0].boxes  # Detected boxes
    classes = results[0].names  # Class labels

    # Prepare bounding box data
    bounding_boxes = []

    # Rescale bounding boxes back to the original image dimensions
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        label = classes[int(box.cls)]
        confidence = box.conf[0].item()  # Get confidence score

        # Apply confidence threshold for the specific class
        if label in confidence_thresholds and confidence < confidence_thresholds[label]:
            continue  # Skip if confidence is below threshold

        # Rescale the bounding box coordinates
        x1 = int(x1 * orig_width / img_resized.shape[1])
        y1 = int(y1 * orig_height / img_resized.shape[0])
        x2 = int(x2 * orig_width / img_resized.shape[1])
        y2 = int(y2 * orig_height / img_resized.shape[0])

        # Add bounding box info to be sent back
        bounding_boxes.append({
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'label': label,
            'confidence': confidence
        })

    # Convert the image to base64 for real-time display
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img_base64 = image_to_base64(pil_img)

    # Send the processed image and bounding box data back to the client
    emit('image', {'image': img_base64, 'boxes': bounding_boxes})
    print("Confidence Thresholds:", confidence_thresholds)

if __name__ == '__main__':
    app.run(ssl_context=('/etc/ssl/certs/selfsigned.crt', '/etc/ssl/private/selfsigned.key'), host='0.0.0.0', port=5000)
