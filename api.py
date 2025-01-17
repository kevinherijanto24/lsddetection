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

app = Flask(__name__)

# Enable CORS for all origins
CORS(app)

socketio = SocketIO(app, cors_allowed_origins="*")  # Allow all origins

# Global variable to store the current model
current_model_path = 'yolov11n_modelLumpySkinwith2class.pt'
model = YOLO(current_model_path)

# Helper function to convert image to base64 for web display
def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Helper function to downscale image
def resize_image(img, width=320):
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
    global model, current_model_path
    new_model_path = data.get('model_path')
    try:
        # Load the new model
        model = YOLO(new_model_path)
        current_model_path = new_model_path
        emit('model_changed', {'success': True, 'model': new_model_path})
    except Exception as e:
        emit('model_changed', {'success': False, 'error': str(e)})

@socketio.on('stream')
def handle_stream(data):
    # Decode the received image data
    img_data = base64.b64decode(data['image'])
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Store the original dimensions for later rescaling of the bounding boxes
    orig_height, orig_width = img.shape[:2]

    # Downscale the image before passing it to the model
    img_resized = resize_image(img, width=160)  # Resize to 320px width

    # Perform the inference using YOLO
    results = model(img_resized)
    boxes = results[0].boxes
    classes = results[0].names

    # Prepare bounding box data
    bounding_boxes = []

    # Rescale the bounding boxes back to the original image dimensions
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # Rescale the bounding box coordinates
        x1 = int(x1 * orig_width / img_resized.shape[1])
        y1 = int(y1 * orig_height / img_resized.shape[0])
        x2 = int(x2 * orig_width / img_resized.shape[1])
        y2 = int(y2 * orig_height / img_resized.shape[0])

        # Draw the bounding box and label on the image
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        label = classes[int(box.cls)]
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Add bounding box info to be sent back
        bounding_boxes.append({
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'label': label
        })

    # Convert the image to base64 for real-time display
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img_base64 = image_to_base64(pil_img)

    # Send the processed image and bounding box data back to the client
    emit('image', {'image': img_base64, 'boxes': bounding_boxes})

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=5000)
