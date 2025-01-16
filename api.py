from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import cv2
from ultralytics import YOLO
import base64
import io
from PIL import Image
import numpy as np
import logging

# Flask application setup
app = Flask(__name__)

# Enable CORS for all origins
CORS(app)

# Enable Socket.IO with WebSocket support
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# Load the YOLO model
model = YOLO('yolov11n_modelLumpySkinwith2class.pt')

# Configure logging
logging.basicConfig(level=logging.INFO)

# Helper function to convert image to base64
def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Helper function to downscale image
def resize_image(img, width=320):
    aspect_ratio = float(img.shape[1]) / float(img.shape[0])
    new_height = int(width / aspect_ratio)
    resized_img = cv2.resize(img, (width, new_height))
    return resized_img

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('stream')
def handle_stream(data):
    try:
        # Decode the received image data
        img_data = base64.b64decode(data['image'])
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Store the original dimensions
        orig_height, orig_width = img.shape[:2]

        # Downscale the image
        img_resized = resize_image(img, width=320)

        # Perform inference using YOLO
        results = model(img_resized)
        boxes = results[0].boxes if results and results[0].boxes else []
        classes = results[0].names if results else []

        # Draw bounding boxes on the original image
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Rescale bounding box coordinates
            x1 = int(x1 * orig_width / img_resized.shape[1])
            y1 = int(y1 * orig_height / img_resized.shape[0])
            x2 = int(x2 * orig_width / img_resized.shape[1])
            y2 = int(y2 * orig_height / img_resized.shape[0])

            # Draw the bounding box and label
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = classes[int(box.cls)]
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Convert the processed image to base64
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img_base64 = image_to_base64(pil_img)

        # Send the processed image back to the client
        emit('image', {'image': img_base64})

    except Exception as e:
        logging.error(f"Error in processing stream: {e}")
        emit('error', {'message': 'Error processing image.'})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
