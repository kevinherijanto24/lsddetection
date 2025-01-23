from ultralytics import YOLO

model = YOLO('yolov11n_modelLumpySkinwith2class_old.pt')
results = model(1, show=True)
for result in results:
  boxes = result.boxes
  classes = result.names