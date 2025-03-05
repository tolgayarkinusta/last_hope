from ultralytics import YOLO

model = YOLO("YOLOboat.pt")  # Load a model
model.export(format="engine", int8=False, half = True)