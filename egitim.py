from ultralytics import YOLO

model = YOLO("yolo11x.pt")

train_results = model.train(
    data="D:\yarkin\yolo11\balonSamandira.v5i.yolov11\data.yaml",
    epochs = 400,
    imgsz=640,
    device=0,
)