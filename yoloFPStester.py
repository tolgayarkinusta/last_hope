import cv2
import supervision as sv
from ultralytics import YOLO

model = YOLO('YOLOboat.pt')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kameradan görüntü gelmiyor...")

bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, conf=0.60)[0]
    detections = sv.Detections.from_ultralytics(results)

    annotated_image = bounding_box_annotator.annotate(
        scene=frame, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections)

    koordinatlar = detections.xyxy.tolist()
    adlar = detections.class_id.tolist()


    cv2.imshow("Webcam", annotated_image)  # burada arduino kart ile arduino ide üzerinden haberleşme yaptık

    k = cv2.waitKey(1)

    if k % 256 == 27:
        print("Esc tuşuna basıldı.. Kapatılıyor..")
        break

