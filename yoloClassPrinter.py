import cv2
import supervision as sv
from ultralytics import YOLO

model = YOLO('YOLOboat.pt')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Kameradan görüntü gelmiyor...")

bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()  # Varsayılan ayarlar

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.60)[0]
    detections = sv.Detections.from_ultralytics(results)

    annotated_image = bounding_box_annotator.annotate(scene=frame, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

    # Her tespit için class id'yi kutunun sol üst köşesine yazdırıyoruz
    for bbox, cls in zip(detections.xyxy.tolist(), detections.class_id.tolist()):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.putText(annotated_image, str(cls), (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Webcam", annotated_image)
    k = cv2.waitKey(1)
    if k % 256 == 27:
        print("Esc tuşuna basıldı.. Kapatılıyor..")
        break
