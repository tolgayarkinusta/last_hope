import pyzed.sl as sl
import cv2
import numpy as np
import math
import time
from ultralytics import YOLO
import os
import sys
import supervision as sv
import torch
import torchvision

# Load a model
#model = YOLO("balonx50.pt")
engine = YOLO('balonYeniNano170.engine')
#model.to("cuda")


bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Check if the script is running with root privileges
#if os.geteuid() != 0:
#    print("This script requires elevated privileges. Please run it with `sudo` or as root.")
#    #sys.exit(1)

#from MainSystem import USV#

# = USV#("/dev/ttyACM0", baud=115200)
print("Arming vehicle...")
#.arm_vehicle()
print("Vehicle armed!")
print("Setting mode...")
#.set_mode("MANUAL")
print("Mode set!")


# Constants
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1
COLOR_RED = (0, 0, 255)
THICKNESS = 2
DEPTH_CENTER_COLOR = (255, 0, 0)
DEPTH_CENTER_RADIUS = 5
width = None  # Başlangıçta tanımlayın

def initialize_camera():
    # ZED kamera nesnesi oluştur
    zed = sl.Camera()
    # ZED başlatma parametreleri ayarla
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # 720p çözünürlük
    init_params.camera_fps = 15  # 30 FPS
    init_params.depth_mode = sl.DEPTH_MODE.QUALITY # depth mode best quality at neural_plus
    init_params.coordinate_units = sl.UNIT.METER #using metric system
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE # default for the opencv
    init_params.depth_minimum_distance = 0.3
    init_params.depth_maximum_distance = 20
    init_params.camera_disable_self_calib = False
    init_params.depth_stabilization = 1 #titreme azaltıcı
    init_params.sensors_required = False # true yaparsan imu açılmadan kamera açılmaz
    init_params.enable_image_enhancement = True #true was always the default
    init_params.async_grab_camera_recovery = False #set true if u want to keep processing if cam gets shutdown
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        raise Exception("Failed to open ZED camera. Exiting.")
    return zed


# Render text over the frame
def render_text(frame, text, position):
    cv2.putText(frame, text, position, FONT, FONT_SCALE, COLOR_RED, THICKNESS)

# Basic class to handle the timestamp of the different sensors to know if it is a new sensors_data or an old one
class TimestampHandler:
    def __init__(self):
        self.t_imu = sl.Timestamp()
    # check if the new timestamp is higher than the reference one, and if yes, save the current as reference
    def is_new(self, sensor):
        if (isinstance(sensor, sl.IMUData)):
            new_ = (sensor.timestamp.get_microseconds() > self.t_imu.get_microseconds())
            if new_:
                self.t_imu = sensor.timestamp
            return new_

def main():
    zed = initialize_camera()
    global width
    camera_info = zed.get_camera_information()
    width = camera_info.camera_configuration.resolution.width
    print(width)
    height = camera_info.camera_configuration.resolution.height
    print(height)
    center_x = width // 2
    center_y = height // 2
    print("Kamera çözünürlüğü: ", width, "x", height)
    print("Görüntü orta noktası: ", (center_x, center_y))

    # Sensor timestamp yönetimi
    ts_handler = TimestampHandler()

    # Görüntü ve derinlik için Mat nesneleri oluşturuluyor
    image = sl.Mat()
    depth = sl.Mat()
    sensors_data = sl.SensorsData()

    fps_previous_time = time.time()

    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Görüntü ve derinlik verilerini al
            zed.retrieve_image(image, sl.VIEW.LEFT)
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            frame = cv2.cvtColor(image.get_data(), cv2.COLOR_BGRA2BGR)

            # YOLO motorundan sonuçlar liste olarak geliyor
            results = engine.track(source=frame, conf=0.50)

            # Liste içindeki her bir sonuç üzerinde döngü ile işleme yapıyoruz
            for result in results:
                detections = sv.Detections.from_ultralytics(result)
                frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
                frame = label_annotator.annotate(scene=frame, detections=detections)

                # Tespit koordinatlarını ve sınıf id'lerini alıyoruz
                coordinates = detections.xyxy.tolist()
                class_ids = detections.class_id.tolist()

                # Her tespit kutusunun sağ üst köşesine derinlik değerini yazdırıyoruz
                for box in coordinates:
                    x1, y1, x2, y2 = map(int, box)
                    depth_val = depth.get_value( (x2+x1)/2, (y1+y2)/2 )[1] #todo: that shi
                    if not np.isnan(depth_val):
                        text = f"{depth_val:.2f} m"
                        cv2.putText(frame, text, (x2 - 60, y1 + 20), FONT, 0.7, COLOR_RED, 2)

                # Renk tespit bayrakları ve pozisyonlarını sıfırlıyoruz
                red_detected = False
                green_detected = False
                yellow_detected = False
                blue_detected = False
                black_detected = False

                red_positions = []
                green_positions = []
                yellow_positions = []
                blue_positions = []
                black_positions = []

                # Her tespit için sınıfa göre bayrak ve pozisyon listelerini oluşturuyoruz
                for i, class_id in enumerate(class_ids):
                    if class_id == 3:  # Kırmızı
                        red_detected = True
                        red_positions.append(coordinates[i])
                    elif class_id == 4:  # Sarı
                        yellow_detected = True
                        yellow_positions.append(coordinates[i])
                    elif class_id == 2:  # Yeşil
                        green_detected = True
                        green_positions.append(coordinates[i])
                    elif class_id == 1:  # Mavi
                        blue_detected = True
                        blue_positions.append(coordinates[i])
                    elif class_id == 0:  # Siyah
                        black_detected = True
                        black_positions.append(coordinates[i])

            # FPS hesaplaması
            fps_current_time = time.time()
            fps = 1 / (fps_current_time - fps_previous_time)
            fps_previous_time = fps_current_time
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Görüntüyü ekranda göster
            frame_resized = cv2.resize(frame, (960, 540))
            cv2.imshow("ZED Camera", frame_resized)

            k = cv2.waitKey(1)
            if k % 256 == 27:
                print("Esc tuşuna basıldı.. Kapatılıyor..")
                break

    cv2.destroyAllWindows()
    zed.close()


if __name__ == "__main__":
    main()
