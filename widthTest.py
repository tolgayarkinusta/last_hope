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
    init_params.camera_fps = 30  # 30 FPS
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

# Convert quaternion to specified angle (yaw, roll, pitch)
def quaternion_to_angle(quaternion, angle_type):
    ox, oy, oz, ow = quaternion
    if angle_type == "yaw":
        siny_cosp = 2 * (ow * oz + ox * oy)
        cosy_cosp = 1 - 2 * (oy * oy + oz * oz)
        return math.degrees(math.atan2(siny_cosp, cosy_cosp))
    elif angle_type == "pitch":
        sinp = 2 * (ow * oy - oz * ox)
        return math.degrees(math.asin(sinp)) if abs(sinp) < 1 else math.copysign(90, sinp)
    elif angle_type == "roll":
        sinr_cosp = 2 * (ow * ox + oy * oz)
        cosr_cosp = 1 - 2 * (ox * ox + oy * oy)
        return math.degrees(math.atan2(sinr_cosp, cosr_cosp))
    return 0

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


def greenRedOnly(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected, blue_detected, black_detected, green_positions, red_positions):
    # Öncelikle sadece yeşil ve kırmızı tespit edilmiş mi kontrol edelim
    if red_detected and green_detected and not (yellow_detected or blue_detected or black_detected):
        # --- En yakın kırmızı tespitin bulunması ---
        closest_red_depth = float('inf')
        closest_red_center = None
        for box in red_positions:
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            depth_val = depth.get_value(cx, cy)[1]
            if np.isnan(depth_val):
                continue
            if depth_val < closest_red_depth:
                closest_red_depth = depth_val
                closest_red_center = (cx, cy)

        # --- En yakın yeşil tespitinin bulunması ---
        closest_green_depth = float('inf')
        closest_green_center = None
        for box in green_positions:
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            depth_val = depth.get_value(cx, cy)[1]
            if np.isnan(depth_val):
                continue
            if depth_val < closest_green_depth:
                closest_green_depth = depth_val
                closest_green_center = (cx, cy)

        if closest_red_center is None or closest_green_center is None:
            return  # Her iki renk tespiti yoksa işleme devam etme

        # --- Derinlik farkı kontrolü ---
        depth_diff = abs(closest_red_depth - closest_green_depth)
        if depth_diff > 2.5:
            # Fark 2.5 metreden büyükse:
            if closest_green_depth > closest_red_depth:
                print("Derinlik farkı > 2.5m, yeşil derinlik > kırmızı: redOnly çağrılıyor.")
            else:
                print("Derinlik farkı > 2.5m, yeşil derinlik <= kırmızı: greenOnly çağrılıyor.")
        else:
            # Fark 2.5 metreden büyük değilse:
            # --- Nokta 1: Kırmızı ve yeşil tespitlerinin orta noktası ---
            point1 = (int((closest_red_center[0] + closest_green_center[0]) / 2),
                      int((closest_red_center[1] + closest_green_center[1]) / 2))
            cv2.circle(frame, point1, 5, (255, 0, 0), -1)  # Nokta1'i mavi ile işaretle

            # --- Nokta 2: Ekran görüntüsünün alt orta noktası ---
            frame_height = frame.shape[0]
            point2 = (center_x, frame_height)
            cv2.circle(frame, point2, 5, (0, 255, 255), -1)  # Nokta2'yi sarı ile işaretle

            # --- Çizgi 1: Nokta1 ile Nokta2 arasındaki çizgi ---
            cv2.line(frame, point1, point2, (255, 0, 0), 2)

            # --- Çizgi 2: Ekran merkezi ile Nokta2 arasındaki çizgi ---
            screen_center = (center_x, center_y)
            cv2.line(frame, screen_center, point2, (0, 255, 255), 2)

            # --- İki çizgi arasındaki açıyı hesaplama ---
            # Line1 vektörü: point1 -> point2
            vec1 = (point2[0] - point1[0], point2[1] - point1[1])
            # Line2 vektörü: screen_center -> point2
            vec2 = (point2[0] - screen_center[0], point2[1] - screen_center[1])
            # Açıyı atan2 kullanarak hesapla (signed açı)
            angle1 = math.atan2(vec1[1], vec1[0])
            angle2 = math.atan2(vec2[1], vec2[0])
            angle_rad = angle1 - angle2
            # Açıyı -pi ile pi arasına normalize et
            angle_rad = math.atan2(math.sin(angle_rad), math.cos(angle_rad))
            angle_deg = math.degrees(angle_rad)

            # Açıyı görselleştir:
            cv2.putText(frame, f"Angle: {angle_deg:.1f}", (center_x - 100, center_y + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # --- Motor düzeltme komutları ---
            # Temel ileri hız
            base_speed = 1550
            # Proportional gain (örneğin: 2)
            k = 2
            correction = int(k * abs(angle_deg))
            if angle_deg > 0:
                # Pozitif açı durumunda: sol motor azalt, sağ motor artır
                left_command = base_speed - correction
                right_command = base_speed + correction
            else:
                # Negatif açı durumunda: sol motor artır, sağ motor azalt
                left_command = base_speed + correction
                right_command = base_speed - correction

            print(f"Açı: {angle_deg:.1f} derece, motor komutları: Sol={left_command}, Sağ={right_command}")
            #.set_servo(5, left_command)
            #.set_servo(6, right_command)


def greenYellowOnly(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected, blue_detected, black_detected, green_positions, yellow_positions):
    # Öncelikle sadece yeşil ve sarı tespit edilmiş mi kontrol edelim
    if yellow_detected and green_detected and not (red_detected or blue_detected or black_detected):

        # --- En yakın yeşil tespitinin bulunması ---
        closest_green_depth = float('inf')
        closest_green_center = None
        for box in green_positions:
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            depth_val = depth.get_value(cx, cy)[1]
            if np.isnan(depth_val):
                continue
            if depth_val < closest_green_depth:
                closest_green_depth = depth_val
                closest_green_center = (cx, cy)

        # --- En yakın sarı tespitinin bulunması ---
        closest_yellow_depth = float('inf')
        closest_yellow_center = None
        for box in yellow_positions:
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            depth_val = depth.get_value(cx, cy)[1]
            if np.isnan(depth_val):
                continue
            if depth_val < closest_yellow_depth:
                closest_yellow_depth = depth_val
                closest_yellow_center = (cx, cy)

        # Derinlik farkı 2.5 metreden büyük değilse:
        # getwidth() fonksiyonundan dönen sonuç kontrol ediliyor
        width_measure = getWidth(frame,closest_green_depth, closest_yellow_depth,
                                 closest_green_center[0],closest_green_center[1],
                                 closest_yellow_center[0],closest_yellow_center[1])
        if width_measure < 1.5:
            print("Genişlik < 1.5m, greenOnly çağrılıyor.")


def redYellowOnly(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected, blue_detected, black_detected, yellow_positions, red_positions):
    # Öncelikle sadece kırmızı ve sarı tespit edilmiş mi kontrol edelim
    if yellow_detected and red_detected and not (green_detected or blue_detected or black_detected):

        # --- En yakın kırmızı tespitinin bulunması ---
        closest_red_depth = float('inf')
        closest_red_center = None
        for box in red_positions:
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            depth_val = depth.get_value(cx, cy)[1]
            if np.isnan(depth_val):
                continue
            if depth_val < closest_red_depth:
                closest_red_depth = depth_val
                closest_red_center = (cx, cy)

        # --- En yakın sarı tespitinin bulunması ---
        closest_yellow_depth = float('inf')
        closest_yellow_center = None
        for box in yellow_positions:
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            depth_val = depth.get_value(cx, cy)[1]
            if np.isnan(depth_val):
                continue
            if depth_val < closest_yellow_depth:
                closest_yellow_depth = depth_val
                closest_yellow_center = (cx, cy)

        if closest_red_center is None or closest_yellow_center is None:
            return  # Her iki tespit bulunamazsa işleme devam etme


        else:
            # Derinlik farkı 2.5 metreden büyük değilse:
            width_measure = getWidth(frame,closest_yellow_depth, closest_red_depth,
                                     closest_yellow_center[0],closest_yellow_center[1],
                                     closest_red_center[0],closest_red_center[1])
            if width_measure < 1.5:
                print("Genişlik ölçümü < 1.5m: redOnly çağrılıyor.")



def greenRedYellowOnly(frame, depth, center_x, center_y,
                         green_detected, red_detected, yellow_detected,
                         blue_detected, black_detected,
                         green_positions, red_positions, yellow_positions):
    # Üç renk tespit edilmiş olmalı
    if not (red_detected and yellow_detected and green_detected):
        return

    # --- En yakın kırmızı tespitinin bulunması ---
    closest_red_depth = float('inf')
    closest_red_center = None
    for box in red_positions:
        x1, y1, x2, y2 = box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        depth_val = depth.get_value(cx, cy)[1]
        if np.isnan(depth_val):
            continue
        if depth_val < closest_red_depth:
            closest_red_depth = depth_val
            closest_red_center = (cx, cy)

    # --- En yakın sarı tespitinin bulunması ---
    closest_yellow_depth = float('inf')
    closest_yellow_center = None
    for box in yellow_positions:
        x1, y1, x2, y2 = box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        depth_val = depth.get_value(cx, cy)[1]
        if np.isnan(depth_val):
            continue
        if depth_val < closest_yellow_depth:
            closest_yellow_depth = depth_val
            closest_yellow_center = (cx, cy)

    # --- En yakın yeşil tespitinin bulunması ---
    closest_green_depth = float('inf')
    closest_green_center = None
    for box in green_positions:
        x1, y1, x2, y2 = box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        depth_val = depth.get_value(cx, cy)[1]
        if np.isnan(depth_val):
            continue
        if depth_val < closest_green_depth:
            closest_green_depth = depth_val
            closest_green_center = (cx, cy)

    # Eğer herhangi bir tespit bulunamazsa, işleme devam etme
    if closest_red_center is None or closest_yellow_center is None or closest_green_center is None:
        return

    # --- getWidth ölçümleri ---
    # getWidth fonksiyonu, iki nokta (örneğin, (x, y) formatında) alarak aralarındaki mesafeyi hesapladığını varsayıyoruz.
    getWidth(frame,closest_green_depth,closest_yellow_depth,
                                  closest_green_center[0],closest_green_center[1],
                                  closest_yellow_center[0],closest_yellow_center[1])
    getWidth(frame,closest_red_depth, closest_yellow_depth,
                                closest_red_center[0], closest_red_center[1],
                                closest_yellow_center[0], closest_yellow_center[1])
    getWidth(frame, closest_red_depth, closest_green_depth,
                               closest_red_center[0], closest_red_center[1],
                               closest_green_center[0],closest_green_center[1])

def getWidth(frame, distance1, distance2, x1, y1, x2, y2):
    """
    ZED 2i kamerası kullanarak iki nesne arasındaki gerçek mesafeyi hesaplar.
    Aynı zamanda, hesaplanan mesafe değeri ile iki nokta arasına siyah çizgi çizer ve
    çizgi üzerinde gri renkte mesafe yazısını ekler.

    Parametreler:
    - frame: Üzerine çizim yapılacak OpenCV görüntüsü (numpy dizisi).
    - distance1: İlk nesnenin tekneye olan uzaklığı (metre cinsinden).
    - distance2: İkinci nesnenin tekneye olan uzaklığı (metre cinsinden).
    - x1, y1: İlk nesnenin görüntüdeki merkezi koordinatları (piksel cinsinden).
    - x2, y2: İkinci nesnenin görüntüdeki merkezi koordinatları (piksel cinsinden).

    Dönüş:
    - İki nesne arasındaki gerçek dünya mesafesi (metre cinsinden).
    """
    global width
    fov = 110
    image_width = width

    # Görüntüdeki x koordinatlarından açıyı hesapla
    angle1 = ((x1 - image_width / 2) / (image_width / 2)) * (fov / 2)
    angle2 = ((x2 - image_width / 2) / (image_width / 2)) * (fov / 2)

    # Açıyı radyana çevir
    angle1 = np.radians(angle1)
    angle2 = np.radians(angle2)

    # Nesnelerin dünya koordinatlarındaki pozisyonlarını hesapla
    x1_world = distance1 * np.cos(angle1)
    y1_world = distance1 * np.sin(angle1)

    x2_world = distance2 * np.cos(angle2)
    y2_world = distance2 * np.sin(angle2)

    # İki nokta arasındaki Öklidyen mesafeyi hesapla
    real_distance = np.sqrt((x2_world - x1_world) ** 2 + (y2_world - y1_world) ** 2)

    # Siyah bir çizgi çiz: (0, 0, 0) renk kodu siyahı temsil eder.
    cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 0), thickness=2)

    # İki nokta arasının orta noktasını hesapla
    mid_point = (int((x1 + x2) / 2), int((y1 + y2) / 2))

    # Hesaplanan mesafeyi metin olarak oluştur (iki ondalık basamak)
    text = f"{real_distance:.2f} m"

    # Metni gri renkte (örneğin, (128, 128, 128)) çizgi üzerine yaz
    cv2.putText(frame, text, mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 2)

    return real_distance


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
                    depth_val = depth.get_value( (x2+x1)/2, (y1+y2)/2 )[1]
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

                # Fonksiyon çağrıları
                greenRedOnly(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected, blue_detected, black_detected, green_positions, red_positions)
                greenYellowOnly(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected, blue_detected, black_detected, green_positions, yellow_positions)
                redYellowOnly(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected, blue_detected, black_detected, yellow_positions, red_positions)
                greenRedYellowOnly(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected, blue_detected, black_detected, yellow_positions, red_positions, green_positions)
            # Sensör verilerini güncelle
            if zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.IMAGE):
                if ts_handler.is_new(sensors_data.get_imu_data()):
                    quaternion = sensors_data.get_imu_data().get_pose().get_orientation().get()
                    magnetometer_data = sensors_data.get_magnetometer_data()
                    magnetic_heading_info = (
                        f"Magnetic Heading: {magnetometer_data.magnetic_heading:.0f} "
                        f"({magnetometer_data.magnetic_heading_state}) "
                        f"[{magnetometer_data.magnetic_heading_accuracy:.0f}]"
                    )
                    yaw = quaternion_to_angle(quaternion, "yaw")
                    roll = quaternion_to_angle(quaternion, "roll")
                    pitch = quaternion_to_angle(quaternion, "pitch")
                    render_text(frame, f"Yaw: {yaw:.0f}", (frame.shape[1] - 200, 30))
                    render_text(frame, f"Roll: {roll:.0f}", (frame.shape[1] - 200, 60))
                    render_text(frame, f"Pitch: {pitch:.0f}", (frame.shape[1] - 200, 90))
                    render_text(frame, magnetic_heading_info, (frame.shape[1] - 1300, 30))

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
