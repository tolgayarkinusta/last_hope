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
manual_mode = False
# Load a model
model = YOLO("balonYeniNano170.engine")
#model.to("cuda")


bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Check if the script is running with root privileges
#if os.geteuid() != 0:
#    print("This script requires elevated privileges. Please run it with `sudo` or as root.")
#    #sys.exit(1)

from MainSystem import USVController

controller = USVController("COM10", baud=57600)
print("Arming vehicle...")
controller.arm_vehicle()
print("Vehicle armed!")
print("Setting mode...")
controller.set_mode("MANUAL")
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
    runtime_params = sl.RuntimeParameters(enable_fill_mode = True
                                          )
    init_params.camera_resolution = sl.RESOLUTION.HD720  # 720p çözünürlük
    init_params.camera_fps = 30  # 30 FPS
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA# depth mode best quality at neural_plus
    init_params.coordinate_units = sl.UNIT.METER #using metric system
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE # default for the opencv
    init_params.depth_minimum_distance = 0.20
    init_params.depth_maximum_distance = 40
    init_params.camera_disable_self_calib = False
    init_params.depth_stabilization = 50 #titreme azaltıcı #todo: positional tracking + depth
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

def nothing(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected, blue_detected, black_detected):
    if not green_detected and not red_detected and not yellow_detected and not blue_detected and not black_detected:
        controller.set_servo(5, 1550)
        controller.set_servo(6, 1550)



# Global olarak while döngüsü dışında tanımlanmalı:
green_detection_start_time = None
def greenOnly(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected, blue_detected, black_detected, green_positions):
    if green_detected and not (red_detected or yellow_detected or blue_detected or black_detected):
        closest_depth = float('inf')
        closest_center = None
        for box in green_positions:
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            depth_val = depth.get_value(cx, cy)[1]
            if np.isnan(depth_val):
                continue
            if depth_val < closest_depth:
                closest_depth = depth_val
                closest_center = (cx, cy)
        if closest_center is not None:
            cv2.line(frame, (center_x, center_y), closest_center, (0, 255, 0), thickness=2)
            # Bekleme kontrolü kaldırıldı, direkt mesafe kontrolü:
            if closest_depth<0.39:
                print("Yeşil nesne 2 metreden yakın: geriye gidiliyor.")
                controller.set_servo(5, 1450)
                controller.set_servo(6, 1450)
            elif 0.39<= closest_depth <=2.01:
                print("Yeşil nesne 2-5 metre arası: yerinde sola dönülüyor.")
                controller.set_servo(5, 1450)
                controller.set_servo(6, 1550)
            elif closest_depth>5.02:
                print("Yeşil nesne 5 metreden uzakta: sadece sağ motor çalıştırılarak hareket ediliyor.")
                controller.set_servo(5, 1500)
                controller.set_servo(6, 1550)
    else:
        global green_detection_start_time
        green_detection_start_time = None



red_detection_start_time = None
def redOnly(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected, blue_detected, black_detected, red_positions):
    if red_detected and not (green_detected or yellow_detected or blue_detected or black_detected):
        closest_depth = float('inf')
        closest_center = None
        for box in red_positions:
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            depth_val = depth.get_value(cx, cy)[1]
            if np.isnan(depth_val):
                continue
            if depth_val < closest_depth:
                closest_depth = depth_val
                closest_center = (cx, cy)
        if closest_center is not None:
            cv2.line(frame, (center_x, center_y), closest_center, (0, 0, 255), thickness=2)
            if closest_depth<0.39:
                print("Kırmızı nesne 2 metreden yakın: geriye gidiliyor.")
                controller.set_servo(5, 1450)
                controller.set_servo(6, 1450)
            elif 0.39<= closest_depth <=2.01:
                print("Kırmızı nesne 2-5 metre arası: yerinde sağa dönülüyor.")
                controller.set_servo(5, 1550)
                controller.set_servo(6, 1450)
            elif closest_depth>5.02:
                print("Kırmızı nesne 5 metreden uzakta: sadece sol motor çalıştırılarak hareket ediliyor.")
                controller.set_servo(5, 1550)
                controller.set_servo(6, 1500)
    else:
        global red_detection_start_time
        red_detection_start_time = None


yellow_detection_start_time = None  # Global değişken, sarı tespiti başlangıç zamanını tutar
def yellowOnly(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected, blue_detected, black_detected, yellow_positions):
    if yellow_detected and not (green_detected or red_detected or blue_detected or black_detected):
        closest_depth = float('inf')
        closest_center = None
        # Tüm sarı kutular etrafına dikdörtgen çiziliyor:
        for box in yellow_positions:
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            depth_val = depth.get_value(cx, cy)[1]
            if np.isnan(depth_val):
                continue
            if depth_val < closest_depth:
                closest_depth = depth_val
                closest_center = (cx, cy)
        if closest_center is not None:
            cv2.line(frame, (center_x, center_y), closest_center, (0, 0, 255), thickness=2)
            if closest_depth < 1:
                print("sarı nesne 1 metreden yakın: geriye gidiliyor.")
                controller.set_servo(5, 1450)
                controller.set_servo(6, 1450)
            else:
                print("sarı nesne 1 metreden uzak:")
                controller.set_servo(5, 1540)
                controller.set_servo(6, 1540)

    else:
        global yellow_detection_start_time
        yellow_detection_start_time = None



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
        if depth_diff>5.02:
            # Fark 2.5 metreden büyükse:
            if closest_green_depth > closest_red_depth:
                print("Derinlik farkı>5.02m, yeşil derinlik > kırmızı: redOnly çağrılıyor.")
                redOnlyRecursive(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected, blue_detected, black_detected, red_positions)
            else:
                print("Derinlik farkı>5.02m, yeşil derinlik <= kırmızı: greenOnly çağrılıyor.")
                greenOnlyRecursive(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected, blue_detected, black_detected, green_positions)
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
            base_speed = 1540
            # Proportional gain (örneğin: 2)
            k = 1.65
            correction = int(k * abs(angle_deg))
            if angle_deg > 0:
                # Pozitif açı durumunda: sol motor azalt, sağ motor artır
                left_command = base_speed + correction
                right_command = base_speed - correction
            else:
                # Negatif açı durumunda: sol motor artır, sağ motor azalt
                left_command = base_speed - correction
                right_command = base_speed + correction

            print(f"Açı: {angle_deg:.1f} derece, motor komutları: Sol={left_command}, Sağ={right_command}")
            controller.set_servo(5, left_command)
            controller.set_servo(6, right_command)


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

        # Eğer her iki tespit de bulunamadıysa çık
        if closest_green_center is None or closest_yellow_center is None:
            return

        # --- Derinlik farkı kontrolü ---
        depth_diff = abs(closest_green_depth - closest_yellow_depth)
        if depth_diff>5.02:
            # Derinlik farkı 2.5 metreden büyükse:
            if closest_yellow_depth > closest_green_depth:
                print("Derinlik farkı>5.02m, sarı derinlik > yeşil: greenOnly çağrılıyor.")
                greenOnlyRecursive(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected, blue_detected, black_detected, green_positions)
            else:
                print("Derinlik farkı>5.02m, sarı derinlik <= yeşil: yellowOnly çağrılıyor.")
                yellowOnlyRecursive(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected, blue_detected, black_detected, yellow_positions)
        else:
            # Derinlik farkı 2.5 metreden büyük değilse:
            # getwidth() fonksiyonundan dönen sonuç kontrol ediliyor
            width_measure = getWidth(closest_green_depth, closest_yellow_depth,
                                     closest_green_center[0], closest_yellow_center[0])
            if width_measure < 0.80:
                print("Genişlik < 1.5m, greenOnly çağrılıyor.")
                greenOnlyRecursive(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected, blue_detected, black_detected, green_positions)

            else:
                # --- İki tespit arasında orta nokta (nokta1) hesaplanıyor ---
                point1 = (int((closest_green_center[0] + closest_yellow_center[0]) / 2),
                          int((closest_green_center[1] + closest_yellow_center[1]) / 2))
                cv2.circle(frame, point1, 5, (255, 0, 0), -1)  # Nokta1 mavi ile işaretleniyor

                # --- Ekran görüntüsünün alt orta noktası (nokta2) ---
                frame_height = frame.shape[0]
                point2 = (center_x, frame_height)
                cv2.circle(frame, point2, 5, (0, 255, 255), -1)  # Nokta2 sarı ile işaretleniyor

                # --- Nokta1 ve Nokta2'yi birleştiren çizgi (çizgi1) ---
                cv2.line(frame, point1, point2, (255, 0, 0), 2)
                # --- Ekran merkezi ile Nokta2 arasında çizgi (çizgi2) ---
                screen_center = (center_x, center_y)
                cv2.line(frame, screen_center, point2, (0, 255, 255), 2)

                # --- İki çizgi arasındaki açıyı hesaplama ---
                # Vektör hesaplamaları:
                vec1 = (point2[0] - point1[0], point2[1] - point1[1])
                vec2 = (point2[0] - screen_center[0], point2[1] - screen_center[1])
                angle1 = math.atan2(vec1[1], vec1[0])
                angle2 = math.atan2(vec2[1], vec2[0])
                angle_rad = angle1 - angle2
                # Açıyı normalize et (-pi, pi arası)
                angle_rad = math.atan2(math.sin(angle_rad), math.cos(angle_rad))
                angle_deg = math.degrees(angle_rad)

                # Açıyı görselleştir:
                cv2.putText(frame, f"Angle: {angle_deg:.1f}", (center_x - 100, center_y + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # --- Motor kontrolü (proportional kontrol) ---
                base_speed = 1540  # Temel hız değeri
                k = 1.65  # Proportional gain
                correction = int(k * abs(angle_deg))
                if angle_deg > 0:
                    # Pozitif açı: sol motor hızını azalt, sağ motoru artır
                    left_command = base_speed + correction
                    right_command = base_speed - correction
                else:
                    # Negatif açı: sol motoru artır, sağ motor hızını azalt
                    left_command = base_speed - correction
                    right_command = base_speed + correction

                print(f"Açı: {angle_deg:.1f} derece, motor komutları: Sol={left_command}, Sağ={right_command}")
                controller.set_servo(5, left_command)
                controller.set_servo(6, right_command)

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

        # --- Derinlik farkı kontrolü ---
        depth_diff = abs(closest_red_depth - closest_yellow_depth)
        if depth_diff>5.02:
            # Eğer derinlik farkı 2.5 metreden büyükse:
            if closest_yellow_depth > closest_red_depth:
                print("Derinlik farkı>5.02m, sarı derinlik kırmızıdan büyük: redOnly çağrılıyor.")
                redOnlyRecursive(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected, blue_detected, black_detected, red_positions)

            else:
                print("Derinlik farkı>5.02m, sarı derinlik kırmızıdan küçük veya eşit: yellowOnly çağrılıyor.")
                yellowOnlyRecursive(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected, blue_detected, black_detected, yellow_positions)

        else:
            # Derinlik farkı 2.5 metreden büyük değilse:
            width_measure = getWidth(closest_yellow_depth, closest_red_depth,
                                     closest_yellow_center[0], closest_red_center[0])
            if width_measure < 0.80:
                print("Genişlik ölçümü < 1.5m: redOnly çağrılıyor.")
                redOnlyRecursive(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected, blue_detected, black_detected, red_positions)

            else:
                # --- Nokta1: Kırmızı ve sarı tespitlerinin orta noktası ---
                point1 = (int((closest_red_center[0] + closest_yellow_center[0]) / 2),
                          int((closest_red_center[1] + closest_yellow_center[1]) / 2))
                cv2.circle(frame, point1, 5, (255, 0, 0), -1)  # Nokta1 mavi renkle işaretleniyor

                # --- Nokta2: Ekran görüntüsünün alt orta noktası ---
                frame_height = frame.shape[0]
                point2 = (center_x, frame_height)
                cv2.circle(frame, point2, 5, (0, 255, 255), -1)  # Nokta2 sarı renkle işaretleniyor

                # --- Çizgi 1: Nokta1 ile Nokta2 arasındaki çizgi ---
                cv2.line(frame, point1, point2, (255, 0, 0), 2)
                # --- Çizgi 2: Ekran merkezi ile Nokta2 arasındaki çizgi ---
                screen_center = (center_x, center_y)
                cv2.line(frame, screen_center, point2, (0, 255, 255), 2)

                # --- İki çizgi arasındaki açıyı hesaplama ---
                # Vektör hesaplamaları:
                vec1 = (point2[0] - point1[0], point2[1] - point1[1])
                vec2 = (point2[0] - screen_center[0], point2[1] - screen_center[1])
                angle1 = math.atan2(vec1[1], vec1[0])
                angle2 = math.atan2(vec2[1], vec2[0])
                angle_rad = angle1 - angle2
                # Açıyı normalize et (-pi, pi arası)
                angle_rad = math.atan2(math.sin(angle_rad), math.cos(angle_rad))
                angle_deg = math.degrees(angle_rad)

                # Açıyı görselleştir:
                cv2.putText(frame, f"Angle: {angle_deg:.1f}", (center_x - 100, center_y + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # --- Motor kontrolü (proportional kontrol) ---
                base_speed = 1540  # Temel hız değeri
                k = 1.65  # Proportional gain
                correction = int(k * abs(angle_deg))
                if angle_deg > 0:
                    # Pozitif açı: sol motor hızını azalt, sağ motoru artır
                    left_command = base_speed + correction
                    right_command = base_speed - correction
                else:
                    # Negatif açı: sol motoru artır, sağ motor hızını azalt
                    left_command = base_speed - correction
                    right_command = base_speed + correction

                print(f"Açı: {angle_deg:.1f} derece, motor komutları: Sol={left_command}, Sağ={right_command}")
                controller.set_servo(5, left_command)
                controller.set_servo(6, right_command)


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
    width_green_yellow = getWidth(closest_green_depth,closest_yellow_depth,
                                  closest_green_center[0], closest_yellow_center[0])
    width_red_yellow = getWidth(closest_red_depth, closest_yellow_depth,
                                closest_red_center[0],closest_yellow_center[0])

    # --- Fonksiyon seçimi ---
    if width_green_yellow > width_red_yellow:
        print("getWidth: Yeşil-Sarı arasındaki mesafe, Kırmızı-Sarı arasından büyük. greenYellowOnly çağrılıyor.")
        greenYellowOnlyRecursive(frame, depth, center_x, center_y,
                        green_detected, red_detected, yellow_detected,
                        blue_detected, black_detected,
                        green_positions, yellow_positions)
    else:
        print("getWidth: Kırmızı-Sarı arasındaki mesafe, Yeşil-Sarı arasından büyük ya da eşit. redYellowOnly çağrılıyor.")
        redYellowOnlyRecursive(frame, depth, center_x, center_y,
                      green_detected, red_detected, yellow_detected,
                      blue_detected, black_detected,
                      yellow_positions, red_positions)


def getWidth(distance1, distance2, x1, x2):
    """
    ZED 2i kamerası kullanarak iki nesne arasındaki gerçek mesafeyi hesaplar.
    Parametreler:
    - distance1: İlk nesnenin tekneye olan uzaklığı (metre cinsinden)
    - distance2: İkinci nesnenin tekneye olan uzaklığı (metre cinsinden)
    - x1: İlk nesnenin görüntüdeki merkezi x koordinatı (piksel cinsinden)
    - x2: İkinci nesnenin görüntüdeki merkezi x koordinatı (piksel cinsinden)
    - fov: Kameranın yatay görüş açısı (derece cinsinden)
    - image_width: Görüntünün genişliği (piksel cinsinden)
    Dönüş:
    - İki nesne arasındaki gerçek dünya mesafesi (metre cinsinden)
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

    return real_distance

def redOnlyRecursive(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected, blue_detected,
              black_detected, red_positions):
    closest_depth = float('inf')
    closest_center = None
    for box in red_positions:  # Kırmızı tespit kutuları arasında
        x1, y1, x2, y2 = box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        depth_val = depth.get_value(cx, cy)[1]
        if np.isnan(depth_val):
            continue
        if depth_val < closest_depth:
            closest_depth = depth_val
            closest_center = (cx, cy)
    if closest_center is not None:
        # Görsel amaçlı: Ekran merkezinden en yakın kırmızı kutunun orta noktasına çizgi çiz
        cv2.line(frame, (center_x, center_y), closest_center, (0, 0, 255), thickness=2)

        if closest_depth<0.39:
                print("Kırmızı nesne 2 metreden yakın: geriye gidiliyor.")
                controller.set_servo(5, 1400)
                controller.set_servo(6, 1400)
        elif 0.39<= closest_depth <=2.01:
                print("Kırmızı nesne 2-5 metre arası: yerinde sağa dönülüyor.")
                controller.set_servo(5, 1600)  # Sol motor ileri
                controller.set_servo(6, 1400)  # Sağ motor geri
        elif closest_depth>5.02:
                print("Kırmızı nesne 5 metreden uzakta: sadece sol motor çalıştırılarak hareket ediliyor.")
                controller.set_servo(5, 1550)  # Sol motor hafif ileri
                controller.set_servo(6, 1500)  # Sağ motor nötr

def greenOnlyRecursive(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected, blue_detected,
              black_detected, green_positions):
    closest_depth = float('inf')
    closest_center = None
    for box in green_positions:  # Yeşil tespit kutuları arasında
        x1, y1, x2, y2 = box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        # Tespit edilen kutunun orta noktasının derinlik değerini al
        depth_val = depth.get_value(cx, cy)[1]
        if np.isnan(depth_val):
            continue
        if depth_val < closest_depth:
            closest_depth = depth_val
            closest_center = (cx, cy)
    if closest_center is not None:
        # Görsel amaçlı: ekranın orta noktası ile en yakın yeşil kutunun orta noktası arasında yeşil çizgi çiz
        cv2.line(frame, (center_x, center_y), closest_center, (0, 255, 0), thickness=2)

        if closest_depth<0.39:
                print("Yeşil nesne 2 metreden yakın: geriye gidiliyor.")
                controller.set_servo(5, 1400)
                controller.set_servo(6, 1400)
        elif 0.39<= closest_depth <=2.01:
                print("Yeşil nesne 2-5 metre arası: yerinde sola dönülüyor.")
                controller.set_servo(5, 1400)  # sol motor geriye
                controller.set_servo(6, 1600)  # sağ motor ileri
        elif closest_depth>5.02:
                print("Yeşil nesne 5 metreden uzakta: sadece sağ motor çalıştırılarak hareket ediliyor.")
                controller.set_servo(5, 1500)  # sol motor nötr
                controller.set_servo(6, 1550)  # sağ motor hafif ileri

def yellowOnlyRecursive(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected, blue_detected,
              black_detected, yellow_positions):
    # En yakın sarı nesnenin mesafesini al
    min_distance = float('inf')  # Varsayılan olarak büyük bir değer atandı
    for box in yellow_positions:
        x1, y1, x2, y2 = map(int, box)
        cx = int((x1 + x2) / 2)  # Nesnenin merkez x koordinatı
        cy = int((y1 + y2) / 2)  # Nesnenin merkez y koordinatı
        depth_val = depth.get_value(cx, cy)[1]  # Derinlik ölçümü

        if depth_val > 0 and depth_val < min_distance:
            min_distance = depth_val  # En yakın sarı nesneyi güncelle

    # Eğer en yakın sarı nesne 5 metreden uzaksa hareketi iptal et
    if min_distance > 1:
        return

    # Görsel eklemeler: Tespit edilen tüm sarı kutucuklar etrafına sarı dikdörtgen çiziliyor
    for box in yellow_positions:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
    # Ekranın orta noktasına "Geri" yazısı ekleniyor
    cv2.putText(frame, "Going Astern!!", (center_x - 50, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    controller.set_servo(5, 1400)  # Sol motor: geri
    controller.set_servo(6, 1400)  # Sağ motor: geri

def greenYellowOnlyRecursive(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected, blue_detected,
                        black_detected, green_positions, yellow_positions):
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

            # Eğer her iki tespit de bulunamadıysa çık
            if closest_green_center is None or closest_yellow_center is None:
                return

            # --- Derinlik farkı kontrolü ---
            depth_diff = abs(closest_green_depth - closest_yellow_depth)
            if depth_diff>5.02:
                # Derinlik farkı 2.5 metreden büyükse:
                if closest_yellow_depth > closest_green_depth:
                    print("Derinlik farkı>5.02m, sarı derinlik > yeşil: greenOnly çağrılıyor.")
                    greenOnlyRecursive(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected,
                                       blue_detected, black_detected, green_positions)
                else:
                    print("Derinlik farkı>5.02m, sarı derinlik <= yeşil: yellowOnly çağrılıyor.")
                    yellowOnlyRecursive(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected,
                                        blue_detected, black_detected, yellow_positions)
            else:
                # Derinlik farkı 2.5 metreden büyük değilse:
                # getwidth() fonksiyonundan dönen sonuç kontrol ediliyor
                width_measure = getWidth(closest_green_depth, closest_yellow_depth,
                                         closest_green_center[0], closest_yellow_center[0])
                if width_measure < 0.80:
                    print("Genişlik < 1.5m, greenOnly çağrılıyor.")
                    greenOnlyRecursive(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected,
                                       blue_detected, black_detected, green_positions)

                else:
                    # --- İki tespit arasında orta nokta (nokta1) hesaplanıyor ---
                    point1 = (int((closest_green_center[0] + closest_yellow_center[0]) / 2),
                              int((closest_green_center[1] + closest_yellow_center[1]) / 2))
                    cv2.circle(frame, point1, 5, (255, 0, 0), -1)  # Nokta1 mavi ile işaretleniyor

                    # --- Ekran görüntüsünün alt orta noktası (nokta2) ---
                    frame_height = frame.shape[0]
                    point2 = (center_x, frame_height)
                    cv2.circle(frame, point2, 5, (0, 255, 255), -1)  # Nokta2 sarı ile işaretleniyor

                    # --- Nokta1 ve Nokta2'yi birleştiren çizgi (çizgi1) ---
                    cv2.line(frame, point1, point2, (255, 0, 0), 2)
                    # --- Ekran merkezi ile Nokta2 arasında çizgi (çizgi2) ---
                    screen_center = (center_x, center_y)
                    cv2.line(frame, screen_center, point2, (0, 255, 255), 2)

                    # --- İki çizgi arasındaki açıyı hesaplama ---
                    # Vektör hesaplamaları:
                    vec1 = (point2[0] - point1[0], point2[1] - point1[1])
                    vec2 = (point2[0] - screen_center[0], point2[1] - screen_center[1])
                    angle1 = math.atan2(vec1[1], vec1[0])
                    angle2 = math.atan2(vec2[1], vec2[0])
                    angle_rad = angle1 - angle2
                    # Açıyı normalize et (-pi, pi arası)
                    angle_rad = math.atan2(math.sin(angle_rad), math.cos(angle_rad))
                    angle_deg = math.degrees(angle_rad)

                    # Açıyı görselleştir:
                    cv2.putText(frame, f"Angle: {angle_deg:.1f}", (center_x - 100, center_y + 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # --- Motor kontrolü (proportional kontrol) ---
                    base_speed = 1540  # Temel hız değeri
                    k = 1.65  # Proportional gain
                    correction = int(k * abs(angle_deg))
                    if angle_deg > 0:
                        # Pozitif açı: sol motor hızını azalt, sağ motoru artır
                        left_command = base_speed + correction
                        right_command = base_speed - correction
                    else:
                        # Negatif açı: sol motoru artır, sağ motor hızını azalt
                        left_command = base_speed - correction
                        right_command = base_speed + correction

                    print(f"Açı: {angle_deg:.1f} derece, motor komutları: Sol={left_command}, Sağ={right_command}")
                    controller.set_servo(5, left_command)
                    controller.set_servo(6, right_command)

def redYellowOnlyRecursive(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected, blue_detected, black_detected, yellow_positions, red_positions):
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

        # --- Derinlik farkı kontrolü ---
        depth_diff = abs(closest_red_depth - closest_yellow_depth)
        if depth_diff>5.02:
            # Eğer derinlik farkı 2.5 metreden büyükse:
            if closest_yellow_depth > closest_red_depth:
                print("Derinlik farkı>5.02m, sarı derinlik kırmızıdan büyük: redOnly çağrılıyor.")
                redOnlyRecursive(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected, blue_detected, black_detected, red_positions)

            else:
                print("Derinlik farkı>5.02m, sarı derinlik kırmızıdan küçük veya eşit: yellowOnly çağrılıyor.")
                yellowOnlyRecursive(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected, blue_detected, black_detected, yellow_positions)

        else:
            # Derinlik farkı 2.5 metreden büyük değilse:
            width_measure = getWidth(closest_yellow_depth, closest_red_depth,
                                     closest_yellow_center[0], closest_red_center[0])
            if width_measure < 0.80:
                print("Genişlik ölçümü < 1.5m: redOnly çağrılıyor.")
                redOnlyRecursive(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected, blue_detected, black_detected, red_positions)

            else:
                # --- Nokta1: Kırmızı ve sarı tespitlerinin orta noktası ---
                point1 = (int((closest_red_center[0] + closest_yellow_center[0]) / 2),
                          int((closest_red_center[1] + closest_yellow_center[1]) / 2))
                cv2.circle(frame, point1, 5, (255, 0, 0), -1)  # Nokta1 mavi renkle işaretleniyor

                # --- Nokta2: Ekran görüntüsünün alt orta noktası ---
                frame_height = frame.shape[0]
                point2 = (center_x, frame_height)
                cv2.circle(frame, point2, 5, (0, 255, 255), -1)  # Nokta2 sarı renkle işaretleniyor

                # --- Çizgi 1: Nokta1 ile Nokta2 arasındaki çizgi ---
                cv2.line(frame, point1, point2, (255, 0, 0), 2)
                # --- Çizgi 2: Ekran merkezi ile Nokta2 arasındaki çizgi ---
                screen_center = (center_x, center_y)
                cv2.line(frame, screen_center, point2, (0, 255, 255), 2)

                # --- İki çizgi arasındaki açıyı hesaplama ---
                # Vektör hesaplamaları:
                vec1 = (point2[0] - point1[0], point2[1] - point1[1])
                vec2 = (point2[0] - screen_center[0], point2[1] - screen_center[1])
                angle1 = math.atan2(vec1[1], vec1[0])
                angle2 = math.atan2(vec2[1], vec2[0])
                angle_rad = angle1 - angle2
                # Açıyı normalize et (-pi, pi arası)
                angle_rad = math.atan2(math.sin(angle_rad), math.cos(angle_rad))
                angle_deg = math.degrees(angle_rad)

                # Açıyı görselleştir:
                cv2.putText(frame, f"Angle: {angle_deg:.1f}", (center_x - 100, center_y + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # --- Motor kontrolü (proportional kontrol) ---
                base_speed = 1540  # Temel hız değeri
                k = 1.65  # Proportional gain
                correction = int(k * abs(angle_deg))
                if angle_deg > 0:
                    # Pozitif açı: sol motor hızını azalt, sağ motoru artır
                    left_command = base_speed + correction
                    right_command = base_speed - correction
                else:
                    # Negatif açı: sol motoru artır, sağ motor hızını azalt
                    left_command = base_speed - correction
                    right_command = base_speed + correction

                print(f"Açı: {angle_deg:.1f} derece, motor komutları: Sol={left_command}, Sağ={right_command}")
                controller.set_servo(5, left_command)
                controller.set_servo(6, right_command)

def main():
    zed = initialize_camera()
    global width
    # Kamera çözünürlüğünü al
    camera_info = zed.get_camera_information()
    width = camera_info.camera_configuration.resolution.width
    print(width)
    height = camera_info.camera_configuration.resolution.height
    print(height)
    # Görüntüde merkez noktasını hesapla
    center_x = width // 2
    center_y = height // 2
    print("Kamera çözünürlüğü: ", width, "x", height)
    print("Görüntü orta noktası: ", (center_x, center_y))

    # Used to store the sensors timestamp to know if the sensors_data is a new one or not
    ts_handler = TimestampHandler()

    # Görüntü ve derinlik verilerini almak için Mat nesneleri oluştur
    image = sl.Mat()
    depth = sl.Mat()
    # Sensör verisi al
    sensors_data = sl.SensorsData()

    # For FPS calculation
    fps_previous_time = 0

    # Sonsuz bir döngüde görüntü akışı
    while True:
        # Kameradan bir yeni kare alın
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Görüntü ve derinlik verilerini al
            zed.retrieve_image(image, sl.VIEW.LEFT)
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            # OpenCV formatına dönüştür
            frame = cv2.cvtColor(image.get_data(), cv2.COLOR_BGRA2BGR)  # BGRA -> BGR
            results = model(frame, conf=0.45)[0]

            # yolo sonuçlarının sv.Detections formatına dönüştürülmesi
            detections = sv.Detections.from_ultralytics(results)

            # tespitlerin sınırlarının ve etiketlerinin oluşturulması
            frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
            frame = label_annotator.annotate(scene=frame, detections=detections)

            # tespitlerin koordinatlarının sınıflarının alınması
            coordinates = detections.xyxy.tolist()
            class_ids = detections.class_id.tolist()

            # Her tespit kutusunun sağ üst köşesine derinlik değerini yazdırmak için:
            for box in coordinates:
                x1, y1, x2, y2 = map(int, box)  # tamsayıya çeviriyoruz
                # Sağ üst köşe koordinatları: (x2, y1)
                depth_val = depth.get_value((x2 + x1) / 2, (y1 + y2) / 2)[1]# Eğer depth değeri geçerliyse (NaN değilse) yazdır
                if not np.isnan(depth_val):
                    text = f"{depth_val:.2f} m"
                    # Yazıyı kutunun sağ üst köşesine ekleyelim; konum ayarını isteğinize göre değiştirebilirsiniz
                    cv2.putText(frame, text, (x2 - 60, y1 + 20), FONT, 0.7, COLOR_RED, 2)

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

            # Tuş kontrolü: 'm' tuşu ile modlar arasında geçiş yapılır.
            key = cv2.waitKey(1) & 0xFF
            if key == ord('m'):
                global manual_mode
                manual_mode = not manual_mode
                if manual_mode:
                    print("Manuel mod aktif. Otomatik sürüş durdu.")
                else:
                    print("Otomatik mod aktif. Manuel kontrol devre dışı.")
                # Küçük bir gecikme, tuşun sürekli algılanmasını önlemek için
                time.sleep(0.2)

            # Manuel mod aktifse, WASD tuşlarıyla kontrol yapılır.
            if manual_mode:
                cv2.putText(frame, "MANUEL MOD", (50, 50), FONT, 1, (0, 255, 255), 2)


                if key == ord('w'):
                    # İleri hareket: her iki motor ileri
                    print("manual")
                    controller.set_servo(5, 1670)
                    controller.set_servo(6, 1670)
                elif key == ord('s'):
                    # Geri hareket: her iki motor geri
                    print("manual")
                    controller.set_servo(5, 1200)
                    controller.set_servo(6, 1200)
                elif key == ord('a'):
                    # Sola dönüş: sol motor yavaş, sağ motor hızlı
                    print("manual")
                    controller.set_servo(5, 1400)
                    controller.set_servo(6, 1600)
                elif key == ord('d'):
                    # Sağa dönüş: sol motor hızlı, sağ motor yavaş
                    print("manual")
                    controller.set_servo(5, 1600)
                    controller.set_servo(6, 1400)
                elif key == ord('l'):
                    # Tuşlara basılmadığında motorlar nötr konumda kalır
                    controller.set_servo(5, 1500)
                    controller.set_servo(6, 1500)
            else:
                # Otomatik modda, tüm renk fonksiyonları çalışır:
                nothing(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected, blue_detected, black_detected)
                greenOnly(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected, blue_detected, black_detected, green_positions)
                redOnly(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected, blue_detected, black_detected, red_positions)
                yellowOnly(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected, blue_detected, black_detected, yellow_positions)
                greenRedOnly(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected, blue_detected, black_detected, green_positions, red_positions)
                greenYellowOnly(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected, blue_detected, black_detected, green_positions, yellow_positions)
                redYellowOnly(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected, blue_detected, black_detected, yellow_positions, red_positions)
                greenRedYellowOnly(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected, blue_detected, black_detected, green_positions, red_positions,yellow_positions)

                # retrieve the current sensors sensors_data
                if zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.IMAGE): #time_reference.image for synchorinzed timestamps
                    # Check if the data has been updated since the last time
                    # IMU is the sensor with the highest rate
                    if ts_handler.is_new(sensors_data.get_imu_data()):

                        # Filtered orientation quaternion
                        quaternion = sensors_data.get_imu_data().get_pose().get_orientation().get()
                        # Access the magnetometer data
                        magnetometer_data = sensors_data.get_magnetometer_data()

                        # Access the magnetic heading and state
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

                # Orta noktanın derinlik bilgisini al
                #depth_value = depth.get_value(center_x, center_y)[1]  # Sadece metre cinsinden değeri al

                # Derinlik bilgisini sağ üst köşede göster imu bilgisini sol üst köşede göster
                #depth_info_text = f"Center Depth: {depth_value:.2f} m" if not np.isnan(depth_value) else "Couldn't Calculate..: NaN"
                #render_text(frame, depth_info_text, (10, 50))
                #cv2.circle(frame, (center_x, center_y), DEPTH_CENTER_RADIUS, DEPTH_CENTER_COLOR, -1)

                # Merkez noktasını görselleştir
                #cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)

            # Calculate FPS
            fps_current_time = time.time()
            fps = 1 / (fps_current_time - fps_previous_time)
            fps_previous_time = fps_current_time
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Görüntüyü göster
            frame_resized = cv2.resize(frame, (960, 540))  # Resize the frame to desired dimensions960, 540
            cv2.imshow("ZED Camera", frame_resized)
            #avoidBuoys()

            k = cv2.waitKey(1)

            if k % 256 == 27:
                print("Esc tuşuna basıldı.. Kapatılıyor..")
                controller.stop_motors()
                break

    # Kaynakları serbest bırak ve kamerayı kapat
    cv2.destroyAllWindows()
    zed.close()

if __name__ == "__main__":
    main()
