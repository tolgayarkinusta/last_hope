import pyzed.sl as sl
import cv2
import numpy as np
import math
import time
from ultralytics import YOLO
import supervision as sv

#import torch , torchvision; from torch.autograd import backward kaldırıldı
from headingFilter import KalmanFilter #todo: heading filtresi testi
magnetic_filter = KalmanFilter(process_variance=1e-3, measurement_variance=1e-1)

# Load a model
model = YOLO("balonYeniNano170.engine")
#model.to("cuda")
from config import MOTOR_PWM, CONTROL_PARAMS

threshold = CONTROL_PARAMS["threshold"]
usv_width = CONTROL_PARAMS["usv_width"]
baseHigher2 = MOTOR_PWM["base++"]
baseHigher1 = MOTOR_PWM["base+"]
baseLower1 = MOTOR_PWM["base-"]
base_speed = MOTOR_PWM["base_speed"]
neutral = MOTOR_PWM["neutral"]
backwards = MOTOR_PWM["backwards"]
k = CONTROL_PARAMS["k"]
close_depth = CONTROL_PARAMS["close_depth"]
medium_depth = CONTROL_PARAMS["medium_depth"]
far_depth = CONTROL_PARAMS["far_depth"]

bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

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
    runtime_params = sl.RuntimeParameters(enable_fill_mode = True)
    init_params.camera_resolution = sl.Resolution.RESOLUTION.HD720  # 720p çözünürlük #todo:Resolution sorun çıkarmadıysa bu todoyu kaldır
    init_params.camera_fps = 60  # 30 FPS
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA# depth mode best quality at neural_plus
    init_params.coordinate_units = sl.UNIT.METER #using metric system
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE # default for the opencv
    init_params.depth_minimum_distance = 0.20
    init_params.depth_maximum_distance = 40
    init_params.camera_disable_self_calib = False
    init_params.depth_stabilization = 50 #titreme azaltıcı
    init_params.sensors_required = False # true yaparsan imu açılmadan kamera açılmaz
    init_params.enable_image_enhancement = True #true was always the default
    init_params.async_grab_camera_recovery = False #set true if u want to keep processing if cam gets shutdown
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        raise Exception("Failed to open ZED camera. Exiting.")
    return zed

# Initialize positional tracking
def initialize_positional_tracking(zed):
    # Enable positional tracking with default parameters
    py_transform = sl.Transform()  # First create a Transform object for TrackingParameters object
    tracking_parameters = sl.PositionalTrackingParameters(_init_pos=py_transform)
    tracking_parameters.enable_pose_smoothing = True  # set true to lower the loop_closures. effect Default False
    tracking_parameters.set_floor_as_origin = False  # Set the floor as the reference point Default False todo: True ile dene
    tracking_parameters.enable_area_memory = True  # Persistent memory for localization, Whether the camera can remember its surroundings. Default True
    tracking_parameters.enable_imu_fusion = True  # When set to False, only the optical odometry will be used. Default True
    tracking_parameters.set_as_static = False  # Whether to define the camera as static. Default False
    tracking_parameters.depth_min_range = 0.40 # It may be useful for example if any steady objects are in front of the camera and may perturb the positional tracking algorithm. Dfault no min range: -1

    err = zed.enable_positional_tracking(tracking_parameters)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Enable positional tracking : " + repr(err) + ". Exit program.")
        zed.close()
        exit()

# Initialize spatial mapping
def initialize_spatial_mapping(zed):
    # Enable spatial mapping
    mapping_parameters = sl.SpatialMappingParameters(map_type=sl.SPATIAL_MAP_TYPE.MESH) #.mesh or .fused_point_cloud
    mapping_parameters.resolution_meter = 0.10  # Define resolution (0.05m for fine mapping)
    mapping_parameters.range_meter = 20  # Maximum range for the mesh
    mapping_parameters.use_chunk_only = True  # Allow chunks of the map #true for better performance #false for accuracy
    mapping_parameters.max_memory_usage = 2048 # The maximum CPU memory (in MB) allocated for the meshing process. #default 2048
    #Whether to inverse the order of the vertices of the triangles.If your display process does not handle front and back face culling,
    mapping_parameters.reverse_vertex_order = False # you can use this to correct it. Default: False. only for mesh

    error = zed.enable_spatial_mapping(mapping_parameters)
    if error != sl.ERROR_CODE.SUCCESS:
        raise Exception(f"Spatial mapping initialization failed: {error}")

# Render text over the frame
def render_text(frame, text, position):
    cv2.putText(frame, text, position, FONT, FONT_SCALE, COLOR_RED, THICKNESS)

# Basic class to handle the timestamp of the different sensors to know if it is a new sensors_data or an old one
class TimestampHandler:
    def __init__(self):
        self.t_imu = sl.Timestamp()
    # check if the new timestamp is higher than the reference one, and if yes, save the current as reference
    def is_new(self, sensor):
        if isinstance(sensor, sl.IMUData):
            new_ = (sensor.timestamp.get_microseconds() > self.t_imu.get_microseconds())
            if new_:
                self.t_imu = sensor.timestamp
            return new_

def nothing(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected, blue_detected, black_detected):
        controller.set_servo(5, base_speed)
        controller.set_servo(6, base_speed)

def greenOnly(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected, blue_detected, black_detected, green_positions):
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
            if closest_depth < close_depth:
                print("Yeşil nesne yakın mesafede: geriye gidiliyor.")
                controller.set_servo(5, backwards-20)
                controller.set_servo(6, backwards+30)
            elif close_depth <= closest_depth < medium_depth:
                print("Yeşil nesne orta mesafede arası: yerinde sola dönülüyor.")
                controller.set_servo(5, 1465)
                controller.set_servo(6, 1540)
            elif closest_depth > medium_depth:
                print("Yeşil nesne uzak mesafede: sadece sağ motor çalıştırılarak hareket ediliyor.")
                controller.set_servo(5, baseLower1)
                controller.set_servo(6, baseHigher1)

def redOnly(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected, blue_detected, black_detected, red_positions):
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
            if closest_depth<close_depth:
                print("Kırmızı nesne yakın: geriye gidiliyor.")
                controller.set_servo(5, backwards+30)
                controller.set_servo(6, backwards-20)
            elif closest_depth <= closest_depth < medium_depth:
                print("Kırmızı nesne orta: yerinde sağa dönülüyor.")
                controller.set_servo(5, 1540)
                controller.set_servo(6, 1465)
            elif closest_depth>medium_depth:
                print("Kırmızı nesne uzakta: sadece sol motor çalıştırılarak hareket ediliyor.")
                controller.set_servo(5, baseHigher1)
                controller.set_servo(6, baseLower1)

def yellowOnly(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected, blue_detected, black_detected, yellow_positions):

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
            if closest_depth < close_depth:
                print("sarı nesne yakın: geriye gidiliyor.")
                controller.set_servo(5, backwards-10)
                controller.set_servo(6, backwards-10)
            elif closest_depth < medium_depth:
                print("sarı nesne orta uzaklıkta: geri daha yavaş:")
                controller.set_servo(5, base_speed-10)
                controller.set_servo(6, base_speed-10)
            else:
                controller.set_servo(5, base_speed-5)
                controller.set_servo(6, base_speed-5)

def greenRedOnly(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected, blue_detected, black_detected, green_positions, red_positions):
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
                closest_red_depth = depth_val # en yakın kırmızının uzaklığı
                closest_red_center = (cx, cy) # en yakın kırmızının orta nokta koordinatları
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
                closest_green_depth = depth_val # en yakın yeşilin uzaklığı
                closest_green_center = (cx, cy) # en yakın yeşilin orta nokta koordinatları

        if closest_red_center is None or closest_green_center is None:
            return  # Her iki renk tespiti yoksa işleme devam etme

        if closest_green_depth < close_depth or closest_red_depth < close_depth:
            controller.set_servo(5, backwards)
            controller.set_servo(6, backwards)
        else:
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

        # Eğer herhangi bir mesafe close_depth'den büyükse geri hareket komutu verelim
        if closest_green_depth < close_depth or closest_yellow_depth < close_depth:
            controller.set_servo(5, backwards)
            controller.set_servo(6, backwards)
        else:
            # a, daha kısa mesafe; b, daha uzun mesafe olsun
            if closest_green_depth < closest_yellow_depth:
                a = closest_green_depth
            else:
                a = closest_yellow_depth
            # Ekranın alt orta noktası (frame yüksekliğinin altı, genişliğin ortası)
            height, wwidth = frame.shape[:2]
            bottom_center = (wwidth // 2, height)

            # Yeşil ve sarı tespit noktalarından ekran alt orta noktasına çizgiler çizelim
            cv2.line(frame, closest_green_center, bottom_center, (0, 255, 0), 2)   # Yeşil çizgi
            cv2.line(frame, closest_yellow_center, bottom_center, (0, 255, 255), 2)  # Sarı çizgi

            # Çizgilerin oluşturduğu vektörler
            vec_green = (bottom_center[0] - closest_green_center[0], bottom_center[1] - closest_green_center[1])
            vec_yellow = (bottom_center[0] - closest_yellow_center[0], bottom_center[1] - closest_yellow_center[1])

            # Vektör büyüklüklerini hesaplayalım
            mag_green = math.hypot(vec_green[0], vec_green[1])
            mag_yellow = math.hypot(vec_yellow[0], vec_yellow[1])

            # Eğer vektörlerden herhangi birinin büyüklüğü 0 ise işlem yapma
            if mag_green == 0 or mag_yellow == 0:
                return

            # Dot product ile açıyı hesapla
            dot = vec_green[0] * vec_yellow[0] + vec_green[1] * vec_yellow[1]
            cosine_angle = dot / (mag_green * mag_yellow)
            # Numerik stabilite için cosine_angle değerini [-1,1] aralığına sınırlayalım
            cosine_angle = max(min(cosine_angle, 1), -1)
            alfa = math.acos(cosine_angle)

            # path_width hesaplanması: sin(alfa) * a
            path_width = math.sin(alfa) * a

            # Eğer path_width, usv_width'den büyükse geçiş uygun
            if path_width > usv_width:
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
            else:
                controller.set_servo(5, 1300)
                controller.set_servo(6, 1450)

def redYellowOnly(frame, depth, center_x, center_y, red_detected, green_detected, yellow_detected, blue_detected, black_detected, red_positions, yellow_positions):
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

        # Eğer her iki tespit de bulunamadıysa çık
        if closest_red_center is None or closest_yellow_center is None:
            return

        # Eğer herhangi bir mesafe close_depth'den büyükse geri hareket komutu verelim
        if closest_red_depth < close_depth or closest_yellow_depth < close_depth:
            controller.set_servo(5, backwards)
            controller.set_servo(6, backwards)
        else:
            # a, daha kısa mesafe; b, daha uzun mesafe olsun (burada sadece a kullanıyoruz)
            if closest_red_depth < closest_yellow_depth:
                a = closest_red_depth
            else:
                a = closest_yellow_depth

            # Ekranın alt orta noktası (frame yüksekliğinin altı, genişliğin ortası)
            height, wwidth = frame.shape[:2]
            bottom_center = (wwidth // 2, height)

            # Kırmızı ve sarı tespit noktalarından ekran alt orta noktasına çizgiler çizelim
            cv2.line(frame, closest_red_center, bottom_center, (0, 0, 255), 2)   # Kırmızı çizgi
            cv2.line(frame, closest_yellow_center, bottom_center, (0, 255, 255), 2)  # Sarı çizgi

            # Çizgilerin oluşturduğu vektörler
            vec_red = (bottom_center[0] - closest_red_center[0], bottom_center[1] - closest_red_center[1])
            vec_yellow = (bottom_center[0] - closest_yellow_center[0], bottom_center[1] - closest_yellow_center[1])

            # Vektör büyüklüklerini hesaplayalım
            mag_red = math.hypot(vec_red[0], vec_red[1])
            mag_yellow = math.hypot(vec_yellow[0], vec_yellow[1])

            # Eğer vektörlerden herhangi birinin büyüklüğü 0 ise işlem yapma
            if mag_red == 0 or mag_yellow == 0:
                return

            # Dot product ile açıyı hesapla
            dot = vec_red[0] * vec_yellow[0] + vec_red[1] * vec_yellow[1]
            cosine_angle = dot / (mag_red * mag_yellow)
            # Numerik stabilite için cosine_angle değerini [-1,1] aralığına sınırlayalım
            cosine_angle = max(min(cosine_angle, 1), -1)
            alfa = math.acos(cosine_angle)

            # path_width hesaplanması: sin(alfa) * a
            path_width = math.sin(alfa) * a

            # Eğer path_width, usv_width'den büyükse geçiş uygun
            if path_width > usv_width:
                # --- İki tespit arasında orta nokta (nokta1) hesaplanıyor ---
                point1 = (int((closest_red_center[0] + closest_yellow_center[0]) / 2),
                          int((closest_red_center[1] + closest_yellow_center[1]) / 2))
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
            else:
                controller.set_servo(5, 1450)
                controller.set_servo(6, 1300)

def greenRedYellowOnly(frame, depth, center_x, center_y,
                                     green_detected, red_detected, yellow_detected,
                                     blue_detected, black_detected,
                                     green_positions, red_positions, yellow_positions):
        print("fonksyiona girildi...")
        # --- Yeşil tespitinin en yakını ---
        closest_green_depth = float('inf')
        closest_green_center = None
        if green_detected:
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
        # --- Kırmızı tespitinin en yakını ---
        closest_red_depth = float('inf')
        closest_red_center = None
        if red_detected:
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
        # --- Sarı tespitinin en yakını ---
        closest_yellow_depth = float('inf')
        closest_yellow_center = None
        if yellow_detected:
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
        print("derinlikler alındı...")

        if closest_yellow_depth < close_depth or closest_green_depth < close_depth or closest_red_depth < close_depth:
            controller.set_servo(5,backwards)
            controller.set_servo(6,backwards)
            print("çok yakınız.")

        else:
            print("validation...")
            # Geçerlilik kontrolü: eşik değeri altındaysa valid sayalım.
            valid_green = (closest_green_center is not None) and (closest_green_depth < threshold)
            valid_red   = (closest_red_center   is not None) and (closest_red_depth   < threshold)
            valid_yellow= (closest_yellow_center is not None) and (closest_yellow_depth < threshold)
            print("valid_count hesaplanıyor...")
            valid_count = sum([valid_green, valid_red, valid_yellow])
            print(valid_count)

            # --- Durum 1: Tüm nesneler eşik içerisinde ise ---
            if valid_count == 3:

                height, wwidth = frame.shape[:2]
                bottom_center = (wwidth // 2, height)
                print("nesnelerin 3ü de threshold içerisinde")
                # Yeşil-Sarı ikilisi için path_width hesaplama:
                a_gy = min(closest_green_depth, closest_yellow_depth)
                vec_green = (bottom_center[0] - closest_green_center[0], bottom_center[1] - closest_green_center[1])
                vec_yellow = (bottom_center[0] - closest_yellow_center[0], bottom_center[1] - closest_yellow_center[1])
                mag_green = math.hypot(vec_green[0], vec_green[1])
                mag_yellow = math.hypot(vec_yellow[0], vec_yellow[1])
                if mag_green == 0 or mag_yellow == 0:
                    path_width_gy = 0
                else:
                    dot_gy = vec_green[0]*vec_yellow[0] + vec_green[1]*vec_yellow[1]
                    cosine_angle_gy = dot_gy / (mag_green * mag_yellow)
                    cosine_angle_gy = max(min(cosine_angle_gy, 1), -1)
                    alfa_gy = math.acos(cosine_angle_gy)
                    path_width_gy = math.sin(alfa_gy) * a_gy

                # Kırmızı-Sarı ikilisi için path_width hesaplama:
                a_ry = min(closest_red_depth, closest_yellow_depth)
                vec_red = (bottom_center[0] - closest_red_center[0], bottom_center[1] - closest_red_center[1])
                mag_red = math.hypot(vec_red[0], vec_red[1])
                if mag_red == 0 or mag_yellow == 0:
                    path_width_ry = 0
                else:
                    dot_ry = vec_red[0]*vec_yellow[0] + vec_red[1]*vec_yellow[1]
                    cosine_angle_ry = dot_ry / (mag_red * mag_yellow)
                    cosine_angle_ry = max(min(cosine_angle_ry, 1), -1)
                    alfa_ry = math.acos(cosine_angle_ry)
                    path_width_ry = math.sin(alfa_ry) * a_ry

                # Hangi ikilinin yol genişliği daha büyükse, ilgili fonksiyon çağrılır.
                if path_width_gy >= path_width_ry:
                    # Yeşil-Sarı ikilisi seçildi
                    greenYellowOnly(frame, depth, center_x, center_y,
                                    green_detected, red_detected, yellow_detected,
                                    blue_detected, black_detected, green_positions, yellow_positions)
                else:
                    # Kırmızı-Sarı ikilisi seçildi
                    redYellowOnly(frame, depth, center_x, center_y,
                                  red_detected, green_detected, yellow_detected,
                                  blue_detected, black_detected, red_positions, yellow_positions)

            # --- Durum 2: Sadece 2 nesne eşik değeri içerisinde ise ---
            elif valid_count == 2:
                print("2 valid")

                # Hangi renk eşik değerini aşıyorsa ignore edelim ve diğer ikiliye uygun fonksiyonu çağıralım.
                if not valid_green:
                    # Yeşil eşik değerinin üzerinde, dolayısıyla kırmızı ve sarı kullanılacak.
                    redYellowOnly(frame, depth, center_x, center_y,
                                  red_detected, green_detected, yellow_detected,
                                  blue_detected, black_detected, red_positions, yellow_positions)
                elif not valid_red:
                    # Kırmızı eşik değerinin üzerinde, dolayısıyla yeşil ve sarı kullanılacak.
                    greenYellowOnly(frame, depth, center_x, center_y,
                                    green_detected, red_detected, yellow_detected,
                                    blue_detected, black_detected, green_positions, yellow_positions)
                elif not valid_yellow:
                    # Sarı eşik değerinin üzerinde.
                    print("greenRedOnly çalışıyor")

                    greenRedOnly(frame, depth, center_x, center_y,
                                    green_detected, red_detected, yellow_detected,
                                    blue_detected, black_detected, green_positions, red_positions)
                    print("greenRedOnly çalışıyor")

            # --- Durum 3: Sadece 1 nesne eşik değeri içerisinde ise (diğer 2 eşik değerini aşıyor) ---
            elif valid_count == 1:
                print("nesnelerin 1i threshold içerisinde")

                if valid_green:
                    greenOnly(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected, blue_detected, black_detected, green_positions)
                elif valid_red:
                    redOnly(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected, blue_detected, black_detected, red_positions)
                elif valid_yellow:
                    yellowOnly(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected, blue_detected, black_detected, yellow_positions)
            else:
                print("Hiçbir nesne eşik değeri altında değil.")
                controller.set_servo(5, base_speed)
                controller.set_servo(6, base_speed)


def main():
    global width
    print("Initializing Camera...")
    zed = initialize_camera()
    print("Camera initialized! Initializing positional tracking...")
    initialize_positional_tracking(zed)
    print("Tracking initialized! Initializing spatial mapping...")
    initialize_spatial_mapping(zed)
    print("Mapping initialized!")

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
    pose = sl.Pose()
    mesh = sl.Mesh()

    # mevcut koordinatları al
    translation = pose.get_translation(sl.Translation()).get()  # [tx, ty, tz]
    current_x = translation[0]
    current_y = translation[2]

    # Sensör verisi al
    sensors_data = sl.SensorsData()

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
            manual_mode = False
            if key == ord('m'):
                manual_mode = not manual_mode
                if manual_mode is True:
                    print("Manuel mod aktif. Otomatik sürüş durdu.")
                else:
                    print("Otomatik mod aktif. Manuel kontrol devre dışı.")
                # Küçük bir gecikme, tuşun sürekli algılanmasını önlemek için
                time.sleep(0.2)

            # Manuel mod aktifse, WASD tuşlarıyla kontrol yapılır.
            if manual_mode:
                cv2.putText(frame, "MANUEL MOD", (50, 50), FONT, 1, (0, 255, 255), 2)#todo: ekran orta noktasına al
                if key == ord('w'):
                    # İleri hareket: her iki motor ileri
                    print("manual")
                    controller.set_servo(5, 1600)
                    controller.set_servo(6, 1600)
                elif key == ord('s'):
                    # Geri hareket: her iki motor geri
                    print("manual")
                    controller.set_servo(5, 1300)
                    controller.set_servo(6, 1300)
                elif key == ord('a'):
                    # Sola dönüş: sol motor yavaş, sağ motor hızlı
                    print("manual")
                    controller.set_servo(5, 1420)
                    controller.set_servo(6, 1580)
                elif key == ord('d'):
                    # Sağa dönüş: sol motor hızlı, sağ motor yavaş
                    print("manual")
                    controller.set_servo(5, 1580)
                    controller.set_servo(6, 1420)
                elif key == ord('l'):
                    # Tuşlara basılmadığında motorlar nötr konumda kalır
                    controller.set_servo(5, 1500)
                    controller.set_servo(6, 1500)
            else:

                # Otomatik modda, tüm renk fonksiyonları çalışır:
                if not green_detected and not red_detected and not yellow_detected and not blue_detected and not black_detected:
                    nothing(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected, blue_detected, black_detected)
                if green_detected and not (red_detected or yellow_detected or blue_detected or black_detected):
                    greenOnly(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected, blue_detected, black_detected, green_positions)
                if red_detected and not (green_detected or yellow_detected or blue_detected or black_detected):
                    redOnly(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected, blue_detected, black_detected, red_positions)
                if yellow_detected and not (green_detected or red_detected or blue_detected or black_detected):
                    yellowOnly(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected, blue_detected, black_detected, yellow_positions)
                if red_detected and green_detected and not (yellow_detected or blue_detected or black_detected):
                    greenRedOnly(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected, blue_detected, black_detected, green_positions, red_positions)
                if yellow_detected and green_detected and not (red_detected or blue_detected or black_detected):
                    greenYellowOnly(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected, blue_detected, black_detected, green_positions, yellow_positions)
                if yellow_detected and red_detected and not (green_detected or blue_detected or black_detected):
                    redYellowOnly(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected, blue_detected, black_detected, yellow_positions, red_positions)
                if yellow_detected and red_detected and green_detected and not (blue_detected or black_detected):
                    greenRedYellowOnly(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected, blue_detected, black_detected, green_positions, red_positions,yellow_positions)

            # retrieve the current sensors sensors_data
            if zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.IMAGE): #time_reference.image for synchorinzed timestamps
                # Check if the data has been updated since the last time
                # IMU is the sensor with the highest rate
                if ts_handler.is_new(sensors_data.get_imu_data()):
                    # Access the magnetometer data
                    magnetometer_data = sensors_data.get_magnetometer_data()
                    # Get the raw magnetic heading  # Apply low-pass filter
                    magnetic_heading = magnetic_filter.update(sensors_data.get_magnetometer_data().magnetic_heading)
                    # Access the magnetic heading and state
                    magnetic_heading_info = (
                        f"Magnetic Heading: {magnetic_heading:.0f} "
                        f"({magnetometer_data.magnetic_heading_state}) "
                        f"[{magnetometer_data.magnetic_heading_accuracy:.0f}]"
                    )
                    render_text(frame, magnetic_heading_info, (frame.shape[1] - 1300, 30))


            cv2.putText(frame, f"FPS: {int(zed.get_current_fps())}", (10, 30), FONT, 1, (0, 255, 0),2)
            cv2.putText(frame, f"{str(zed.get_spatial_mapping_state())}", (10, 60), FONT, 0.5, (20, 220, 20), 1)
            cv2.putText(frame, f"POSITIONAL_TRACKING_STATE.{str(zed.get_position(pose, sl.REFERENCE_FRAME.WORLD))}",(10, 90), FONT, 0.5, (20, 220, 20), 1)
            cv2.putText(frame, f"Coordinates X,Y: {current_x:.1f} {current_y:.1f}",(10, 120), FONT, 0.75, (0, 150, 240), 1,)

            # Görüntüyü göster
            frame_resized = cv2.resize(frame, (960, 540))  # Resize the frame to desired dimensions960, 540
            cv2.imshow("ZED Camera", frame_resized)

            if key % 256 == 27:
                print("Esc tuşuna basıldı.. Kapatılıyor..")
                controller.stop_motors()
                break

    # Kaynakları serbest bırak ve kamerayı kapat
    cv2.destroyAllWindows()
    zed.close()

if __name__ == "__main__":
    main()
