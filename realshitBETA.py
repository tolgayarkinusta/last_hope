import pyzed.sl as sl
import cv2
import numpy as np
import math
import time
from ultralytics import YOLO
import supervision as sv
# import torch , torchvision; from torch.autograd import backward kaldırıldı
from headingFilter import KalmanFilter  # todo: heading filtresi testi
from config import MOTOR_PWM, CONTROL_PARAMS
from MainSystem import USVController

# --- CONFIGURATION ---
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

# --- GLOBAL VARIABLES ---
# Load a model
model = YOLO("balonYeniNano170.engine")
# model.to("cuda")

# Harita için ayarlar:
map_width = 800  # Harita genişliği (piksel)
map_height = 800  # Harita yüksekliği (piksel)
scale = 20  # 1 metre = 50 piksel (uyarlamayı ihtiyaç duyarsan değiştirebilirsin)
# Başlangıç konumunu harita merkezine yerleştiriyoruz:
map_center = (map_width // 2, map_height // 2)
# Kalıcı harita (persistent map): başlangıçta mavi bir tuval
light_blue = (173, 216, 230)  # Açık mavi renk (R, G, B)
map_image = np.full((map_height, map_width, 3), light_blue, dtype=np.uint8) * 255

magnetic_filter = KalmanFilter(process_variance=1e-3, measurement_variance=1e-1)

bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

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
    runtime_params = sl.RuntimeParameters(enable_fill_mode=True)
    init_params.camera_resolution = sl.RESOLUTION.HD720  # 720p çözünürlük
    init_params.camera_fps = 60  # 30 FPS
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # depth mode best quality at neural_plus
    init_params.coordinate_units = sl.UNIT.METER  # using metric system
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP  # finale
    init_params.depth_minimum_distance = 0.20
    init_params.depth_maximum_distance = 40
    init_params.camera_disable_self_calib = False
    init_params.depth_stabilization = 50  # titreme azaltıcı
    init_params.sensors_required = False  # true yaparsan imu açılmadan kamera açılmaz
    init_params.enable_image_enhancement = True  # true was always the default
    init_params.async_grab_camera_recovery = False  # set true if u want to keep processing if cam gets shutdown
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
    tracking_parameters.depth_min_range = 0.40  # It may be useful for example if any steady objects are in front of the camera and may perturb the positional tracking algorithm. Dfault no min range: -1
    tracking_parameters.mode = sl.POSITIONAL_TRACKING_MODE.GEN_1  # gen_1 for performance, gen_2 for accuracy #todo: GEN_1 ve GEN_2 kıyaslamasını yap.

    err = zed.enable_positional_tracking(tracking_parameters)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Enable positional tracking : " + repr(err) + ". Exit program.")
        zed.close()
        exit()


# Initialize spatial mapping
def initialize_spatial_mapping(zed):
    # Enable spatial mapping
    mapping_parameters = sl.SpatialMappingParameters(map_type=sl.SPATIAL_MAP_TYPE.MESH)  # .mesh or .fused_point_cloud
    mapping_parameters.resolution_meter = 0.10  # Define resolution (0.05m for fine mapping)
    mapping_parameters.range_meter = 20  # Maximum range for the mesh
    mapping_parameters.use_chunk_only = True  # Allow chunks of the map #true for better performance #false for accuracy
    mapping_parameters.max_memory_usage = 2048  # The maximum CPU memory (in MB) allocated for the meshing process. #default 2048
    # Whether to inverse the order of the vertices of the triangles.If your display process does not handle front and back face culling,
    mapping_parameters.reverse_vertex_order = False  # you can use this to correct it. Default: False. only for mesh

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


def nothing(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected, blue_detected,
            black_detected):
    controller.set_servo(5, base_speed)
    controller.set_servo(6, base_speed)


def greenOnly(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected, blue_detected,
              black_detected, green_positions):
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
        if closest_depth <= close_depth:
            print("Yeşil nesne yakın mesafede: geriye gidiliyor.")
            controller.set_servo(5, backwards - 20)
            controller.set_servo(6, backwards + 30)
        elif closest_depth <= medium_depth:
            print("Yeşil nesne orta mesafede arası: yerinde sola dönülüyor.")
            controller.set_servo(5, 1465)
            controller.set_servo(6, 1540)
        else:
            print("Yeşil nesne uzak mesafede: sadece sağ motor çalıştırılarak hareket ediliyor.")
            controller.set_servo(5, baseLower1)
            controller.set_servo(6, baseHigher1)


def redOnly(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected, blue_detected,
            black_detected, red_positions):
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
        if closest_depth <= close_depth:
            print("Kırmızı nesne yakın: geriye gidiliyor.")
            controller.set_servo(5, backwards + 30)
            controller.set_servo(6, backwards - 20)
        elif closest_depth <= medium_depth:
            print("Kırmızı nesne orta: yerinde sağa dönülüyor.")
            controller.set_servo(5, 1540)
            controller.set_servo(6, 1465)
        else:
            print("Kırmızı nesne uzakta: sadece sol motor çalıştırılarak hareket ediliyor.")
            controller.set_servo(5, baseHigher1)
            controller.set_servo(6, baseLower1)


def yellowOnly(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected, blue_detected,
               black_detected, yellow_positions):
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
            controller.set_servo(5, backwards - 10)
            controller.set_servo(6, backwards - 10)
        elif closest_depth < medium_depth:
            print("sarı nesne orta uzaklıkta: geri daha yavaş:")
            controller.set_servo(5, base_speed - 10)
            controller.set_servo(6, base_speed - 10)
        else:
            controller.set_servo(5, base_speed - 5)
            controller.set_servo(6, base_speed - 5)


def greenRedOnly(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected, blue_detected,
                 black_detected, green_positions, red_positions):
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
            closest_red_depth = depth_val  # en yakın kırmızının uzaklığı
            closest_red_center = (cx, cy)  # en yakın kırmızının orta nokta koordinatları
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
            closest_green_depth = depth_val  # en yakın yeşilin uzaklığı
            closest_green_center = (cx, cy)  # en yakın yeşilin orta nokta koordinatları

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


def greenYellowOnly(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected, blue_detected,
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
        cv2.line(frame, closest_green_center, bottom_center, (0, 255, 0), 2)  # Yeşil çizgi
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


def redYellowOnly(frame, depth, center_x, center_y, red_detected, green_detected, yellow_detected, blue_detected,
                  black_detected, red_positions, yellow_positions):
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
        cv2.line(frame, closest_red_center, bottom_center, (0, 0, 255), 2)  # Kırmızı çizgi
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
        controller.set_servo(5, backwards)
        controller.set_servo(6, backwards)
        print("çok yakınız.")

    else:
        print("validation...")
        # Geçerlilik kontrolü: eşik değeri altındaysa valid sayalım.
        valid_green = (closest_green_center is not None) and (closest_green_depth < threshold)
        valid_red = (closest_red_center is not None) and (closest_red_depth < threshold)
        valid_yellow = (closest_yellow_center is not None) and (closest_yellow_depth < threshold)
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
                dot_gy = vec_green[0] * vec_yellow[0] + vec_green[1] * vec_yellow[1]
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
                dot_ry = vec_red[0] * vec_yellow[0] + vec_red[1] * vec_yellow[1]
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
                greenOnly(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected,
                          blue_detected, black_detected, green_positions)
            elif valid_red:
                redOnly(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected, blue_detected,
                        black_detected, red_positions)
            elif valid_yellow:
                yellowOnly(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected,
                           blue_detected, black_detected, yellow_positions)
        else:
            print("Hiçbir nesne eşik değeri altında değil.")
            controller.set_servo(5, base_speed)
            controller.set_servo(6, base_speed)

def avoid_buoys(frame, depth, center_x, center_y,
                       green_detected, red_detected, yellow_detected,
                       blue_detected, black_detected,
                       green_positions, red_positions, yellow_positions):

    # Otomatik modda, tüm renk fonksiyonları çalışır:
    if not green_detected and not red_detected and not yellow_detected and not blue_detected and not black_detected:
        nothing(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected,
                blue_detected, black_detected)
    if green_detected and not (red_detected or yellow_detected or blue_detected or black_detected):
        greenOnly(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected,
                  blue_detected, black_detected, green_positions)
    if red_detected and not (green_detected or yellow_detected or blue_detected or black_detected):
        redOnly(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected,
                blue_detected, black_detected, red_positions)
    if yellow_detected and not (green_detected or red_detected or blue_detected or black_detected):
        yellowOnly(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected,
                   blue_detected, black_detected, yellow_positions)
    if red_detected and green_detected and not (yellow_detected or blue_detected or black_detected):
        greenRedOnly(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected,
                     blue_detected, black_detected, green_positions, red_positions)
    if yellow_detected and green_detected and not (red_detected or blue_detected or black_detected):
        greenYellowOnly(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected,
                        blue_detected, black_detected, green_positions, yellow_positions)
    if yellow_detected and red_detected and not (green_detected or blue_detected or black_detected):
        redYellowOnly(frame, depth, center_x, center_y, green_detected, red_detected, yellow_detected,
                      blue_detected, black_detected, yellow_positions, red_positions)
    if yellow_detected and red_detected and green_detected and not (blue_detected or black_detected):
        greenRedYellowOnly(frame, depth, center_x, center_y, green_detected, red_detected,
                           yellow_detected, blue_detected, black_detected, green_positions,
                           red_positions, yellow_positions)

def blueOnly(frame, depth, blue_positions):
    closest_depth = float('inf')
    closest_center = None
    for box in blue_positions:
        x1, y1, x2, y2 = box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        depth_val = depth.get_value(cx, cy)[1]
        if np.isnan(depth_val):
            continue
        if depth_val < closest_depth:
            closest_depth = depth_val
            closest_center = (cx, cy)

    # --- Nokta 1: Mavi tespitin orta noktası ---
    point1 = (int(closest_center[0]), int(closest_center[0]))
    cv2.circle(frame, point1, 6, (0, 0, 0), -1)  # Nokta1'i siyah ile işaretle


def draw_map_with_heading(current_x, current_y, magnetic_heading):
    """
    2D harita üzerinde aracın izini günceller ve o anki magnetic_heading yönünde ok çizer.
    Önceki ok silinir; ancak aracın izi kalıcı olarak haritada tutulur.
    """
    global map_image, map_center, scale

    # World koordinatlarını piksele dönüştür (başlangıç konumunu merkez kabul ederek)
    pixel_x = int(map_center[0] + current_x * scale)
    pixel_y = int(map_center[1] - current_y * scale)  # Görüntü koordinatlarında y ters yönde artar

    # Kalıcı haritaya aracın izini ekle (küçük daire)
    cv2.circle(map_image, (pixel_x, pixel_y), 2, (0, 0, 255), -1)

    # Haritadan kopya alarak geçici çizim yap; bu kopyaya ok çizilecek
    display_map = map_image.copy()

    # Ok çizimi için ayarlar:
    arrow_length = 40  # ok uzunluğu (piksel)
    # Manyetik başlığı radyana çevir (0° kuzeyi temsil edecek şekilde)
    rad = math.radians(magnetic_heading)
    # Okun ucunu hesapla:
    arrow_tip_x = pixel_x + int(arrow_length * math.sin(rad))
    arrow_tip_y = pixel_y - int(arrow_length * math.cos(rad))

    # O anki yönü gösteren ok çiz:
    cv2.arrowedLine(display_map, (pixel_x, pixel_y), (arrow_tip_x, arrow_tip_y), (0, 255, 0), 2, tipLength=0.3)

    # Haritayı göster:
    cv2.imshow("Vehicle Map", display_map)
    cv2.waitKey(1)


def navigate_to_start(frame, current_x, current_y, magnetic_heading, start_x, start_y):
    """
    Mevcut konum ile başlangıç konumu arasındaki yön, hakiki rota ile eşleşecek şekilde
    motor komutları gönderilir. Aradaki açı farkı ±10° içindeyse düz gidilir.
    Başlangıç konumuna yaklaşıldığında motorlar durdurulur.
    """

    # Hedef vektör hesaplaması:
    dx = start_x - current_x
    dy = start_y - current_y
    # İki nokta arasındaki mesafeyi hesapla:
    distance = math.hypot(dx, dy)
    # Varış için mesafe eşiği (örneğin 0.5 metre)
    arrival_threshold = 0.5

    if distance < arrival_threshold:
        print("Başlangıç konumuna ulaşıldı. Motorlar durduruluyor.")
        controller.stop_motors()
        return

    # Hedef açıyı hesapla (0° kuzeyi temsil eder; atan2 kullanarak)
    # Burada: atan2(dx, dy) kullanıyoruz; böylece dx pozitif olduğunda hedef açı pozitif olur.
    target_angle = math.degrees(math.atan2(dx, dy))

    # Hedef açı ile mevcut manyetik başlık arasındaki fark:
    error = target_angle - magnetic_heading
    # Farkı -180 ile 180 arasında normalize et:
    error = (error + 180) % 360 - 180

    # Eğer fark ±10° içindeyse düz git:
    if abs(error) <= 20:
        left_command = base_speed
        right_command = base_speed
    else:
        # Orantısal kontrol: k katsayısını kullanarak düzeltme
        correction = int(k * abs(error / 5))
        if error > 0:
            # Hedef açı, manyetik başlıktan büyükse; sol motor hız artırılır, sağ motor yavaşlatılır.
            left_command = base_speed + correction
            right_command = base_speed - correction
        else:
            # Hedef açı, manyetik başlıktan küçükse; sol motor yavaşlatılır, sağ motor hız artırılır.
            left_command = base_speed - correction
            right_command = base_speed + correction

    print(f"Navigating home: distance = {distance:.2f} m, target_angle = {target_angle:.1f}°, "
          f"magnetic_heading = {magnetic_heading:.1f}°, error = {error:.1f}°")
    controller.set_servo(5, left_command)
    controller.set_servo(6, right_command)

    cv2.putText(frame, f"Navigating home: distance = {distance:.2f} m", (10, 150), FONT, 1, (0, 0, 0), 1, )
    cv2.putText(frame, f"target_angle = {target_angle:.1f}°", (10, 170), FONT, 1, (0, 0, 0), 1, )
    cv2.putText(frame, f"magnetic_heading = {magnetic_heading:.1f}°", (10, 190), FONT, 1, (0, 0, 0), 1, )
    cv2.putText(frame, f"error = {error:.1f}°", (10, 210), FONT, 1, (0, 0, 0), 1, )


# --- MISSION_DOCKING ---
def docking(frame, depth, current_x, current_y, magnetic_heading, center_x, center_y, map_image,
            detections):

    if not hasattr(docking, "phase"):
        docking.phase = "compute_line"
        docking.landing_candidates = []
        docking.current_candidate_index = 0
        docking.wait_start_time = 0

    # --- AŞAMA 1: Çizginin Hesaplanması ---
    if docking.phase == "compute_line":
        shape_boxes = []
        for i, class_id in enumerate(detections.class_id.tolist()):
            if class_id in [10, 11, 12]:
                shape_boxes.append(detections.xyxy.tolist()[i])
        if len(shape_boxes) < 2:
            cv2.putText(frame, "Intermediate shapes not enough...", (50, 250), FONT, 1, (0, 255, 255), 2)
            return False

        shape_world_points = []
        for box in shape_boxes:
            x1, y1, x2, y2 = map(int, box)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            d_val = depth.get_value(cx, cy)[1]
            if np.isnan(d_val):
                continue
            offset = cx - center_x
            angle_offset = math.degrees(math.atan2(offset, center_x))
            global_angle = magnetic_heading + angle_offset
            dx = d_val * math.cos(math.radians(global_angle))
            dy = d_val * math.sin(math.radians(global_angle))
            shape_x = current_x + dx
            shape_y = current_y + dy
            shape_world_points.append((shape_x, shape_y))
            pixel_x = int(map_center[0] + shape_x * scale)
            pixel_y = int(map_center[1] - shape_y * scale)
            cv2.circle(map_image, (pixel_x, pixel_y), 5, (0, 0, 255), -1)

        if len(shape_world_points) < 2:
            cv2.putText(frame, "Insufficient valid shape points...", (50, 270), FONT, 1, (0, 255, 255), 2)
            return False

        # İlk iki noktayı kullanarak doğru oluşturma:
        p1, p2 = shape_world_points[0], shape_world_points[1]
        pixel1 = (int(map_center[0] + p1[0] * scale), int(map_center[1] - p1[1] * scale))
        pixel2 = (int(map_center[0] + p2[0] * scale), int(map_center[1] - p2[1] * scale))
        cv2.line(map_image, pixel1, pixel2, (255, 0, 0), 2)
        mid_x = (p1[0] + p2[0]) / 2
        mid_y = (p1[1] + p2[1]) / 2
        vx = p2[0] - p1[0]
        vy = p2[1] - p1[1]
        mag = math.hypot(vx, vy)
        if mag == 0:
            return False
        perp_x = -vy / mag
        perp_y = vx / mag
        candidate1 = (mid_x + 3 * perp_x, mid_y + 3 * perp_y)
        candidate2 = (mid_x - 3 * perp_x, mid_y - 3 * perp_y)
        dist1 = math.hypot(current_x - candidate1[0], current_y - candidate1[1])
        dist2 = math.hypot(current_x - candidate2[0], current_y - candidate2[1])
        if dist1 < dist2:
            chosen_candidates = [candidate1, candidate2]
        else:
            chosen_candidates = [candidate2, candidate1]
        docking.landing_candidates = chosen_candidates
        docking.current_candidate_index = 0
        docking.phase = "approach"
        cv2.putText(frame, "Line computed, moving to approach phase.", (50, 250), FONT, 1, (0, 255, 0), 2)
        return False

    # --- AŞAMA 2: Yaklaşma ---
    elif docking.phase == "approach":
        candidate = docking.landing_candidates[docking.current_candidate_index]
        candidate_pixel = (int(map_center[0] + candidate[0] * scale), int(map_center[1] - candidate[1] * scale))
        cv2.circle(map_image, candidate_pixel, 8, (0, 255, 0), -1)

        # Seçilen şekli tespit et (class_id 10, 11, 12)
        selected_shape_box = None
        for i, class_id in enumerate(detections.class_id.tolist()):
            if class_id in [10, 11, 12]:
                selected_shape_box = detections.xyxy.tolist()[i]
                break

        if selected_shape_box is not None:
            x1, y1, x2, y2 = map(int, selected_shape_box)
            shape_center_x = int((x1 + x2) / 2)
            deviation = shape_center_x - center_x
            # Şekil ekranın yatay merkezinden sapıyorsa motor komutlarıyla düzelt.
            if abs(deviation) > 20:
                cv2.putText(frame, "Centering shape...", (50, 280), FONT, 1, (255, 255, 0), 2)
                if deviation > 0:
                    controller.set_servo(5, base_speed - 20)
                    controller.set_servo(6, base_speed + 20)
                else:
                    controller.set_servo(5, base_speed + 20)
                    controller.set_servo(6, base_speed - 20)
                return False

            # Şekil merkezde ise, port durumunu kontrol et.
            center_depth = depth.get_value(center_x, center_y)[1]
            if center_depth < 1.5:
                cv2.putText(frame, "Port full, switching candidate...", (50, 300), FONT, 1, (0, 0, 255), 2)
                if docking.current_candidate_index < len(docking.landing_candidates) - 1:
                    docking.current_candidate_index += 1
                controller.stop_motors()
                return False
            else:
                # Port boş; shape'ın derinliğine göre yaklaşma.
                shape_depth = depth.get_value(shape_center_x, int((y1+y2)/2))[1]
                navigate_to_start(frame, current_x, current_y, magnetic_heading, candidate[0], candidate[1])
                if shape_depth < 1.0:
                    controller.stop_motors()
                    cv2.putText(frame, "Port reached, waiting...", (50, 320), FONT, 1, (0, 255, 0), 2)
                    docking.phase = "wait"
                    docking.wait_start_time = time.time()
                    return False
        else:
            # Şekil tespit edilemediyse, doğrudan landing candidate'a ilerle.
            navigate_to_start(frame, current_x, current_y, magnetic_heading, candidate[0], candidate[1])
        return False

    # --- AŞAMA 3: Bekleme ---
    elif docking.phase == "wait":
        elapsed = time.time() - docking.wait_start_time
        cv2.putText(frame, f"Waiting: {elapsed:.1f}s", (50, 340), FONT, 1, (0, 255, 0), 2)
        if elapsed >= 3:
            docking.phase = "reverse"
        return False

    # --- AŞAMA 4: Geri Çekilme ---
    elif docking.phase == "reverse":
        candidate = docking.landing_candidates[docking.current_candidate_index]
        controller.set_servo(5, backwards)
        controller.set_servo(6, backwards)
        dist_to_candidate = math.hypot(current_x - candidate[0], current_y - candidate[1])
        cv2.putText(frame, f"Reversing: {dist_to_candidate:.2f}m", (50, 360), FONT, 1, (255, 0, 0), 2)
        if dist_to_candidate > 5.0:
            controller.stop_motors()
            cv2.putText(frame, "Intermediate Mission 1 complete.", (50, 380), FONT, 1, (0, 255, 0), 2)
            del docking.phase
            del docking.landing_candidates
            del docking.current_candidate_index
            del docking.wait_start_time
            return True
        return False

    return False

def speed_challenge(frame, depth, current_x, current_y, magnetic_heading, center_x, center_y, map_image, detections):
    """
    Intermediate Mission 2:
      - Phase "search_for_15": Araç, class id 15 olan nesneyi tespit edene kadar (sol:1540, sağ:1580)
        motor komutlarıyla arama yapar.
      - Phase "approach_15": Nesne tespit edildiğinde, nesne ekranın merkezine alınır ve
        nesneye yaklaşılır. Mesafe 7 m’nin altına düştüğünde, nesneyle arasındaki mesafe 2-3 m aralığında tutulur.
        Bu durum, class id 20 tespit edilene kadar sürdürülür.
      - Phase "record_pos1_and_search_for_30": Class id 20 tespit edildiğinde, mevcut konum pos1 kaydedilir
        ve araç yavaşça dönerek class id 30 aramaya başlar. 30 saniye içinde tespit olmazsa görev tamamlanır.
      - Phase "approach_30": Class id 30 tespit edildiğinde, nesne ekranın merkezine alınır ve
        nesneye yaklaşılır; mesafe 4 m’nin altına düştüğünde pos2 kaydedilir.
      - Phase "mark_and_rotate": Pos2 kullanılarak nesne haritada işaretlenir, ardından araç 2 m çapında (1 m yarıçap)
        nesnenin işareti etrafında dönerek pos2’ye geri döner.
      - Phase "return_to_pos1": Araç pos1’e yönelir ve görev tamamlanır.
    """
    if not hasattr(speed_challenge, "phase"):
        speed_challenge.phase = "search_for_15"
        speed_challenge.pos1 = None
        speed_challenge.pos2 = None
        speed_challenge.search_30_start = None
        # rotation phase için başlangıç zamanı
        speed_challenge.rotate_start = None

    # --- PHASE 1: SEARCH FOR OBJECT WITH CLASS ID 15 ---
    if speed_challenge.phase == "search_for_15":
        found_15 = any(class_id == 15 for class_id in detections.class_id.tolist())
        if not found_15:
            cv2.putText(frame, "Searching for object 15...", (50, 250), FONT, 1, (255,255,0), 2)
            controller.set_servo(5, 1540)
            controller.set_servo(6, 1580)
            return False
        else:
            cv2.putText(frame, "Object 15 detected, approaching...", (50, 250), FONT, 1, (0,255,0), 2)
            speed_challenge.phase = "approach_15"
            return False

    # --- PHASE 2: APPROACH OBJECT WITH CLASS ID 15 ---
    elif speed_challenge.phase == "approach_15":
        obj15_box = None
        for i, class_id in enumerate(detections.class_id.tolist()):
            if class_id == 15:
                obj15_box = detections.xyxy.tolist()[i]
                break
        if obj15_box is not None:
            x1, y1, x2, y2 = map(int, obj15_box)
            obj_center_x = int((x1 + x2) / 2)
            obj_center_y = int((y1 + y2) / 2)
            deviation = obj_center_x - center_x
            if abs(deviation) > 20:
                cv2.putText(frame, "Centering object 15...", (50, 280), FONT, 1, (255,255,0), 2)
                if deviation > 0:
                    controller.set_servo(5, base_speed - 20)
                    controller.set_servo(6, base_speed + 20)
                else:
                    controller.set_servo(5, base_speed + 20)
                    controller.set_servo(6, base_speed - 20)
                return False
            # Nesne merkezde: derinlik ölçümü
            obj_depth = depth.get_value(obj_center_x, obj_center_y)[1]
            cv2.putText(frame, f"Obj15 depth: {obj_depth:.2f} m", (50, 310), FONT, 1, (0,255,0), 2)
            if obj_depth > 7:
                # Henüz 7 m'ye yaklaşılmadı: normal yaklaşma komutları verilebilir.
                # Örneğin, navigate_to_start kullanılarak nesnenin olduğu yönde ilerlenebilir.
                navigate_to_start(frame, current_x, current_y, magnetic_heading, current_x, current_y)
                return False
            else:
                # Mesafe 7 m'nin altına düştüğünde: artık nesneyle arasındaki mesafe 2-3 m aralığında tutulmalı.
                if obj_depth < 2:
                    controller.set_servo(5, backwards)
                    controller.set_servo(6, backwards)
                elif obj_depth > 3:
                    controller.set_servo(5, base_speed + 10)
                    controller.set_servo(6, base_speed + 10)
                else:
                    controller.stop_motors()
                cv2.putText(frame, "Maintaining 2-3 m distance from object 15", (50, 340), FONT, 1, (0,255,0), 2)
                # Class id 20 tespit edilip edilmediğini kontrol et:
                found_20 = any(class_id == 20 for class_id in detections.class_id.tolist())
                if found_20:
                    cv2.putText(frame, "Object 20 detected, recording pos1", (50, 370), FONT, 1, (0,255,0), 2)
                    speed_challenge.pos1 = (current_x, current_y)
                    speed_challenge.phase = "search_for_30"
                    speed_challenge.search_30_start = time.time()
                return False
        else:
            speed_challenge.phase = "search_for_15"
            return False

    # --- PHASE 3: SEARCH FOR OBJECT WITH CLASS ID 30 ---
    elif speed_challenge.phase == "search_for_30":
        cv2.putText(frame, "Searching for object 30...", (50, 250), FONT, 1, (255,255,0), 2)
        # Yavaşça dönme komutları (örneğin, sol motor yavaş, sağ motor hızlı)
        controller.set_servo(5, 1400)
        controller.set_servo(6, 1600)
        if time.time() - speed_challenge.search_30_start > 30:
            cv2.putText(frame, "Object 30 not found in 30s. Mission complete.", (50, 280), FONT, 1, (0,0,255), 2)
            controller.stop_motors()
            return True  # Görev tamamlandı.
        found_30 = any(class_id == 30 for class_id in detections.class_id.tolist())
        if found_30:
            cv2.putText(frame, "Object 30 detected, approaching...", (50, 310), FONT, 1, (0,255,0), 2)
            speed_challenge.phase = "approach_30"
        return False

    # --- PHASE 4: APPROACH OBJECT WITH CLASS ID 30 ---
    elif speed_challenge.phase == "approach_30":
        obj30_box = None
        for i, class_id in enumerate(detections.class_id.tolist()):
            if class_id == 30:
                obj30_box = detections.xyxy.tolist()[i]
                break
        if obj30_box is not None:
            x1, y1, x2, y2 = map(int, obj30_box)
            obj30_center_x = int((x1 + x2) / 2)
            obj30_center_y = int((y1 + y2) / 2)
            deviation = obj30_center_x - center_x
            if abs(deviation) > 20:
                cv2.putText(frame, "Centering object 30...", (50, 280), FONT, 1, (255,255,0), 2)
                if deviation > 0:
                    controller.set_servo(5, base_speed - 20)
                    controller.set_servo(6, base_speed + 20)
                else:
                    controller.set_servo(5, base_speed + 20)
                    controller.set_servo(6, base_speed - 20)
                return False
            obj30_depth = depth.get_value(obj30_center_x, obj30_center_y)[1]
            cv2.putText(frame, f"Obj30 depth: {obj30_depth:.2f} m", (50, 310), FONT, 1, (0,255,0), 2)
            # Nesneye yaklaşma: navigate_to_start kullanılarak (uygun şekilde ayarlanmalı)
            navigate_to_start(frame, current_x, current_y, magnetic_heading, current_x, current_y)
            if obj30_depth < 4:
                controller.stop_motors()
                cv2.putText(frame, "Object 30 reached.", (50, 340), FONT, 1, (0,255,0), 2)
                speed_challenge.pos2 = (current_x, current_y)
                speed_challenge.phase = "mark_and_rotate"
            return False
        else:
            speed_challenge.phase = "search_for_30"
            return False

    # --- PHASE 5: MARK AND ROTATE AROUND OBJECT 30 ---
    elif speed_challenge.phase == "mark_and_rotate":
        # Pos2'yi haritada işaretle
        if speed_challenge.pos2 is not None:
            pixel_pos2 = (int(map_center[0] + speed_challenge.pos2[0] * scale),
                          int(map_center[1] - speed_challenge.pos2[1] * scale))
            cv2.circle(map_image, pixel_pos2, 10, (0,255,255), -1)
            cv2.putText(frame, "Marking object on map.", (50, 280), FONT, 1, (0,255,255), 2)
        # Aracın, işaret etrafında 2 m çapında (1 m yarıçap) dönmesini simüle ediyoruz.
        if speed_challenge.rotate_start is None:
            speed_challenge.rotate_start = time.time()
        elapsed_rotate = time.time() - speed_challenge.rotate_start
        cv2.putText(frame, f"Rotating: {elapsed_rotate:.1f}s", (50, 310), FONT, 1, (0,255,255), 2)
        controller.set_servo(5, 1400)
        controller.set_servo(6, 1600)
        if elapsed_rotate >= 10:  # Örneğin 10 saniyede tam dönüş varsayalım.
            controller.stop_motors()
            speed_challenge.rotate_start = None
            speed_challenge.phase = "return_to_pos1"
        return False

    # --- PHASE 6: RETURN TO POS1 ---
    elif speed_challenge.phase == "return_to_pos1":
        if speed_challenge.pos1 is not None:
            cv2.putText(frame, "Returning to pos1...", (50, 280), FONT, 1, (0,255,0), 2)
            navigate_to_start(frame, current_x, current_y, magnetic_heading, speed_challenge.pos1[0], speed_challenge.pos1[1])
            dist_to_pos1 = math.hypot(current_x - speed_challenge.pos1[0], current_y - speed_challenge.pos1[1])
            cv2.putText(frame, f"Dist to pos1: {dist_to_pos1:.2f} m", (50, 310), FONT, 1, (0,255,0), 2)
            if dist_to_pos1 < 0.5:
                controller.stop_motors()
                cv2.putText(frame, "Intermediate Mission 2 complete.", (50, 340), FONT, 1, (0,255,0), 2)
                # Temizlik
                del speed_challenge.phase
                del speed_challenge.pos1
                del speed_challenge.pos2
                del speed_challenge.search_30_start
                del speed_challenge.rotate_start
                return True
        return False

    return False

def special_mission(frame, depth, current_x, current_y, magnetic_heading, center_x, center_y, map_image, detections):
    """
    Special Mission:
      - Eğer class id 40 veya 50 tespit edilirse, mevcut konum ve manyetik heading kaydedilir.
      - Eğer class id 40 tespit edilmişse:
          * Nesne ekranın ortasında olacak şekilde merkezlenir.
          * Nesneye 1 metreye kadar yaklaşılır.
          * Nesne ile arasındaki mesafe 1 ile 1.5 m aralığında tutulur.
          * Bu mesafe sağlanırken servo pin 4'e 2000 PWM 5 saniye uygulanır,
            ardından 1000 PWM'ye döndürülür.
          * Kaydedilen konuma dönülür, dönüş sonrası kaydedilen manyetik heading ±10 farkla
            hizalanana kadar araç etrafında döner.
          * Görev tamamlanır.
      - Eğer class id 50 tespit edilmişse:
          * Yukarıdaki adımlar, servo pin 3 için uygulanır.
    """
    if not hasattr(special_mission, "phase"):
        special_mission.phase = "init"
        special_mission.recorded_pos = None
        special_mission.recorded_heading = None
        special_mission.mission_type = None  # 40 veya 50
        special_mission.maintain_start = None
        special_mission.servo_hold_start = None

    # --- PHASE 1: INIT ---
    if special_mission.phase == "init":
        mission_type = None
        for i, class_id in enumerate(detections.class_id.tolist()):
            if class_id == 40:
                mission_type = 40
                special_mission.object_box = detections.xyxy.tolist()[i]
                break
            elif class_id == 50:
                mission_type = 50
                special_mission.object_box = detections.xyxy.tolist()[i]
                break
        if mission_type is None:
            cv2.putText(frame, "Special mission: waiting for object 40/50...", (50, 250), FONT, 1, (255,0,255), 2)
            return False
        else:
            special_mission.mission_type = mission_type
            special_mission.recorded_pos = (current_x, current_y)
            special_mission.recorded_heading = magnetic_heading
            cv2.putText(frame, f"Special mission triggered, type {mission_type} recorded", (50, 270), FONT, 1, (0,255,0), 2)
            special_mission.phase = "approach"
            return False

    # --- PHASE 2: APPROACH ---
    elif special_mission.phase == "approach":
        obj_box = special_mission.object_box
        if obj_box is None:
            special_mission.phase = "init"
            return False
        x1, y1, x2, y2 = map(int, obj_box)
        obj_center_x = int((x1 + x2) / 2)
        obj_center_y = int((y1 + y2) / 2)
        deviation = obj_center_x - center_x
        if abs(deviation) > 20:
            cv2.putText(frame, "Centering special object...", (50, 280), FONT, 1, (255,255,0), 2)
            if deviation > 0:
                controller.set_servo(5, base_speed - 20)
                controller.set_servo(6, base_speed + 20)
            else:
                controller.set_servo(5, base_speed + 20)
                controller.set_servo(6, base_speed - 20)
            return False
        obj_depth = depth.get_value(obj_center_x, obj_center_y)[1]
        cv2.putText(frame, f"Special object depth: {obj_depth:.2f} m", (50, 310), FONT, 1, (0,255,0), 2)
        # Yaklaşma: nesne 1 metreye ulaşana kadar yaklaş
        if obj_depth > 1.0:
            # İlerleme komutları: örneğin navigate_to_start benzeri komutlarla ileri hareket.
            # Burada, nesnenin yönüne doğru ilerlemek için motor komutları verilebilir.
            navigate_to_start(frame, current_x, current_y, magnetic_heading, current_x, current_y)  # Bu örnek; gerçek uygulamada hedef nesnenin konumuna göre ayarlanmalı.
            return False
        else:
            # Nesne 1 m'ye ulaştı; aynı zamanda derinlik 1-1.5 m aralığındaysa
            if 1.0 <= obj_depth <= 1.5:
                cv2.putText(frame, "Special object reached desired distance", (50, 340), FONT, 1, (0,255,0), 2)
                special_mission.phase = "maintain"
                special_mission.maintain_start = time.time()
            else:
                # Eğer çok yakınsa (< 1.0), biraz geri git
                controller.set_servo(5, backwards)
                controller.set_servo(6, backwards)
            return False

    # --- PHASE 3: MAINTAIN DISTANCE (1-1.5 m) FOR 5 SECONDS ---
    elif special_mission.phase == "maintain":
        obj_box = special_mission.object_box
        x1, y1, x2, y2 = map(int, obj_box)
        obj_center_x = int((x1 + x2) / 2)
        obj_center_y = int((y1 + y2) / 2)
        obj_depth = depth.get_value(obj_center_x, obj_center_y)[1]
        cv2.putText(frame, f"Maintaining distance: {obj_depth:.2f} m", (50, 360), FONT, 1, (0,255,0), 2)
        if obj_depth < 1.0:
            controller.set_servo(5, backwards)
            controller.set_servo(6, backwards)
        elif obj_depth > 1.5:
            controller.set_servo(5, base_speed + 10)
            controller.set_servo(6, base_speed + 10)
        else:
            controller.stop_motors()
        if time.time() - special_mission.maintain_start >= 5:
            special_mission.phase = "servo_hold"
            special_mission.servo_hold_start = time.time()
        return False

    # --- PHASE 4: SERVO HOLD ---
    elif special_mission.phase == "servo_hold":
        if special_mission.mission_type == 40:
            controller.set_servo(4, 2000)
        elif special_mission.mission_type == 50:
            controller.set_servo(3, 2000)
        cv2.putText(frame, "Activating special servo...", (50, 380), FONT, 1, (0,255,0), 2)
        if time.time() - special_mission.servo_hold_start >= 5:
            if special_mission.mission_type == 40:
                controller.set_servo(4, 1000)
            elif special_mission.mission_type == 50:
                controller.set_servo(3, 1000)
            special_mission.phase = "return"
        return False

    # --- PHASE 5: RETURN TO RECORDED POSITION ---
    elif special_mission.phase == "return":
        if special_mission.recorded_pos is not None:
            cv2.putText(frame, "Returning to recorded position...", (50, 400), FONT, 1, (0,255,0), 2)
            navigate_to_start(frame, current_x, current_y, magnetic_heading, special_mission.recorded_pos[0], special_mission.recorded_pos[1])
            dist_to_recorded = math.hypot(current_x - special_mission.recorded_pos[0], current_y - special_mission.recorded_pos[1])
            cv2.putText(frame, f"Dist to record: {dist_to_recorded:.2f} m", (50, 420), FONT, 1, (0,255,0), 2)
            if dist_to_recorded < 0.5:
                controller.stop_motors()
                special_mission.phase = "rotate"
        return False

    # --- PHASE 6: ROTATE TO ALIGN WITH RECORDED HEADING (±10°) ---
    elif special_mission.phase == "rotate":
        cv2.putText(frame, "Rotating to align heading...", (50, 440), FONT, 1, (0,255,0), 2)
        heading_error = special_mission.recorded_heading - magnetic_heading
        heading_error = (heading_error + 180) % 360 - 180  # Normalize error to [-180, 180]
        if abs(heading_error) > 10:
            correction = int(k * abs(heading_error / 5))
            if heading_error > 0:
                controller.set_servo(5, base_speed + correction)
                controller.set_servo(6, base_speed - correction)
            else:
                controller.set_servo(5, base_speed - correction)
                controller.set_servo(6, base_speed + correction)
            return False
        else:
            controller.stop_motors()
            cv2.putText(frame, "Special Mission complete.", (50, 460), FONT, 1, (0,255,0), 2)
            # Cleanup
            del special_mission.phase
            del special_mission.recorded_pos
            del special_mission.recorded_heading
            del special_mission.mission_type
            if hasattr(special_mission, "maintain_start"):
                del special_mission.maintain_start
            if hasattr(special_mission, "servo_hold_start"):
                del special_mission.servo_hold_start
            return True

    return False

def special_mission_triggered():
    # Üçgen veya cross tespit edildi mi kontrol et
    global cross_detected, triangle_detected
    if cross_detected or triangle_detected:
        return True
    return False

def main():
    global width, manual_mode, magnetic_heading
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
    # mesh = sl.Mesh()

    # Sensör verisi al
    sensors_data = sl.SensorsData()

    translation = pose.get_translation(sl.Translation()).get()  # [tx, ty, tz]
    start_x = translation[0]
    start_y = -(translation[1])
    print("Başlangıç konumu kaydedildi:", start_x, start_y)

    manual_mode = False

    # --- MISSION CONSTANTS ---
    MISSION_AVOID_BUOYS = 1
    MISSION_DOCKING = 2  # You can add more intermediate missions later
    MISSION_SPEED_CHALLENGE = 3
    MISSION_RETURN_HOME = 4

    MISSION_SPECIAL = 99    #ball and water delivery

    mission_state = MISSION_AVOID_BUOYS
    prev_mission_state = mission_state

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

            # retrieve the current sensors sensors_data
            if zed.get_sensors_data(sensors_data,
                                    sl.TIME_REFERENCE.IMAGE):  # time_reference.image for synchorinzed timestamps
                # Check if the data has been updated since the last time
                # IMU is the sensor with the highest rate
                if ts_handler.is_new(sensors_data.get_imu_data()):
                    # Access the magnetometer data
                    magnetometer_data = sensors_data.get_magnetometer_data()
                    # Get the raw magnetic heading  # Apply low-pass filter
                    # magnetic_heading = magnetic_filter.update(sensors_data.get_magnetometer_data().magnetic_heading)
                    magnetic_heading = sensors_data.get_magnetometer_data().magnetic_heading

                    # Access the magnetic heading and state
                    magnetic_heading_info = (
                        f"Magnetic Heading: {magnetic_heading:.0f} "
                        f"({magnetometer_data.magnetic_heading_state}) "
                        f"[{magnetometer_data.magnetic_heading_accuracy:.1f}]"
                    )
                    render_text(frame, magnetic_heading_info, (frame.shape[1] - 1300, 30))

            # mevcut koordinatları al
            translation = pose.get_translation(sl.Translation()).get()  # [tx, ty, tz]
            current_x = translation[0]
            current_y = -(translation[1])

            draw_map_with_heading(current_x, current_y, magnetic_heading)

            # Her tespit kutusunun sağ üst köşesine derinlik değerini yazdırmak için:
            for box in coordinates:
                x1, y1, x2, y2 = map(int, box)  # tamsayıya çeviriyoruz
                # Sağ üst köşe koordinatları: (x2, y1)
                depth_val = depth.get_value((x2 + x1) / 2, (y1 + y2) / 2)[
                    1]  # Eğer depth değeri geçerliyse (NaN değilse) yazdır
                if not np.isnan(depth_val):
                    text = f"{depth_val:.2f} m"
                    # Yazıyı kutunun sağ üst köşesine ekleyelim; konum ayarını isteğinize göre değiştirebilirsiniz
                    cv2.putText(frame, text, (x2 - 60, y1 + 20), FONT, 0.7, COLOR_RED, 2)

            red_detected = False
            green_detected = False
            yellow_detected = False
            blue_detected = False
            black_detected = False
            cross_detected = False
            triangle_detected = False

            red_positions = []
            green_positions = []
            yellow_positions = []
            blue_positions = []
            black_positions = []
            cross_positions = []
            triangle_positions = []

            #while not (şekiller detected and şekil tespitinden 7 saniye geçmişse)
            for i, class_id in enumerate(class_ids):
                if class_id == 4:  # Kırmızı
                    red_detected = True
                    red_positions.append(coordinates[i])
                elif class_id == 6:  # Sarı
                    yellow_detected = True
                    yellow_positions.append(coordinates[i])
                elif class_id == 3:  # Yeşil
                    green_detected = True
                    green_positions.append(coordinates[i])
                elif class_id == 1:  # Mavi
                    blue_detected = True
                    blue_positions.append(coordinates[i])
                elif class_id == 0:  # Siyah
                    black_detected = True
                    black_positions.append(coordinates[i])
                elif class_id == 2:  # siyah artı
                    cross_detected = True
                    cross_positions.append(coordinates[i])
                elif class_id == 5:  # siyah üçgen
                    triangle_detected = True
                    triangle_positions.append(coordinates[i])

            # Tuş kontrolü: 'm' tuşu ile modlar arasında geçiş yapılır.
            key = cv2.waitKey(1) & 0xFF
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
                cv2.putText(frame, "MANUEL MOD", (50, 50), FONT, 1, (0, 255, 255), 2)  # todo: ekran orta noktasına al
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

                # --- AUTONOMOUS MISSIONS (ALL MUST BE PLACED IN THIS ELSE CONDITION) ---

                # Check if the special mission trigger condition is met
                if special_mission_triggered():
                    # Save the current mission state if we're not already in the special mission
                    if mission_state != MISSION_SPECIAL:
                        prev_mission_state = mission_state
                        mission_state = MISSION_SPECIAL

                if mission_state == MISSION_AVOID_BUOYS:
                    avoid_buoys(frame, depth, center_x, center_y,
                       green_detected, red_detected, yellow_detected,
                       blue_detected, black_detected,
                       green_positions, red_positions, yellow_positions)
                    cv2.putText(frame, "Avoiding buoys", (50, 350), FONT, 1, (255, 255, 0), 2)
                    # Transition condition example: if buoys have been successfully passed
                    # if buoys_passed_condition():
                    mission_state = MISSION_DOCKING

                elif mission_state == MISSION_DOCKING:
                    done = docking(frame, depth, current_x, current_y, magnetic_heading, center_x,
                                                  center_y, map_image, detections)
                    cv2.putText(frame, "Docking", (50, 350), FONT, 1, (255, 255, 0), 2)
                    if done:
                        mission_state = MISSION_SPEED_CHALLENGE

                elif mission_state == MISSION_SPEED_CHALLENGE:
                    # Placeholder for your second intermediate mission.
                    done = speed_challenge(frame, depth, current_x, current_y, magnetic_heading, center_x, center_y, map_image, detections)
                    cv2.putText(frame, "Speed challenge", (50, 350), FONT, 1, (255, 255, 0), 2)
                    if done:
                        mission_state = MISSION_RETURN_HOME

                elif mission_state == MISSION_RETURN_HOME:
                    navigate_to_start(frame, current_x, current_y, magnetic_heading, start_x, start_y)
                    cv2.putText(frame, "Going back to home", (50, 350), FONT, 1, (255, 255, 0), 2)

                elif mission_state == MISSION_SPECIAL:
                    done = special_mission(frame, depth, current_x, current_y, magnetic_heading, center_x, center_y, map_image, detections)
                    cv2.putText(frame, "Delivery Mission Active", (50, 400), FONT, 1, (255, 0, 255), 2)
                    if done:
                        # When done, revert back:
                        mission_state = prev_mission_state

            cv2.putText(frame, f"FPS: {int(zed.get_current_fps())}", (10, 30), FONT, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"{str(zed.get_spatial_mapping_state())}", (10, 60), FONT, 0.5, (20, 220, 20), 1)
            cv2.putText(frame, f"POSITIONAL_TRACKING_STATE.{str(zed.get_position(pose, sl.REFERENCE_FRAME.WORLD))}",
                        (10, 90), FONT, 0.5, (20, 220, 20), 1)
            cv2.putText(frame, f"Coordinates X,Y: {current_x:.1f} {current_y:.1f}", (10, 120), FONT, 0.75,
                        (0, 150, 240), 1, )

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
