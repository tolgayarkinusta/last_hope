import pyzed.sl as sl
import cv2
import numpy as np
import math
import time

# Constants
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1
COLOR_RED = (0, 0, 255)
THICKNESS = 2
DEPTH_CENTER_COLOR = (255, 0, 0)
DEPTH_CENTER_RADIUS = 5

def initialize_camera():
    # ZED kamera nesnesi oluştur
    zed = sl.Camera()
    # ZED başlatma parametreleri ayarla
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # 720p çözünürlük
    init_params.camera_fps = 60  # 30 FPS
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL # depth mode best quality at neural_plus
    init_params.coordinate_units = sl.UNIT.METER #using metric system
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE # default for the opencv
    init_params.depth_minimum_distance = 0.20
    init_params.depth_maximum_distance = 40
    init_params.camera_disable_self_calib = True
    init_params.depth_stabilization = 80 #titreme azaltıcı
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


def main():
    zed = initialize_camera()

    # Used to store the sensors timestamp to know if the sensors_data is a new one or not
    ts_handler = TimestampHandler()

    # Görüntü ve derinlik verilerini almak için Mat nesneleri oluştur
    image = sl.Mat()
    depth = sl.Mat()
    # Sensör verisi al
    sensors_data = sl.SensorsData()

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
            depth_value = depth.get_value(center_x, center_y)[1]  # Sadece metre cinsinden değeri al

            # Derinlik bilgisini sağ üst köşede göster imu bilgisini sol üst köşede göster
            depth_info_text = f"Center Depth: {depth_value:.2f} m" if not np.isnan(depth_value) else "Couldn't Calculate..: NaN"
            render_text(frame, depth_info_text, (10, 50))
            cv2.circle(frame, (center_x, center_y), DEPTH_CENTER_RADIUS, DEPTH_CENTER_COLOR, -1)

            # Merkez noktasını görselleştir
            cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)

            # Calculate FPS
            fps_current_time = time.time()
            fps = 1 / (fps_current_time - fps_previous_time)
            fps_previous_time = fps_current_time
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Görüntüyü göster
            frame_resized = cv2.resize(frame, (960, 540))  # Resize the frame to desired dimensions
            cv2.imshow("ZED Camera", frame_resized)

            # 'q' tuşuna basıldığında çık
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Kaynakları serbest bırak ve kamerayı kapat
    cv2.destroyAllWindows()
    zed.close()

if __name__ == "__main__":
    main()
