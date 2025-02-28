from pymavlink import mavutil
import math
import cv2
import numpy
import math
import time
import threading

def map_value_to_range(value, in_min=-1, in_max=1, out_min=-2000, out_max=2000):
    return int((value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)


class USVController:
    def __init__(self, connection_string="COM10", baud=57600):
        # connecting to the vehicle
        print("trying to connect")
        self.master = mavutil.mavlink_connection(connection_string, baud=baud)
        print("Connected to MAVLINK")
        print("Heartbeat waiting...")
        self.master.wait_heartbeat()
        print("Heartbeat found!")
        self.first_position = None  # add for gps
        self.current_position = None  # add for gps

    def arm_vehicle(self):
        # arming vehicle
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1, 0, 0, 0, 0, 0, 0
        )
        print("Waiting for the vehicle to arm...")
        self.master.motors_armed_wait()  # Waiting for arming the vehicle
        print('Armed:!!!!!!!!!!!!!!!!!!!!!!!!!')

    def disarm_vehicle(self):
        # Disarming the vehicle
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            0, 0, 0, 0, 0, 0, 0
        )
        print("Waiting for the vehicle to disarm")
        self.master.motors_disarmed_wait()  # Waiting for disarming the vehicle
        print('Disarmed!')

    def print_motor_outputs(self):
        print(self.master.recv_match(type='SERVO_OUTPUT_RAW', blocking=True))

    def go_to_vehicle_raw(self, x=0, y=0, z=500, r=0, buttons=0, s=0, t=0, see_motor_output=0):
        # Manual controls
        self.master.mav.manual_control_send(
            self.master.target_system,
            x,  # x is straight 2000 is full front -2000 full back
            y,  # y is sideways 2000 is full right -2000 full left
            z,  # z is up 1500 is full up -500 full down (pwm is 1100 up 1900 down all z axis motors)
            r,
            buttons,
            s,
            t
        )
        if see_motor_output:
            self.print_motor_outputs()

    def to_straight_pose(self):  # todo: not working well (maybe pid and deadbands)
        # straightening the vehicle all axis
        msg = self.master.recv_match(type='ATTITUDE', blocking=True)
        if msg:
            pitch = msg.pitch  # Pitch angle (radian)
            yaw = msg.yaw
            roll = msg.roll

            # radians to degrees
            yaw_degrees = yaw * (180.0 / math.pi)
            roll_degrees = roll * (180.0 / math.pi)
            pitch_degrees = pitch * (180.0 / math.pi)

            # Command for straighten the vehicle
            self.go_to_vehicle_raw(
                0,
                0,
                0,
                self.up_or_down_r(yaw_degrees),
                0,
                self.up_or_down_x_and_y(pitch_degrees),
                self.up_or_down_x_and_y(roll_degrees)
            )

    def set_servo(self, servo_pin, pwm_value, see_motor_output=0):
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
            0,
            servo_pin,
            pwm_value,
            0, 0, 0, 0, 0
        )
        if see_motor_output:
            self.print_motor_outputs()


    def set_mode(self, mode_name):
        # (example: 'STABILIZE', 'MANUAL', 'DEPTH_HOLD')

        mode_id = self.master.mode_mapping()[mode_name]

        self.master.mav.set_mode_send(
            self.master.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id
        )
        print("Aracın modu " + mode_name + " olarak değiştirildi.")

    @staticmethod
    def up_or_down_x_and_y(a):
        # control values for pitch and roll
        return -500 if a < 0 else 500

    @staticmethod
    def up_or_down_r(a):
        # control values for yaw
        return -500 if a < 0 else 500

    def go_full_straight(self, see_motor_output=0):
        self.go_to_vehicle_raw(2000, 0, 0, 0, 0, 0, 0, see_motor_output)

    def go_full_back(self, see_motor_output=0):
        self.go_to_vehicle_raw(-2000, 0, 0, 0, 0, 0, 0, see_motor_output)

    def go_full_right(self, see_motor_output=0):
        self.go_to_vehicle_raw(0, 2000, 0, 0, 0, 0, 0, see_motor_output)

    def go_full_left(self, see_motor_output=0):
        self.go_to_vehicle_raw(0, -2000, 0, 0, 0, 0, 0, see_motor_output)

    def go_full_up(self, see_motor_output=0):
        self.go_to_vehicle_raw(0, 0, 1500, 0, 0, 0, 0, see_motor_output)

    def go_full_down(self, see_motor_output=0):
        self.go_to_vehicle_raw(0, 0, -1500, 0, 0, 0, 0, see_motor_output)

    def go_to_vehicle(self, x=0, y=0, z=0, r=0, buttons=0, s=0, t=0, see_motor_output=0):
        x_mapped = map_value_to_range(x)
        y_mapped = map_value_to_range(y)
        z_mapped = map_value_to_range(z, -1, 1, -500, 1500)
        r_mapped = map_value_to_range(r)
        self.go_to_vehicle_raw(x_mapped, y_mapped, z_mapped, r_mapped, buttons, s, t, see_motor_output)

    def test_motors(self):  # motorları 3 saniyeliğine test eder ve durdurur.
        time.sleep(0.1)
        self.set_servo(1, 1750, )
        self.set_servo(3, 1750, )
        time.sleep(3)
        self.stop_motors()

    def stop_motors(self):  # motorları durdur.

        self.set_servo(5, 1500, 1)
        self.set_servo(6, 1500, 1)


    def turn_port(self):  # teknenin pruvasını sancağa x derece çevir.
        time.sleep(0.1)
        self.set_servo(1, 1200, 1)
        self.set_servo(3, 1800, 1)
        time.sleep(0.1)
        self.stop_motors()

    def turn_starboard(self):  # teknenin pruvasını iskeleye x derece çevir.
        time.sleep(0.1)
        self.set_servo(1, 1800, 1)
        self.set_servo(3, 1200, 1)
        time.sleep(0.1)
        self.stop_motors()

    def go_forward_xs(self):  # 1 saniyeliğine ve %20 ileri hızda her iki motoru çalıştır.
        time.sleep(0.1)
        self.set_servo(1, 1600, 1)
        self.set_servo(3, 1600, 1)
        time.sleep(1)
        self.stop_motors()

    def go_forward_30(self):  # 1 saniyeliğine ve %30 ileri hızda her iki motoru çalıştır.
        time.sleep(0.1)
        self.set_servo(1, 1650, 1)
        self.set_servo(3, 1650, 1)
        time.sleep(1)
        self.stop_motors()

    def go_backward_20(self):  # 1 saniyeliğine ve %20 geri hızda her iki motoru çalıştır.
        time.sleep(0.1)
        self.set_servo(1, 1400, 1)
        self.set_servo(3, 1400, 1)
        time.sleep(1)
        self.stop_motors()

    def print_gps_info(self): #mevut gps mevki görüntüle
        # GPS_GLOBAL_INT mesajını al
        print("getting gps info")
        msg = self.master.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
        print("GPS info received")
        if msg:
            lat = msg.lat / 1e7  # Enlem (derece cinsinden)
            lon = msg.lon / 1e7  # Boylam (derece cinsinden)
            alt = msg.alt / 1e3  # Yükseklik (metre cinsinden)


            # GPS bilgilerini yazdır
            print(f"Enlem: {lat:.6f}°, Boylam: {lon:.6f}°, Yükseklik: {alt:.2f} metre")

    def get_current_position(self): #mevcut gps mevkii al
        #anlık GPS konumu alınır
        msg = self.master.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
        if msg:
            enlem = msg.lat / 1e7
            boylam = msg.lon / 1e7
            return enlem, boylam
        return None, None

    def save_initial_position(self): #mevcut(başlangıç) gps mevkii kaydet
        #başlangıç noktasını kaydet
        self.first_position = self.get_current_position()
        if self.first_position:
            print(f"Başlangıç noktası kaydedildi: Enlem: {self.first_position[0]}, Boylam: {self.first_position[1]}")
        else:
            print("GPS verileri alınamıyor.")

    def update_current_position(self): #mevcut(bitiş) gps mevkii kaydet
        #anlık GPS konumu güncelleme
        self.current_position = self.get_current_position()
        if self.current_position:
            print(f"Güncel konum: Enlem: {self.current_position[0]}, Boylam: {self.current_position[1]}")
        else:
            print("GPS verileri alınamıyor.")

    def haversine(self, lat1, lon1, lat2, lon2): #todo: add to git if success
        R = 6371000  # Dünya'nın yarıçapı (metre cinsinden)
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def check_distance(self, threshold):    #todo: add to git if success
        if self.first_position:
            self.update_current_position()
            if self.current_position:
                distance = self.haversine(
                    self.first_position[0], self.first_position[1],
                    self.current_position[0], self.current_position[1]
                )
                print(f"Mesafe: {distance:.2f} metre")
                if distance >= threshold:
                    print("STOP")
                    self.set_servo(1,1500)
                    self.set_servo(3,1500)
                    return True
        return False

    def calculate_heading(self): #başlangıç ve bitiş gps mevkiileri arasındaki headingi hesapla
        #ilk ve son GPS konumları arasındaki heading açısını hesapla
        if not self.first_position or not self.current_position:
            print("Başlangıç veya Bitiş konumu eksik, heading hesaplanamıyor.")
            return None

        enlem1, boylam1 = self.first_position
        enlem2, boylam2 = self.current_position

        #enlem ve boylamları radyan cinsine çevirme
        enlem1_radyan = math.radians(enlem1)
        enlem2_radyan = math.radians(enlem2)
        delta_boylam = math.radians(boylam2 - boylam1)

        #heading hesaplama
        x = math.sin(delta_boylam) * math.cos(enlem2_radyan)
        y = math.cos(enlem1_radyan) * math.sin(enlem2_radyan) - math.sin(enlem1_radyan) * math.cos(enlem2_radyan) * math.cos(delta_boylam)
        initial_kerteriz = math.atan2(x, y)

        #açıyı dereceye çevirme
        initial_kerteriz = math.degrees(initial_kerteriz)
        pusula_kerteriz = (initial_kerteriz + 360) % 360
        return pusula_kerteriz

    def give_desired_heading(self): #hesaplanan mevkiyi 'kerteriz' olarak kaydet ve terminale yazdır
        kerteriz = self.calculate_heading()
        print(f"Evin kerterizi: {kerteriz}")

    def get_compass_heading(self): #todo Pusula açısı alınacak
        msg = self.master.recv_match(type='ATTITUDE', blocking=True)
        if msg:
            # yaw (radyan cinsinden) al
            yaw = msg.yaw

            # Yaw'ı dereceye çevir
            yaw_degrees = yaw * (180.0 / math.pi)

            # Dereceyi 0 ile 360 arasında normalize et
            heading_degrees = (yaw_degrees + 360) % 360

            return heading_degrees


'''
    def set_servo(self, servo_pin, pwm_value, see_motor_output=0):
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
            0,
            servo_pin,
            pwm_value,
            0, 0, 0, 0, 0
        )
        if see_motor_output:
            self.print_motor_outputs()
'''










