import pyzed.sl as sl
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import math
import time

# Motor control constants
STOP_PWM = 1500  # PWM value to stop motor
REVERSE_PWM = 1450  # PWM value for reverse
MIN_PWM = 1540  # Minimum PWM for normal operation
STRAIGHT_PWM = 1560  # PWM for straight movement
MAX_PWM = 1570  # Maximum PWM value


class USVController:
    def __init__(self, port="COM10", baud=115200):
        from MainSystem import USVController as BaseController  # from MainSystem olacak
        self.controller = BaseController(port, baud)
        self.right_motor = 5  # Right motor pin
        self.left_motor = 6  # Left motor pin

        # State variables
        self.hazard_detection_time = None
        self.is_reversing = False
        self.last_detection_time = time.time()
        self.is_turning = False

    def set_servo(self, pin, pwm):
        """Control motor PWM values"""
        self.controller.set_servo(pin, pwm)

    def initialize(self):
        """Initialize the USV"""
        print("Arming vehicle...")
        self.controller.arm_vehicle()
        print("Vehicle armed!")
        print("Setting mode...")
        self.controller.set_mode("MANUAL")
        print("Mode set!")


def calculate_triangle_metrics(point1, point2, depth1, depth2, frame_width, horizontal_fov=90):
    """
    Calculate triangle metrics with USV as vertex and two points forming base
    Args:
        point1, point2: x-coordinates in pixels
        depth1, depth2: depth values in meters
        frame_width: width of frame
        horizontal_fov: camera's horizontal field of view in degrees
    Returns:
        vertex_angle: angle between the two points from USV perspective
        base_width: actual distance between the two points
    """
    # Calculate angles from center for both points
    center_x = frame_width / 2
    angle_per_pixel = horizontal_fov / frame_width

    angle1 = (point1 - center_x) * angle_per_pixel
    angle2 = (point2 - center_x) * angle_per_pixel

    # Convert to radians for calculations
    angle1_rad = math.radians(angle1)
    angle2_rad = math.radians(angle2)

    # Calculate vertex angle (angle between the two points from USV perspective)
    vertex_angle = abs(angle1 - angle2)

    # Calculate base width using law of cosines
    # c² = a² + b² - 2ab*cos(C)
    base_width = math.sqrt(depth1 ** 2 + depth2 ** 2 -
                           2 * depth1 * depth2 * math.cos(math.radians(vertex_angle)))

    return vertex_angle, base_width


def calculate_insole_length(vertex_angle, base_width, depth):
    """
    Calculate the height (insole) of the triangle from vertex to base
    Args:
        vertex_angle: angle at vertex in degrees
        base_width: width of the triangle base in meters
        depth: depth to the base in meters
    Returns:
        insole_length: height of the triangle in meters
    """
    vertex_angle_rad = math.radians(vertex_angle)
    # Using the sine formula: height = base * sin(vertex_angle/2)
    insole_length = base_width * math.sin(vertex_angle_rad / 2)
    return insole_length


def get_depth_at_point(point_2d, depth_map):
    """Get depth value at given point"""
    x, y = int(point_2d[0]), int(point_2d[1])
    if x >= 0 and x < depth_map.shape[1] and y >= 0 and y < depth_map.shape[0]:
        return depth_map[y, x]
    return None


def find_optimal_path(detections, depth_map, frame_width):
    """
    Find optimal path using triangle geometry with USV as vertex
    Args:
        detections: List of detected buoys with bounding boxes and class IDs
        depth_map: Depth information from ZED camera
        frame_width: Width of the camera frame
    Returns:
        target_x: Target x-coordinate for navigation
        target_depth: Target depth for navigation
    """
    # Find all buoys with their positions and depths
    red_buoys = []  # Left side
    green_buoys = []  # Right side
    hazard_buoys = []  # Yellow and black buoys

    for detection in detections:
        x1, y1, x2, y2 = detection.bbox.xyxy[0]
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        depth = get_depth_at_point((center_x, center_y), depth_map)

        if depth is not None:
            if detection.class_id == 0:  # Red buoy
                red_buoys.append((center_x, depth))
            elif detection.class_id == 1:  # Green buoy
                green_buoys.append((center_x, depth))
            elif detection.class_id in [2, 3]:  # Hazard buoys (yellow=2 or black=3)
                hazard_buoys.append((center_x, depth, detection.class_id))

    # Handle single buoy detection cases
    if not red_buoys and not green_buoys:
        return None, None  # No buoys detected
    elif not red_buoys:  # Only green buoy detected
        closest_green = min(green_buoys, key=lambda x: x[1])
        return closest_green[0], closest_green[1]  # Turn left
    elif not green_buoys:  # Only red buoy detected
        closest_red = min(red_buoys, key=lambda x: x[1])
        return closest_red[0], closest_red[1]  # Turn right

    # Find closest red and green buoys
    closest_red = min(red_buoys, key=lambda x: x[1])
    closest_green = min(green_buoys, key=lambda x: x[1])

    # Calculate triangle metrics between red and green buoys
    gate_angle, gate_width = calculate_triangle_metrics(
        closest_red[0], closest_green[0],
        closest_red[1], closest_green[1],
        frame_width
    )

    # Find hazards between the buoys
    path_hazards = []
    for hx, hdepth, _ in hazard_buoys:
        # Calculate angles to both gate buoys
        h_red_angle, h_red_dist = calculate_triangle_metrics(
            hx, closest_red[0], hdepth, closest_red[1], frame_width)
        h_green_angle, h_green_dist = calculate_triangle_metrics(
            hx, closest_green[0], hdepth, closest_green[1], frame_width)

        # If hazard is between the buoys
        if h_red_angle + h_green_angle <= gate_angle:
            path_hazards.append((hx, hdepth, h_red_angle / gate_angle))

    # Calculate target point
    if path_hazards:
        # Sort hazards by depth
        path_hazards.sort(key=lambda x: x[1])
        closest_hazard = path_hazards[0]

        # Position ratio (0 = near red buoy, 1 = near green buoy)
        hazard_ratio = closest_hazard[2]

        if hazard_ratio < 0.5:  # Hazard is closer to red buoy
            # Aim for 70% of the way to the green buoy
            target_x = closest_red[0] + (closest_green[0] - closest_red[0]) * 0.7
            target_depth = closest_hazard[1]
        else:  # Hazard is closer to green buoy
            # Aim for 30% of the way from red buoy
            target_x = closest_red[0] + (closest_green[0] - closest_red[0]) * 0.3
            target_depth = closest_hazard[1]
    else:
        # No hazards, aim for center
        target_x = (closest_red[0] + closest_green[0]) / 2
        target_depth = (closest_red[1] + closest_green[1]) / 2

    return target_x, target_depth


def main():
    # Initialize ZED camera with recommended settings
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30
    init_params.depth_mode = sl.DEPTH_MODE.QUALITY
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
    init_params.depth_minimum_distance = 0.20
    init_params.depth_maximum_distance = 40
    init_params.camera_disable_self_calib = True
    init_params.depth_stabilization = 80
    init_params.sensors_required = False
    init_params.enable_image_enhancement = True

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED camera")
        return

    # Initialize YOLO model
    model = YOLO("balonLarge54.pt")

    # Initialize USV controller
    controller = USVController()
    controller.initialize()

    runtime_params = sl.RuntimeParameters()
    image = sl.Mat()
    depth = sl.Mat()

    while True:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # Get camera data
            zed.retrieve_image(image, sl.VIEW.LEFT)
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            frame = image.get_data()
            # Convert RGBA to RGB
            if frame.shape[2] == 4:  # If image has 4 channels (RGBA)
                frame = frame[:, :, :3]  # Keep only RGB channels
            depth_map = depth.get_data()
            frame_width = frame.shape[1]

            # Run detection
            results = model(frame)[0]
            detections = sv.Detections.from_ultralytics(results)

            current_time = time.time()

            # Handle no detection timeout
            if len(detections) == 0:
                if current_time - controller.last_detection_time > 2.5:
                    if not controller.is_turning:
                        # Reverse for 2 seconds
                        controller.set_servo(controller.right_motor, REVERSE_PWM)
                        controller.set_servo(controller.left_motor, REVERSE_PWM)
                        time.sleep(2.0)
                        controller.is_turning = True
                    else:
                        # Turn until proper orientation
                        controller.set_servo(controller.right_motor, MIN_PWM)
                        controller.set_servo(controller.left_motor, REVERSE_PWM)
                    continue
            else:
                controller.last_detection_time = current_time

            # Find optimal path
            target_x, target_depth = find_optimal_path(detections, depth_map, frame_width)

            # Extract buoy positions for navigation logic
            red_buoys = []
            green_buoys = []
            hazard_buoys = []
            for detection in detections:
                x1, y1, x2, y2 = detection.bbox.xyxy[0]
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                depth = get_depth_at_point((center_x, center_y), depth_map)

                if depth is not None:
                    if detection.class_id == 0:  # Red buoy
                        red_buoys.append((center_x, depth))
                    elif detection.class_id == 1:  # Green buoy
                        green_buoys.append((center_x, depth))
                    elif detection.class_id in [2, 3]:  # Hazard buoys
                        hazard_buoys.append((center_x, depth))

            # Handle hazard detection (yellow or black buoys)
            if hazard_buoys:
                if not controller.is_reversing:
                    controller.hazard_detection_time = current_time
                    controller.is_reversing = True

                # Always reverse when seeing hazard buoys
                controller.set_servo(controller.right_motor, REVERSE_PWM)  # 1450 PWM
                controller.set_servo(controller.left_motor, REVERSE_PWM)  # 1450 PWM

                # If we see navigation buoys, continue reversing for 2 more seconds
                if len(red_buoys) > 0 or len(green_buoys) > 0:
                    if current_time - controller.hazard_detection_time < 2.0:
                        continue
                    else:
                        controller.is_reversing = False
                        controller.hazard_detection_time = None
                continue

            # Handle orientation verification during turn
            if controller.is_turning:
                if red_buoys and green_buoys:
                    closest_red = min(red_buoys, key=lambda x: x[1])
                    closest_green = min(green_buoys, key=lambda x: x[1])

                    if closest_green[0] > closest_red[0]:  # Correct orientation
                        controller.is_turning = False
                    else:  # Continue turning
                        controller.set_servo(controller.right_motor, MIN_PWM)
                        controller.set_servo(controller.left_motor, REVERSE_PWM)
                        continue

            # Normal navigation
            if red_buoys and green_buoys:
                closest_red = min(red_buoys, key=lambda x: x[1])
                closest_green = min(green_buoys, key=lambda x: x[1])
                target_x = (closest_red[0] + closest_green[0]) / 2

                # Calculate error for proportional control
                error = target_x - frame_width / 2
                error_ratio = abs(error) / (frame_width / 4)

                if error > 0:  # Turn right
                    # Stop right motor, gradually increase left up to 1570
                    controller.set_servo(controller.right_motor, STOP_PWM)  # 1500 PWM
                    left_pwm = STRAIGHT_PWM + int((MAX_PWM - STRAIGHT_PWM) * min(error_ratio, 1.0))
                    controller.set_servo(controller.left_motor, left_pwm)  # 1560-1570 PWM
                elif error < 0:  # Turn left
                    # Stop left motor, gradually increase right up to 1570
                    right_pwm = STRAIGHT_PWM + int((MAX_PWM - STRAIGHT_PWM) * min(error_ratio, 1.0))
                    controller.set_servo(controller.right_motor, right_pwm)  # 1560-1570 PWM
                    controller.set_servo(controller.left_motor, STOP_PWM)  # 1500 PWM
                else:  # Go straight at 1560 PWM
                    controller.set_servo(controller.right_motor, STRAIGHT_PWM)  # 1560 PWM
                    controller.set_servo(controller.left_motor, STRAIGHT_PWM)  # 1560 PWM
            elif green_buoys:  # Only green buoy - turn left slowly until another buoy is seen
                controller.set_servo(controller.right_motor, STRAIGHT_PWM)  # 1560 PWM
                controller.set_servo(controller.left_motor, STOP_PWM)  # 1500 PWM
                continue  # Keep turning until another buoy is detected
            elif red_buoys:  # Only red buoy - turn right slowly until another buoy is seen
                controller.set_servo(controller.right_motor, STOP_PWM)  # 1500 PWM
                controller.set_servo(controller.left_motor, STRAIGHT_PWM)  # 1560 PWM
                continue  # Keep turning until another buoy is detected
            else:  # No valid navigation buoys
                controller.set_servo(controller.right_motor, STOP_PWM)
                controller.set_servo(controller.left_motor, STOP_PWM)

            # Display frame
            cv2.imshow("Navigation", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Cleanup
    controller.set_servo(controller.right_motor, STOP_PWM)
    controller.set_servo(controller.left_motor, STOP_PWM)
    zed.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()