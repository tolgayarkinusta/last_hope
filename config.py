# config.py
MOTOR_PWM = {
    "base_speed": 1540, #düz ileri

    "left_forward": 1670,
    "right_forward": 1670,
    "left_turn": 1400,
    "right_turn": 1600,

    "neutral": 1500, #motorlar stop

    "backwards": 1370, #düz geri

    "base+": 1555,
    "base++": 1560,
    "base+++": 1570,

    "base-": 1530,

}
CONTROL_PARAMS = {
    "k": 1, #açı çarpanı

    "usv_width": 0.45,

    "close_depth": 0.40,
    "medium_depth": 1.5, #0.85
    "far_depth": 5.00,

    "threshold": 3.5, #3.2

    "max_depth": 2.01,
    "depth_threshold": 2.71,
    "width_threshold": 0.80
}
