import numpy as np

pkg_name = 'dira'
node_name = 'DiRa'

class Param:
    def __init__(self):
        self.base_speed = 10
        self.min_speed = 10

        self.speed_decay = 2
        self.max_steer_angle = 60.0

        self.steer_angle_scale = 1
        self.middle_pos_scale = 1

        self.sign_size_1 = 33
        self.sign_size_2 = 37

        self.speed_slow_down = 1

        self.delay_time = 1
