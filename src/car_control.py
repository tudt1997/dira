#!/usr/bin/env python

import cv2
import math
import numpy as np
import rospkg
import rospy
# from model_keras import nvidia_model
from std_msgs.msg import Float32, String, Bool
from road_segmentation import RoadSegmentation
from param import pkg_name, node_name

path = rospkg.RosPack().get_path(pkg_name)
#import keras

import time
#import tensorflow as tf

#print(tf.__version__)
#print(keras.__version__)
#from keras.backend.tensorflow_backend import set_session

#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.2
#set_session(tf.Session(config=config))


class CarControl:
    def __init__(self, param):
        rospy.init_node(node_name, anonymous=True)
        self.speed_pub = rospy.Publisher("/set_speed_car_api", Float32, queue_size=1)
        self.steerAngle_pub = rospy.Publisher("/set_steer_car_api", Float32, queue_size=1)

        rospy.Rate(10)
        self.param = param

        self.stopping = True
        self.time = 0
        self.current_speed = 0

        # Load keras model
        #self.model = nvidia_model()
        #self.model.load_weights(path + '/param/semi-3_2-weights.11-0.00358.h5')
        #self.model._make_predict_function()

        self.h, self.w = 160, 320
        self.carPos = (160, 240)
        self.last_detected = 0
        self.sign_type = 0
        self.is_turning = False
        self.time_detected = 0

        self.fps = 0
        self.fps_timer = time.time()

        self.road_segmentation = RoadSegmentation(path)
        self.cover_left = np.array([[0, 0], [200, 0], [100, 240]])
        self.cover_right = np.array([[120, 0], [320, 0], [220, 240]])
        self.triangle_cnt = np.array([[0, 0]])

    def control(self, img, sign, is_go, danger_zone, slow_down):
        steer_angle = self.cal_steer_angle(img, sign, danger_zone)
        speed = 0

        if not rospy.is_shutdown():

            # if steer_angle > 25:
            #     steer_angle *= self.steer_angle_scale
            # print(steer_angle, isGo)

            if not is_go:
                self.stopping = True
                self.current_speed = 0
                # steer_angle = 0
            elif self.stopping:
                self.time = time.time()
                self.stopping = False
            elif time.time() - self.time > self.param.delay_time and self.current_speed == 0:
                self.current_speed = self.param.min_speed
            elif slow_down > 2:
                self.current_speed = self.param.min_speed
            elif slow_down > 0:
                self.current_speed = max(self.param.min_speed, self.current_speed - self.param.speed_slow_down)

            if self.current_speed >= self.param.min_speed:
                speed = max(self.param.min_speed,
                            self.current_speed - self.param.speed_decay * (self.param.base_speed - self.param.min_speed) * abs(steer_angle ** 2) / (self.param.max_steer_angle ** 2))

                if self.current_speed < self.param.base_speed and slow_down == 0:
                    self.current_speed += 0.4

            # print("slow down: ", slow_down)

            # print(self.current_speed)
            # print('Steer angle: {:.2f}'.format(steer_angle))

            self.speed_pub.publish(speed)
            if is_go:
                self.steerAngle_pub.publish(-steer_angle)
        return self.is_turning, steer_angle, speed

    def cal_steer_angle(self, img, sign, danger_zone):
        # fps counter
        self.fps += 1
        if time.time() - self.fps_timer > 1:
            print("fps ", self.fps)
            self.fps_timer = time.time()
            self.fps = 0
        steerAngle = 0

        if sign != 0:
            self.time_detected = time.time()
            # turn right
            if sign == 1:
                self.triangle_cnt = self.cover_left
            # turn left
            else:
                self.triangle_cnt = self.cover_right

        img_array = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_bv = self.bird_view(img_array)

        # cover in 1 sec
        # if time.time() - self.time_detected < 1:
        #    cv2.drawContours(img_bv, [self.triangle_cnt], 0, (0, 0, 0), -1)
        # timer = time.time()

        # middle_pos = float(self.model.predict(img_bv[None, :, :, :], batch_size=1)) * 160
        x = self.road_segmentation.get_points(cv2.resize(img_array, (320, 160)))

        
        if time.time() - self.time_detected < 1 and x[0] < 20 and x[1] > 300:
            middle_pos = x[sign]
        else:
            middle_pos = (x[1] + x[0]) / 2
			
        # print(x, middle_pos)

        #if abs(middle_pos) > 100:
        #    middle_pos = middle_pos * self.param.middle_pos_scale + 160
        #else:
        #    middle_pos = middle_pos + 160

        # if (sign == 1 and middle_pos < 160) or (sign == 2 and middle_pos > 160):
        #    middle_pos = 320 - middle_pos

        # img_bv_flipped = cv2.flip(img_bv, 1)
        # middle_pos_flipped = float(self.model.predict(img_bv_flipped[None, :, :, :], batch_size=1)) * 160 + 160

        # middle_pos = (middle_pos + 320 - middle_pos_flipped) / 2
        print("middle_pos",middle_pos)
        # avoid obstacles
        if danger_zone != (0, 0):

            # 2 objects
            if danger_zone[0] == -1:
                print("2 obstacles")
                middle_pos = danger_zone[1]
            # single object
            else:
                center_danger_zone = int((danger_zone[0] + danger_zone[1]) / 2)
                # print(danger_zone, center_danger_zone)
                if danger_zone[0] + 30 < middle_pos < danger_zone[1] - 30:
                    # obstacle's on the right
                    if middle_pos < center_danger_zone:
                        print("on the right")
                        middle_pos = danger_zone[0]
                    # left
                    else:
                        print("on the left")
                        middle_pos = danger_zone[1]
        # print("drive ",time.time()-timer)

        cv2.line(img, (int(middle_pos), self.h / 2), (self.w / 2, self.h), (255, 0, 0), 2)
        # cv2.imshow("Bird view", img_bv[:, :, :])

        # Distance between MiddlePos and CarPos
        distance_x = middle_pos - self.w / 2
        distance_y = self.h - self.h / 2

        # Angle to middle position
        steerAngle = math.atan(float(distance_x) / distance_y) * 180 / math.pi
        # cv2.waitKey(1)

        # QIK MATH
        # steerAngle = ((middle_pos - 160) / 160) * 60

        return steerAngle

    def unwarp(self, img, src, dst):
        h, w = img.shape[:2]
        M = cv2.getPerspectiveTransform(src, dst)

        unwarped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)
        return unwarped

    def bird_view(self, source_img):
        h, w = source_img.shape[:2]
        # define source and destination points for transform

        src = np.float32([(100, 120),
                          (220, 120),
                          (0, 210),
                          (320, 210)])

        dst = np.float32([(120, 0),
                          (w - 120, 0),
                          (120, h),
                          (w - 120, h)])

        # change perspective to bird's view
        unwarped = self.unwarp(source_img, src, dst)
        return unwarped
