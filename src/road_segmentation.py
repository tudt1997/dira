# import rospkg
from floodfill import fill
from p2c import p2c_main
from model import Model
import numpy as np
import cv2

class RoadSegmentation():
    def __init__(self, path, roi=0.5):
        self.model = Model(path)
        self.roi = roi

    def get_points(self, img):
        segmented_img, _ = self.model.predict(img)
	floodfilled_img = fill(np.uint8(segmented_img))
        #cv2.imshow('segmented image', floodfilled_img * 255.)
        #cv2.waitKey(1)
        floodfilled_img = floodfilled_img.astype(np.int32)
        x_left, x_right = p2c_main.get_center_points_by_roi(floodfilled_img, self.roi)
        return x_left, x_right
