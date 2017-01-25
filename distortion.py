from calibrate import load_calibration_data
import cv2
import pickle

class Distortion:
    def __init__(self, calibration_data_filepath):
        self.load_calibration_data(calibration_data_filepath)

    def load_calibration_data(self, filepath):
        self.mtx, self.dist = load_calibration_data(filepath)

    def undistort(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
