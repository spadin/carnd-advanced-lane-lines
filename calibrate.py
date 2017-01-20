from image_helper import rgb_image, rgb_to_gray
import cv2
import glob
import numpy as np
import pickle

images_glob = glob.glob("images/calibration/calibration*.jpg")
image_shape = rgb_image(images_glob[0]).shape[1::-1]
chessboard_shape = (9, 6)
output_file = "./calibration.pkl"

objpoints = []
imgpoints = []

for filepath in images_glob:
    img = rgb_image(filepath)
    gray = rgb_to_gray(img)
    pattern_was_found, corners = cv2.findChessboardCorners(gray, chessboard_shape, None)
    cols, rows = chessboard_shape
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:,:2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

    if pattern_was_found:
        objpoints.append(objp)
        imgpoints.append(corners)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_shape, None, None)

with open(output_file, "wb") as f:
    data = {}
    data["ret"] = ret
    data["mtx"] = mtx
    data["dist"] = dist
    data["rvecs"] = rvecs
    data["tvecs"] = tvecs
    pickle.dump(data, f)
