# Note:
# The following code is heavily inspired from the code in the "Finding Lane
# Lines" lesson

from distortion import Distortion
from image_helper import rgb_image
from transform import TopDownTransform
from lane import Lane
from lanes import Lanes
import cv2
import numpy as np
import thresholding
import matplotlib.pyplot as plt

def detect_lane_lines(image, last_lanes=None):
    if(last_lanes is None):
        return full_detect_lane_lines(image )
    else:
        return quick_detect_lane_lines(image, last_lanes)

def quick_detect_lane_lines(image, last_lanes):
    nonzero = image.nonzero()
    nonzero_x, nonzero_y = np.array(nonzero[1]), np.array(nonzero[0])

    last_left_p = np.poly1d(last_lanes.left.pixels.fit)
    last_right_p = np.poly1d(last_lanes.right.pixels.fit)

    margin = 100

    left_lane_indices = ((nonzero_x > (last_left_p(nonzero_y) - margin)) &
                         (nonzero_x < (last_left_p(nonzero_y) + margin)))

    right_lane_indices = ((nonzero_x > (last_right_p(nonzero_y) - margin)) &
                          (nonzero_x < (last_right_p(nonzero_y) + margin)))

    # Again, extract left and right line pixel positions
    left_x = nonzero_x[left_lane_indices]
    left_y = nonzero_y[left_lane_indices]
    right_x = nonzero_x[right_lane_indices]
    right_y = nonzero_y[right_lane_indices]

    left = Lane(left_x, left_y)
    right = Lane(right_x, right_y)

    return Lanes(left, right), image

def full_detect_lane_lines(image):
    # Settings
    window_margin = 100          # This will be +/- on left and right sides of the window
    min_pixels_to_recenter = 50  # Minimum number of pixels before recentering the window
    num_windows = 9              # Number of sliding windows

    image_height, image_width = image.shape

    # Incoming image should already be  undistorted, transformed top-down, and
    # passed through thresholding. Takes histogram of lower half of the image.
    histogram = np.sum(image[image_height//2:,:], axis=0)

    # Placeholder for the image to be returned
    out_image = np.dstack((image, image, image))*255

    # Find peaks on left and right halves of the image
    midpoint = image_width//2
    base_left_x = np.argmax(histogram[:midpoint])
    base_right_x = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows based on num_windows
    window_height = image_height//num_windows

    # Get points of non-zero pixels in image
    nonzero = image.nonzero()
    nonzero_x, nonzero_y = np.array(nonzero[1]), np.array(nonzero[0])

    # Initialize current position, will be updated in each window
    current_left_x = base_left_x
    current_right_x = base_right_x

    # This is where the lane indices will be stored
    left_lane_indices = []
    right_lane_indices = []

    for window in range(num_windows):
        # Get the window boundaries
        window_y_low = image_height - (window + 1) * window_height
        window_y_high = image_height - window * window_height

        window_left_x_low = current_left_x - window_margin
        window_left_x_high = current_left_x + window_margin

        window_right_x_low = current_right_x - window_margin
        window_right_x_high = current_right_x + window_margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_image, (window_left_x_low, window_y_low), (window_left_x_high, window_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_image, (window_right_x_low, window_y_low), (window_right_x_high, window_y_high), (0, 255, 0), 2)

        # Identify the non-zero points within the window
        good_left_indices = ((nonzero_y >= window_y_low) &
                             (nonzero_y < window_y_high) &
                             (nonzero_x >= window_left_x_low) &
                             (nonzero_x < window_left_x_high)).nonzero()[0]

        good_right_indices = ((nonzero_y >= window_y_low) &
                              (nonzero_y < window_y_high) &
                              (nonzero_x >= window_right_x_low) &
                              (nonzero_x < window_right_x_high)).nonzero()[0]

        # Append the indices to the list
        left_lane_indices.append(good_left_indices)
        right_lane_indices.append(good_right_indices)

        if(len(good_left_indices) > min_pixels_to_recenter):
            current_left_x = np.int(np.mean(nonzero_x[good_left_indices]))

        if(len(good_right_indices) > min_pixels_to_recenter):
            current_right_x = np.int(np.mean(nonzero_x[good_right_indices]))

    # Concatenate indices so it becomes a flat array
    left_lane_indices = np.concatenate(left_lane_indices)
    right_lane_indices = np.concatenate(right_lane_indices)

    # Extract the land right lane pixels
    left_x, left_y = nonzero_x[left_lane_indices], nonzero_y[left_lane_indices]
    right_x, right_y = nonzero_x[right_lane_indices], nonzero_y[right_lane_indices]

    left = Lane(left_x, left_y)
    right = Lane(right_x, right_y)

    return Lanes(left, right), out_image

if __name__ == "__main__":
    transform = TopDownTransform()
    distortion = Distortion(calibration_data_filepath="./calibration.pkl")

    filepath = "images/test/straight_lines1.jpg"
    image = rgb_image(filepath)
    image = distortion.undistort(image)
    image = transform.transform_to_top_down(image)
    image = thresholding.pipeline(image)

    lanes, out_image = detect_lane_lines(image)

