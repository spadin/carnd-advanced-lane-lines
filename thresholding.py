from image_helper import rgb_to_gray, rgb_to_hls
import cv2
import numpy as np

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    thresh_min, thresh_max = thresh

    # Convert to grayscale
    gray = rgb_to_gray(img)

    # Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    elif orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

    # Take the absolute value of the derivative or gradient
    sobel = np.absolute(sobel)

    # Scale to 8-bit (0 - 255) then convert to type = np.uint8
    sobel = np.uint8(255 * sobel / np.max(sobel))

    # Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(sobel)
    binary_output[(sobel >= thresh_min) & (sobel <= thresh_max)] = 1

    # Return this mask as your binary_output image
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = rgb_to_gray(img)

    # Take the gradient in x and y separately
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Calculate the magnitude
    sobel_xy = np.sqrt(sobel_x**2 + sobel_y**2)

    # Scale to 8-bit (0 - 255) and convert to type = np.uint8
    sobel_xy = np.uint8(255 * sobel_xy / np.max(sobel_xy))

    # Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(sobel_xy)
    binary_output[(sobel_xy >= mag_thresh[0]) & (sobel_xy <= mag_thresh[1])] = 1

    # Return this mask as your binary_output image
    return binary_output

def dir_thresh(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Convert to grayscale
    gray = rgb_to_gray(img)

    # Take the gradient in x and y separately
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Take the absolute value of the x and y gradients
    sobel_x = np.absolute(sobel_x)
    sobel_y = np.absolute(sobel_y)

    # Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    direction = np.arctan2(sobel_y, sobel_x)

    # Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(direction)
    binary_output[(direction >= thresh[0]) & (direction <= thresh[1])] = 1

    # Return this mask as your binary_output image
    return binary_output

def saturation_thresh(img, thresh=(0, 255)):
    # Convert to HLS colorspace
    hls = rgb_to_hls(img)

    # Use only the saturation channel
    S = hls[:,:,2]

    # Create a binary mask where saturation thresholds are met
    binary_output = np.zeros_like(S)
    binary_output[((S > thresh[0]) & (S < thresh[1]))] = 1

    return binary_output

def red_thresh(img, thresh=(0, 255)):
    # Use only the red color channel
    R = img[:,:,0]

    # Create a binary mask where the red thresholds are met
    binary_output = np.zeros_like(R)
    binary_output[((R > thresh[0]) & (R < thresh[1]))] = 1

    return binary_output

def pipeline(img):
    # Choose a Sobel kernel size
    ksize = 17 # Choose a larger odd number to smooth gradient measurements

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(30, 100))
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(75, 150))
    mag_binary = mag_thresh(img, sobel_kernel=ksize, mag_thresh=(80, 200))
    dir_binary = dir_thresh(img, sobel_kernel=ksize, thresh=(0.7, 1.3))
    yellows = saturation_thresh(img, thresh=(180, 255))
    whites = red_thresh(img, thresh=(215, 255))

    combined = np.zeros_like(dir_binary)
    combined[
        ((whites == 1) & (yellows == 1)) |
        ((gradx == 1) & (grady == 1)) |
        ((mag_binary == 1) & (dir_binary == 1))] = 1

    return combined
