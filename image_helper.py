import cv2
import matplotlib.image as mpimg

def rgb_image(filepath):
    """Load an image as RGB"""
    return mpimg.imread(filepath)

def rgb_to_gray(img):
    """Convert an RGB image to Grayscale"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def rgb_to_hls(img):
    """Convert an RGB image to HLS"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
