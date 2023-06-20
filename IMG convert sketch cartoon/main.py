import cv2
import numpy as np


def img_to_sketch(path):
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (25, 25), 0, 0)
    img_sketch = cv2.divide(img_gray, img_blur, scale=256)
    return img_sketch


def img_to_cartoon(path):
    img = cv2.imread(path)

    # covert to gray and using edgedetection
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_gray_img = cv2.GaussianBlur(gray_img, (5, 5), 1)
    edges = cv2.adaptiveThreshold(blur_gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 7, 5)
    # cartoon img
    color = cv2.bilateralFilter(img, 9, 250, 250)
    cartoon = cv2.bitwise_and(color, color, mask=edges)

    return cartoon