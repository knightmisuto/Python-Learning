import cv2
import numpy as np

img = cv2.imread("Data/face_1.jpg")
blurred_img = cv2.blur(img, (75,75))

imgHeight, imgWidth, _ = img.shape

mask = np.zeros((imgHeight, imgWidth, 3), dtype=np.uint8)
mask = cv2.circle(mask, (imgWidth//2, imgHeight//2), 200, (255,255,255), -1)

out = np.where(mask==np.array([255, 255, 255]), blurred_img, img)

cv2.imshow('mask', mask)
cv2.imshow("img", out)
cv2.waitKey(0)