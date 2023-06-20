import cv2
import numpy as np
from Useful_functions_opencv import stackImages
import os

def empty(a):
    pass

img_list = os.listdir("Data/Data")

path = 'Data/Data/8k_reverse_2.png'

cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640, 240)
cv2.createTrackbar("R Min", "TrackBars", 0, 255, empty)
cv2.createTrackbar("R Max", "TrackBars", 255, 255, empty)
cv2.createTrackbar("G Min", "TrackBars", 0, 255, empty)
cv2.createTrackbar("G Max", "TrackBars", 255, 255, empty)
cv2.createTrackbar("B Min", "TrackBars", 0, 255, empty)
cv2.createTrackbar("B Max", "TrackBars", 255, 255, empty)

while True:
    img = cv2.imread(path)
    r_min = cv2.getTrackbarPos("R Min", "TrackBars")
    r_max = cv2.getTrackbarPos("R Max", "TrackBars")
    g_min = cv2.getTrackbarPos("G Min", "TrackBars")
    g_max = cv2.getTrackbarPos("G Max", "TrackBars")
    b_min = cv2.getTrackbarPos("B Min", "TrackBars")
    b_max = cv2.getTrackbarPos("B Max", "TrackBars")
    lower = np.array([r_min, g_min, b_min])
    upper = np.array([r_max, g_max, b_max])
    mask = cv2.inRange(img, lower, upper)
    imgResult = cv2.bitwise_and(img, img, mask=mask)
    imgStack = stackImages(1,([img,mask,imgResult]))
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print("Number of Contours found = " + str(len(contours)))
    for cnt in contours:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        x = int(x)
        y = int(y)
        cv2.rectangle(img, (x-15, y-15), (x+15,y+15), (0,255,0), 1)
    cv2.imshow("Stacked Images", imgStack)
    cv2.imshow("img", img)
    if cv2.waitKey(1) == ord('q'):
        break

# num = 0
#
# img = cv2.imread(path)
# r_min = 200
# r_max = 255
# g_min = 215
# g_max = 255
# b_min = 150
# b_max = 255
# lower = np.array([r_min, g_min, b_min])
# upper = np.array([r_max, g_max, b_max])
# mask = cv2.inRange(img, lower, upper)
# imgResult = cv2.bitwise_and(img, img, mask=mask)
# imgStack = stackImages(1,([img,mask,imgResult]))
# contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# print("Number of Contours found = " + str(len(contours)))
# # for cnt in contours:
# #     (x, y), radius = cv2.minEnclosingCircle(cnt)
# #     x = int(x)
# #     y = int(y)
# #     crop = img[int(y - 30 / 2):int(y + 30 / 2), int(x - 30 / 2):int(x + 30 / 2)]
#
# cv2.imshow("img", imgStack)
# cv2.waitKey(0)