import cv2
import numpy as np
from Useful_functions_opencv import stackImages

def empty(a):
    pass

path = 'E:/Samples/Sample_OW/Pic1.png'
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
    img = cv2.resize(img,(640,480))
    r_min = cv2.getTrackbarPos("R Min", "TrackBars")
    r_max = cv2.getTrackbarPos("R Max", "TrackBars")
    g_min = cv2.getTrackbarPos("G Min", "TrackBars")
    g_max = cv2.getTrackbarPos("G Max", "TrackBars")
    b_min = cv2.getTrackbarPos("B Min", "TrackBars")
    b_max = cv2.getTrackbarPos("B Max", "TrackBars")
    lower = np.array([b_min, g_min, r_min])
    upper = np.array([b_max, g_max, r_max])
    mask = cv2.inRange(img, lower, upper)
    imgResult = cv2.bitwise_and(img, img, mask=mask)
    imgStack = stackImages(1,([img,mask,imgResult]))
    cv2.imshow("Stacked Images", imgStack)
    if cv2.waitKey(1) == ord('q'):
        break