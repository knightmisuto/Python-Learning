import cv2
import numpy as np
import math


def nothing():
    pass

cv2.namedWindow("Value")
cv2.createTrackbar("ValueThresh", "Value", 0, 255, nothing)
cv2.createTrackbar("ValueKernel_X", "Value", 0, 300, nothing)
cv2.createTrackbar("ValueKernel_Y", "Value", 0, 300, nothing)
cv2.createTrackbar("Value_w,h", "Value", 0, 300, nothing)

cap = cv2.VideoCapture("Videotest/Cars_passing_test_3.mp4")

_, frame2 = cap.read()
coordinate_prev = []

width = 1024
height = 720

while True:
    _, frame1 = cap.read()
    if frame1 is None:
        break

    frame1 = cv2.resize(frame1, (width, height))
    frame2 = cv2.resize(frame2, (width, height))
    coordinate_new = []
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    frame1_blur = cv2.medianBlur(frame1_gray, 5)
    frame2_blur = cv2.medianBlur(frame2_gray, 5)
    delta = cv2.absdiff(frame1_blur, frame2_blur)
    x = cv2.getTrackbarPos("ValueThresh", "Value")
    y = cv2.getTrackbarPos("ValueKernel_X", "Value")
    w = cv2.getTrackbarPos("ValueKernel_Y", "Value")
    z = cv2.getTrackbarPos("Value_w,h", "Value")
    thresh = cv2.threshold(delta, x, 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((y, w), np.uint8)
    dilation = cv2.dilate(thresh, kernel)
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if w > z and h > z:
            coordinate_new.append((x, y))
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 1)

    coordinate_new = np.array(coordinate_new)
    print(coordinate_new.shape)
    cv2.imshow("dilation", dilation)
    cv2.imshow("frame", frame1)
    frame2 = frame1
    coordinate_prev = coordinate_new

    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()