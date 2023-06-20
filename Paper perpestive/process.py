import cv2
import numpy as np
import utlis

webCamFeed = False
pathImage = "Data/1.jpg"
cap = cv2.VideoCapture(0)

# utlis.initializeTrackbars()
count = 0


def Analyze(pathImage):
    if webCamFeed:
        success, img = cap.read()
    else:
        img = cv2.imread(pathImage)
    widthImg, heightImg, _ = img.shape
    img = cv2.resize(img, (heightImg // 3, widthImg // 3))
    # imgBlank = np.zeros((heightImg // 5, widthImg // 5, 3), np.uint8)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    thres = (64, 255)
    imgThreshold = cv2.Canny(imgBlur, thres[0], thres[1])
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    paper_contour = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            paper_contour = cnt

    # x, y, w, h = cv2.boundingRect(paper_contour)
    # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    rect = cv2.minAreaRect(paper_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    box = np.array(box)
    # x_start = min(box[:, 0])
    # x_end = max(box[:, 0])
    # y_start = min(box[:, 1])
    # y_end = max(box[:, 1])
    points = utlis.reorder_points(box)

    # list = ['A', 'B', 'C', 'D']
    #
    # idx = 0
    # for coor in points:
    #     cv2.putText(img, list[idx], (coor[0], coor[1]), cv2.FONT_HERSHEY_SIMPLEX
    #                 , 1, (0,255,255), 1, cv2.LINE_AA)
    #     idx += 1

    # cv2.drawContours(img, [points], -1, (0, 0, 255), 2)

    # warped = utlis.transform(img, points, rect)

    out = utlis.transform(img, points)
    out = cv2.flip(out, 1)
    if out.shape[0] < out.shape[1]:
        out = cv2.rotate(out, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        out = cv2.rotate(out, cv2.ROTATE_180)

    return out
