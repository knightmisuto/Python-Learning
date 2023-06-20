import cv2
import requests
import mediapipe as mp
import numpy as np
import time
import autopy
import math

pTime = 0
cTime = 0
video = cv2.VideoCapture(0)
#URL from ip webcam + /video
url = "http://192.168.0.101:4747/video"
video.open(url)

text_file = open("Data/fingers.txt")
fingers_list = text_file.read().split("\n")

def fingerUp(idx_finger, fingers_list):
    if fingers_list[idx_finger][1] > fingers_list[idx_finger+1][1]:
        return True
    else:
        return False

def findDistance(x1, y1, x2, y2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while True:
        _, frame = video.read()
        if not _:
            continue

        imageHeight, imageWidth, _ = frame.shape
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)
        frame.flags.writeable = False
        results = hands.process(frame)

        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks_list = []
                # mp_drawing.draw_landmarks(
                #     frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                for point in mp_hands.HandLandmark:
                    if str(point) in fingers_list:
                        normalizedLandmark = hand_landmarks.landmark[point]
                        x = normalizedLandmark.x
                        y = normalizedLandmark.y
                        landmarks_list.append([x, y])
                index_finger_up = fingerUp(0, landmarks_list)
                middle_finger_up = fingerUp(2, landmarks_list)
                ring_finger_up = fingerUp(4, landmarks_list)
                pinky_finger_up = fingerUp(6, landmarks_list)

                if index_finger_up:
                    x1 = landmarks_list[1][0]
                    y1 = landmarks_list[1][1]
                    cv2.circle(frame, (int(x1 * imageWidth), int(y1 * imageHeight)), 5, (0,255,0), -1)
                    autopy.mouse.move(x1 * 1920, y1 * 1080)
                    x1 = int(x1 * imageWidth)
                    y1 = int(y1 * imageHeight)
                    if middle_finger_up:
                        x2 = landmarks_list[3][0]
                        y2 = landmarks_list[3][1]
                        x2 = int(x2 * imageWidth)
                        y2 = int(y2 * imageHeight)
                        cv2.circle(frame, (x2, y2), 5, (0,255,0), -1)
                        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        distance = int(round(findDistance(x1,y1,x2,y2), 0))
                        if distance < 50:
                            autopy.mouse.click()
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == ord("q"):
            break
cv2.destroyAllWindows()