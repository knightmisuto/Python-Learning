import cv2
import requests
import mediapipe as mp
import numpy as np
import time

video = cv2.VideoCapture(0)
#URL from ip webcam + /video
url = "http://192.168.0.101:4747/video"
video.open(url)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence= 0.5,
    min_tracking_confidence=0.5) as hands:
    while True:
        _, frame = video.read()
        if not _:
            continue

        imageHeight, imageWidth, _ = frame.shape
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        results = hands.process(frame)

        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks_list = []
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == ord("q"):
            break
cv2.destroyAllWindows()