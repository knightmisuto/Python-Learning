import cv2
import mediapipe as mp
import numpy as np

img = cv2.imread("Data/hand_1.jpg")
imageHeight, imageWidth, _ = img.shape

text_file = open("Data/fingers.txt")
fingers_list = text_file.read().split("\n")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
results = hands.process(imgRGB)

def fingerUp(idx_finger, fingers_list):
    if fingers_list[idx_finger][1] > fingers_list[idx_finger+1][1]:
        return True
    else:
        return False

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
                landmarks_list.append([x,y])
        index_finger_up = fingerUp(0, landmarks_list)
        print(index_finger_up)

cv2.circle(img, (100,100), 20, (0,255,0), -1)

cv2.imshow("img", img)
cv2.waitKey(0)