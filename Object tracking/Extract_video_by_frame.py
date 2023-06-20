import cv2
from tqdm import tqdm
import os

VIDEOs_PATH = 'E:/AI learning/Videotest'
VIDEO_NAME = 'Cars_passing_test_3.mp4'

VIDEO = os.path.join(VIDEOs_PATH,VIDEO_NAME)

cap = cv2.VideoCapture(VIDEO)

FRAMES_OUT = "Frames"

PROCESSED_FRAMES = "Processed frames"

if not os.path.isdir(os.path.join(VIDEOs_PATH,FRAMES_OUT)):
    os.mkdir(os.path.join(VIDEOs_PATH,FRAMES_OUT))
if not os.path.isdir(os.path.join(VIDEOs_PATH,PROCESSED_FRAMES)):
    os.mkdir(os.path.join(VIDEOs_PATH,PROCESSED_FRAMES))


total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

while cap.get(cv2.CAP_PROP_POS_FRAMES) < cap.get(cv2.CAP_PROP_FRAME_COUNT):
    _,frame = cap.read()
    if _:
        num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        cv2.imwrite('{frames_dir}/frame_{num:05d}.jpg'.format(
            frames_dir=os.path.join(VIDEOs_PATH,FRAMES_OUT), num=num), frame)
