import cv2
import numpy as np

video = cv2.VideoCapture("Videotest/Cars_passing_test_3.mp4")

# Randomly select 500 frames
frameIds = video.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=700)

# Store selected frames in an array
frames = []
for fid in frameIds:
    video.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = video.read()
    frames.append(frame)

# Calculate the median along the time axis
medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)

cv2.imwrite("Imagetest/Background2.jpg", medianFrame)