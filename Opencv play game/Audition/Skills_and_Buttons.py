import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import time

location = "Data"

down_normal = np.load(os.path.join(location, "down_normal.npy"))
down_reverse = np.load(os.path.join(location, "down_reverse.npy"))
end_normal = np.load(os.path.join(location, "end_normal.npy"))
end_reverse = np.load(os.path.join(location, "end_reverse.npy"))
home_normal = np.load(os.path.join(location, "home_normal.npy"))
home_reverse = np.load(os.path.join(location, "home_reverse.npy"))
left_normal = np.load(os.path.join(location, "left_normal.npy"))
left_reverse = np.load(os.path.join(location, "left_reverse.npy"))
pgdn_normal = np.load(os.path.join(location, "pgdn_normal.npy"))
pgdn_reverse = np.load(os.path.join(location, "pgdn_reverse.npy"))
pgup_normal = np.load(os.path.join(location, "pgup_normal.npy"))
pgup_reverse = np.load(os.path.join(location, "pgup_reverse.npy"))
right_normal = np.load(os.path.join(location, "right_normal.npy"))
right_reverse = np.load(os.path.join(location, "right_reverse.npy"))
up_normal = np.load(os.path.join(location, "up_normal.npy"))
up_reverse = np.load(os.path.join(location, "up_reverse.npy"))

Skills = [down_normal, down_reverse, end_normal, end_reverse,
           home_normal, home_reverse, left_normal, left_reverse,
           pgdn_normal, pgdn_reverse, pgup_normal, pgup_reverse,
           right_normal, right_reverse, up_normal, up_reverse]

Buttons = ["down2", "down1", "end2", "end1",
           "home2", "home1", "left2", "left1",
           "pgdn2", "pgdn1", "pgup2", "pgup1",
           "right2", "right1", "up2", "up1"]

img = cv2.imread("Data/up_normal/1792.png")
img = img.ravel()
img = img.reshape(-1, img.shape[0])

def Buttons_with_skill(img, skills=Skills):
    results = []
    for skill in skills:
        simi = cosine_similarity(img, skill)
        results.append(np.argmax(simi))
    results = np.array(results)
    index = np.argmax(results, axis=0)
    return Buttons[index], results

time_start = time.time()
print(Buttons_with_skill(img))
end = time.time()
print(end-time_start)