import numpy as np
import cv2
import os
from tqdm import tqdm
from glob import glob

folders_list = glob("E:/Nghich ngom/Hack Audition/Data/*/")

Buttons_list = ["down_normal", "down_reverse", "end_normal", "end_reverse",
           "home_normal", "home_reverse", "left_normal", "left_reverse",
           "pgdn_normal", "pgdn_reverse", "pgup_normal", "pgup_reverse",
           "right_normal", "right_reverse", "up_normal", "up_reverse"]

Skills = []
Buttons = []

for folder in tqdm(folders_list):
    images_list = os.listdir(folder)
    name = folder.split("\\")[-2]
    if name != "Data":
        for image in images_list:
            img = cv2.imread(folder + image)
            img = np.array(img)
            Skills.append(img)
            Buttons.append(Buttons_list.index(name))

Skills = np.array(Skills)
Buttons = np.array(Buttons)

np.save("Data/Data/Skills", Skills)
np.save("Data/Data/Buttons", Buttons)