import cv2
import os
from PIL import Image
import numpy as np

list_image = os.listdir("Data/Keys")

# for image in list_image:
#     img = cv2.imread("Data/Keys/" + image)
#     img = cv2.resize(img, (30,30))
#     cv2.imwrite("Data/Keys/" + image, img)

Skills = []
Buttons = []

for image in list_image:
    img = cv2.imread("Data/Keys/" + image)
    img = np.array(img)
    Skills.append(img.ravel())
    Buttons.append(image[:-5])
    print(image[:-5])

Skills = np.array(Skills)
Buttons = np.array(Buttons)

# print(Buttons)

np.save("Data/Skills", Skills)
np.save("Data/Buttons", Buttons)
