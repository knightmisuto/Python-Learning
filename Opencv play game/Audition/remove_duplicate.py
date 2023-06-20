import os
import cv2
import numpy as np
import math
from tqdm import tqdm
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity


location = "Data/data"

list_image = os.listdir(location)

for image_1 in tqdm(list_image):
    img_1 = cv2.imread(os.path.join(location, image_1))
    for image_2 in list_image:
        if image_2 != image_1:
            img_2 = cv2.imread(os.path.join(location, image_2))
            if img_2.shape == (30,30,3):
                res = cv2.absdiff(img_1, img_2)
                res = res.astype(np.uint8)
                percentage = ((res.size - np.count_nonzero(res)) * 100) / res.size
                if percentage > 90:
                    os.remove(os.path.join(location, image_2))
                    list_image.remove(image_2)
            else:
                os.remove(os.path.join(location, image_2))
                list_image.remove(image_2)

img_1 = cv2.imread("Data/data/76.png")
# for image_2 in list_image:
#     img_2 = cv2.imread(os.path.join(location, image_2))
img_2 = cv2.imread("Data/data/4.png")
# res = cv2.absdiff(img_1, img_2)
# res = res.astype(np.uint8)
# percentage = ((res.size - np.count_nonzero(res)) * 100)/ res.size
# print(percentage)
# if img_2.shape != (30,30,3):
#     print("Not right")
# else:
#     print("right")