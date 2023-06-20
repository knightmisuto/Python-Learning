import cv2
import os
from tqdm import tqdm

Path_to_images = "E:\Wider_face_dataset\images"

def write_xml(img_name,Path_to_images):
    filePath = os.path.join(Path_to_images,img_name)
    img = cv2.imread(filePath)
    return img.shape


for img_name in os.listdir(Path_to_images):
    file_name, ext = os.path.splitext(img_name)
    if ext != ".xml":
        print(ext)