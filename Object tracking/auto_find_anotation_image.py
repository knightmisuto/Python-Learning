import cv2
import numpy as np
import face_recognition
import os
from pathlib import Path
from tqdm import tqdm


Folder = 'E:/Wider_face_dataset/WIDER_train/images'
New_dir = 'E:/Wider_face_dataset/images'

for sub_folder in os.listdir(Folder):
    for img_name in tqdm(os.listdir(os.path.join(Folder,sub_folder))):
        img = face_recognition.load_image_file(os.path.join(Folder,sub_folder,img_name))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        faces_loc = face_recognition.face_locations(img)
        if len(faces_loc) == 0:
            pass
        else:
            for face in faces_loc:
                cv2.rectangle(img, (face[3],face[0]), (face[1],face[2]), (0,255,0), 1)

            cv2.imshow("img", img)
            if cv2.waitKey(0) == ord('y'):
                Path(os.path.join(Folder,sub_folder,img_name)).rename(New_dir+"/"+img_name)