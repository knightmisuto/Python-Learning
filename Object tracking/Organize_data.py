import xml.etree.ElementTree as ET
from tqdm import tqdm
import os
import shutil

path = "E:/Wider_face_dataset/images"
glob_path = "E:/Wider_face_dataset"
glob_des = ['train', 'test']

for file in tqdm(os.listdir(path)):
    if os.path.splitext(file)[-1] == '.xml':
        xml_file = os.path.join(path,file)
        tree = ET.parse(xml_file)
        root = tree.getroot()
        if root[0].text == 'train':
            destinaton = root[0].text
            new_path = os.path.join(glob_path,destinaton)
            shutil.move(xml_file,new_path)
        if root[0].text == 'test':
            destinaton = root[0].text
            new_path = os.path.join(glob_path,destinaton)
            shutil.move(xml_file,new_path)

for des in glob_des:
    for file in tqdm(os.listdir(os.path.join(glob_path,des))):
        file_name = os.path.splitext(file)[0]
        for root, dirs, files in os.walk(path):
            for name in files:
                if name == file_name:
                    file_des = os.path.join(root, name)
                    new_path = os.path.join(glob_path,des)
                    shutil.move(file_des,new_path)