import face_recognition
import os
from tqdm import tqdm
import cv2
import random

def bounding_box_for_xml(xmin,ymin,xmax,ymax,bbx_label):
    xml = "\t<object>\n"
    xml = xml + "\t\t<name>" + bbx_label + "</name>\n"
    xml = xml + "\t\t<pose>Unspecified</pose>\n"
    xml = xml + "\t\t<truncated>0</truncated>\n"
    xml = xml + "\t\t<difficult>0</difficult>\n"
    xml = xml + "\t\t<bndbox>\n"
    xml = xml + "\t\t\t<xmin>" + str(xmin) + "</xmin>\n"
    xml = xml + "\t\t\t<xmax>" + str(xmax) + "</xmax>\n"
    xml = xml + "\t\t\t<ymin>" + str(ymin) + "</ymin>\n"
    xml = xml + "\t\t\t<ymax>" + str(ymax) + "</ymax>\n"
    xml = xml + "\t\t</bndbox>\n"
    xml = xml + "\t</object>\n"
    return xml

def write_xml(img_name,Path_to_images,Path_to_xml):
    filePath = os.path.join(Path_to_images,img_name)
    img = cv2.imread(filePath)
    width = img.shape[1]
    height = img.shape[0]
    depth = img.shape[-1]
    folder = ""
    rnd = random.randint(0,9)
    if rnd < 2:
        folder = "test"
    else:
        folder = "train"

    xml = "<annotation>\n\t<folder>" + folder + "</folder>\n"
    xml = xml + "\t<filename>" + img_name + "</filename>\n"
    xml = xml + "\t<path>" + filePath + "</path>\n"
    xml = xml + "\t<source>\n\t\t<database>Unknown</database>\n\t</source>\n"
    xml = xml + "\t<size>\n"
    xml = xml + "\t\t<width>" + str(width) + "</width>\n"
    xml = xml + "\t\t<height>" + str(height) + "</height>\n"
    xml = xml + "\t\t<depth>" + str(depth) + "</depth>\n"
    xml = xml + "\t</size>\n"
    xml = xml + "\t<segmented>0</segmented>\n"

    img = face_recognition.load_image_file(os.path.join(Path_to_images, img_name))
    faces_loc = face_recognition.face_locations(img)
    for face in faces_loc:
        xmin = face[3]
        ymin = face[0]
        xmax = face[1]
        ymax = face[2]
        bbx_label = "face"
        xml = xml + bounding_box_for_xml(xmin, ymin, xmax, ymax, bbx_label)

    xml = xml + "</annotation>"

    xmlFilePath = os.path.join(Path_to_xml, img_name + ".xml")
    with open(xmlFilePath, 'w') as f:
        f.write(xml)

Path_to_images = "E:/Wider_face_dataset/images"
Path_to_xml = "E:/Wider_face_dataset/xml"

for file in tqdm(os.listdir(Path_to_images)):
    file_name, ext = os.path.splitext(file)
    if ext == ".xml":
        pass
    else:
        write_xml(file,Path_to_images,Path_to_xml)