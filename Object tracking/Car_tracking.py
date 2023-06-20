#This script tend to use car detection model for detect car in video and estimate speed
#import everything we need in this script
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import time
from math import sqrt

#Path config to model and video folder
PATH_to_object_detection_folder = 'models/research'

TYPE_DETECT = 'object_detection'

PATH_TO_VIDEO = 'Videotest'

#import function from model folder
from models.research.object_detection.utils import label_map_util

MODEL_FOLDER = 'inference_graph'
VIDEO_NAME = 'Cars_passing_test_3.mp4'

img_width = 640
img_height = 480

#Function for draw bounding box with opencv
def draw_bounding_box(image, x, y, w, h):
    return cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),1)

def get_bounding_box_area(x1, y1, x2, y2):
    area = ((x2-x1) + (y2-y1)) * 2
    return area

#Algrothim for find location of car in past frame
def find_nearest_white(img, target):
    nonzero = cv2.findNonZero(img)
    distances = np.sqrt((nonzero[:,:,0] - target[0]) ** 2 + (nonzero[:,:,1] - target[1]) ** 2)
    nearest_index = np.argmin(distances)
    return nonzero[nearest_index][0]

cap = cv2.VideoCapture(os.path.join(PATH_TO_VIDEO,VIDEO_NAME))

#Path to model folder
PATH_TO_CKPT = os.path.join(PATH_to_object_detection_folder,TYPE_DETECT,MODEL_FOLDER,'car_detection_ssd_mobilenet_v1_coco.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(PATH_to_object_detection_folder,TYPE_DETECT,'training','labelmap.pbtxt')

# Path to video
PATH_TO_VIDEO = os.path.join(PATH_TO_VIDEO,VIDEO_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 1

# Load the label map.
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.compat.v1.GraphDef()
  with tf.compat.v1.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

black_img1 = None


with detection_graph.as_default():
  with tf.compat.v1.Session(graph=detection_graph) as sess:
    while True:
        start = time.time()
        _, frame = cap.read()
        fps = int(round(cap.get(cv2.CAP_PROP_FPS),0))
        black_img2 = np.zeros((img_height, img_width,3),np.uint8)
        black_img2 = cv2.cvtColor(black_img2, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame,(img_width,img_height))
        cv2.putText(frame, str(fps), (5,20),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1,
                            cv2.LINE_AA)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(frame, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')


        # Actual detection.
        (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
        # Visualization of the results of a detection.

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)

        #only dislay object that have more than 70% accurate
        result = np.where(scores >= 0.7)[0]
        for index in result:
            #draw bounding box for each detected car
            y1, x1, y2, x2 = boxes[index]
            x = int(round(x1*img_width,0))
            y = int(round(y1*img_height,0))
            w = int(round(x2*img_width - x,0))
            h = int(round(y2*img_height - y,0))
            center_x = x + w//2
            center_y = y + h//2
            draw_bounding_box(frame, x, y, w, h)
            cv2.circle(black_img2, (center_x, center_y), 1, (255, 255, 255), 2)
            if black_img1 is None:
                pass
            else:
                end = time.time()
                target = (center_x, center_y)
                nearest_target = find_nearest_white(black_img1, target)
                target = (center_x, center_y)
                distance = sqrt((nearest_target[0] - x)**2 + (nearest_target[1] - y)**2)
                distance = distance
                timer = end - start
                diff = (1/30) / timer
                v = distance / timer
                v_sec = (1//(1/fps) * v) / diff
                #estimate speed of car in video
                #Still need to working on for improvement
                estimate_v = (v_sec * 3600) / 3779528.0352161
                re_calculate = center_y / img_height
                estimate_v = estimate_v / re_calculate
                #Display current speed on bounding box
                cv2.putText(frame, (str(int(round(estimate_v,0)))+" km/h"), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                            cv2.LINE_AA)


        cv2.imshow('object detection', frame)

        black_img1 = black_img2

        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break