import os
import cv2
import numpy as np
import tensorflow as tf
import sys

PATH_to_object_detection_folder = 'E:/tensorflow/models/research/'

TYPE_DETECT = 'object_detection'

PATH_TO_IMGS = 'E:/Wider_face_dataset/WIDER_train/images/3--Riot'

sys.path.insert(0, PATH_to_object_detection_folder)

from object_detection.utils import label_map_util

MODEL_FOLDER = 'inference_graph'
IMG_NAME = '3_Riot_Riot_3_93.jpg'

cap = cv2.VideoCapture(os.path.join(PATH_TO_IMGS,IMG_NAME))

PATH_TO_CKPT = os.path.join(PATH_to_object_detection_folder,TYPE_DETECT,MODEL_FOLDER,'face_detection_ssd_mobilenet_v1_coco.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(PATH_to_object_detection_folder,TYPE_DETECT,'training','labelmap.pbtxt')

# Path to video
PATH_TO_IMG = os.path.join(PATH_TO_IMGS,IMG_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 1


# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v1.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.compat.v1.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

def draw_bounding_box(image, x1, x2, y1, y2):
    return cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)
# Load image using OpenCV and
# expand image dimensions to have shape: [1, None, None, 3]
# i.e. a single-column array, where each item in the column has the pixel RGB value

image = cv2.imread(os.path.join(PATH_TO_IMGS,IMG_NAME))
img_width = image.shape[1]
img_height = image.shape[0]
image = cv2.resize(image,(img_width,img_height))
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_expanded = np.expand_dims(image_rgb, axis=0)

# Perform the actual detection by running the model with the image as input
(boxes, scores, classes, num) = sess.run(
    [detection_boxes, detection_scores, detection_classes, num_detections],
    feed_dict={image_tensor: image_expanded})

# Draw the results of the detection (aka 'visulaize the results')
boxes = np.squeeze(boxes)
scores = np.squeeze(scores)

result = np.where(scores >= 0.9)[0]
for index in result:
    draw_bounding_box(image,int(boxes[index][1]*img_width),int(boxes[index][3]*img_width),int(boxes[index][0]*img_height),int(boxes[index][2]*img_height))

# All the results have been drawn on image. Now display the image.
cv2.imshow('Object detector', image)

# Press any key to close the image
cv2.waitKey(0)

# Clean up
cv2.destroyAllWindows()
