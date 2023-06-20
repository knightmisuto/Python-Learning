import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import time


PATH_to_object_detection_folder = 'E:/tensorflow/models/research/'

TYPE_DETECT = 'object_detection'

PATH_TO_VIDEO = 'E:/AI learning/Videotest'

sys.path.insert(0, PATH_to_object_detection_folder)

from object_detection.utils import label_map_util

MODEL_FOLDER = 'inference_graph'
VIDEO_NAME = 'Conan_show.mp4'

img_width = 640
img_height = 480

def draw_bounding_box(image, x, y, w, h):
    return cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),1)

cap = cv2.VideoCapture(os.path.join(PATH_TO_VIDEO,VIDEO_NAME))

PATH_TO_CKPT = os.path.join(PATH_to_object_detection_folder,TYPE_DETECT,MODEL_FOLDER,'face_detection_ssd_mobilenet_v1_coco.pb')

# Path to label map file
PATH_TO_LABELS = "E:\Face_detection\ssd_mobilenet_v2_coco_2018_03_29\labelmap.pbtxt"

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

        result = np.where(scores >= 0.9)[0]
        for index in result:
            y1, x1, y2, x2 = boxes[index]
            x = int(round(x1*img_width,0))
            y = int(round(y1*img_height,0))
            w = int(round(x2*img_width - x,0))
            h = int(round(y2*img_height - y,0))
            center_x = x + w//2
            center_y = y + h//2
            draw_bounding_box(frame, x, y, w, h)


        cv2.imshow('object detection', frame)

        black_img1 = black_img2

        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break