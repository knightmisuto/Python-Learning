import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import time
from math import sqrt
from Try_more import Kalman_filter

PATH_to_object_detection_folder = 'E:/tensorflow/models/research/'

TYPE_DETECT = 'object_detection'

PATH_TO_VIDEO = 'E:/AI learning/Videotest'

sys.path.insert(0, PATH_to_object_detection_folder)

from object_detection.utils import label_map_util

MODEL_FOLDER = 'inference_graph'
VIDEO_NAME = 'Conan_show.mp4'

img_width = 640
img_height = 480