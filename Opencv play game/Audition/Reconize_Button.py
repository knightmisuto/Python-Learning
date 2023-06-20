import cv2
import numpy as np
from skimage.measure import compare_ssim as ssim
from PIL import ImageGrab
import win32gui
from sklearn.metrics.pairwise import cosine_similarity
import datetime
from pyKey import pressKey, releaseKey
import time

Keys2Hex = {"left2": "NUM4", "left1": "NUM4", "right1": "NUM6", "right2": "NUM6",
            "up2": "NUM8", "up1": "NUM8", "down1": "NUM2", "down2": "NUM2",
            "pgup2": "NUM9", "pgup1": "NUM9", "pgdn2": "NUM3", "pgdn1": "NUM3",
            "home2": "NUM7", "home1": "NUM7", "end2": "NUM1", "end1": "NUM1"}

Skills = np.load("Data/Skills.npy", allow_pickle=True)
Buttons = np.load("Data/Buttons.npy", allow_pickle=True)

windows_list = []
toplist = []
def enum_win(hwnd, result):
    win_text = win32gui.GetWindowText(hwnd)
    windows_list.append((hwnd, win_text))
win32gui.EnumWindows(enum_win, toplist)

game_hwnd = 0
for (hwnd, win_text) in windows_list:
    if "Audition" in win_text:
        game_hwnd = hwnd


def Buttons_with_skill(img, skills=Skills, button=Buttons):
    simi = cosine_similarity(img, skills)
    index = np.argmax(simi, axis=1)
    key = button[index[0]]
    return key

def sort_contours(cnts, method="left-to-right"):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0
	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
	# construct the list of bounding boxes and sort them from top to
	# bottom
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))
	# return the list of sorted contours and bounding boxes
	return (cnts)

path = 'Data/Skills/Key_thuong_3.png'

center_x = 960
center_y = 540
width = 1024
height = 768

r_min = 200
r_max = 255
g_min = 215
g_max = 255
b_min = 150
b_max = 255

IMG_NAME = 0

while True:
	position = win32gui.GetWindowRect(game_hwnd)
	(x1,y1,x2,y2) = position
	screen = np.array(ImageGrab.grab(position))
	screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
	# crop_1 = screen[int(height / 1.44):int(height / 1.30), int(width / 3.8):int(width / 1.35)]
	crop_1 = screen[int(height / 1.44):int(height / 1.30), int(width / 6.5):int(width / 1.2)]
	lower = np.array([r_min, g_min, b_min])
	upper = np.array([r_max, g_max, b_max])
	mask = cv2.inRange(crop_1, lower, upper)

	contours, hierarchy =cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

	#cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
	start = datetime.datetime.now()
	Keys = []
	if len(contours) < 6 or len(contours) > 11:
		pass
	else:
		print("Number of Contours found = " + str(len(contours)))
		for cnt in sort_contours(contours, method="left-to-right"):
			(x,y),radius = cv2.minEnclosingCircle(cnt)
			x = int(x)
			y = int(y)
			crop_2 = crop_1[int(y-30/2):int(y+30/2), int(x-30/2):int(x+30/2)]
			crop_2 = crop_2.ravel()
			if crop_2.shape[0] == 0:
				pass
			else:
				crop_2 = crop_2.reshape(-1, crop_2.shape[0])
				try:
					Keys.append(str(Buttons_with_skill(crop_2)))
				except:
					pass
	if len(Keys) != 0:
		print(np.array(Keys))
	for key in Keys:
		pressKey(Keys2Hex[key])
		time.sleep(0.015)
		releaseKey(Keys2Hex[key])
		time.sleep(0.02)
	end = datetime.datetime.now()


	cv2.imshow("img", crop_1)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		cv2.destroyAllWindows()
		break