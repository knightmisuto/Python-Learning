import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from Useful_functions_opencv import stackImages

Skills = np.load("Data/Skills.npy", allow_pickle=True)
Buttons = np.load("Data/Buttons.npy", allow_pickle=True)

path = 'Data/Data/8k_reverse_4.png'

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

num = 0

img = cv2.imread(path)
r_min = 200
r_max = 255
g_min = 215
g_max = 255
b_min = 150
b_max = 255
lower = np.array([r_min, g_min, b_min])
upper = np.array([r_max, g_max, b_max])
mask = cv2.inRange(img, lower, upper)
imgResult = cv2.bitwise_and(img, img, mask=mask)
imgStack = stackImages(1,([img,mask,imgResult]))
contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print("Number of Contours found = " + str(len(contours)))
Keys = []
for cnt in sort_contours(contours, method="left-to-right"):
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    x = int(x)
    y = int(y)
    crop = img[int(y - 30 / 2):int(y + 30 / 2), int(x - 30 / 2):int(x + 30 / 2)]
    # cv2.imwrite("Data/Keys/"+str(num)+".png",crop)
    num+=1
    crop = crop.ravel()
    if crop.shape[0] == 0:
        pass
    else:
        crop = crop.reshape(-1, crop.shape[0])
        try:
            Keys.append(str(Buttons_with_skill(crop)))
        except:
            pass
print(np.array(Keys))

cv2.imshow("img", img)
cv2.waitKey(0)