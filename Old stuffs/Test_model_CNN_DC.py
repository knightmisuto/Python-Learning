import tensorflow as tf
import cv2
from tqdm import tqdm
import os

CATEGORIES = ["Cat", "Dog"]
Dog_count = 1
Cat_count = 1


TEST_PATH = "DogvsCat/Test"
model = tf.keras.models.load_model("DogvsCat4x3-CNN.model")


def prepare(filepath):
    IMG_SIZE = 100
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

for img in tqdm(os.listdir(TEST_PATH)):
    path = os.path.join(TEST_PATH, img)
    image = cv2.imread(path)
    prediction = model.predict([prepare(path)])
    result = int(prediction[0][0])
    if result == 0:
        cv2.imwrite("DogvsCat/Result/Cats/Cat_{}.jpg".format(str(Cat_count)), image)
        Cat_count += 1
    if result == 1:
        cv2.imwrite("DogvsCat/Result/Dogs/Dog_{}.jpg".format(str(Dog_count)), image)
        Dog_count += 1

print("Dog: ",Dog_count)
print("Cat: ",Cat_count)