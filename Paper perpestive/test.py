from tkinter import *
from PIL import Image, ImageTk

n = 5
root = Tk()

path = "E:/Machine Learning/Paper perpestive/Data/1.jpg"
img = Image.open(path)
[imageSizeWidth, imageSizeHeight] = img.size
newImageSizeWidth = int(imageSizeWidth // n)
newImageSizeHeight = int(imageSizeHeight / n)
img = img.resize((newImageSizeWidth, newImageSizeHeight))
img = ImageTk.PhotoImage(img)
panel = Label(root, image=img)
panel.photo = img
panel.grid(column=2,row=2)

root.mainloop()