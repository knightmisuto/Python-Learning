from tkinter import *
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename


def Open_file():
    filepath = askopenfilename(initialdir="/",
                               title="Select a File",
                               filetypes=(("image files",
                                           ".jpeg .png .jpg"),
                                          ("all files",
                                           ".*")))
    return filepath

def Open_img():
    path = Open_file()
    n = 5
    img = Image.open(path)
    [imageSizeWidth, imageSizeHeight] = img.size
    newImageSizeWidth = int(imageSizeWidth // n)
    newImageSizeHeight = int(imageSizeHeight / n)
    img = img.resize((newImageSizeWidth, newImageSizeHeight), Image.ANTIALIAS)
    tkimage = ImageTk.PhotoImage(img)
    panel = Label(window, image=tkimage)

    panel.image = img
    panel.grid(column=1, row=0)