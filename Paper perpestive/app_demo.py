from tkinter import *
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename, asksaveasfilename
import process
import cv2

window = Tk()
window.title("Document scanner")
window.rowconfigure(0, minsize=800, weight=1)
window.columnconfigure(1, minsize=800, weight=1)

panel_1 = Label(window)
panel_2 = Label(window)
test_out = None


def Open_file():
    filepath = askopenfilename(initialdir="/",
                               title="Select a File",
                               filetypes=(("image files",
                                           ".jpeg .png .jpg"),
                                          ("all files",
                                           ".*")))
    return filepath


def Save_img():
    filepath = asksaveasfilename(
        defaultextension="png",
        filetypes=[("Image", "*.png"), ("All Files", "*.*")],
    )
    cv2.imwrite(filepath, test_out)


def Open_img():
    global panel_1
    global panel_2
    n = 5
    path = Open_file()
    panel_1.destroy()
    panel_2.destroy()
    img = Image.open(path)
    [imageSizeWidth, imageSizeHeight] = img.size
    newImageSizeWidth = int(imageSizeWidth // n)
    newImageSizeHeight = int(imageSizeHeight // n)
    img = img.resize((newImageSizeWidth, newImageSizeHeight))
    img = ImageTk.PhotoImage(img)
    panel_1 = Label(window, image=img)
    panel_1.photo = img
    panel_1.grid(column=1, row=0)

    img_out = Analyze_img(path)
    panel_2 = Label(window, image=img_out)
    panel_2.image = img_out
    panel_2.grid(column=2, row=0)


def Analyze_img(path):
    global test_out
    out = process.Analyze(path)
    test_out = out
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    tkimageout = ImageTk.PhotoImage(image=Image.fromarray(out))
    return tkimageout


frame_buttons = Frame(window)
btn_open = Button(frame_buttons, text="Open image", command=Open_img)
btn_save = Button(frame_buttons, text="Save As", command=Save_img)

btn_open.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
btn_save.grid(row=1, column=0, sticky="ew", padx=5)

frame_buttons.grid(row=0, column=0, sticky="ns")

window.mainloop()