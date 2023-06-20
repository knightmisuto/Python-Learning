from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename, asksaveasfilename
import cv2
import main

window = Tk()
window.geometry("1240x768")
window.title("IMG transform")

frame_master = Frame(window)
frame_master.pack(fill=BOTH, expand=1)

# canvas_1 = Canvas(frame_master)
# canvas_1.pack(side=LEFT, fill=BOTH, expand=1)
#
# v_scroll = ttk.Scrollbar(frame_master, orient=VERTICAL, command=canvas_1.yview)
# h_scroll = ttk.Scrollbar(frame_master, orient=HORIZONTAL, command=canvas_1.xview)
# v_scroll.pack(side=RIGHT, fill=Y)
# h_scroll.pack(side=BOTTOM, fill=X)
#
# canvas_1.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
# canvas_1.bind("<Configure>", lambda e: canvas_1.configure(scrollregion=canvas_1.bbox("all")))
#
# frame_squad = Frame(canvas_1)
#
# canvas_1.create_window(0, 0, window=frame_squad, anchor="nw")

frame_1 = Frame(frame_master)
frame_1.grid(row=0, column=0, sticky="nsew")
frame_2 = Frame(frame_master)
frame_2.grid(row=0, column=1, sticky="nsew")
label_1 = Label(window)
label_2 = Label(window)

PATH = None
out = None


def Open_file():
    filepath = askopenfilename(initialdir="/",
                               title="Select a File",
                               filetypes=(("image files",
                                           ".jpeg .png .jpg"),
                                          ("all files",
                                           ".*")))
    return filepath


def open_img():
    global label_1
    global PATH
    path = Open_file()
    PATH = path
    label_1.destroy()
    label_2.destroy()
    img = Image.open(path)
    img = ImageTk.PhotoImage(img)
    label_1 = Label(frame_2, image=img)
    label_1.image = img
    label_1.grid(row=0, column=0)


def Save_img():
    filepath = asksaveasfilename(
        defaultextension="png",
        filetypes=[("Image", "*.png"), ("All Files", "*.*")],
    )
    cv2.imwrite(filepath, out)


def preview_sketch():
    global label_2
    global out
    label_2.destroy()
    img_sketch = main.img_to_sketch(PATH)
    out = img_sketch
    img_sketch = ImageTk.PhotoImage(image=Image.fromarray(img_sketch))
    label_2 = Label(frame_2, image=img_sketch)
    label_2.image = img_sketch
    label_2.grid(row=1, column=0)
    sketch_save_btn = Button(frame_1, text="Save sketch image", command=Save_img)
    sketch_save_btn.grid(row=3, column=0, sticky="ew", padx=5, pady=5)


def preview_cartoon():
    global label_2
    global out
    label_2.destroy()
    img_cartoon = main.img_to_cartoon(PATH)
    out = img_cartoon
    img_cartoon = cv2.cvtColor(img_cartoon, cv2.COLOR_BGR2RGB)
    img_cartoon = ImageTk.PhotoImage(image=Image.fromarray(img_cartoon))
    label_2 = Label(frame_2, image=img_cartoon)
    label_2.image = img_cartoon
    label_2.grid(row=1, column=0)
    cartoon_save_btn = Button(frame_1, text="Save cartoon image", command=Save_img)
    cartoon_save_btn.grid(row=3, column=0, sticky="ew", padx=5, pady=5)


button_open = Button(frame_1, text="Open image", command=open_img)
button_preview_sketch = Button(frame_1, text="Preview sketch", command=preview_sketch)
button_preview_cartoon = Button(frame_1, text="Preview cartoon", command=preview_cartoon)

button_open.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
button_preview_sketch.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
button_preview_cartoon.grid(row=2, column=0, sticky="ew", padx=5, pady=5)

window.mainloop()
