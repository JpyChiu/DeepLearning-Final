import tkinter as tk
#from tkinter import *
from PIL import Image,ImageTk
import tkinter.font as tkFont
from tkinter import filedialog
import os

class CNN:
    def __init__(self, age):
        self.age1 = age

    def temp(self):
        return self.age1

class Interface:
    def __init__(self, root):
        self.root = root
        root.title("人臉年齡偵測系統")
        self.ft = tkFont.Font(family='Helvetica', size=30, weight=tkFont.BOLD)
        self.ftBtn = tkFont.Font(family='Helvetica', size=15, weight=tkFont.BOLD)
        self.label = tk.Label(self.root,text="歡迎使用年齡偵測系統",font=self.ft,bg='black',fg='SkyBlue1').pack(fill=tk.X)
        self.img=ImageTk.PhotoImage(file = "C:/Users/user/PycharmProjects/test/helloworld.jpg")
        self.imgLabel = tk.Label(self.root, image=self.img).pack()
        self.inBtn = tk.Button(self.root, text="選擇檔案：",font=self.ftBtn, command=self.select_img).pack(fill=tk.X)

    def hideRoot(self):
        self.root.withdraw()

    def select_img(self):
        self.tl = tk.Toplevel()
        #self.tl.geometry("+300+80")
        self.tl.withdraw()
        self.file_path = filedialog.askopenfilename(initialdir = "C:/Users/user/PycharmProjects/test/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
        self.rel_path = os.path.relpath(self.file_path)
        self.tl.update()
        self.tl.deiconify()
        self.im=Image.open(self.rel_path)
        self.im = self.im.resize((224, 224), Image.ANTIALIAS)
        self.img=ImageTk.PhotoImage(self.im)
        self.imLabel=tk.Label(self.tl,image=self.img).grid(row=0, column=1)
        self.hideRoot()
        self.start_testing()
        self.ftAge = tkFont.Font(family='Helvetica', size=15, weight=tkFont.BOLD)
        self.ageLabel = tk.Label(self.tl, height=5, text=("預測年齡:"+self.age+"歲"), font=self.ftAge).grid(row=1,column=1)
        self.closeBtn = tk.Button(self.tl, text="結束程式", font=self.ftBtn, command=self.close_window).grid(row=2, column=0)
        self.nextBtn = tk.Button(self.tl, text="看下一張", font=self.ftBtn, command=self.restart_window).grid(row=2, column=2)
        self.tl.mainloop()

    def close_window(self):
        self.tl.destroy()
        self.root.destroy()

    def restart_window(self):
        self.close_window()
        newRoot = tk.Tk()
        self.__init__(newRoot)

    def start_testing(self):
        self.tmp = CNN(self.rel_path)
        self.age = self.tmp.temp()

if __name__ == '__main__':
    root = tk.Tk()
    app = Interface(root)
    root.mainloop()
