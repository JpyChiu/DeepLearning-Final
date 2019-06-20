from __future__ import print_function, division
import sys as sy
sy.path.append("..")
from keras.layers import Input, Dense, Flatten, Dropout
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam
import datetime
import numpy as np
from sklearn.metrics import accuracy_score
from keras.models import load_model
import csv
import cv2
#from get_accuracy import get_accuracy
import tkinter as tk
from PIL import Image,ImageTk
import tkinter.font as tkFont
from tkinter import filedialog
import os

class Predict_Age:
    def __init__(self, imgName):
        my_CNN_Model = load_model('./test07.h5')#('./test07.h5')
        self.imgName = os.path.realpath(imgName)
        test_data = []
        test_data.append(cv2.imread(self.imgName))
        test_data = np.array(test_data) / 127.5 - 1.
        predict_label = my_CNN_Model.predict(test_data)
        aged_range = np.argmax(predict_label, axis=1)
        if aged_range == float(10):
            self.predict_age = "11" #第11類 無人臉
        else:
            self.predict_age = str(aged_range* 10 + 1)+"~"+str((aged_range + 1) * 10)#1~10歲類推

    def get_age(self):
        return self.predict_age

class Interface:
    def __init__(self, root):
        self.root = root
        root.title("人臉年齡偵測系統")
        ft = tkFont.Font(family='Helvetica', size=30, weight=tkFont.BOLD)
        self.ftBtn = tkFont.Font(family='Helvetica', size=15, weight=tkFont.BOLD)
        label = tk.Label(self.root,text="歡迎使用年齡偵測系統",font=ft,bg='black',fg='SkyBlue1').pack(fill=tk.X)
        im=Image.open("helloworld.jpg")
        self.img=ImageTk.PhotoImage(im)
        imgLabel = tk.Label(self.root, image=self.img).pack()
        inBtn = tk.Button(self.root, text="選擇檔案：",font=self.ftBtn, command=self.select_img).pack(fill=tk.X)

    def hideRoot(self):
        self.root.withdraw()

    def select_img(self):
        self.tl = tk.Toplevel()
        self.tl.geometry("+300+80")
        self.tl.withdraw()
        file_path = filedialog.askopenfilename(initialdir = "C:/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
        self.rel_path = os.path.realpath(file_path)
        self.tl.update()
        self.tl.deiconify()
        im=Image.open(self.rel_path)
        im = im.resize((224, 224), Image.ANTIALIAS)
        img=ImageTk.PhotoImage(im)
        imLabel=tk.Label(self.tl,image=img).grid(row=0, column=1)
        self.hideRoot()
        self.start_testing()
        ftAge = tkFont.Font(family='Helvetica', size=15, weight=tkFont.BOLD)
        if(self.age == "11"):
            output = "您選擇的圖片中辨識不到人臉"
        else:
            output = "預測年齡:"+self.age+"歲"
        ageLabel = tk.Label(self.tl, height=5, text=(output), font=ftAge).grid(row=1,column=1)
        closeBtn = tk.Button(self.tl, text="結束程式", font=self.ftBtn, command=self.close_window).grid(row=2, column=0)
        nextBtn = tk.Button(self.tl, text="看下一張", font=self.ftBtn, command=self.restart_window).grid(row=2, column=2)
        self.tl.mainloop()

    def close_window(self):
        self.tl.destroy()
        self.root.destroy()

    def restart_window(self):
        self.close_window()
        newRoot = tk.Tk()
        self.__init__(newRoot)

    def start_testing(self):
        tmp = Predict_Age(self.rel_path)
        self.age = tmp.get_age()

if __name__ == '__main__':
    root = tk.Tk()
    app = Interface(root)
    root.mainloop()


# class ConvolutionalNeuralNetworks():
#     def __init__(self):
#         # Input shape
#         self.img_rows = 224
#         self.img_cols = 224
#         self.channels = 3
#         self.img_shape = (self.img_rows, self.img_cols, self.channels)

#之前的code
# if __name__ == '__main__':
#     '''testing model'''
#     imgs, labels = [], []
#     #my_CNN = ConvolutionalNeuralNetworks()
#     #my_CNN_Model = my_CNN.build_CNN_Network()
#     my_CNN_Model = load_model('./model_35%2.h5')#('./test07.h5')
#
#
#     #demo的檔案假設在input.in中
#     file = open('./input.in', 'r')
#     #讀入檔名
#     dataset_name = file.readline().rstrip()
#     ##這裡應該將dataset_name讀成圖片再轉成二維矩陣 丟入model.predict
#     #使用cv2讀圖片 一維list:內容為像素
#     #使用append 而不用=是因為要讓np array 的shape 維度 = 4
#     test_data = []
#     #for a in enumerate(dataset_name):
#
#     test_data.append(cv2.imread(dataset_name))
#     #轉成np.array 得到左上到右下每一點的像素(R,G,B)   再把(R,G,B)分別除以127.5-1 得到三個小數
#     test_data = np.array(test_data) / 127.5 - 1.
#     #丟入model.predict 由於假設丟入一張圖片
#     #所以predict_label 大概會是一維矩陣塞入一堆奇怪的數字:
#     ##index: 0  1  2  3  4  5  6  7        8         9
#     ##       !  @  #  $  %  %  ^  ^ 最大數(我們要的)  *
#     predict_label = my_CNN_Model.predict(test_data)
#     ##接著使用predict_label判斷年齡層
#     #因此用np.argmax(pred_labels, axis=1)抓到最大數字的index 等於年齡層
#     #年齡層
#     #0:1~10
#     #1:11~20
#     #2:21~30
#     #3:31~40
#     #4:41~50
#     #5:51~60
#     #6:61~70
#     #7:71~80
#     #8:81~90
#     #9:91~100
#
#     aged_range = np.argmax(predict_label, axis=1)
#     print("We guess you are %d ~ %d years old" % ((aged_range* 10 + 1),(aged_range + 1) * 10) )
#
#     ###接下來給demo平台使用這個結果去做事
#     ###交給林韶恩...
