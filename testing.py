from __future__ import print_function, division
import numpy as np
from keras.models import load_model
import cv2
import tkinter as tk
from PIL import Image,ImageTk
import tkinter.font as tkFont
from tkinter import filedialog

class Predict_Age:
    def __init__(self, img_path):
        my_CNN_Model = load_model('./test07.h5')#('./test07.h5')
        test_data = []
        img = cv2.imread(img_path)#從圖片路徑讀取圖片
        img = cv2.resize(img, (224,224), interpolation=cv2.INTER_CUBIC)#將圖片調整為224*224
        test_data.append(img)
        test_data = np.array(test_data) / 127.5 - 1.
        #這裡應該將dataset_name讀成圖片再轉成二維矩陣 丟入model.predict
        #使用cv2讀圖片 一維list:內容為像素
        #使用append 而不用=是因為要讓np array 的shape 維度 = 4
        predict_label = my_CNN_Model.predict(test_data)
        aged_range = np.argmax(predict_label, axis=1)
        if aged_range==10:
            self.predict_age = "11" #第11類 無人臉
        else:
            self.predict_age = str(aged_range* 10 + 1)+"~"+str((aged_range + 1) * 10)#1~10歲類推
        #轉成np.array 得到左上到右下每一點的像素(R,G,B)   再把(R,G,B)分別除以127.5-1 得到三個小數
        #丟入model.predict 由於假設丟入一張圖片
        #所以predict_label 大概會是一維矩陣塞入一堆奇怪的數字:
        ##index: 0  1  2  3  4  5  6  7        8         9
        ##       !  @  #  $  %  %  ^  ^ 最大數(我們要的)  *
        ##接著使用predict_label判斷年齡層
        #因此用np.argmax(pred_labels, axis=1)抓到最大數字的index 等於年齡層

    def get_age(self):
        return self.predict_age

class Interface:
    def __init__(self, root):
        self.root = root
        root.title("人臉年齡偵測系統")
        ft = tkFont.Font(family='Helvetica', size=30, weight=tkFont.BOLD)#設定標題字型
        self.ftBtn = tkFont.Font(family='Helvetica', size=15, weight=tkFont.BOLD)#設定按鈕字型
        #標題label，bg=被景色，fg=文字顏色，font=字型，grid為視窗元件排版
        label = tk.Label(self.root,text="歡迎使用年齡偵測系統",font=ft,bg='black',fg='SkyBlue1').grid(row=0, column=0, columnspan=2, sticky = tk.E+tk.W)
        im=Image.open("helloworld.jpg")#PIL將圖片讀入
        self.img=ImageTk.PhotoImage(im)
        imgLabel = tk.Label(self.root, image=self.img).grid(row=1, column=0, columnspan=2)#主畫面圖片label
        #選擇檔案按鈕，command呼叫function，sticky = tk.E+tk.W為橫向拓展
        inBtn = tk.Button(self.root, text="選擇檔案",font=self.ftBtn, command=self.select_one_img, bg='light grey').grid(row=2, column=0, sticky = tk.E+tk.W)
        multi_inBtn = tk.Button(self.root, text="選擇多個檔案",font=self.ftBtn, command=self.select_multi_img, bg='light grey').grid(row=2, column=1, sticky = tk.E+tk.W)

    def hideRoot(self):#隱藏Tk()視窗
        self.root.withdraw()

    def select_multi_img(self):#選擇多張照片
        #輸入圖片的路徑
        #file_paths事由圖片路徑組成的陣列
        self.file_paths = filedialog.askopenfilenames(initialdir = "C:/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("png files","*.png")))
        self.cnt = -1#紀錄讀到第幾張圖片
        self.cnt_next()

    def cnt_next(self):
        self.cnt += 1
        is_multi = True#是讀入多個圖檔
        if self.cnt != 0:#不是第一個就把之前開的視窗關閉
            self.tl.destroy()
        if self.cnt+1 == len(self.file_paths):#讀到最後一個圖檔，is_multi = False
            is_multi = False
        self.select_img(self.file_paths[self.cnt],is_multi)

    def select_one_img(self):#選擇一張照片
        file_path = filedialog.askopenfilename(initialdir = "C:/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("png files","*.png")))
        self.select_img(file_path,False)#進select_img function，只讀一個圖檔

    def select_img(self, file_path, is_multi):
        self.tl = tk.Toplevel()
        self.tl.geometry("+300+80")#設定UI在螢幕上的位置
        self.tl.withdraw()#隱藏UI
        if not file_path:#在選擇圖片時按下取消，回到主畫面(防呆)
            self.restart_window()
        else:
            self.path = file_path
            self.tl.update()#重新顯示視窗
            self.tl.deiconify()
            im=Image.open(self.path)
            im = im.resize((224, 224), Image.ANTIALIAS)#PIL的image resize
            img=ImageTk.PhotoImage(im)
            imLabel=tk.Label(self.tl,image=img).grid(row=0, column=1)#輸入圖片的label
            self.hideRoot()#隱藏主畫面
            self.start_testing()#開始進行testing
            ftAge = tkFont.Font(family='Helvetica', size=15, weight=tkFont.BOLD)#年齡字型
            if self.age == "11":
                output = "您選擇的圖片中辨識不到人臉"
            else:
                output = "預測年齡:"+self.age+"歲"
            ageLabel = tk.Label(self.tl, height=5, text=output, font=ftAge).grid(row=1,column=1)
            closeBtn = tk.Button(self.tl, text="結束程式", font=self.ftBtn, command=self.close_window).grid(row=2, column=0)
            newBtn = tk.Button(self.tl, text="看新圖片", font=self.ftBtn, command=self.restart_window).grid(row=2, column=1)
            #如果不是輸入多張圖片或是讀到多張圖的最後一張，就把nextBtn鎖起來(state=tk.DISABLED)
            if is_multi:
                nextBtn = tk.Button(self.tl, text="看下一張", font=self.ftBtn, command=self.cnt_next).grid(row=2, column=2)
            else:
                nextBtn = tk.Button(self.tl, text="看下一張", font=self.ftBtn, command=self.cnt_next, state=tk.DISABLED).grid(row=2, column=2)
            self.tl.mainloop()

    def close_window(self):
        self.tl.destroy()#關閉Toplevel()視窗
        self.root.destroy()#關閉tk()視窗

    def restart_window(self):#重開UI介面
        self.close_window()
        newRoot = tk.Tk()
        self.__init__(newRoot)

    def start_testing(self):
        tmp = Predict_Age(self.path)#進入class Predict_Age，讀入model預測年齡
        self.age = tmp.get_age()#回傳預測年齡範圍

if __name__ == '__main__':
    root = tk.Tk()#建立Tk()物件
    app = Interface(root)#進入class Interface的init
    root.mainloop()#讓視窗持續顯示
