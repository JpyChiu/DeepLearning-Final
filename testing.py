from __future__ import print_function, division
import sys
sys.path.append("..")
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


class Where_is_Wally():
    def __init__(self):
        # Input shape
        self.img_rows = 224
        self.img_cols = 224
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)


if __name__ == '__main__':
    '''testing model'''
    imgs, labels = [], []
    #my_CNN = Where_is_Wally()
    #my_CNN_Model = my_CNN.build_CNN_Network()
    my_CNN_Model = load_model('./test07.h5')


    #demo的檔案假設在input.in中
    file = open('./input.in', 'r')
    #讀入檔名
    dataset_name = file.readline().rstrip()
    ##這裡應該將dataset_name讀成圖片再轉成二維矩陣 丟入model.predict
    #使用cv2讀圖片 一維list:內容為像素
    #使用append 而不用=是因為要讓np array 的shape 維度 = 4
    test_data = []
    #for a in enumerate(dataset_name):

    test_data.append(cv2.imread(dataset_name))
    #轉成np.array 得到左上到右下每一點的像素(R,G,B)   再把(R,G,B)分別除以127.5-1 得到三個小數
    test_data = np.array(test_data) / 127.5 - 1.
    #丟入model.predict 由於假設丟入一張圖片 
    #所以predict_label 大概會是一維矩陣塞入一堆奇怪的數字:
    ##index: 0  1  2  3  4  5  6  7        8         9
    ##       !  @  #  $  %  %  ^  ^ 最大數(我們要的)  *   
    predict_label = my_CNN_Model.predict(test_data)
    ##接著使用predict_label判斷年齡層
    #因此用np.argmax(pred_labels, axis=1)抓到最大數字的index 等於年齡層
    #年齡層
    #0:1~10
    #1:11~20
    #2:21~30
    #3:31~40
    #4:41~50
    #5:51~60
    #6:61~70
    #7:71~80
    #8:81~90
    #9:91~100

    aged_range = np.argmax(predict_label, axis=1)
    print("We guess you are %d ~ %d years old" % ((aged_range* 10 + 1),(aged_range + 1) * 10) )
<<<<<<< HEAD
=======

    ###接下來給demo平台使用這個結果去做事
    ###交給林韶恩...
    
    
    
>>>>>>> 1f7a6cf2db11390187b5cb418888d3d92dd867e5

    ###接下來給demo平台使用這個結果去做事
    ###交給林韶恩...
