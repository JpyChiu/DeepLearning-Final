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
    my_CNN = Where_is_Wally()
    my_CNN_Model = my_CNN.build_CNN_Network()
    my_CNN_Model.load_models('../test07.h5')  
    
    
    #demo的檔案假設在input.in中
    file = open('./input.in', 'r')
    #讀入檔名
    dataset_name = file.readline().rstrip()
    file.close()
    ##這裡應該將dataset_name讀成圖片再轉成二維矩陣 丟入model.predict
    #使用cv2讀圖片 一維list:內容為像素
    test_data = cv2.imread(dataset_name)
    #轉成np.array
    test_data = np.array(test_data) / 127.5 - 1.
    #丟入model.predict
    predict_label = my_CNN_Model.predict(test_data)
    ##接著使用predict_label判斷年齡層
    ###留給謝宗倫
    
    ###
    
    
    

