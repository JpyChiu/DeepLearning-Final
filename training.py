from __future__ import print_function, division
from keras.layers import Input, Dense, Flatten, Dropout
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam
import datetime
import numpy as np
from dataloader import DataLoader
from sklearn.metrics import accuracy_score
import csv
import cv2
#import pandas as pd

class Where_is_Wally():
    def __init__(self):
        # Input shape
        self.img_rows = 224
        self.img_cols = 224
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Configure data loader
<<<<<<< HEAD
        self.train_dataset_name = './bbox_train.csv' #已改 使用0~500張的train.jpg
        self.val_dataset_name = './bbox_val.csv' #已改 使用501~1000張的train.jpg
=======
        self.train_dataset_name = './dataset/bbox_train.csv' #已改 使用0~100張的train.jpg
        self.val_dataset_name = './dataset/bbox_val.csv' #已改 使用101~200張的train.jpg
>>>>>>> 1f7a6cf2db11390187b5cb418888d3d92dd867e5
        self.data_loader =DataLoader(train_dataset=self.train_dataset_name,val_dataset=self.val_dataset_name,img_res=(self.img_rows, self.img_cols))
        
        # Build the network
        optimizer = Adam(lr=0.002, beta_1=0.9, beta_2=0.999)
        self.CNN_Network = self.build_CNN_Network()
        self.CNN_Network.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # Input images
        Xtr = Input(shape=self.img_shape)

        # Output labels
        Ypred = self.CNN_Network(Xtr)

    def build_CNN_Network(self):
        def conv2d(layer_input, filters, f_size=4, stride=2, bn=True):
            d = Conv2D(filters, kernel_size=f_size, strides=stride, padding='valid')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d


        def maxpooling2d(layer_input, f_size, stride=2):
            d = MaxPooling2D(pool_size=f_size, strides=stride, padding='valid')(layer_input)
            return d

        def flatten(layer_input):
            d = Flatten()(layer_input)
            return d

        def dense(layer_input, f_size, dr=True, lastLayer=True):
            if lastLayer:
                d = Dense(f_size, activation='softmax')(layer_input)
            else:
                d = Dense(f_size, activation='linear')(layer_input)
                d = LeakyReLU(alpha=0.2)(d)
                if dr:
                    d = Dropout(0.5)(d)
            return d

        # LeNet-5 layers #224*224    6*6
        d0 = Input(shape=self.img_shape)  # Image input
        d1 = conv2d(d0, filters=6, f_size=5, stride=1, bn=True) #卷積
        d2 = maxpooling2d(d1, f_size=2, stride=2)   #可能是2x2範圍內取最大值 
        d3 = conv2d(d2, filters=16, f_size=5, stride=1, bn=True)
        d4 = maxpooling2d(d3, f_size=2, stride=2)
        d5 = flatten(d4)
        d6 = dense(d5, f_size=120, dr=True, lastLayer=False)
        d7 = dense(d6, f_size=84, dr=True, lastLayer=False)
        d8 = dense(d7, f_size=10, dr=False, lastLayer=True) #由於有10種class 所以改成10
        return Model(d0, d8)

    def train(self, epochs, batch_size=1, sample_interval=50):
        start_time = datetime.datetime.now()

        for epoch in range(epochs):
            for batch_i, (Xtr, labels) in enumerate(self.data_loader.load_batch(batch_size)):
                # Training
                crossentropy_loss = self.CNN_Network.train_on_batch(Xtr, labels)

                elapsed_time = datetime.datetime.now() - start_time

                # Plot the progress
                print("[Epoch %d/%d] [Batch %d/%d] [Training loss: %f, Training acc: %3d%%] time: %s" % (
                    epoch + 1, epochs,
                    batch_i + 1, self.data_loader.n_batches - 1,
                    crossentropy_loss[0], 100 * crossentropy_loss[1],
                    elapsed_time))
                
                print(batch_i)

                # If at save interval => do validation and save model
                if (batch_i + 1) % sample_interval == 124:
                    self.validation(epoch)

    def validation(self, epoch):
        Xte, Xte_labels = self.data_loader.load_data(batch_size=1024)

        pred_labels = self.CNN_Network.predict(Xte)
        print("Validation acc: " + str(
            int(accuracy_score(np.argmax(Xte_labels, axis=1), np.argmax(pred_labels, axis=1)) * 100)) + "%")
        self.CNN_Network.save('./test07.h5')#修正 改成存model
        
if __name__ == '__main__':
    #     '''training model'''
    my_CNN_Model = Where_is_Wally()
    my_CNN_Model.train(epochs=20, batch_size=4, sample_interval=1266)