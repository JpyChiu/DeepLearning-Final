import numpy as np
import scipy.misc
import csv
import cv2
from PIL import Image
# from glob import glob

class DataLoader():
    def __init__(self, train_dataset, val_dataset, img_res=96):
        #訓練集檔名
        self.train_dataset_name = train_dataset
        #測試集檔名
        self.val_dataset_name = val_dataset
        self.img_res = img_res
        self.train_data = []
        self.train_labels = []
        self.test_data = []
        self.test_labels = []
        
        # 開啟csv檔案 存取訓練集
        with open(self.train_dataset_name, newline='') as csvfile:
            # 讀取 CSV 檔案內容
            rows = csv.reader(csvfile)
            for a, b in zip(rows, range(0, 500)):  ##假設是list 第一列的資料就是 檔名 labels
                #採用cv2 讀取像素 附加到list尾端 形成一維list    
                self.train_data.append(cv2.imread(a[0]))#100張圖片的像素 依序存進list                             
                self.train_labels.append(int(a[1]))
                if b == 499:
                    break
              
             
        #將測試集的一維list存成np.array       
        self.train_data = np.array(self.train_data)
        self.train_labels = np.array(self.train_labels)
        #訓練集end
    
        # 開啟csv檔案 存取測試集
        with open(self.val_dataset_name, newline='') as csvfile:
            # 讀取 CSV 檔案內容
            rows = csv.reader(csvfile)
            for a, b in zip(rows, range(0, 500)):
                #採用cv2 讀取像素 附加到list尾端 形成一維list    
                self.test_data.append(cv2.imread(a[0]))             
                self.test_labels.append(int(a[1]))
                if b == 499:
                    break         
               
        #將測試集的list存成np.array         
        self.test_data = np.array(self.test_data)
        self.test_labels = np.array(self.test_labels)
        #測試集end
        
    #把圖片跟年齡都放好  回傳兩者   
    def load_batch(self, batch_size=1):#假設讀500張圖片
        #n_batches = 100 / 4 = 125
        self.n_batches = int(len(self.train_data) / batch_size)
        
        for i in range(self.n_batches - 1):#0 ~ 124        
            batch = self.train_data[i * batch_size:(i + 1) * batch_size]#[0 * 4 : 1 * 4]  [0:4] = 取第0, 1, 2, 3項   #第0個第4個colcol取到
            labels = self.train_labels[i * batch_size:(i + 1) * batch_size]#與batch 相同 一次取batch_size數的項

            batch_label = []
            for label in labels:#4個
                label_one_hot_encodes = self.one_hot_encode(label, num_classes=10)#改num_classes=10
                batch_label.append(label_one_hot_encodes)
            
            Xtr_label = np.array(batch_label)
            Xtr = np.array(batch) / 127.5 - 1.
            
            yield Xtr, Xtr_label#修正縮排
    
    #把圖片放好 回傳圖片 
    def load_data(self, batch_size=1):
        #np.random.rand(batch_size))隨機數應該是浮點數 所以要強制轉成int
        #indices = 維度(10) * 0~4
        indices = (len(self.test_labels) * np.random.rand(batch_size)).astype(int)
        #取第indices個維度的所有值 意思是隨機從batch_size(=4)中抓一個出來測
        batch_images = self.test_data[indices, :]
        Xte = np.array(batch_images) / 127.5 - 1.
        
        batch_label = []
        for label in self.test_labels[indices]:
            label_one_hot_encodes = self.one_hot_encode(label, num_classes=10)#改num_classes=10
            batch_label.append(label_one_hot_encodes)#修正縮排
        
        Xte_label = np.array(batch_label)
        
        return Xte, Xte_label

    def one_hot_encode(self, y, num_classes=10):#改num_classes=10
        return np.squeeze(np.eye(num_classes)[y.reshape(-1)]) #分成4個[0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]這樣的形式 #np.eye(N) NxN對角矩陣
