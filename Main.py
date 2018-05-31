# coding: utf-8
'''
Created on 2018. 5. 25.

@author: Insup Jung
'''

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit #데이터셋에서 일정한 비율로 train과 test set을 나눠준다.

if __name__ == '__main__':
    
    img_dir = "./EOSINOPHIL/_0_5239.jpeg"
    
    img1 = mpimg.imread(img_dir)
    print(type(img1))
    img = Image.open(img_dir)
    print(type(img))
    print(type(np.asarray(img)))
    img.thumbnail((120, 160), Image.ANTIALIAS)
    imgplot = plt.imshow(img)
    plt.show()
    print(np.asarray(img).shape)
    
    
    print(img1.shape, img1.dtype)
    print(img1[30][30][1])
    
    dataset_dir = "./images/TRAIN"
    
    classes = []
    labels = []
    features = []
    
    for root, dirs, files in os.walk(dataset_dir, topdown=False):
        for name in dirs:
            classes.append(name)
    
    
    
    for class_name in classes:
        root_dir = os.path.join(dataset_dir, class_name)
        for name in os.listdir(root_dir):
            img_path = os.path.join(root_dir, name)
            try:
                
                #img_ndarray = mpimg.imread(img_path)
                #features.append(img_ndarray)
                img_ndarray = Image.open(img_path)
                img_ndarray.thumbnail((120,160), Image.ANTIALIAS)
                img_ndarray = np.asarray(img_ndarray)
                features.append(img_ndarray)
                #print(type(img_ndarray))
                
                labels.append(class_name)
            except OSError as e:
                print(img_path)
                print(e)
            except ValueError as e:
                print(img_path)
                print(e)
    
    # 그림의 라벨을 가져오는 Encoder
    le = preprocessing.LabelEncoder()
    le.fit(classes)
    #print(le.classes_) #Labels가 그대로 출력됨
    
    labels_int = le.transform(labels) #
    print(labels_int) #해당 폴더 안에 있는 모든 그림에 label을 가져온다. 폴더 안에 있는 그림에 차례대로
    print(labels_int.shape) #shape이 (9957,)이 나온다.
    
    ohe = preprocessing.OneHotEncoder()
    ohe.fit(labels_int.reshape(-1,1))
    labels_one_hot = ohe.transform(labels_int.reshape(-1,1))
    print(labels_one_hot.toarray().shape)
    print(np.asarray(features).shape) #(9957, 240, 320, 3)이 나온다.
    features = np.asarray(features)
    #features = np.reshape(features, [-1, features.shape[1]*features.shape[2]*features.shape[3]])
    print(features.shape)
    
    data = {}
    data["features"] = np.asarray(features)
    data["labels"] = np.asarray(labels_one_hot.toarray())
    print(type(data["features"]))
    
    print(np.shape(data["features"]))
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    train_set={}
    test_set={}
    i=0
    for train_index, test_index in sss.split(data["features"], data["labels"]):
        print(i)
        i=i+1
        print(train_index)
        print(np.shape(train_index))
        train_set["features"] = data["features"][train_index]
        train_set["labels"] = data["labels"][train_index]
        print(np.shape(train_set["features"]))
        
        test_set["features"] = data["features"][test_index]
        test_set["labels"] = data["labels"][test_index]
    
    print(type(train_set))
    np.save("blood_cell.npy", data)
    np.save("blood_cell_train.npy", train_set)
    np.save("blood_cell_test.npy", test_set)
    
    pass