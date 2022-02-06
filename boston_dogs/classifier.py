# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import re
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPool2D,Dropout,Flatten,Activation
from keras.preprocessing.image import load_img,img_to_array, ImageDataGenerator
import os
from sklearn.model_selection import train_test_split

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

y= pd.read_csv('../../labels_new.csv')
y.head()

print(y.breed.unique())
print("\n")
print(y.breed.nunique())
print("\n")
print(y.isna().sum())
print("\n")
print(y.breed.value_counts())
print("\n")

H=128
W=128
C=3

train_file_location = '../../train/'
train_data = y.assign(img_path = lambda x : train_file_location + x['id'] + '.jpg')
train_data.head()

X = np.array([img_to_array(load_img(img,target_size = (H,W))) for img in train_data['img_path'].values.tolist()])
print(X.shape)
Y = pd.get_dummies(train_data['breed'])
print(Y.shape)

X_train,X_test,Y_train,Y_test  = train_test_split(X,Y,test_size = 0.25)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

model=Sequential()

model.add(Conv2D(32,(3,3),input_shape=(H,W,C)))
model.add(Activation('relu'))

model.add(MaxPool2D((2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))

model.add(MaxPool2D((2,2)))

model.add(Conv2D(128,(3,3)))
model.add(Activation('relu'))

model.add(MaxPool2D((2,2)))

model.add(Conv2D(128,(3,3)))
model.add(Activation('relu'))

model.add(MaxPool2D((2,2)))

model.add(Flatten())

model.add(Dropout(0.5))

model.add(Dense(512))
model.add(Activation('relu'))

model.add(Dense(Y.shape[1]))
model.add(Activation('softmax'))

model.summary()

batch=32

model.compile(
      optimizer='adam',
      loss='categorical_crossentropy',
      metrics=['accuracy'])

trained_model=model.fit(X_train,Y_train,
         epochs=20,
         batch_size=batch,
         steps_per_epoch=X_train.shape[0]//batch,
         validation_steps=X_test.shape[0]//batch,
         validation_data=(X_test,Y_test),
         verbose=2)



test_datagen = ImageDataGenerator()

test_set = test_datagen.flow_from_directory(
    '../../',
    target_size = (128,128),
    classes=['test'])

y_pred = model.predict(test_set)
print(y_pred)

submission = pd.read_csv('../../sample_submission_new2.csv')
submission.head()

file_list = test_set.filenames
id_list = []
for name in file_list:
    m = re.sub('test/', '', name)
    m = re.sub('.jpg', '', m)
    id_list.append(m)

submission['id'] = file_list
submission.iloc[:,1:] = y_pred
submission.head()

r = submission.set_index('id')
r.to_csv('../../submission.csv')
