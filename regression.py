from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import csv
import numpy as np
i=0
epochs=1000;
x_train=np.full((699,1),0)
y_train=np.full((699,1),0)
y_train=y_train.astype('float64')
with open('train.csv') as csvfile:
 reader = csv.DictReader(csvfile)
 for row in reader:
  x_train[i]=int(row['x'])
  y_train[i]=np.float64(row['y'])
  print(y_train[i])
  i=i+1;
model=Sequential()
model.add(Dense(4,input_dim=1,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(1,activation='linear'))
model.summary()
model.compile(loss='mse',optimizer='adam')
model.fit(x_train[:550],y_train[:550],epochs=epochs,verbose=1,validation_data=(x_train[550:], y_train[550:]))
y_pred=model.predict(x_train[4])
print(y_train[4])
print(y_pred)

