# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 21:19:28 2021

@author: raghad
"""
# import required  library 
import tensorflow as tf
from tensorflow import keras
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from math import sqrt
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Bidirectional, TimeDistributed,Dropout
#from sklearn.metrics import  classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.utils import np_utils


# load dataset and preprocessing 
dataset = read_csv("voiceData.csv", header=0, index_col=0)

# Handle empty values- if any 
dataset.fillna(dataset.mean(), inplace=True)

# get the values from the dataframe
values = dataset.values

# integer encode direction for string attribute- if any
#encoder = LabelEncoder()
#values[:,4] = encoder.fit_transform(values[:,4])

# ensure all values are float
#values = values.astype('float32')

# Handle Inf values- if any
#values[values >= 1E308] = 0


X, y = values[:, :-1],  values[:, -1:]
X = X.astype('float32')

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
y = np_utils.to_categorical(y)


# # normalization of features
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, shuffle=True, random_state=42)




train_X = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))

test_X = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
print(train_X.shape, y_train.shape, test_X.shape, y_test.shape)


model = Sequential()
model.add (LSTM(512, input_shape=(train_X.shape[1], train_X.shape[2])))

model.add(Dropout(0.02))

model.add(Dense(4, activation='softmax'))
#model.compile(loss='accuracy', optimizer='adam')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# train the model
history = model.fit(train_X, y_train, epochs=100, batch_size=500,validation_split=0.2, verbose=2)

#saving the model 
model.save("model.h5")
#plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()


pred_y = model.predict(test_X)
y_pred=np.argmax(pred_y, axis=1)
y_test=np.argmax(y_test, axis=1)

from sklearn.metrics import classification_report
print('\nClassification Report\n')
print(classification_report(y_test, y_pred, target_names=['Class 1', 'Class 2', 'Class 3', 'Class 4']))



