# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 13:07:57 2021

@author: raghad
"""

# import required  library
import tensorflow as tf
from tensorflow import keras
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np

from keras.models import load_model

import os
import subprocess
import time
import csv
from flask import Flask
app = Flask(__name__)


@app.route('/language')
def getanguage():
    filepath = '/home/raghad/Desktop/turkish.wav'
    smile = "/home/raghad/Downloads/opensmile-3.0-linux-x64/"

    name = "output"
    outputpath="/home/raghad/Desktop/out/"+name+".csv"
    datapath = "/home/raghad/Desktop/out/test1.csv"
    try:
        if filepath.endswith(".wav"):
            import_command = smile + "bin/SMILExtract -C " + smile + "config/misc/emo_large.conf -I " +filepath+" -O "+outputpath
            command = import_command.split()
            p = subprocess.Popen(command,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT)
            time.sleep(4)


            with open(datapath, 'w') as out:
                data = ""
                with open(outputpath, 'r') as input:
                    for row in reversed(list(csv.reader(input))):
                        data = ','.join(row)
                        print(data)

                        break

                    out.write(data)
                    out.write("\n")

                input.close()


                ## load model
                model = load_model('/home/raghad/Desktop/test_LD/model.h5')
                # summarize model.
                model.summary()

                # load dataset and preprocessing
                dataset = read_csv(datapath, header=None, index_col=0)

                # Handle empty values- if any
                dataset.fillna(dataset.mean(), inplace=True)

                # get the values from the dataframe
                values = dataset.values

                X_test, y = values[:, 1:-1], values[:, -1:]

                X_test = X_test.astype('float32')

                # normalization of features
                # scaler = MinMaxScaler(feature_range=(0, 1))
                # X_test = scaler.fit_transform(X_test)

                test_X = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
                # test_X = X_test.reshape((X_test.shape))

                print(test_X.shape)

                # Let's check:
                result=""
                Dict = {"0": "Arabic", "1": "Spanish", "2": "French", "3": "Turkish"}
                pred_y = model.predict(test_X)
                y_pred = np.argmax(pred_y, axis=1)
                print(Dict[str(y_pred[0])], pred_y[0][y_pred[0]])
                result=Dict[str(y_pred[0])]+"   "+str( pred_y[0][y_pred[0]])




        else:
            return "your file must end with .wav "
    except:
        return "Exception happened while extracting audio features"



    return result

if __name__ == '__main__':
        app.run(host="0.0.0.0")