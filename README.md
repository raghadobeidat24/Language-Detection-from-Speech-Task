# Language-Detection-from-Speech-Task
Spoken Language detection has many applications in speech recognition, multilingual machine translations. This project will try to classify four languages (Arabic, Spanish, French, and Turkish) from the spoken utterances.
 We will implement two models LSTM, MLP models, and train them to classify the languages using Keras. We will use OPENSMILE (open-source tool) for features extraction, this tool can extract large features spaces from different types of audio files.
 
 # Dataset
 The given dataset is a media speech dataset of four languages (Arabic, Spanish, French, and Turkish), which is built with the purpose of testing Automated Speech Recognition (ASR) systems performance.  
The dataset consists of short speech segments automatically extracted from media videos available on YouTube and manually transcribed, with some pre- and post-processing, The dataset contains 10 hours of speech for each language provided, it has 10023 samples of (.wav) audio records.

The dataset is divided into 2 sets:

•	train (6715 samples)

•	test (3308 samples)

The dataset paper https://arxiv.org/abs/2103.16193  and it  can be found in the following git repository: https://github.com/NTRLab/MediaSpeech

# Exploratory Data Analysis
Tableau and Excel have been used for data visualization. Additionally, we have used ffmpeg (command-line tool)/ wave python library to get audio length (duration) for each language.

(Gender and age for the speaker were not provided by the dataset publishers).
The following figures represent utterances per class (language).

We can see that we have almost 2500 sample per language, our dataset is perfectly balanced. This is extremely important for classification since if we have an imbalanced dataset i.e.(we have a lot more samples in a language than the others) our model will be biased and won’t work as well.


![Sheet 3](https://user-images.githubusercontent.com/87562803/126233804-e84e9fb3-dbd0-4711-8078-676f67aa53ae.png)


![chart2](https://user-images.githubusercontent.com/87562803/126233914-5d4ee08b-10d3-4de7-ad42-0a448be72917.PNG)


The following figure represents lengths of the utterances. We can see from the figure the audio files have 14 seconds duration for most samples.

![duration](https://user-images.githubusercontent.com/87562803/126688555-f7ce9111-f043-4b7f-8a0d-56b019ba83de.PNG)


# Feature Extraction 

We have used OpenSMILE - https://www.audeering.com/opensmile/-
A trending tool used by many researchers and companies in different fields (speech recognition, emotion recognition, and music information), it can extract large features spaces from different types of audio files or in real-time., it works using command-line under Linux, Mac, and windows.

It contains several configuration files that extract prosodic features, PLP features, MFCC features and Chroma features.
In this project we have extracted 6551 features using emo_large configuration(config/emo large.conf). The output is a csv file features for every audio file.


The dataset voice featrues are available on google drive link https://drive.google.com/file/d/1EaSHd1nUxPykW5l1d4K4OTxzPOWiJZsc/view?usp=sharing

I did not upload it on repository due to size of the file

# Models to detect the language of the speaker

We have implemented two models: Long Short-Term Memory (LSTM) and Multilayer Perceptron’s (MLP) models to train our data to classify the languages using Keras

LSTM model is well-suited to classify, process and predict time series given time lags of unknown duration. This includes both sequences of text and sequences of spoken language represented as a time series.

MLP model is the classical type of neural network and it is suitable for classification prediction problems where inputs are assigned a class or label.

The following two tables presents the results of the implemented models with different measurements (Accuracy, Precision, Recall, F-score)

![lstmacc](https://user-images.githubusercontent.com/87562803/126688324-c8f8e4c7-5205-4a07-8ea2-b79385f4fb3f.PNG)

![mlpacc](https://user-images.githubusercontent.com/87562803/126689079-0357f8fd-bf42-4b77-ae2b-9e3907948b4c.PNG)


We can see that the LSTM increases the accuracy by 1.05%. the LSTM model performs better than MLP especially in the precision value for class1 0.97, in the recall value of class2 0.97and in f-score value of class2 0.96
