# Language-Detection-from-Speech-Task
Spoken Language detection has many applications in speech recognition, multilingual machine translations. This project will try to classify four languages (Arabic, Spanish, French, and Turkish) from the spoken utterances.
 We will implement two models LSTM, MLP models, and train them to classify the languages using Keras. We will use OPENSMILE (open-source tool) for features extraction, this tool can extract large features spaces from different types of audio files.
 
 # Dataset
 The given dataset is a media speech dataset of four languages (Arabic, Spanish, French, and Turkish), which is built with the purpose of testing Automated Speech Recognition (ASR) systems performance.  
The dataset consists of short speech segments automatically extracted from media videos available on YouTube and manually transcribed, with some pre- and post-processing, The dataset contains 10 hours of speech for each language provided, it has 10023 samples of (.wav) audio records.
The dataset is divided into 2 directories:
•	train (6715 samples)
•	test (3308 samples)

The dataset paper https://arxiv.org/abs/2103.16193  and it  can be found in the following git repository: https://github.com/NTRLab/MediaSpeech

# Exploratory Data Analysis
Tableau and Excel have been used for data visualization. Additionally, we have used ffmpeg (command-line tool)/ wave python library to get audio length (duration) for each language.
(Gender and age for the speaker were not provided by the dataset publishers).
The following figures represent utterances per class (language). We can see that we have almost 2500 sample per language, our dataset is perfectly balanced. This is extremely important for classification since if we have an imbalanced dataset i.e.(we have a lot more samples in a language than the others) our model will be biased and won’t work as well.


![Sheet 3](https://user-images.githubusercontent.com/87562803/126233804-e84e9fb3-dbd0-4711-8078-676f67aa53ae.png)
