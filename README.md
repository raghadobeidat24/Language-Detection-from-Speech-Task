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

