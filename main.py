import numpy as np
import pandas as pd
import os
import librosa
import audioread
import scipy.io.wavfile
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import librosa.display
import sklearn

from DataManager import DataManager
from AudioFeatureExtractor import AudioFeatureExtractor
from FeatureExtractor import FeatureExtractor
from MHMM import hmm2

# load data
# audio_filename = "../Huawei/18-03-2019-17-01-19/1552928479438_audio_01.wav"
# file_path = "../Huawei/18-03-2019-17-01-19"
# data_manager = DataManager()
# data_dict, fs_dict = data_manager.read_files(file_path)

# set-up feature extractors
# win_size = 0.2 # second
# overlap = 0.5
# sensors_fe = FeatureExtractor()
# sensors_fe.set_sampling_freq(fs_dict)
# sensors_fe.set_sampler(win_size, overlap)
# audio_fe = AudioFeatureExtractor()
# audio_fe.set_sampler(win_size, overlap)
# audio = audio_fe.read_audio(audio_filename)


# sample
# sensors_features = np.array([])
# audio_features = np.array([])
# for i in range(200):
#     if i % 10 == 0:
#         print(i)
#     # sample
#     sensors_sample = sensors_fe.sample(data_dict, fs_dict, i)
#     audio_sample = audio_fe.sample(audio, i)

#     # extract features
#     feature = sensors_fe.extract_features(sensors_sample, rt_type="flat").reshape(1,-1)
#     v = feature.shape[1]
#     sensors_features = np.append(sensors_features, sensors_fe.extract_features(sensors_sample, rt_type="flat")).reshape(-1,v)
#     audio_features = np.append(audio_features, audio_fe.extract_features(audio_sample, rt_type="flat", average=True)).reshape(-1,17)



import numpy as np
from hmmlearn import hmm

model = hmm.GaussianHMM(n_components=3, covariance_type="full")
model.startprob_ = np.array([0.6, 0.3, 0.1])
model.transmat_ = np.array([[0.7, 0.2, 0.1],
                             [0.3, 0.5, 0.2],
                             [0.3, 0.3, 0.4]])
model.means_ = np.array([[0.0], [3.0], [5.0]])
model.covars_ = np.tile(np.identity(1), (3, 1, 1))
X, Z = model.sample(100)

# train
model = hmm2(3)
model.set_emmision_model("gaussian", 1)
print(model.fit(X.reshape(1,-1)))