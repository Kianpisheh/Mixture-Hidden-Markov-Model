import librosa
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import librosa.display
import sklearn
import numpy as np

class AudioFeatureExtractor:

    def __init__(self):
        self._winsize = 0
        self._overlap = 0.5
        self.audio = None
        self._sr = None

    def read_audio(self, filepath):
        audio, self._sr = librosa.load(filepath, offset=0, sr=None)
        return audio

    def set_sampler(self, win_size, overlap):
        self._winsize = win_size
        self._overlap = overlap

    def sample(self, x, i):
        start = int(self._sr * self._winsize * (i*(1-self._overlap)))
        end = start + int(self._sr * self._winsize)
        return x[start:end]

    def extract_features(self, sample, rt_type="flat", average=False):
        features = {}
        features["zero_cross"] = np.sum(librosa.zero_crossings(sample, pad=False))
        mfccs = librosa.feature.mfcc(sample, sr=self._sr)
        mfccs_mean = mfccs.mean()
        mfccs_std = mfccs.std() 
        #features["mfcc"] = sklearn.preprocessing.scale(mfccs, axis=1)
        features["mfccs_mean"] = mfccs_mean
        features["mfccs_std"]  =mfccs_std
        features["roll_off"] = librosa.feature.spectral_rolloff(sample, sr=self._sr, roll_percent=0.2)
        features["flatness"] = librosa.feature.spectral_flatness(sample)
        features["chroma"] = librosa.feature.chroma_stft(sample, sr=self._sr)

        if average:
            for key, feature in features.items():
                if feature.ndim > 1:
                    features[key] = np.mean(feature, axis=1)

        if rt_type == "list":
            return list(features.values())
        elif rt_type == "dict":
            return features
        elif rt_type == "flat":
            output = np.array([])
            for feature in features.values():
                output = np.append(output, feature.reshape(-1,))
            return output
        return None
            
    def show_chromogram(self, chroma):
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
        plt.colorbar()
        plt.title('Chromagram')
        plt.tight_layout()
        plt.show()

