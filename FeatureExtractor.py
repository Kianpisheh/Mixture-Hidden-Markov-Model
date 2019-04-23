import numpy as np
import pandas as pd
from scipy.fftpack import fftshift
import matplotlib.pyplot as plt
from scipy import stats
import copy

#TODO
# spectral features
# pairwise correlations (intra ans inter sensors)
# autocorrlation
# averaged derivatives
# specttogram
# entropy features

"""
get the sample number and convert it into the time window 
you need to pick

extract feature vector for each sensor

output?
"""

class FeatureExtractor:
    """Extract features of the given signal"""

    def __init__(self):
        self._winsize = 0
        self._overlap = 0.5
        self._sensor_to_features = {"linear Acceleration": ["stats", "freq"]}#,
                                    # "mag-akm09911": ["stats", "freq"],
                                    # "orientation": ["stats", "freq"],
                                    # "light-bh1745": ["stats", "freq"],
                                    # "proximity-pa224": ["raw"],
                                    # "gyroscope-icm20690": ["stats", "freq"],
                                    # "gravity": ["stats", "freq"],
                                    # "accelerometer-icm20690": ["stats", "freq"],
                                    # "rotation Vector": ["stats", "freq"],
                                    # "uncalibrated Magnetic Field": ["stats", "freq"],
                                    # "game Rotation Vector": ["stats", "freq"],
                                    # "uncalibrated Gyroscope": ["stats", "freq"],
                                    # "step Detector": ["raw"],
                                    # "step counter": ["raw"],
                                    # "geomagnetic Rotation": ["stats", "freq"]}

    def sampling_freq(self, x):
        """x: a numpy array of signal timestamps"""
        return np.diff(x), np.diff(x).mean()

    def set_sampler(self, win_size, overlap):
        self._winsize = win_size
        self._overlap = overlap

    def set_sampling_freq(self, dict_fs):
        self._fs = copy.deepcopy(dict_fs)
        
    def sample(self, data, fs, i):
        #TODO: what if the signal is not available?
        # only sample what we want
        """
        Args:
            data: dictionary of dataframes or a dataframe
            fs: dictionary of sensors sampling frequency
            i: the sample number

        Returns:
            features: a dictionary of dataframes samples or
                    a dataframe sample
        """

        if not isinstance(data, dict):
            if not isinstance(data, pd.DataFrame):
                print("The sample type should be either dataframe or dictionary of dataframes")
                return None
            else:
                dict_ = {data.columns[1].split("_")[0]: data}
                data = dict_

        samples = {}
        for key, df in data.items():
            if not (key in self._sensor_to_features):
                continue
            start_time = df["timestamp"].iloc[0]
            df = df.set_index("timestamp")
            start = start_time + self._winsize * 1000 * i * (1-self._overlap)
            end = start + self._winsize * 1000
            start_idx = df.index.get_loc(key=start, method="nearest", tolerance=2000)
            end_idx = df.index.get_loc(key=end, method="nearest", tolerance=2000)
            end_idx += 1 if start_idx == end_idx else end_idx
            samples[key] = df.iloc[start_idx:end_idx] if end_idx >= start_idx else None
        return samples

    def extract_features(self, samples, rt_type="flat"):
        """ Extratc features of the given sample.

            Args: 
                sample: a dictionary of dataframes or a dataframe
            Returns:
                a numpy array of concatenated features or a dictionary 
                of numpy arrays
        """

        if not isinstance(samples, dict):
            if not isinstance(samples, pd.DataFrame):
                print("The sample type should be either dataframe or dictionary of dataframes")
                return None
            else:
                dict_ = {samples.columns[1].split("_")[0]: samples}
                samples = dict_

        features = {}
        for key, df in samples.items():
            feature_vector = np.array([])
            if not (key in self._sensor_to_features): # not interested in this sensor
                continue
            if "stats" in self._sensor_to_features[key]:
                feature_vector = np.append(feature_vector, self.stats_features(df.values))
            if "freq" in self._sensor_to_features[key]:
                feature_vector = np.append(feature_vector, self.spectral_features((df.values), key))
            if "raw" in self._sensor_to_features[key]:
                feature_vector = np.append(feature_vector, df.values.mean(axis=0))
            features[key] = feature_vector

        if rt_type == "flat":
            output = np.array([])
            for feature in features.values():
                output = np.append(output, feature.reshape(-1,))
            return output
        return features

    def stats_features(self, x):
        """
        Args:
            x: a numpy array of size m x ch 
                (ch: num of sensors channel, m: num of sample points)
        Returns: 
            a numpy array of time-domain features
        """

        x = x.reshape(1,-1) if x.ndim == 1 else x

        stats_fe = np.mean(x, axis=0).reshape(1,-1)
        stats_fe = np.concatenate((stats_fe, np.std(x, axis=0).reshape(1,-1)), axis=0)
        stats_fe = np.concatenate((stats_fe, np.median(x, axis=0).reshape(1,-1)), axis=0)
        stats_fe = np.concatenate((stats_fe, stats.skew(x, axis=0).reshape(1,-1)), axis=0)
        stats_fe = np.concatenate((stats_fe, stats.kurtosis(x, axis=0).reshape(1,-1)), axis=0)
        stats_fe = np.concatenate((stats_fe, np.quantile(x, .25, axis=0).reshape(1,-1)), axis=0)
        stats_fe = np.concatenate((stats_fe, np.quantile(x, .5, axis=0).reshape(1,-1)), axis=0)
        stats_fe = np.concatenate((stats_fe, np.quantile(x, .75, axis=0).reshape(1,-1)), axis=0)
        stats_fe = np.concatenate((stats_fe, self.__zero_crossings(x, ax=0).reshape(1,-1)), axis=0)
        stats_fe = np.concatenate((stats_fe, self.__mean_crossings(x, ax=0).reshape(1,-1)), axis=0)
        return stats_fe

    def spectral_features(self, x, key):
        """
        Return a set of features in freq domain.

        Args:
            x: a numpy array of size m x ch 
                (ch: num of sensors channel, m: num of sample points) 
        Returns:
            ft: fft of the input signal (numpy array)
        """

        # convert the input into a (num_sample x num_ch) numpy array
        x = x.reshape(1,-1) if x.ndim == 1 else x

        #self.__log_energy_band(x)
        ft, freq = self.__calc_fft(x, key)
        freq_fe = self.__spectral_energy(ft, ax=0)
        freq_fe = np.concatenate((freq_fe, x.mean(axis=0).reshape(1,-1)), axis=0)
        freq_fe = np.concatenate((freq_fe, x.var(axis=0).reshape(1,-1)), axis=0)
        return freq_fe

    # Helper methods

    def __calc_fft(self, x, key):
        N = x.shape[0]
        ft = np.absolute(np.fft.fft(x))[:int(N/2)]
        freq = np.arange(0, self._fs[key], self._fs[key] / N)[:int(N/2)]
        return ft, freq

    def __pick_numeric_features(self, data):
        non_numeric_sensors = ["wifi_log", "battery_log", "location_log", "step counter",
                                "step Detector", "HALL sensor", "tilt dector"]
        output = data.copy()
        for key in data:
            if key in non_numeric_sensors:
                output.pop(key)
        return output

    def __spectral_energy(self, ft, ax=0):
        return np.linalg.norm(ft, axis=ax).reshape(1,-1) / ft.shape[0] 

    def __zero_crossings(self, x, ax=0):
        return np.sum((np.diff(np.sign(x), axis=ax) != 0), axis=0)

    def __mean_crossings(self, x, ax=0):
        return np.sum((np.diff(np.sign(x-x.mean()), axis=ax) != 0), axis=0)




    