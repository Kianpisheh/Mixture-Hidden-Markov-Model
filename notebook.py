#%%
import numpy as np
import pandas as pd
import os


class DataManager:

    def __init__(self):
        self.times = np.array([])
        self.times_idx = np.array([])
        self.data = None
        self.current_values = np.array([])
        self.prev_time = 0
        self.sensor_pointers = {}

    def __set_times(self, times, times_idx):
        self.times = np.array(times)
        self.times_idx = np.array(times_idx)

    def __sort_times(self):
        idx = self.times.argsort()
        self.times.sort()
        self.times_idx = self.times_idx[idx]

    def __update_dataframe(self, sensor_idx: str, data_dict, sensor_df):
        # pick the relevant sensor dataframe
        sensor_df = data_dict[sensor_idx]
        # pick the relevant sensor sample
        row = self.sensor_pointers[sensor_idx]
        sub_series = sensor_df.iloc[row]
        # set relevant sensor data
        for key, value in sub_series.iteritems():
            (self.data.iloc[-1])[key] = value

    def __append_to_dataframe(self, sensor_idx: str, data_dict, time, sensor_df):
        # create an empty series
        sample_series = pd.Series(np.nan, index=column_headers)
        # set timestamp
        sample_series["timestamp"] = time
        # pick the relevant sensor dataframe
        sensor_df = data_dict[sensor_idx]
        # pick the relevant sensor sample
        row = self.sensor_pointers[sensor_idx]
        sub_series = sensor_df.iloc[row]
        self.sensor_pointers[sensor_idx] += 1
        # set relevant sensor data
        for key, value in sub_series.iteritems():
            sample_series[key] = value
        # update the unified dataframe
        self.data = self.data.append(sample_series, ignore_index=True)


    def read_data_files(self, file_path: str):
        """
        file_path: directory of csv files
        return: a dictionary of dataframes with file names keys
        """
        file_names = os.listdir(files_path)[1:]
        # read sensor data
        data = {}
        for file_name in file_names:
            file_dir = files_path + "/" + file_name
            if not file_name.equal("label") and not non_empty_file(file_dir) and \
                (file_name.endswith(".csv") or (file_name.endswith(".txt"))):
                df = pd.read_csv(file_dir, header=None, delimiter=',')
                if not file_name.endswith(".txt"):
                    df = (df.iloc[:,1:]).copy()
                    df.drop(df.columns[-1], inplace=True, axis=1)
                if df.shape[1] > 0:
                    column_headers = ["timestamp"]
                    file_name = file_name.split(".")[0]
                    file_name = "_".join(file_name.split("_")[1:])
                    for i in range(df.shape[1]-1): 
                        column_headers.append(file_name + "_" + str(i))
                    data[file_name] = df
                    data.get(file_name).columns = column_headers
        return data


    def create_unified_data(self, data_dict):
        column_headers = ["timestamp"]
        # pick timestamps of all the sensors
        num_cols = 0
        times, times_idx = [], []
        for _, key in enumerate(data_dict.keys()):
            df = data_dict[key]
            ts_size = df["timestamp"].values.size 
            times.extend(df["timestamp"].values)
            times_idx.extend([key]*ts_size)
            num_cols += df.shape[1] - 1
            column_headers.extend(df.columns[1:])
            self.sensor_pointers[key] = 0
        num_cols += 1
        self.__set_times(times, times_idx)
        self.__sort_times()
        # create the unified dataframe
        self.data = pd.DataFrame({}, columns=column_headers)     
        prev_time = 0
        i = 0
        for time, sensor_idx in zip(self.times, self.times_idx):
            pass

           
        self.data.to_csv("./unified_data.csv", index=False)
        
def non_empty_file(file_path):
    return False if os.path.isfile(file_path) and \
        os.path.getsize(file_path) > 0 else True



#%%
files_path = "./Huawei/18-03-2019-17-01-19"
file_names = os.listdir(files_path)[1:]

#%%
# read sensor data
    

#%%
k = list(data.keys())


#%% 
sensorData = SensorData()
sensorData.create_unified_data(data)


#%%



