import numpy as np
import pandas as pd
import os

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


class DataManager:

    def __init__(self):
        self.times = np.array([])
        self.times_idx = np.array([])
        self.data = None
        self.column_headers = None
        self.current_values = np.array([])
        self.sensor_pointers = {}
        self.sensor_list = []

    def __update_dataframe(self, sensor_idx: str, data_dict, time):
        sensor_df = data_dict[sensor_idx]
        row = self.sensor_pointers[sensor_idx]
        sub_series = sensor_df.iloc[row]
        
        # add a new sample to the df 
        if (time != self.data.iloc[-1]["timestamp"]) or (self.data.values.size == 0):
            sample_series = pd.Series(np.nan, index=self.column_headers)
            sample_series["timestamp"] = time
            self.sensor_pointers[sensor_idx] += 1
            for key, value in sub_series.iteritems():
                sample_series[key] = value
            self.data = self.data.append(sample_series, ignore_index=True)     

        # update the last sample (row) in df
        else:
            for key, value in sub_series.iteritems():
                (self.data.iloc[-1])[key] = value
           
    def __non_empty_file(self, file_path):
        return False if os.path.isfile(file_path) and \
            os.path.getsize(file_path) > 0 else True

    def read_files(self, file_path: str):
        """
        file_path: directory of csv files
        return: a dictionary of dataframes with file names keys
        """
        file_names = os.listdir(file_path)[1:]
        # read sensor data
        data, fs = {}, {}
        for file_name in file_names:
            file_dir = file_path + "/" + file_name
            if (not file_name.endswith("label.txt")) and (not self.__non_empty_file(file_dir)) and \
                (file_name.endswith(".csv") or (file_name.endswith(".txt"))):
                df = pd.read_csv(file_dir, header=None, delimiter=',')
                if not file_name.endswith(".txt"):
                    file_name = file_name.split(".")[0]
                    self.sensor_list.extend(file_name.split("_")[1:])
                    file_name = "_".join(file_name.split("_")[1:])
                    fs[file_name] = np.diff((df.iloc[:,0]).values).mean()
                    df = (df.iloc[:,1:]).copy()
                    df.drop(df.columns[-1], inplace=True, axis=1)
                if df.shape[1] > 0:
                    column_headers = ["timestamp"]
                    for i in range(df.shape[1]-1): 
                        column_headers.append(file_name + "_" + str(i))
                    data[file_name] = df
                    data.get(file_name).columns = column_headers
        return data, fs

    def unified_data(self, data_dict, merge_time=5, save=False):
        # find union of timestamps
        idx = pd.DataFrame({}).index
        for df in data_dict.values():
            idx = idx.union(df.set_index("timestamp").index)
        # downsample the final timestamp
        ts= pd.Series(idx, index=pd.to_timedelta(idx, "ms"))
        down_sampled_ts = ts.resample(str(merge_time) + "ms").max()
        # set the unified dataframe
        self.data = pd.DataFrame({})
        self.data["timestamp"] = down_sampled_ts.index / np.timedelta64('1', 'ms')
        # combine sensor data
        for df in data_dict.values():
            df["timestamp"] = pd.to_timedelta(df["timestamp"], "ms") / np.timedelta64('1', 'ms')
            self.data = pd.merge_asof(self.data, df, on="timestamp", tolerance=pd.to_timedelta(1, "ms") / np.timedelta64('1', 'ms'))
        if save:
            self.data.to_csv("./unified_data.csv", index=True)
        return self.data


    def plot_feature(self, data, feature_name=None, activity=None):
        """
        it gets the dataframe "data" and plots the feature_name data against 
        timestamp.
        """
        
        if data is None:
            print("No data tp draw")
            return 

        if feature_name is None:
            feature_name = data.columns[1]

        if not activity:
            data.plot(kind="line", x="timestamp", y=feature_name)
        else:
            activity = activity.upper()
            activity = "label:" + activity
            # pick time vs desired label
            desired_data = data[["timestamp", activity]]
            ax = data.dropna().plot(kind="line", x="timestamp", y=feature_name)
            activity_on = data[desired_data[activity] == 1]
            if not activity_on.empty:
                activity_on.plot(kind="scatter", x="timestamp", y=feature_name, ax=ax, color='r')
            else:
                print("No label to draw")
        plt.show()

    def __get_act_intervals(self, data: pd.DataFrame, act: str):
        act = act.strip()
        desired_act = data[["timestamp", act]].values
        c = np.nonzero(np.diff(desired_act[:,1]))[0]
        segments, segment_labels = [], []
        segments.append([desired_act[0,0], desired_act[c[0]+1,0]])
        segment_labels.append(desired_act[0,1])
        for i in range(len(c)-1):
            segments.append([desired_act[c[i]+1,0], desired_act[c[i+1]+1,0]])
            segment_labels.append(desired_act[c[i]+1,1])
        segments.append([desired_act[c[-1]+1,0], desired_act[len(desired_act)-1,0]])
        segment_labels.append(desired_act[c[-1]+1,1])
        return np.array(segments), np.array(segment_labels)

    def plot_activities(self, user_id: str, activity: list, ax=None):
        data = pd.read_csv(self.address + "/" + user_id, delimiter=',')
        time_stamp = data["timestamp"].values
        x_labels = np.linspace(time_stamp[0], time_stamp[-1])
        x_labels = datetime.datetime.fromtimestamp(x_labels[0])
        for i, act in enumerate(activity):
            act_intervals, act_labels = self.__get_act_intervals(data, act)
            act_intervals = act_intervals[np.where(act_labels == 1)[0]]
            width = act_intervals[:,1] - act_intervals[:,0]
            if not (width.size == 0):
                # generate random html color code
                color = ""
                for _ in range(6):
                    color += hex(np.random.randint(15)).split("x")[-1]
                color = ("#" + color).upper()
                # get datetime
                ax.barh(i*3+1.5, width, height=3, left=act_intervals[:,0], color=color, label=(act[6:]+str(i*3)))
                ax.legend(bbox_to_anchor=(0,-0.07), ncol=7, loc=2, prop={'size': 8})
        ax.set_ylim(-10, 3*len(activity))
        plt.show()
    

     

