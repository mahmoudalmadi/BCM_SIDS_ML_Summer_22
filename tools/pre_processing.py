import numpy as np
import pandas as pd
import os
from pandas.core.common import SettingWithCopyWarning

from datetime import datetime
import pickle
import warnings

def convert_to_lower(x):
    if isinstance(x,str):
        
        return str(x).lower()
    else:
        return x
def convert_to_float(x):
    '''
    Converts x to float
    x should be a string
    '''
    if isinstance(x,str):
        x = x.strip()
        if x == "":
            return -1e-7
        else:
            x = float(x)
            return x
    else:
        return float(x)

def handle_time_meta(x):
    '''
    This cleans the time data in the meta_data file.
    
    Returns: Datetime object with well formatted time
    
    '''
    x = x.split(":")
    micro = "0"
    second = "0"
    minute = "0"
    hour = "0"
    temp = ""
    if len(x) == 1:
        second = x[-1].split(".")
        if len(second) == 1:
            second = second[-1]
        else:
            micro = second[-1]
            second = second[-2]
        temp = hour + ":" + minute + ":" + second + ":" + micro
    if len(x) == 2:
        second = x[-1].split(".")
        if len(second) == 1:
            second = second[-1]
        else:
            micro = second[-1]
            second = second[-2]
        minute = x[-2]
        temp = hour + ":" + minute + ":" + second + ":" + micro
    if len(x) == 3:
        second = x[-1].split(".")
        if len(second) == 1:
            second = second[-1]
        else:
            micro = second[-1]
            second = second[-2]
        minute = x[-2]
        hour = x[-3]
        temp = hour + ":" + minute + ":" + second + ":" + micro
        
    return datetime.strptime(temp,"%H:%M:%S:%f")

def handle_time_ecg(x):
    '''
    x - a time string from ecg data
    
    Converts the datetime of a specific row into the proper format
    
    returns - Datetime object with well formatted time
    '''
    x = x.split(" ")
    x = x[1].split(":")
    micro = "0"
    second = "0"
    minute = "0"
    hour = "0"
    temp = ""
    if len(x) == 1:
        second = x[-1].split(".")
        if len(second) == 1:
            second = second[-1]
        else:
            micro = second[-1]
            second = second[-2]
        temp = hour + ":" + minute + ":" + second + ":" + micro
    if len(x) == 2:
        second = x[-1].split(".")
        if len(second) == 1:
            second = second[-1]
        else:
            micro = second[-1]
            second = second[-2]
        minute = x[-2]
        temp = hour + ":" + minute + ":" + second + ":" + micro
    if len(x) == 3:
        second = x[-1].split(".")
        if len(second) == 1:
            second = second[-1]
        else:
            micro = second[-1]
            second = second[-2]
        minute = x[-2]
        hour = x[-3]
        temp = hour + ":" + minute + ":" + second + ":" + micro
        
    return datetime.strptime(temp,"%H:%M:%S:%f")

def preprocess_ecg(df):
    '''
    Given the ecg df (dataframe) this returns the cleaned version of the dataframe
    
    It imputes the missing values, add columns which specify which row had a missing value.
    It cleans up the TimeData column and formats it properly to datetime. (while formatting it ignores the year,month and date as they are not relevant)
    
    '''
    cat_cols = ["TimeDate"]
    df = df.copy(deep = True)
    #We skip the last 5 rows as they have some aggregate statistics
    df = df[:df.shape[0]-5]
    
    # This converts the time date columns data type to be datetime
    # then converts them to seconds
    df["TimeDate"] = df["TimeDate"].apply(handle_time_ecg)
    df["TimeDate"] = df['TimeDate'].dt.hour * 3600 + df['TimeDate'].dt.minute * 60 + df['TimeDate'].dt.second + df['TimeDate'].dt.microsecond * 1e-6

    for i in df.columns:
        if i not in cat_cols:
            df[i] = df[i].apply(convert_to_float)
            df[i] = df[i].astype("float32")
            x = (df[i] == -1e-7)
            # if more than 50% values are missing we create a new column indicating that this row had a 
            # missing value
            if sum(x)/df.shape[0] > 0.5:
                df[f"{i}_missing"] = x
                df[f"{i}_missing"] = df[f"{i}_missing"].astype(int)
            df.replace(-1e-7,np.mean(df[i]))
    return df

def split_data(trial_time,path,type_data,trials,corrections = None):
    '''
    Splits the data trial wise
    
    trial_time - contains the start and end time for each trial for each mice
    path - path of the files that are to be processed.
    type_data - Type of data (ecg or raw or breath)
    trial - dictionary where the data will be stored
    corrections - is required for some files due to instrumentation failure 
    
    [ID,correction value, bad rows to remove]
    
    Example - 
    corrections = [
    ["M20868", 91.45,[3576,3577,3578,3579,3580]],
    ["M21267", 17.45,[0,1,2,3]],
    ["M21269", 96.8,[0,1,2,3]]
    ]
    '''
    
    # this is the list of files available in the given directory
    dir_list = os.listdir(path)
    
    if type_data == "ecg":
        for i in dir_list:
            temp_df = pd.read_csv(path + i,sep = "\t",index_col = False)
            
            temp_df = preprocess_ecg(temp_df)

            for k,v in trial_time.items():
                if k == i[:6]:
                    for j in range(len(v)):
                        start = v[j][0]
                        end = v[j][1]
                        # have to take into account the time when the mice expires during the last trial (no comments indicate the time of expiration)
                        if j == len(v) - 1:
                            # this is the last trial
                            required_df = temp_df.loc[start <= temp_df["TimeDate"]]
                        else:
                            required_df = temp_df.loc[(start <= temp_df["TimeDate"]) & (temp_df["TimeDate"] <= end)]

                        with warnings.catch_warnings():
                            warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
                            required_df["trial_no"] = [j-1]*required_df.shape[0]

                        if k not in trials:
                            trials[k] = {}
                            trials[k][type_data] = [required_df]
                        elif type_data in trials[k]:
                            trials[k][type_data].append(required_df)
                        else:
                            trials[k][type_data] = [required_df]
            trials[i[:6]][type_data] = pd.concat(trials[i[:6]][type_data],axis = 0)
    if type_data == "raw":
        for i in dir_list:
            print(i)
            temp_df = pd.read_csv(path + i,sep = "\t",index_col = False,skiprows = list(range(9)),on_bad_lines = "warn",header = None,encoding="latin1")
            raw_data_columns = ["Timestamp","Breathing_flow_signal","O2_sensor_data","CO2_sensor_data","Chamber_temperature","ECG","Heart_Rate", "Integrated_Flow","Breathing"]
            temp_df.columns = raw_data_columns
            if corrections is not None:
                for val in corrections:
                    if val[0] == i[:6]:
                        temp_df.dropna(axis = 0,inplace = True)
                        temp_df.drop(temp_df.index[val[2]],axis = 0,inplace = True)
                        
                        for j in temp_df.columns:
                            temp_df[j].astype("float64")
                        temp_df["Timestamp"].iloc[int(val[1]*1000):] = temp_df["Timestamp"].iloc[int(val[1]*1000):] + float(val[1])
                        
            for k,v in trial_time.items():
                if k == i[:6]:
                    for j in range(len(v)):
                        start = v[j][0]
                        end = v[j][1]
                        # have to take into account the time when the mice expires during the last trial (no comments indicate the time of expiration)
                        if j == len(v) - 1:
                            # this is the last trial
                            required_df = temp_df.loc[start <= temp_df["Timestamp"]]
                        else:
                            required_df = temp_df.loc[(start <= temp_df["Timestamp"]) & (temp_df["Timestamp"] <= end)]
                            
                        required_df["trial_no"] = [j-1]*required_df.shape[0]
                        
                        if k not in trials:
                            trials[k] = {}
                            trials[k][type_data] = [required_df]
                        elif type_data in trials[k]:
                            trials[k][type_data].append(required_df)
                        else:
                            trials[k][type_data] = [required_df]
                            
            trials[i[:6]][type_data] = pd.concat(trials[i[:6]][type_data],axis = 0)
            for k in trials[i[:6]][type_data].columns:
                trials[i[:6]][type_data][k] = trials[i[:6]][type_data][k].astype("float32")
            
    if type_data == "breath":
        for i in dir_list:
            temp_df = pd.read_csv(path + i,index_col = False)
            for k,v in trial_time.items():
                if k == i[:6]:
                    for j in range(len(v)):
                        start = v[j][0]
                        end = v[j][1]
                        # have to take into account the time when the mice expires during the last trial (no comments indicate the time of expiration)
                        if j == len(v) - 1:
                            # this is the last trial
                            required_df = temp_df.loc[start <= temp_df["Timestamp_Inspiration"]]
                        else:
                            required_df = temp_df.loc[(start <= temp_df["Timestamp_Inspiration"]) & (temp_df["Timestamp_Inspiration"] <= end)]
                        
                        required_df["trial_no"] = [j-1]*required_df.shape[0]
                        
                        if k not in trials:
                            trials[k] = {}
                            trials[k][type_data] = [required_df]
                        elif type_data in trials[k]:
                            trials[k][type_data].append(required_df)
                        else:
                            trials[k][type_data] = [required_df]
            
            trials[i[:6]][type_data] = pd.concat(trials[i[:6]][type_data],axis = 0)
    return trials

        
def get_trial_time(meta_data):
    meta_data.reset_index(drop = True,inplace = True)
    meta_data["Time"] = meta_data["Time"].apply(handle_time_meta)
    meta_data["Time"] = meta_data['Time'].dt.hour * 3600 + meta_data['Time'].dt.minute * 60 + meta_data['Time'].dt.second + meta_data['Time'].dt.microsecond * 1e-6
    meta_data["Comment"] = meta_data["Comment"].apply(convert_to_lower)
    meta_group = meta_data.groupby("source file")

    # stores start time and end time for each trial
    trial_time = {}

    for k,v in meta_group:
        v = v.copy(deep = True)
        v.sort_values("Time",inplace = True)

        num_trials = 0
        for i in range(v.shape[0]-1,-1,-1):
            if isinstance(v["Comment"].iloc[i],int):
                num_trials = v["Comment"].iloc[i]
                break

        count = 0
        start_time = v["Time"].loc[v.index[0]]
        for i in range(v.shape[0]):
            if isinstance(v["Comment"].iloc[i],int):
                count += 1
                if k in trial_time:
                    trial_time[k].append((start_time,v["Time"].iloc[i]))
                else:
                    trial_time[k] = [(start_time,v["Time"].iloc[i])]
                start_time = v["Time"].iloc[i]

                if count == num_trials:
                    if k in trial_time:
                        trial_time[k].append((v["Time"].iloc[i],v["Time"].iloc[-1]))
                    else:
                        trial_time[k] = [(v["Time"].iloc[i],v["Time"].iloc[-1])]
    
    return trial_time

def main_preprocess_data(paths,save_loc):
    '''
    paths - A dictionary with keys being the name and value being the path 
    to the FOLDER where all the corresponding files.
    
    Example - (The keys which are input must be identical to the ones below)
    breath_path_3d : "location to the list of breath files for the gene HM3D"
    ecg_path_3d : "location to the list of ecg files for the gene HM3D"
    exported_path_3d : "location to the list of raw files for the gene HM3D"
    breath_path_4d : "location to the list of breath files for the gene HM4D"
    ecg_path_4d : "location to the list of ecg files for the gene HM4D"
    exported_path_4d : "location to the list of raw files for the gene HM4D"
    meta_data : "location to the meta_data file"
    
    save_loc - The save location where all the processed data must be saved
    
    Returns - Saves preprocessed data into the save location
    The data is a pickle object in the format of key, value pairs.
    Each key is the mouse ID where each value is its corresponding data, Essentially
    a trial no column is added which can be used to analyze the mice trial wise
    
    '''
    
    breath_path_3d = os.path.join(paths,"DBH_FLPo_bactin_Cre_FP_hM3D","hm3d breathlist/")
    ecg_path_3d = os.path.join(paths,"DBH_FLPo_bactin_Cre_FP_hM3D","hm3d ecg list/")
    exported_path_3d = os.path.join(paths,"DBH_FLPo_bactin_Cre_FP_hM3D","hm3d exported data/")
    
    breath_path_4d = os.path.join(paths,"DBH_FLPo_bactin_cre_FP_hM4De","hm4d breathlist/")
    ecg_path_4d = os.path.join(paths,"DBH_FLPo_bactin_cre_FP_hM4De","hm4d ecg list/")
    exported_path_4d = os.path.join(paths,"DBH_FLPo_bactin_cre_FP_hM4De","hm4d exported data/")
    
    meta_data = pd.read_excel(os.path.join(paths,"d2k project metadata.xlsx"))
    
    trial_time = get_trial_time(meta_data)
    
    static_data = {
    "MUID":["M21264","M21267","M21269","M21627","M21628","M21630","M20865","M20867","M20869","M20874","M21480","M21484","M21487","M20864","M20868","M20870","M21481","M21483","M21486","M21488"],
    "Line":["hm4d","hm4d","hm4d","hm4d","hm4d","hm4d","hm3d","hm3d","hm3d","hm3d","hm3d","hm3d","hm3d","hm3d","hm3d","hm3d","hm3d","hm3d","hm3d","hm3d"],
    "Genotype":["Exp","Exp","Exp","Con","Con","Con","Exp","Exp","Exp","Exp","Exp","Exp","Exp","Con","Con","Con","Con","Con","Con","Con"],
    "Weight":[4.06,3.92,5.6,4.54,5.81,5.65,3.41,3.43,4.44,3.8,4.49,3.37,3.72,3.74,4.04,4.61,4.25,4.64,4.79,4.67]
    }
    static_data = pd.DataFrame(static_data)

    static_data["Mouse Type"] = static_data["Line"] + " " + static_data["Genotype"]

    with open(f"{save_loc}/static_data.obj",'wb') as file1:
        pickle.dump(static_data,file1)
    
    
    trial_ecg = {}
    trial_ecg = split_data(trial_time,ecg_path_3d,"ecg",trial_ecg)
    trial_ecg = split_data(trial_time,ecg_path_4d,"ecg",trial_ecg)
    
    
    trial_raw = {}
    # some trials require manual corrections due to restarting of the measuring
    # devices
    corrections = [
        ["M20868", 91.45,[3576,3577,3578,3579,3580]],
        ["M21267", 17.45,[0,1,2,3]],
        ["M21269", 96.8,[0,1,2,3]]
        ]

    trial_raw = split_data(trial_time,exported_path_3d,"raw",trial_raw,corrections = corrections)
    trial_raw = split_data(trial_time,exported_path_4d,"raw",trial_raw,corrections = corrections)
    
    trial_breath = {}
    trial_breath = split_data(trial_time,breath_path_3d,"breath",trial_breath)
    trial_breath = split_data(trial_time,breath_path_4d,"breath",trial_breath)
    
    # saving the data
    with open(f"{save_loc}/trials_raw1.obj",'wb') as file1:
        pickle.dump(trial_raw,file1)

    with open(f"{save_loc}/trials_breath1.obj",'wb') as file1:
        pickle.dump(trial_breath,file1)

    with open(f"{save_loc}/trials_ecg1.obj",'wb') as file1:
        pickle.dump(trial_ecg,file1)

main_preprocess_data(r"C:\Users\Mahmoud Al-Madi\Desktop\BCM_SIDS_ML_Summer_22\Data",r"C:\Users\Mahmoud Al-Madi\Desktop\BCM_SIDS_ML_Summer_22\Data")