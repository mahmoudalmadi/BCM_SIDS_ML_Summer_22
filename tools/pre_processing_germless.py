import pandas as pd
import os
import pickle


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
    

    if type_data == "raw":
        for i in dir_list:
            print(i)
            trial_time = {i[:6]:[(0,20)]}
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

    return trials
            
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
    
    germfree_data_path = os.path.join(paths,"Germfree/")
    
    
    static_data = {
    "MUID":["M30216","M30217", "M32603", "M32604","M32605","M39092","M39093","M39094","M39095","M34401","M34402","M34403"],
    "Sex":["Male", "Male","Male","Male","Male","Female","Female","Female","Male","Female","Female","Female"],
    "Weight":[4.93,5.03,5.07,5.15,5.25,3.67,3.43,3.29,3.4,3.68,3.82,3.55]
    }
    static_data = pd.DataFrame(static_data)

    with open(f"{save_loc}/germless_static_data.obj",'wb') as file1:
        pickle.dump(static_data,file1)
    
    
    trial_raw = {}
    
    trial_raw = split_data({1:(0,1)},germfree_data_path,"raw",trial_raw)
    
    # saving the data
    with open(f"{save_loc}/germless_raw1.obj",'wb') as file1:
        pickle.dump(trial_raw,file1)


main_preprocess_data(r"C:\Users\Mahmoud\Desktop\BCM_SIDS_ML_Summer_22\BCM_SIDS_ML_Summer_22\data",r"C:\Users\Mahmoud\Desktop\BCM_SIDS_ML_Summer_22\BCM_SIDS_ML_Summer_22\data")