
from tools.gradient_based_breathlists_extraction_functions import custom_breathlists_per_mouse

import pickle
import pandas as pd
import numpy as np

file = open(r"C:\Users\Mahmoud Al-Madi\Desktop\BCM_SIDS_ML_Summer_22\Data\trials_breath1.obj", "rb")
trials_breath = pickle.load(file)
file = open(r"C:\Users\Mahmoud Al-Madi\Desktop\BCM_SIDS_ML_Summer_22\Data\static_data.obj", "rb")
static_data = pickle.load(file)
file = open(r"C:\Users\Mahmoud Al-Madi\Desktop\BCM_SIDS_ML_Summer_22\Data\trials_raw1.obj", "rb")
trials_raw = pickle.load(file)
metadata = pd.read_csv(r"C:\Users\Mahmoud Al-Madi\Desktop\BCM_SIDS_ML_Summer_22\Data\d2k project metadata.csv")


#preparing useful dictionaries to be later used for labelling purposes
static_data["Mouse Type"] = static_data["Line"] + " " + static_data["Genotype"]

count = 0
mtype2num = {}
for genotype in static_data["Mouse Type"].unique():
    mtype2num[genotype] = count
    count += 1

num2mtype = {}

for keys, values in mtype2num.items():
    num2mtype[values] = keys

#producing breathlists
experiment_starter_timestamps = {}
custom_breathlists = {}
for mouse in trials_raw.keys():

    custom_breathlists, mouse_mount_timestamp= custom_breathlists_per_mouse(trials_raw, trials_breath,
                                                                            custom_breathlists, metadata,
                                                                            mouse,                                                                         
                                                                            bandpass_lowfreq=1, 
                                                                            bandpass_highfreq=15,
                                                                            filter_order=2,
                                                                            sampling_freq=1000,
                                                                            p2p_min=160,
                                                                            peak_prominence=1,
                                                                            breath_length_cutoff=2500) #in ms

    max_trial = np.array(custom_breathlists[mouse]["Trial Number"])[-1]
    
    custom_breathlists[mouse]["Trials Till Expiration"] = max_trial - custom_breathlists[mouse]["Trial Number"]

    custom_breathlists[mouse]["% of Experiment"] = custom_breathlists[mouse]["Trial Number"] / max_trial 

    custom_breathlists[mouse]["Genotype"] = mtype2num[static_data[static_data["MUID"] == mouse]["Mouse Type"].item()]
                                                                            
    experiment_starter_timestamps[mouse] = mouse_mount_timestamp

