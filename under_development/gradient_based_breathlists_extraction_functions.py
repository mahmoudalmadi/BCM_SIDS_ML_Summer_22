import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from sklearn.cluster import KMeans
import re
from sklearn.decomposition import PCA
from collections import *
from scipy.signal import butter, filtfilt
from scipy.integrate import cumtrapz
from scipy.signal import find_peaks
import operator

#all helper functions that constitute the breathlists extraction code

def butter_bandpass(lowcutoff, highcutoff, fs, order):
    """
    This function takes in the filter parameters and
    does the simple calculations needed to account for the
    Nyquist frequency and using the scipy butter function
    to produce the numerator and polynomials of the IIR filter \n
    Inputs: \n
    1) lowcutoff - lower frequecy cutoff of bandpass filter
    2) highcutoff - upper frequecy cutoff of bandpass filter
    3) fs - sampling frequency
    4) order - filter order \n
    Outputs: \n
    1) a - denominator polynomials of the IIR filter
    2) b - numerator polynomials of the IIR filter

    """
    nyq = 0.5 * fs  #nyquist frequency
    lowcutoff = lowcutoff / nyq        
    highcutoff = highcutoff / nyq
    a,b = butter(order, [lowcutoff, highcutoff], analog=False, btype='bandpass',output = 'ba')
    return a,b

def butter_bandpass_filter(data, lowcutoff, highcutoff, fs, order):
    """
    This function wraps the scipy functions needed to implement a 
    bandpass fillter and ultimately filters the signal \n
    Inputs: \n
    1) data - signal to be filtered
    2) lowcutoff - lower frequecy cutoff of bandpass filter
    3) highcutoff - upper frequecy cutoff of bandpass filter
    4) fs - sampling frequency
    5) order - filter order \n
    Outputs: \n
    1) filtered_signal

    """
    a,b = butter_bandpass(lowcutoff, highcutoff, fs, order=order)
    filtered_signal = filtfilt(a,b, data)
    return filtered_signal

def bandpass_filter(low, high, fs,order, signal, quiet = False):
    """
    This function bandpasses filters a given signal and has an option
    of plotting what the signal looks like before and after filtering \n
    Inputs: \n
    1) low - lower frequecy cutoff of bandpass filter
    2) high - upper frequecy cutoff of bandpass filter
    3) fs - sampling frequency
    4) order - filter order 
    5) signal - signal to be filtered
    6) quiet - (default: False) boolean indicating whether or not to display signal before and after filtering \n
    Outputs: \n
    1) filtered_data

    """
    filtered_data = butter_bandpass_filter(signal, low, high, fs, order)

    if quiet == False:    
        plt.figure(figsize = (8,8))
        plt.plot(signal, label='Raw signal')
        plt.title("Unfiltered signal", size = 10)
        plt.xlabel("Time (ms)", size = 10)
        plt.ylabel("Ventflow", size = 10)

        plt.figure(figsize = (8,8))
        plt.plot(filtered_data)
        plt.title("Bandpass Filtered Signal, Cutoff = {}-{} Hz" .format(low,high), size = 10)
        plt.xlabel("Time (ms)", size = 10)
        plt.ylabel("Ventflow", size = 10)

    return filtered_data

def compute_timestamps(metadata, mouse_id, all = False, breath_only = False, phase = None, phase2 = None, phase3 = None, phase4 = None):
    """
    Given the required phase(s) and metadata, this function returns all the timestamps in which
    the given phase occurs in seconds as a list \n  

    Inputs:
    1) metadata
    2) mouse_id
    3) all - (default: False) boolean indicating whether all timestamps in experiments are requested as opposed to a select few
    4) breath_only - (default: False) Only if all == True. Excludes all HR Recovery timestamps when set to true
    5) phase, phase2, ... - all phases requested. At least 1 and at most 5 phases.

    if all == True, all timestamps in the experiment is returned \n
    Output formatted as follows: \n
        timestamps_in_s["trial"] = [list of timestamps] \n
        timestamps_in_s["other"]["timestamp"] = [all timestamps for non-trial onset] \n
        timestamps_in_s["other"]["comment"] = [names of all timestamps for not-trial onset]  \n
    if all == false, only specified timestamps are returned \n
    Output is just a 
    """
        
    if not all:
        #retreiving all relevant timestamps
        timestamps_in_hms = list(metadata[((metadata["Comment"] == phase) | (metadata["Comment"] == phase2) | (metadata["Comment"] == phase3) | (metadata["Comment"] == phase4)) & (metadata["source file"] == mouse_id)]["Time"])
        timestamps_in_s = []
        
        #converting timstamps from h:m:s to seconds
        for timestamp in timestamps_in_hms:
            processed_timestamp = re.split(':', timestamp)
            timestamp = [float(i) for i in processed_timestamp]   
            if len(timestamp) == 1:
                timestamps_in_s.append(timestamp[0])
            elif len(timestamp) == 2:
                timestamps_in_s.append(60*timestamp[0]+timestamp[1])
            elif len(timestamp) == 3:
                timestamps_in_s.append(3600*timestamp[0]+60*timestamp[1]+timestamp[2])

        return timestamps_in_s

    else: #retreive all timestamps in experiment
        comments = list(metadata[metadata["source file"] == mouse_id]["Comment"]) 
        timestamps_in_hms = list(metadata[metadata["source file"] == mouse_id]["Time"]) 
        timestamps_in_s = defaultdict(lambda: defaultdict(str))

        if breath_only:  #exclude HR recovery timestamps
            other_comments = []
            other_timestamps = []
            for timestamp in range(0,len(comments)):
                if comments[timestamp] not in ["HR recovery","Hr recovery","HR recovery - No signal", "hr recovery"]:
                    other_comments.append(comments[timestamp])
                    other_timestamps.append(timestamps_in_hms[timestamp])
            timestamps_in_hms = other_timestamps
            comments = other_comments

        timestamps_in_s["trial"] = []
        timestamps_in_s["other"]["timestamp"] = []
        timestamps_in_s["other"]["comment"] = []

        #converting all hms timestamps 
        for stage in range(0,len(comments)):
            if comments[stage].isdecimal(): #if timestamps is trial onset, it is assigned elsewehere
                processed_timestamp = re.split(':', timestamps_in_hms[stage])
                timestamp = [float(i) for i in processed_timestamp]   
                if len(timestamp) == 1:
                    timestamps_in_s["trial"].append(timestamp[0])
                    timestamps_in_s["other"]["timestamp"].append((timestamp[0]))
                elif len(timestamp) == 2:
                    timestamps_in_s["trial"].append(60*timestamp[0]+timestamp[1])
                    timestamps_in_s["other"]["timestamp"].append((60*timestamp[0]+timestamp[1]))
                elif len(timestamp) == 3:
                    timestamps_in_s["trial"].append(3600*timestamp[0]+60*timestamp[1]+timestamp[2])
                    timestamps_in_s["other"]["timestamp"].append((3600*timestamp[0]+60*timestamp[1]+timestamp[2]))
                timestamps_in_s["other"]["comment"].append(comments[stage])
            else:
                if not comments[stage][1:].isdecimal():
                    processed_timestamp = re.split(':', timestamps_in_hms[stage])
                    timestamp = [float(i) for i in processed_timestamp]   
                    if len(timestamp) == 1:
                        timestamps_in_s["other"]["timestamp"].append((timestamp[0]))
                    elif len(timestamp) == 2:
                        timestamps_in_s["other"]["timestamp"].append((60*timestamp[0]+timestamp[1]))
                    elif len(timestamp) == 3:
                        timestamps_in_s["other"]["timestamp"].append((3600*timestamp[0]+60*timestamp[1]+timestamp[2]))
                    timestamps_in_s["other"]["comment"].append(comments[stage])

        return timestamps_in_s

def trial_number(metadata, custom_breathlists, breath_start, mouse):
    """
    Given the custom breathlists dictionary mapping mouse IDs to their 
    corresponding custom breathlists, this function adds a column to that dataframe
    indicating the trial number in which each breath took place \n

    Inputs: \n
    1) metadata
    2) custom_breathlists - dictionary mapping each mouse ID to its corresponding custom breathlists
    3) breath_start - dataframe column indicating timestamps of inspiration
    4) mouse - mouse ID
    """

    #retreiving timestamps of each trials
    all_timestamps = compute_timestamps(metadata, mouse, all=True)

    #assigning trial number 0 to all breaths before onset of trial 1
    custom_breathlists[mouse].loc[breath_start < all_timestamps["trial"][0]*1000, "Trial Number"] = 0

    #assigning the last trial number to the breaths that happened in the last breath
    custom_breathlists[mouse].loc[breath_start > all_timestamps["trial"][-1]*1000, "Trial Number"] = len(all_timestamps["trial"])

    #assigning the rest of the trial numbers
    for timestamp in range(1,len(all_timestamps["trial"])):
        custom_breathlists[mouse].loc[(breath_start < all_timestamps["trial"][timestamp]*1000) & (breath_start > all_timestamps["trial"][timestamp-1]*1000), "Trial Number"] = timestamp

phase2number = {"Cal 20 Room Air": 0,
                "Pre-CNO Room Air": 1,
                "Pre-CNo Room Air": 1,
                "Post-CNO Room Air": 2,
                "Post-CNo Room Air": 2,
                "trial onset" : 3,
                "apea starts" : 4,
                "apnea starts" : 4,
                "first gasp" : 5,
                "eupnea starts" :6,
                "eupnea recovery" : 6,
                "Cal 5 Room Air": 7
                }

number2phase = {0 : "Cal 20 Room Air",
                1 : "Pre-CNO Room Air",
                2 : "Post-CNO Room Air",
                3 : "Trial Onset",
                4 : "Apnea Starts",
                5 : "First Gasp",
                6 : "Eupnea Recovery",
                7 : "Cal 5 Room Air"
                }

def phase_type(metadata, custom_breathlists, breath_start, mouse, phase2number, number2phase):
    """
    Given the custom breathlists dictionary mapping mouse IDs to their 
    corresponding custom breathlists, this function adds a column to that dataframe
    indicating the phase in which each breath took place and another column indicating an
    arbitraty code number corresponding to that phase \n

    Inputs: \n
    1) metadata
    2) custom_breathlists - dictionary mapping each mouse ID to its corresponding custom breathlists
    3) breath_start - dataframe column indicating timestamps of inspiration
    4) mouse - mouse ID
    5) phase2number - arbitrary mapping of phase name to an arbitrarily chosen integer
    6) number2phase - inverese of phase2number
    """

    #retrieving all timestamps in the experiment
    all_timestamps = compute_timestamps(metadata, mouse, all=True, breath_only = True)

    custom_breathlists[mouse].loc[breath_start < all_timestamps["other"]["timestamp"][0]*1000, "Phase Number"] = phase2number[all_timestamps["other"]["comment"][0]]
    custom_breathlists[mouse].loc[breath_start < all_timestamps["other"]["timestamp"][0]*1000, "Phase Type"] = number2phase[phase2number[all_timestamps["other"]["comment"][0]]]
    custom_breathlists[mouse].loc[breath_start > all_timestamps["other"]["timestamp"][-1]*1000, "Phase Number"] = phase2number[all_timestamps["other"]["comment"][-1]]
    custom_breathlists[mouse].loc[breath_start > all_timestamps["other"]["timestamp"][-1]*1000, "Phase Type"] = number2phase[phase2number[all_timestamps["other"]["comment"][-1]]]
    
    for timestamp in range(1,len(all_timestamps["other"]["timestamp"])):
        
        if all_timestamps["other"]["comment"][timestamp-1].isdecimal():
            custom_breathlists[mouse].loc[(breath_start < all_timestamps["other"]["timestamp"][timestamp]*1000) & (breath_start > all_timestamps["other"]["timestamp"][timestamp-1]*1000), "Phase Number"] = phase2number["trial onset"]
            custom_breathlists[mouse].loc[(breath_start < all_timestamps["other"]["timestamp"][timestamp]*1000) & (breath_start > all_timestamps["other"]["timestamp"][timestamp-1]*1000), "Phase Type"] = "Trial Onset"
        else:
            custom_breathlists[mouse].loc[(breath_start < all_timestamps["other"]["timestamp"][timestamp]*1000) & (breath_start > all_timestamps["other"]["timestamp"][timestamp-1]*1000), "Phase Number"] = phase2number[all_timestamps["other"]["comment"][timestamp-1]]
            custom_breathlists[mouse].loc[(breath_start < all_timestamps["other"]["timestamp"][timestamp]*1000) & (breath_start > all_timestamps["other"]["timestamp"][timestamp-1]*1000), "Phase Type"] = number2phase[phase2number[all_timestamps["other"]["comment"][timestamp-1]]]

def label_breaths(metadata, custom_breathlists, mouse, breath_start, phase2number, number2phase):
    """
    This function adds 3 columns to the custom breathlists dataframe: \n
    1) column with trial number in which each breath took place
    2) column with phase type in which each breath took place
    3) column with number corresponding to a phase type in which each breath took place based on the arbitrary phase2number assignments
    """
    trial_number(metadata, custom_breathlists, breath_start,mouse)

    phase_type(metadata,custom_breathlists, breath_start, mouse, phase2number, number2phase)

def extract_breath(data,start,end, time_zero):
    """
    Given start and end timestamp of a given breath, this function
    extracts and returns the data corresponding to the breath as a
    normalized array
    """
    start = start - time_zero
    end = end - time_zero

    breath = data[start:end]

    return breath - np.mean(breath)

def clean_breath_data(raw_data, mouse, freqlow, freqhigh, filter_order, sampling_freq):
    """
    This function filters the raw breathing data and fixes an issue present in 3 of the mice \n

    Inputs: \n
    1) raw_data - 2D dictionary mapping each mouse ID to dataframe corresponding to its raw data
    2) mouse - mouse ID
    3) freqlow - lower bound for bandpass filter
    4) freqhigh - upper bound for bandpass filter
    5) filter_order - filter order
    6) sampling_freq - sampling frequency \n

    Outputs: \n
    1) data - raw_data dataframe corresponding to "mouse" but refined for use later
    2) filtered_data - array corresponding to breathing data from raw_data after filtering
    """
    
    #remove section of missing data from the only 3 mice with that problem
    timestamps = np.array(raw_data[mouse]["raw"]["Timestamp"])

    #time between each inspiration timestamps
    lengths = timestamps[1:] - timestamps[:-1]

    #excluding parts of breathing data missing readings for longer than a seconds
    if mouse in ["M20868", "M21267", "M21269"]:
        data = raw_data[mouse]["raw"][np.where(lengths>1)[0][0]+1:]
    else:
        data = raw_data[mouse]["raw"]

    #resetting index for consistency of the data with peak indices
    data.reset_index(inplace= True)

    #filtering data
    filtered_data = bandpass_filter(freqlow,freqhigh,sampling_freq,filter_order,data["Breathing_flow_signal"], quiet = True)
    
    return data, filtered_data
    
def find_breath_timestamps(metadata, raw_data: dict, mouse: str, filtered_data, peak_prominence : float, p2p_min: float):
    """
    Tthis function returns the index of the first breath indicating when the mouse was first mounted on the rig.
    This function also returns the inspiration timestamps of all the breaths \n

    Inputs: \n
    1) raw_data - 2D dictionary mapping each mouse ID to dataframe corresponding to its raw data
    2) mouse - mouse ID
    3) filtered_data - array corresponding to breathing data from raw_data after filtering
    4) peak_prominence - value indicating how much higher the peaks we're looking for need to be than data around it
    5) p2p_min - minimum distance between two peaks 

    Outputs: \n
    1) mouse_online - timestamp during which mouse was mounted onto rig
    2) breath_timestamps - all inspiration timestamps 
    3) expiration_timestamps - all expiration timestamps
    """

    time = raw_data["index"]

    #finding all the timestamps of onset of inspiration and expiration
    y_volume = cumtrapz(filtered_data, x=time)

    peaks, _ = find_peaks(-y_volume, distance = p2p_min, prominence = peak_prominence)
    peaks = peaks + time[0]
    exp_peaks, _ = find_peaks(y_volume, distance = p2p_min, prominence = peak_prominence)
    exp_peaks = exp_peaks + time[0]
    
    #taking care of inconsistencies in a couple of the data files
    calibr = compute_timestamps(metadata, mouse, phase = "Cal 20 Room Air")
    if mouse not in ["M20868", "M21267", "M21269"]:
        narrowed_peaks = peaks[peaks > (calibr[-1]*1000 + 5e4)]
    else:
        narrowed_peaks = peaks

    #identifying where exactly in
    #the data the mouse is actually mounted onto the machine
    for peak in range(0,len(peaks)):
        distance = narrowed_peaks[peak+40] - narrowed_peaks[peak]
        if distance < 40e3:
            starter_peak = peak
            break
    
    #timestamp where the mouse was mounted on the rig
    mouse_online = raw_data["Timestamp"].iloc[narrowed_peaks[starter_peak]]

    #finding all the breaths detected after mouse mounted on rig
    breath_timestamps = narrowed_peaks[starter_peak:]
    expiration_timestamps = exp_peaks[exp_peaks > breath_timestamps[0]]

    peaks1=breath_timestamps
    peaks0=expiration_timestamps

    #accounting for all instances in which consective expirations were detected with
    #no insipirations inbetween and vice versa
    expiration =  [1] * len(peaks0)
    inspiration =  [-1] * len(peaks1)

    breath_out = list(zip(list(peaks0), expiration))
    breath_in = list(zip(list(peaks1), inspiration))

    all_breaths = breath_out + breath_in
    all_breaths.sort(key=operator.itemgetter(0))

    timestamp_type = list(zip(*all_breaths))
    test = np.array(timestamp_type[1][:-1]) + np.array(timestamp_type[1][1:])

    consecutives = np.where(test != 0)[0]

    for index in consecutives:
        del all_breaths[index+1]
        consecutives -= 1

    all_timestamps = np.array(list(zip(*all_breaths))[0])
    breath_timestamps = all_timestamps[0::2]
    expiration_timestamps = all_timestamps[1::2]

    #if last peak deteced was an inspiration, remove it. Making sure last peak detected was expiration
    if len(breath_timestamps) > len(expiration_timestamps):
        breath_timestamps = breath_timestamps[:-1]
        expiration_timestamps = expiration_timestamps[:-1]
    else: #removing last expiration since last inspiration will be discarded later
        expiration_timestamps = expiration_timestamps[:-1]

    return mouse_online, breath_timestamps, expiration_timestamps, time[0]

def extract_all_breaths(metadata, original_breathlists: dict, custom_breathlists: dict, mouse : str, breath_timestamps, expiration_timestamps, filtered_data, breath_length_cutoff:int, time_zero:int):
    """
    Given an dictionary, breath timestamps, and filtered data 
    this function extracts every single breath and information on each breath in
    a dataframe in the dictionary and maps it to mouse ID as follows:
    custom_breathlists[mouseID] = dataframe with information on all breaths
    """

    #calculating duration of all stamps
    breath_durations = breath_timestamps[1:] - breath_timestamps[:-1]

    setting_up_breath_Df = {"Breath Start Timestamp (ms)":breath_timestamps[:-1], "Expiration Timestamp (ms)":expiration_timestamps,
                            "Breath End Timestamp (ms)":breath_timestamps[1:], "Breath Duration (ms)":breath_durations}
    custom_breathlists[mouse] = pd.DataFrame(data=setting_up_breath_Df)

    #removing breaths 
    custom_breathlists[mouse] = custom_breathlists[mouse][custom_breathlists[mouse]["Breath Duration (ms)"] < breath_length_cutoff]
    custom_breathlists[mouse].reset_index(inplace=True)

    all_breaths = []
    breath_height = []

    #volume correction of raw data
    normalization_factor = find_normalization_factor(metadata,original_breathlists, mouse)
    if mouse in ["M21269", "M20868"]:
        filtered_data = filtered_data * 1
    else:
        filtered_data = filtered_data * normalization_factor

    #extracting data corresponding to each breath + calculating height of each breath
    for index, row in custom_breathlists[mouse].iterrows():
        all_breaths.append(extract_breath(filtered_data, row["Breath Start Timestamp (ms)"], row["Breath End Timestamp (ms)"], time_zero))
        breath_height.append(np.max(all_breaths[index]) - np.min(all_breaths[index]))

    custom_breathlists[mouse]["Breath Data"] = all_breaths
    custom_breathlists[mouse]["Breath Height"] = breath_height

    return custom_breathlists

def find_normalization_factor(metadata, breathlists, mouse, calibration_volume = 20):
    """
    Given breathlists, mouse, calibration volume, this function computes 
    the normalization factor by which we multiply the instantaneous
    tidal volume readings to normalize them
    """
    
    calibr = compute_timestamps(metadata, mouse, phase = "Cal 20 Room Air")
    start = calibr[-1] * 1e3
    end = start + 4e4
    
    breathlists[mouse]["breath"]["bpm"] = 60 / breathlists[mouse]["breath"]["Inspiratory_Duration"]
    data_subset = breathlists[mouse]["breath"][(breathlists[mouse]["breath"]["Breath Number"] > start) & (breathlists[mouse]["breath"]["Breath Number"] < end) & ((breathlists[mouse]["breath"]["bpm"] > 60))]
    avg_tidal = np.mean(data_subset["Tidal_Volume_uncorrected"])
    
    normalization_factor = calibration_volume/avg_tidal
    
    return normalization_factor

def custom_breathlists_per_mouse(raw_data, original_breathlists, custom_breathlists, metadata, mouse, bandpass_lowfreq, bandpass_highfreq, filter_order, sampling_freq, p2p_min, peak_prominence, breath_length_cutoff):
    """
    Wrapper functions given raw data per mouse that creates breathlists as dictionary mapping mouseID to dataframe of breathlists\n

    Inputs: \n
    1) raw_data - 2D dictionary containing dataframe of raw data formatted as as raw_data[mouseID]["raw"]
    2) original_breathlists - breathlists provided by the lab. This is used for volume correction
    3) custom_breathlists - 2D dictionary formatted as custom_breathlists[mouseID] so that we can add breathlists for other mice to it
    4) mouse - mouse ID
    5) bandpass_lowfreq - lower bound for butterworth bandpass filter
    6) bandpass_highfreq - upper bound for butterworth bandpass filter
    7) filter_order - butterworth filter order
    8) smapling_freq - sampling frequency of raw data
    9) p2p_min - minimum distance between peaks of two breaths/ minimum length of breath to detect
    10) peak_prominence - how much higher a peak of a breath should be than its surroundings
    11) breath_length_cutoff - max breath duration to cutout detected breaths that include extended periods where the mouse is not breathing

    Outputs: \n
    1) custom_breathlists - dictionary mapping mouse ID to dataframe containing breathlists
    2) mouse_mount_timestamp - timestamp where mouse is mounted onto rig
    """

    #clean up and filter raw_data
    refined_trials_raw, filtered_data = clean_breath_data(raw_data, mouse, freqlow = bandpass_lowfreq,freqhigh= bandpass_highfreq, filter_order = filter_order, sampling_freq = sampling_freq)
    
    #extract all breath timestamps
    mouse_mount_timestamp, breath_timestamps, expiration_timestamps, time_zero = find_breath_timestamps(metadata,refined_trials_raw, mouse, filtered_data,
                                                                        p2p_min=p2p_min,peak_prominence=peak_prominence)

    #extract data for every single breaths (readings for entire breath) and put it in dataframe with the rest of breath information
    custom_breathlists = extract_all_breaths(metadata, original_breathlists, custom_breathlists, mouse, breath_timestamps, expiration_timestamps,
                                            filtered_data, breath_length_cutoff, time_zero)

    #label breaths based on which trial/ phase they are in
    label_breaths(metadata,custom_breathlists, mouse, custom_breathlists[mouse]["Breath Start Timestamp (ms)"], phase2number, number2phase)


    return custom_breathlists, mouse_mount_timestamp

