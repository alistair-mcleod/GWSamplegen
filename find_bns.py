#import mldatafind
import h5py
import numpy as np
from gwpy.timeseries import TimeSeries, TimeSeriesDict
#from utils.mldatafind.segments import query_segments
from utils.mldatafind.find import find_data
from utils.noise_utils import combine_seg_list

#mldatafind.authenticate.authenticate()

#note: in find.py, n_samples definition on line 23 needs to be: n_samples = int(n_channels * duration * sample_rate)
#some strange bug causes it to think it's a 32bit LIGO gps time

write_dir = "/fred/oz016/alistair/GWSamplegen/noise/test"
sample_rate = 2048
min_duration = 1010



def split_segment(
    segment: np.ndarray,
    start_GPS: int,
    prefix: str
):
    """Segments of LIGO noise sometimes have sections where a detector's strain data is good according to the DQ bits, 
    but the detector is not operating. This function splits segments of noise that have these sections 
    of bad data and saves the good sections."""

    if len(segment.shape) == 1:
        times_to_remove = np.isnan(segment)
    else:
        times_to_remove = np.any(np.isnan(segment), axis = 0)

    slice_idx = np.where(np.diff(times_to_remove) == 1)[0]
    slice_idx = np.add(slice_idx, 1)

    if not np.any(np.isnan(segment[:,0])):
        slice_idx = np.insert(slice_idx,0,0)

    if not np.any(np.isnan(segment[:,-1])):
        slice_idx = np.append(slice_idx,segment.shape[1])


    all_arrays = []

    for i in range(0,len(slice_idx),2):
        #first element is start of good section, element after is end of that good section etc.
        #first element gets rounded up to the nearest second, second element gets rounded down

        new_start = int(sample_rate * np.ceil(slice_idx[i]/sample_rate))
        new_end = int(slice_idx[i+1] - slice_idx[i+1]%sample_rate)
        print(new_start,new_end)

        #duration of this sub-segment
        new_duration = (new_end - new_start)//sample_rate
        if new_duration < min_duration:
            print("sub-segment too short!")
            continue
        
        array_to_save = segment[:,new_start:new_end]

        #start time of this sub-segment
        t0 = new_start//sample_rate + start_GPS

        fname = write_dir + "/" + f"{prefix}-{t0}-{new_duration}.npy"
        np.save(fname,array_to_save)
    



ifos = ["H1","L1"]

#C02?
flagname = ":DMT-ANALYSIS_READY:1"

flagname = "_DATA"

#C00
#flagname = ":GDS-CALIB_STRAIN"
#flagname = ":DCS-ANALYSIS_READY_C02:1"

#C01
#flagname = 

#gw_data_find --help



flags = [ifo + flagname for ifo in ifos]


#1239150592 is some time in O3, a bit after GW190425

#for my BNS samples, I use 1010 seconds of noise, then remove the outer 5 seconds due to whitening

#all the noise I need (and more) can be achieved in this 5e5 seconds of data

#segs = query_segments(flags, 1239150592, 1239150592 + 5e5, min_duration = 1010)

#segs = query_segments(flags, 1239213516, 1239213516 + 2e4, min_duration = 1000)


#segs = query_segments(flags, 1239554063, 1239554063 + 1500, min_duration = 1)

#segs = query_segments(["H1:DMT-ANALYSIS_READY:1","L1:DMT-ANALYSIS_READY:1"], 1239150592, 1239150592 + 1e4, min_duration = 1000)
segs = [[1239554617, 1239555563]]

segs, h1, l1 = combine_seg_list('noise/segments/H1_O3a.txt', 'noise/segments/L1_O3a.txt',1239553063,1239555563, min_duration=1000)


print(segs)

#data = find_data(segs, ifos)

#channelname = ":GDS-CALIB_STRAIN"
channelname = ""
channelname = ""

#channelname = ":DCS-CALIB_STRAIN_CLEAN-SUB60HZ_C01"

channels = [ifo + channelname for ifo in ifos]


data = find_data(segs, channels)



for segment in data:
    segment.resample(sample_rate)

    print(segment)
    t0 = int(segment[channels[0]].t0.value)
    length = len(segment[channels[0]])//sample_rate

    arr = np.zeros(shape=(len(ifos),len(segment[channels[0]])))
    prefix = ""

    for i in range(len(ifos)):
        arr[i] = segment[channels[i]]
        prefix += ifos[i][0]

    #at this point, we need to check for NaNs

    if np.any(np.isnan(arr)):
        print("NANS DEETECTED!")
        #split_segment(arr,t0,prefix)
    #else:
    fname = write_dir + "/" + f"{prefix}-{t0}-{length}.npy"
    np.save(fname,arr)




""" 
for ifo in ifos:
    #data = find_data(segs, ["H1","L1"])
    data = find_data(segs, [ifo])
    
    
    for segment in data:
        segment.resample(2048)
 """
"""
data = find_data(segs, ["L1"])
for segment in data:
    print(segment)

"""

