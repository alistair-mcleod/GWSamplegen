"""
============
Helper functions for inserting waveforms into glitchy noise
============

`get_glitchy_times` returns a list of glitchy times and glitchless times from a glitch file and a list of valid times.

`get_glitchy_gps_time` takes a list of valid times from `get_valid_noise_times` and a glitchy time from `get_glitchy_times` 
and offsets the glitchy time to ensure the glitch occurs in the SNR time series. This is necessary for BNS and NSBH samples,
as the peak frequency of the glitch can be tens of seconds from the merger.
"""


import numpy as np
from GWSamplegen.waveform_utils import t_at_f
from typing import Iterator, List, Optional, Sequence, Tuple
from pathlib import Path

def get_glitchy_times(
		glitch_file: Path, 
		duration: int,
		valid_times: List[int], 
		longest_waveform: int, 
		SNR_cutoff: float = 5.0, 
		freq_cutoff: float = 0.0, 
		seconds_before: int = 1, 
		seconds_after: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""
	Given a list of valid times and a glitch file from find_glitches.py, return a list of glitchy times and glitchless times.

	Parameters
	----------

	glitch_file: Path
		Path to a glitch file from find_glitches.py
	duration: int
		length of noise the waveforms are inserted into
	valid_times: List[int]
		List of valid times from get_valid_noise_times
	longest_waveform: int
		Length of the longest waveform in seconds
	SNR_cutoff: float
		Minimum SNR of glitches to consider
	freq_cutoff: float
		Minimum frequency of the glitches to consider
	seconds_before: int
		Number of seconds before the glitch to exclude from the glitchy times
	seconds_after: int
		Number of seconds after the glitch to exclude from the glitchy times

	Returns
	-------
	glitchy_times: np.ndarray
		List of glitchy times
	glitchless_times: np.ndarray
		List of glitchless times
	frequency_list: np.ndarray
		List of frequencies of the glitches

	"""
	glitch_data = np.load(glitch_file, allow_pickle=True).item()

	glitch_array = np.array([glitch_data['time'], glitch_data['frequency'], glitch_data['snr'], 
				glitch_data['tstart'], glitch_data['tend'], glitch_data['fstart'], glitch_data["fend"]]).T

	# select only times with SNR and end frequency above cutoff.
	glitch_array = glitch_array[(glitch_array[:,2] > SNR_cutoff) & (glitch_array[:,6] > freq_cutoff)]

	#also cut glitches that are outside of valid_times
	glitch_array = glitch_array[(glitch_array[:,0] > valid_times[0]) & (glitch_array[:,0] < valid_times[-1] + duration)] 

	#no_glitch = np.array([])
	glitch = []
	frequency_list = []
	snr_list = []
	glitch_idxs = []
	no_glitch = np.ones(len(valid_times), dtype=bool)

	for i in range(len(glitch_array)):
		idx_start = np.searchsorted(valid_times, int(glitch_array[i,3]-seconds_after - 1024//2))
		idx_end =  np.searchsorted(valid_times,int(glitch_array[i,4]+longest_waveform + seconds_before - 1024//2))

		if idx_start == 0 and idx_end == 0:
			continue
		no_glitch[idx_start:idx_end] = False
		#TODO: expand include based on the length of the waveform. longer samples can be injected multiple times into a glitch.
		include = int(glitch_array[i,0] - duration//2 +1)

		#no_glitch = np.hstack((no_glitch, exclude))

		idx = np.searchsorted(valid_times, int(glitch_array[i,0] - 1024//2 +1))
		if idx < len(valid_times) - 1:
			if valid_times[idx] == int(glitch_array[i,0] - 1024//2 +1):
				glitch_idxs.append(int(glitch_array[i,0] - 1024//2 +1))
				frequency_list.append([glitch_array[i,5], glitch_array[i,1], glitch_array[i,6]])
				snr_list.append(glitch_array[i,2])
		
	#no_glitch = np.unique(no_glitch)
	snr_list = np.array(snr_list)
	glitchless_times = valid_times[no_glitch]
	glitchy_times = np.array(glitch_idxs)

	#frequency list now contains fstart and fend for a glitch
	frequency_list = np.array(frequency_list)
	frequency_list = np.clip(frequency_list, freq_cutoff, None)

	print("There are {} glitchy times and {} glitchless times in {}".format(len(glitchy_times), len(glitchless_times), glitch_file[-15:]))

	return glitchy_times, glitchless_times, frequency_list, snr_list

def get_glitchy_gps_time(
		valid_times: List[int],
		mass1: float,
		mass2: float, 
		glitch_time: int, 
		frequencies: float,
		snr: float,
		random_offset_snr_thresh: float = 50
)-> int:
	"""
	Offset a glitchy time to ensure the glitch occurs in the SNR time series. This is necessary for BNS and NSBH samples,
	as the peak frequency of the glitch can be tens of seconds from the merger.

	Parameters
	----------

	valid_times: List[int]
		List of valid times from get_valid_noise_times
	mass1: float
		Primary mass of the waveform
	mass2: float
		Secondary mass of the waveform
	glitch_time: int
		Glitch time from get_glitchy_times
	
	frequencies: float
		fstart, peak frequency, fend of the glitch
	
	snr: float
		SNR of the glitch

	random_offset_snr_thresh: float
		Threshold SNR to use a random offset for the glitch time. 
		If the SNR is below this threshold, the peak frequency is used to offset the glitch time.
		Glitches above the SNR threshold can have the signal injected at a random frequency.

	Returns
	-------
	glitch_time: int
		Offset glitch time

	Warning
	-------
	If the glitch cannot be offset enough to ensure it appears in the SNR time series, 
	and a warning is printed and the closest valid time is returned.
	"""

	t_offsets = t_at_f(mass1,mass2,frequencies)
	if snr > random_offset_snr_thresh:
		#print("Using Random offset")
		#pick a random time between times[0] and times[2] to inject a glitch, with a preference for a time near times[1]
		t_offsets[0] = np.mean(t_offsets[:2])
		#print(t_offsets)
		selected_time = int(np.random.triangular(t_offsets[2], t_offsets[1], t_offsets[0]))
		
	else:
		#print("Can't use random offset")
		#otherwise, just use the peak frequency to offset the glitch
		selected_time = t_offsets[1]
	
	#print("selected time: ", selected_time)

	if int(glitch_time + selected_time) in valid_times:
		return int(glitch_time + selected_time)#, (glitch_time + t_offsets).astype(int)
	else:
		print("WARNING: CANNOT OFFSET GPS TIME ENOUGH TO ENSURE GLITCH APPEARS IN SNR TIME SERIES")
		return valid_times[np.argmin(np.abs(valid_times - int(glitch_time + selected_time)))]