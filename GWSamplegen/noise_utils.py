"""
============
Functions for fetching, loading and selecting noise segments.
============

`load_noise` loads a directory of noise into memory.

`get_valid_noise_times` is a multipurpose function for returning a list of GPS times to inject samples into, and a list of sorted file paths.

`load_psd` loads the PSD of each interferometer from a noise directory.

`fetch_noise_loaded` fetches noise segments from a list of noise files loaded into memory.

`generate_time_slides` generates a list of time slides from a list of noise segments (vital if you want to generate more than ~10000 samples).
"""

import os
import numpy as np
from typing import List, Tuple, Generator
from pycbc.types.timeseries import TimeSeries
from pycbc.types import FrequencySeries
from pycbc.psd import interpolate, inverse_spectrum_truncation
import json
from GWSamplegen.waveform_utils import t_at_f
from pathlib import Path
from importlib import resources as impresources
from GWSamplegen import segments
import h5py
from gwpy.timeseries import TimeSeries as GWPYTimeSeries

#TODO: handle arbitrary groups of interferometers. Assume each noise file in a dir has the same ifos

def load_gps_blacklist(
	f_lower: int, 
	event_file: Path = None
) -> np.ndarray:
	"""Load a .json file containing real event data, for the purpose of removing valid GPS times so that
	we don't accidentally include real events in training sets or background data."""
	
	if event_file is None:
		event_file = impresources.files(segments).joinpath("event_gpstimes.json")
	else:
		event_file = Path(event_file)

	with open(event_file) as f:
		events = json.load(f)['events']
	gps_blacklist = []

	for event in events:

		if events[event]['mass_1_source'] is not None and events[event]['mass_2_source'] is not None \
			and events[event]['redshift'] is not None and events[event]['GPS'] is not None:
			
			m1_det = events[event]['mass_1_source'] * (1 + events[event]['redshift'])
			m2_det = events[event]['mass_2_source'] * (1 + events[event]['redshift'])
			start = np.floor(events[event]['GPS'] - t_at_f(m1_det, m2_det, f_lower))
			end = np.ceil(events[event]['GPS'] + 1)
			gps_blacklist.append(np.arange(start, end))

	return np.sort(np.concatenate(gps_blacklist))



def get_valid_noise_times(
	noise_dir: str,
	noise_len: int,
	min_step: int = 1,
	start_time: int = None,
	end_time: int = None,
	blacklisting: bool = True,
	blacklisting_file: Path = None,
	f_lower = 30
) -> Tuple[List[int], np.ndarray, List[Path]]:
	"""multipurpose function to return a list of valid GPS start times, a list of noise file paths 
	and a list of deconstructed file names.

	Parameters
	----------
	
	noise_dir: str
		directory containing noise files

	noise_len: int
		minimum length of noise segments to consider. Any file shorter than noise_len in noise_dir are ignored.
		This should be the duration of noise you're injecting signals into.

	min_step: int
		If specified, will ensure that valid times are min_step seconds apart from each other. Used for background
		and injection runs to step through noise files. Should be at most noise_len minus the longest filter used.
		If not specified, will produce 1 second steps of noise times which can be used for making training/testing sets.

	start_time: int
		if specified, the start of the time window to consider. Otherwise, all noise in noise_dir will be used.
	
	end_time: int
		if specified, the end of the time window to consider
	
	blacklisting: bool
		if True, will remove any GPS times that are too close to detected events. Defaults to True.
	
	blacklisting_file: Path
		path to a json file containing GPS times of detected events. Defaults to None, in which case the default
		file from the GWSamplegen package is used. This file currently contains all detected events up to and including O3.

	Returns
	-------

	valid_times: List[int] 
		A list of valid start times for noise segments

	paths: np.ndarray
		array of deconstructed file names, giving detector info, segment start time and segment duration
	
	file_list List[Path]: 
		list of noise file paths in chronological order
	"""

	valid_times = np.array([])
	
	#get all strain file paths from the noise directory, then extract their start time and duration
	paths = os.listdir(noise_dir)
	paths = [path.split("-") for path in paths if len(path.split("-")) == 3]

	#paths[0] is the interferometer list
	#paths[1] is the start time
	#paths[2] is the duration

	ifo_list = paths[0][0]
	
	valid_paths = []
	for path in paths:
		if int(path[2][:-4]) >= noise_len:
			if start_time is not None and end_time is not None:
				if int(path[1]) <= start_time and int(path[1]) + int(path[2][:-4]) - start_time >= noise_len:
					valid_paths.append(path)
				
				elif int(path[1]) >= start_time and int(path[1]) + int(path[2][:-4]) <= end_time:
					valid_paths.append(path)
				
				elif int(path[1]) < end_time and int(path[1]) + int(path[2][:-4]) - end_time >= noise_len:
					valid_paths.append(path)

				else:
					pass
			
			else:
				valid_paths.append(path)

	paths = valid_paths
	for path in paths:
		path[1] = int(path[1])
		path[2] = int(path[2][:-4])

		times = np.arange(path[1], path[1]+path[2] - noise_len + 1, min_step)
		if path[1] + path[2] - noise_len not in times:

			if int((path[1] + path[2] - noise_len) - times[-1]) > 2 and min_step != 1:
				#This if-else condition is to solve the edge case of a 1 or 2 second noise segment.
				#only relevant if min_step is not 1.
				times = np.append(times, path[1] + path[2] - noise_len)
			
			else:
				print("ignoring a {} second segment".format(int((path[1] + path[2] - noise_len) - times[-1])))

		valid_times = np.concatenate((valid_times,times))

		if start_time is not None and end_time is not None:
			valid_times = valid_times[(valid_times >= start_time) & (valid_times + noise_len <= end_time) ]
		
	#ensure the file paths are in chronological order
	paths = np.array(paths)
	paths = paths[np.argsort(paths[:,1])]

	valid_times = np.sort(valid_times)
	
	#remove any GPS times that are too close to detected events

	if blacklisting:
		
		gps_blacklist = load_gps_blacklist(f_lower, blacklisting_file)
		n_blacklisted = len(np.where(np.isin(valid_times, gps_blacklist-noise_len//2))[0])
		print("{} GPS times are too close to detected events and have been removed".format(n_blacklisted))
		valid_times = np.delete(valid_times, np.where(np.isin(valid_times, gps_blacklist-noise_len//2)))

	#reconstruct the file paths from the start times and ifo_list
	file_list = [noise_dir +"/"+ ifo_list +"-"+ path[1] +"-"+ path[2] +".npy" for path in paths]

	return valid_times, paths, file_list


def load_noise(
	noise_dir: Path
) -> List[np.ndarray]:
	
	"""Loads noise segments from a directory into memory for fast processing."""

	_,_, fps = get_valid_noise_times(noise_dir,0)

	segments = []

	for fp in fps:
		segments.append(np.load(fp))

	return segments


def fetch_noise_loaded(
	noise_list: List[np.ndarray],
	noise_len: int,
	noise_start_time: List[int],
	sample_rate: int,
	paths: np.ndarray
) -> np.ndarray[float]:

	"""Fetch an array of noise segments from a list of noise files.
	Supports timeslides by taking a list of start times.

	Parameters
	----------
	
	noise_list: List[np.ndarray] 
		List of noise files loaded into memory. Usually generated by `load_noise`.
	
	noise_len: int
		Length of noise segment to fetch, in seconds.

	noise_start_time: List[int]
		list of start times for each detector's noise. 
		If not timesliding, this should just be a list of the same GPS time.

	sample_rate: int
		sample rate of noise.

	paths: array of noise file paths, generated by `get_valid_noise_times`
	
	Returns
	-------

	noises: np.ndarray[float]
		Array of noise segments, with shape (len(noise_start_time), noise_len*sample_rate)
	"""

	noises = np.empty(shape = (len(noise_start_time),noise_len*sample_rate))
	
	for i in range(len(noise_start_time)):
		f_idx = np.searchsorted(paths[:,1].astype('int'), noise_start_time[i],side='right') -1
		start_idx = int((noise_start_time[i] - paths[f_idx,1].astype('int'))*sample_rate)
		noises[i] = np.copy(noise_list[f_idx][i,start_idx:start_idx + noise_len * sample_rate])

	return noises

def generate_time_slides(
	detector_data: List[int], 
	min_distance: int
) -> Generator[Tuple[int, int], None, None]:
	
	"""Generates a list of time slides from a list of noise segments. Currently superseded by `two_det_timeslide`."""

	num_detectors = len(detector_data)
	data_lengths = [len(data) for data in detector_data]
	indices = [list(range(length)) for length in data_lengths]
	used_combinations = set()

	while True:
		min_length = min(data_lengths)
		# Limits the number of possible samples we draw from the generator
		if len(used_combinations) == (min_length - (min_distance - 1)) * (min_length - min_distance):
			print("No more unique combinations available.")
			return
		
		sample_indices = [np.random.choice(indices[i]) for i in range(num_detectors)]
		
		if all(abs(sample_indices[i] - sample_indices[j]) >= min_distance for i in range(num_detectors) for j in range(i+1, num_detectors)):
			combination = tuple(sample_indices)
			
			if combination not in used_combinations:
				used_combinations.add(combination)
				yield tuple(detector_data[i][sample_indices[i]] for i in range(num_detectors))


def two_det_timeslide(
		detector_data: List[List[int]], 
		min_distance: int,
		max_distance: int = 604800
) -> Generator[Tuple[int], None, None]:
	
	"""A generator that returns a time slide from a list of valid noise times.
	Avoids creating time slides that are too similar to previous time slides.
	NOTE: this function actually works for N >= 2 detectors. TODO: rename it.
	
	Parameters
	----------
	
	detector_data: List[List[int]]
		Lists of noise times for each detector. Can be generated by `get_valid_noise_times`, and is compatible with
		noise time lists from `get_glitchy_times`.
		
	min_distance: int
		Minimum distance between time slides. This is the minimum number of seconds between the start times of two noise segments.

	max_distance: int
		Maximum distance between time slides. This is the maximum number of seconds between the start times of two noise segments.
		The default is 604800 seconds, or 1 week.
	"""


	used_combinations = set()

	data_lengths = [len(data) for data in detector_data]
	n_detectors = len(detector_data)
	min_length = min(data_lengths)
	max_combos = (min_length - (min_distance - 1)) * (min_length - min_distance)
	while True:
		#use a Cantor pairing function to generate unique indices
		idx = np.random.randint(0,np.prod(data_lengths))
		sample_indicies = np.unravel_index(idx, data_lengths)

		# Limits the number of possible samples we draw from the generator
		if len(used_combinations) == max_combos:
			print("No more unique combinations available.")
			return

		#get absolute difference between ALL N pairs of noise times
		a, b = np.triu_indices(n_detectors,1)
		separations = np.abs(np.array(sample_indicies)[a] - np.array(sample_indicies)[b])
		if np.all(separations >= min_distance)  and np.all(separations <= max_distance):
			if sample_indicies not in used_combinations:

				used_combinations.add(sample_indicies)
				#yield detector gps times as a tuple
				yield tuple(detector_data[i][sample_indicies[i]] for i in range(n_detectors))


#utilities for downloading noise for later use

def overlapping_intervals(
	arr1: List[Tuple[int]], 
	arr2: List[Tuple[int]]):
	"""
	Find overlapping segments between two lists of segments from different
	detectors.

	Designed to be used with the outputs of `get_seg_list` as the inputs.

	Parameters
	----------

	arr1: List[Tuple[int]]
		List of segments from the first detector
	
	arr2: List[Tuple[int]]
		List of segments from the second detector

	"""
	res = []
	arr1_pos = 0
	arr2_pos = 0
	len_arr1 = len(arr1)
	len_arr2 = len(arr2)
	# Iterate over all intervals and store answer
	while arr1_pos < len_arr1 and arr2_pos < len_arr2:
		arr1_seg = arr1[arr1_pos]
		arr2_seg = arr2[arr2_pos]

		# arr1_seg fully inside of arr2_seg
		if arr1_seg[0] >= arr2_seg[0] and arr1_seg[1] <= arr2_seg[1]:
			res.append(arr1_seg)
			arr1_pos += 1

		# arr2_seg fully inside of arr1_seg
		elif arr2_seg[0] >= arr1_seg[0] and arr2_seg[1] <= arr1_seg[1]:
			res.append(arr2_seg)
			arr2_pos += 1

		# arr1_seg fully below arr2_seg
		elif arr1_seg[0] <= arr2_seg[0] and arr1_seg[1] <= arr2_seg[0]:
			arr1_pos += 1

		# arr2_seg fully below arr1_seg
		elif arr2_seg[0] <= arr1_seg[0] and arr2_seg[1] <= arr1_seg[0]:
			arr2_pos += 1

		# arr1_seg overlaps start of arr2_seg
		elif arr1_seg[0] <= arr2_seg[0] <= arr1_seg[1] <= arr2_seg[1]:
			res.append([arr2_seg[0], arr1_seg[1]])
			arr1_pos += 1

		# arr2_seg overlaps start of arr1_seg
		elif arr2_seg[0] <= arr1_seg[0] <= arr2_seg[1] <= arr1_seg[1]:
			res.append([arr1_seg[0], arr2_seg[1]])
			arr2_pos += 1

	if not res:
		return [[]]

	return res

def exclusive_intervals(arr0, arr1):
	"""
	Find segments where detector arr0 is active but arr1 is not.

	Designed to be used with the outputs of `get_seg_list` as the inputs.

	Parameters
	----------

	arr1: List[Tuple[int]]
		List of segments from the first detector
	
	arr2: List[Tuple[int]]
		List of segments from the second detector

	"""
	res = []
	check_time = 0
	for seg0 in arr0:
		check_time = seg0[0]
		for seg1 in arr1:
			if seg1[1] <= check_time:
				#this seg1 is irrelevant
				continue
			elif seg1[0] >= seg0[1]:
				#no more overlapping seg1s
				if check_time == seg0[0]:
					#the whole seg0 is valid
					res.append(seg0)
				break
			else:
				if seg1[0] > check_time:
					#there is a valid segment before seg1
					res.append([check_time, min(seg1[0], seg0[1])])
				check_time = max(check_time, seg1[1])
				print("check_time updated to ", check_time)
				if check_time >= seg0[1]:
					#no more valid time in seg0
					break
		if check_time < seg0[1]:
			#there is a valid segment at the end of seg0
			res.append([check_time, seg0[1]])
	if not res:
		return [[]]
	return res

def get_seg_list(
	file_name: Path, 
	macrostart: int, 
	macroend: int
) -> List[Tuple[int]]:
	"""
	Get a list of segments from a single detector segment file bounded
	by a GPS time window.

	Parameters
	----------

	file_name: Path
		path to detector's segment list

	macrostart: int
		Start GPS time of overlap window
	
	macroend: int
		End GPS time of overlap window

	Returns
	-------

	good_segs: List[Tuple[int]]
		List of segments from the detector within the GPS time window
	"""
	file = open(file_name, "r")

	good_segs = []

	for i in file.readlines():
		times = i.split(" ")
		if int(times[1]) < macrostart or int(times[0]) > macroend:
			continue
		else:
			good_segs.append([int(times[0]), int(times[1])])

	file.close()

	if good_segs[0][0] < int(macrostart):
		good_segs[0][0] = int(macrostart)
	if good_segs[-1][1] > int(macroend):
		good_segs[-1][1] = int(macroend)

	return good_segs


def combine_seg_list(
	file_h1: Path, 
	file_l1: Path, 
	macrostart: int, 
	macroend: int, 
	min_duration: int
) -> Tuple[List[Tuple[int]], List[Tuple[int]], List[Tuple[int]]]:
	"""
	Find overlapping segments between two detectors within a window
	defined by two GPS times.

	Parameters
	----------

	file_h1: Path
		path to H1 complete segment list
	
	file_l1: Path
		path to L1 complete segment list
	
	macrostart: int
		Start GPS time of overlap window
	
	macroend: int
		End GPS time of overlap window

	min_duration: int
		Minimum duration of segments to consider

	Returns
	-------

	good_segs: List[Tuple[int]]
		List of segments overlapping between the two detectors
	"""
	good_segs_h1 = get_seg_list(file_h1, macrostart, macroend)
	good_segs_l1 = get_seg_list(file_l1, macrostart, macroend)

	good_segs = overlapping_intervals(good_segs_h1, good_segs_l1)

	#remove segments shorter than min_duration
	good_segs = [x for x in good_segs if x[1] - x[0] >= min_duration]

	return good_segs, good_segs_h1, good_segs_l1

def exclusive_seg_list(
		ifo_include: Path,
		ifo_exclude: Path,
		macrostart: int,
		macroend: int,
		min_duration: int,
		*args: Path
) -> Tuple[List[Tuple[int]], List[Tuple[int]], List[List[Tuple[int]]]]:

	"""
	Find segments where detector ifo_include is active but ifo_exclude is not. Additional
	detectors can be excluded by passing in additional segment list paths as args.

	Parameters
	----------

	ifo_include: Path
		path to segment list of the detector to include
	
	ifo_exclude: Path
		path to segment list of the detector to exclude
	
	*args: Path
		Additional paths to segment lists of detectors to exclude
	Returns
	-------
	good_segs: List[Tuple[int]]
		List of segments where ifo_include is active but ifo_exclude is not
	"""
	good_segs_include = get_seg_list(ifo_include, macrostart, macroend)
	excluded = []
	good_segs_exclude = get_seg_list(ifo_exclude, macrostart, macroend)
	excluded.append(good_segs_exclude)
	for ifo in args:
		good_segs_exclude = get_seg_list(ifo, macrostart, macroend)
		excluded.append(good_segs_exclude)
	for ex in excluded:
		good_segs_include = exclusive_intervals(good_segs_include, ex)
	good_segs = good_segs_include
	#remove segments shorter than min_duration
	good_segs = [x for x in good_segs if x[1] - x[0] >= min_duration]
	return good_segs, good_segs_include, excluded


#PSD UTILS

def construct_noise_PSD(
	noise_paths: List[str]
) -> None:
	"""Create and save an averaged PSD of noise segments. 
	This way of doing things is fine so long as the noise is stationary
	(which it is provided the segments do not span longer than ~1-2 weeks.)

	Parameters
	----------

	noise_paths: List[str]
		List of paths to noise segments, from `get_valid_noise_times`. The PSD is automatically saved
		to the same directory as the noise segments.
	"""

	segments = []

	for noise_path in noise_paths:
		segments.append(np.load(noise_path))

	ifo_psds = []

	#iterate through interferometers
	for ifo in range(len(segments[0])):
		
		psds = []
		for i in range(len(segments)):
			#setting NaNs in segments to 0. This hopefully shouldn't be needed!
			segments[i][ifo][np.isnan(segments[i][ifo])] = 0
			seg = TimeSeries(segments[i][ifo],delta_t=1/2048)
			psd = seg.psd(4)
			psds.append(psd.data * len(segments[i][ifo]))
			
		if ifo == 0:
			ifo_psds.append(psd.sample_frequencies.data)
		
		total_length = 0
		for segment in segments:
			total_length += len(segment[ifo])

		psd_avg = np.sum(np.array(psds), axis = 0)/total_length
		ifo_psds.append(psd_avg)

	ifo_psds = np.array(ifo_psds)

	#return ifo_psds
	np.save(os.path.dirname(noise_paths[0]) + "/psd.npy", ifo_psds)


def load_psd(
	noise_dir: str,
	duration: int,
	ifos: List[str],
	f_lower: int,
	sample_rate: int
) -> np.ndarray[float]:
	
	"""Load the PSD of each interferometer from a noise directory.
	
	Parameters
	----------
	
	noise_dir: str
		Path to the directory containing noise files.
	
	duration: int
		Target length of the noise segments in seconds.
	
	ifos: List[str]
		List of interferometers to load PSDs for.
	
	f_lower: int
		Lower frequency cutoff for the PSDs.
	
	sample_rate: int
		Sample rate of the noise segments.
	
	Returns
	-------

	psds: np.ndarray[float]
		Array of PSDs, with shape (len(ifos), len(psd_freqs))
		"""

	with open(noise_dir+ '/args.json') as f:
		args = json.load(f)
		ifo_list = args['detectors']
		#TODO: map ifo list to requested ifos.

	psd = np.load(noise_dir + "/psd.npy")
	psds = {}
	for i in range(len(ifos)):
		psds[ifos[i]] = FrequencySeries(psd[ifo_list.index(ifos[i])+1], delta_f = psd[0][1], dtype = np.complex128)
		psds[ifos[i]] = interpolate(psds[ifos[i]], delta_f= 1/(duration))
		psds[ifos[i]] = inverse_spectrum_truncation(psds[ifos[i]], int(4 * sample_rate),
										low_frequency_cutoff=f_lower)
		
	return psds


def get_valid_noise_times_from_segments(
		segment_tuples: List[Tuple[int,int]],
		noise_len: int,
		min_step: int = 1,
		run: str = "O3a",
		ifos: List[str] = ["H1", "L1"],
		blacklisting = True,
		f_lower = 30,
		) -> np.ndarray:
	"""Get valid noise times from a list of segment tuples. Returns a list of valid GPS times.
	Produces more general GPS time lists than `get_valid_noise_times`, but does not return valid
	noise file paths. 
	
	Parameters
	----------
	segment_tuples: List[Tuple[int,int]]
		List of tuples containing start and end GPS times of segments.
	noise_len: int
		Minimum length of noise segments to consider, in seconds.
	min_step: int
		If specified, will ensure that valid times are min_step seconds apart from each other.
	run: str
		The observation run to use for the noise files. TODO: add O4a and support for multiple runs.
	ifos: List[str]
		List of interferometers to consider. Currently supports times when H1 and L1 are both active, or when only one of them is active.
		TODO: add support for only considering one ifo, irrespective of the state of the other.
	blacklisting: bool
		If True, will remove any GPS times that are too close to detected events.
	f_lower: int
		Lower frequency cutoff for the PSDs, used for blacklisting."""
	
	valid_times = np.array([])
	ifo_1 = "{}_{}.txt".format(ifos[0], run)
	if len(ifos) > 1:
		ifo_2 = "{}_{}.txt".format(ifos[1], run)
	elif ifos[0] == "H1":
		ifo_2 = "{}_{}.txt".format("L1", run)
	elif ifos[0] == "L1":
		ifo_2 = "{}_{}.txt".format("H1", run)
	ifo_1 = impresources.files(segments).joinpath(ifo_1)
	ifo_2 = impresources.files(segments).joinpath(ifo_2)
	ifo_3 = impresources.files(segments).joinpath("V1_{}.txt".format(run)) #TODO: remove the requirement that V1 is excluded

	for start_time, end_time in segment_tuples:
		if ifos == ["H1", "L1"]:
			segs, _, _ = combine_seg_list(ifo_1,ifo_2,start_time,end_time, min_duration=noise_len)
		elif len(ifos) ==1:
			segs, _, _ = exclusive_seg_list(ifo_1, ifo_2, start_time, end_time, noise_len, ifo_3)

		for seg in segs:
			times = np.arange(seg[0], seg[1] - noise_len +1 , min_step)
			if seg[1] - noise_len not in times:
				if (seg[1] - noise_len) - times[-1] > 2 and min_step != 1:
					times = np.append(times, seg[1] - noise_len)
				else:
					
					print("ignoring a {} second segment".format((seg[1] - noise_len) - times[-1]))
			valid_times = np.concatenate([valid_times, times])

		if blacklisting:

			gps_blacklist = load_gps_blacklist(f_lower)
			n_blacklisted = len(np.where(np.isin(valid_times, gps_blacklist-noise_len//2))[0])
			print("{} GPS times are too close to detected events and have been removed".format(n_blacklisted))
			valid_times = np.delete(valid_times, np.where(np.isin(valid_times, gps_blacklist-noise_len//2)))
	valid_times = np.sort(valid_times)
	return valid_times


def psd_preceding(noise, sample_rate, f_lower, delta_f, seconds_before = 10):
	ret = noise[:int(len(noise)//2 - seconds_before*sample_rate)].psd(4).astype('complex128')
	ret = interpolate(ret, delta_f)
	ret = inverse_spectrum_truncation(ret, int(4 * sample_rate), low_frequency_cutoff=f_lower)
	return ret

def segment_psd(segment, duration, detector, delta_t, f_lower):
	psd = get_data_from_OzStar(segment, duration, detector).psd(4)
	psd = interpolate(psd, delta_f = 1/duration)
	psd = inverse_spectrum_truncation(psd, int(4/delta_t), low_frequency_cutoff=f_lower)
	return psd

import multiprocessing as mp
def psd_from_segments(segments, duration, detector, delta_t, f_lower, max_segments = 100):
	psd_seg_list = get_valid_noise_times_from_segments(segments,duration,duration)
	#to avoid processing too many segments
	if len(psd_seg_list) > max_segments:
		psd_seg_list = psd_seg_list[::len(psd_seg_list)//max_segments]
	psds = []
	with mp.Pool(8) as p:
		psds = p.starmap(segment_psd, [(i, duration, detector, delta_t, f_lower) for i in psd_seg_list])
	psd_array = np.array(psds)
	psd = np.median(psd_array, axis=0)
	#return a pycbc frequencyseries
	return FrequencySeries(psd, delta_f = psds[0].delta_f, epoch = psds[0].epoch)


def get_data_from_OzStar(gps_start, duration, ifo, verbose = False, root = "/datasets/LIGO/public/"):
	"""OzStar-specific function for fetching GW data."""
	if gps_start != int(gps_start):
		print("NOTE: you have specified a non-integer GPS time to fetch. Make sure this is what you want!")
	if gps_start >= 1249850209 and gps_start + duration <= 1249850209 + 4096:
		#edge case since Hanford was locked but not in observing mode for GW190814
		print("we're looking for GW190814 data")
		fp = "/fred/oz016/alistair/GWSamplegen/noise/GW190814/GW190814_{}_4096.hdf5".format(ifo)
		f = h5py.File(fp, 'r')
		print("loaded")
		data = f['strain']['Strain'][()]
		start_idx = int((gps_start-1249850209)*4096)

		data = data[start_idx:start_idx+duration*4096]
		data = TimeSeries(data, delta_t = 1/4096, epoch = gps_start)
		data = data.resample(1/2048)
		return data
	elif gps_start >= 1180920447 and gps_start + duration <= 1180920447 + 4096:
		#edge case since Hanford was not in observing mode for GW170608
		print("We're looking for GW170608 data")
		fp = "/fred/oz016/alistair/GWSamplegen/noise/GW170608/GW170608_{}_4096.hdf5".format(ifo)
		f = h5py.File(fp, 'r')
		print("loaded")
		data = f['strain']['Strain'][()]
		start_idx = int((gps_start-1180920447)*4096)

		data = data[start_idx:start_idx+duration*4096]
		data = TimeSeries(data, delta_t = 1/4096, epoch = gps_start)
		data = data.resample(1/2048)
		return data

	root = os.path.join(root, "gwosc.osgstorage.org", "gwdata")
	#by looking in reverse order we avoid prematurely selecting the wrong chunk
	ObsRuns = ["O3b", "O3a", "O2", "O1"]
	for run in ObsRuns:
		if verbose:
			print("Checking run", run)
		chunks = np.sort(np.array(os.listdir(os.path.join(root, run, "strain.4k", "hdf.v1", ifo)), dtype = int))
		chunkstart = np.where((chunks <= int(gps_start)))[0]
		#chunkend = np.where((chunks >= int(gps_start+duration)))[0]
		chunkend = np.where((chunks < int(gps_start+duration)) & (chunks > int(gps_start)))[0]
		if verbose:
			print("chunks:", chunks)
		if len(chunkstart) == 0:
			continue
		else:
			chunkstart = chunkstart[-1]
			if len(chunkend) == 0:
				chunkend = chunkstart
			else:
				chunkend = chunkend[0] 
				if verbose:
					print("crosses two chunks")
			if verbose:
				print("Found chunk", chunkstart, chunkend)
			break

	if chunkend is None:
		print("Bad GPS time! not found in any chunk")
		return None

	segments_start = np.sort(np.array(os.listdir(os.path.join(root, run, "strain.4k", "hdf.v1", ifo,str(chunks[chunkstart])))))
	segments_end = np.sort(np.array(os.listdir(os.path.join(root, run, "strain.4k", "hdf.v1", ifo,str(chunks[chunkend])))))
						
	segments_split_start = np.array([seg.split("-") for seg in segments_start])
	segments_split_end = np.array([seg.split("-") for seg in segments_end])

	if verbose:
		print("segments start", segments_split_start[:,2])
	seg_idx = np.where(segments_split_start[:,2].astype('int') <= gps_start)[0][-1]
	seg_idx_end = np.where(segments_split_end[:,2].astype('int') + 4096 >= gps_start+duration)[0][0]

	if seg_idx != seg_idx_end:
		simple = False
	else:
		simple = True

	if simple:
		dat = GWPYTimeSeries.read(os.path.join(root, run, "strain.4k", "hdf.v1", ifo,str(chunks[chunkstart]),segments_start[seg_idx]), 
					format="hdf5.gwosc", start = gps_start, end = gps_start+duration)
		#if there are NaNs, replace them with zeros and print a warning
		if np.any(np.isnan(dat)):
			print("WARNING: Found NaNs in the data!")
			print("GPS time:", gps_start)
			print("Ifo:", ifo)

		dat = dat.to_pycbc()
		dat = dat.resample(1/2048)

	else:
		dat_start = GWPYTimeSeries.read(os.path.join(root, run, "strain.4k", "hdf.v1", ifo,str(chunks[chunkstart]),segments_start[seg_idx]),
				format="hdf5.gwosc", start = gps_start)
		dat_end = GWPYTimeSeries.read(os.path.join(root, run, "strain.4k", "hdf.v1", ifo,str(chunks[chunkend]),segments_end[seg_idx_end]),
				format="hdf5.gwosc", end = gps_start+duration)	
		dat_start = dat_start.to_pycbc()
		dat_end = dat_end.to_pycbc()
		#resample
		dat_start = dat_start.resample(1/2048)
		dat_end = dat_end.resample(1/2048)
		#pad the end of the first segment 
		dat_start.append_zeros(len(dat_end))
		#concatenate
		dat_start[-len(dat_end):] = dat_end
		dat = dat_start
	if np.any(np.isnan(dat.data)):
		print("WARNING: Found NaNs in the data!")
		print("GPS time:", gps_start)
		print("Ifo:", ifo)	
		print("Returning None for now.")
		return None

	return dat


def get_data_from_local(gps_start, duration, ifo, gwf_files, verbose = False):
	"""Check local .gwf files for data covering the requested gps time. Designed
	to assist with running scripts in a HTCondor environment."""
	
	obsrun_format = {
		"O1": {"frame": "_LOSC_4_V1", "channel": ":LOSC-STRAIN"},
		"O2": {"frame": "_GWOSC_O2_4KHZ_R1", "channel": ":GWOSC-4KHZ_R1_STRAIN"},
		"O3a": {"frame": "_GWOSC_O3a_4KHZ_R1", "channel": ":GWOSC-4KHZ_R1_STRAIN"},
		"O3b": {"frame": "_GWOSC_O3b_4KHZ_R1", "channel": ":GWOSC-4KHZ_R1_STRAIN"},
		"O4a": {"frame": "_GWOSC_O4a_4KHZ_R1", "channel": ":GWOSC-4KHZ_R1_STRAIN"},
	}

	if gps_start >= 1126051217 and gps_start < 1137254417:
		run = "O1"
	elif gps_start >= 1164556817 and gps_start < 1187733618:
		run = "O2"
	elif gps_start >= 1238166018 and gps_start < 1253977218:
		run = "O3a"
	elif gps_start >= 1256655618 and gps_start < 1269363618:
		run = "O3b"
	elif gps_start >= 1368195220 and gps_start < 1389456018:
		run = "O4a"

	if verbose:
		print("Checking local .gwf files for data...")
	data = GWPYTimeSeries.read(gwf_files, channel = f"{ifo}{obsrun_format[run]["channel"]}", format="gwf", start = gps_start, end = gps_start+duration).to_pycbc()
	#print("data:", data)
	#resample to 2048 Hz
	data = data.resample(1/2048)
	return data