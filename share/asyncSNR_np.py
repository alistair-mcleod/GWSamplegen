""" Generate a dataset of SNR time series for a given set of parameters and noise segments."""


import os
import multiprocessing as mp
import argparse
import json
import numpy as np
from GWSamplegen.snr_utils_np import numpy_matched_filter, np_get_cutoff_indices
from GWSamplegen.noise_utils import get_valid_noise_times, load_noise, fetch_noise_loaded, load_psd, get_data_from_OzStar, psd_preceding, load_psds_from_txt
from GWSamplegen.chisq_utils_np import reduced_chisquared_precomputed_SNR
from GWSamplegen.waveform_utils import maximum_f_lower, select_approximant

from pycbc.filter import highpass
from pycbc.detector import Detector
from pycbc.types.timeseries import TimeSeries
from pycbc.waveform import get_fd_waveform, get_td_waveform, td_approximants, fd_approximants
import pycbc.noise
from GWSamplegen.spectrogram_utils import *
import time

#from astropy.utils import iers
#iers.conf.auto_download = False

from GWSamplegen.waveform_utils import get_projected_waveform_mp

def run_batch(n):
	#file_idx = templates_per_file * (np.ravel(template_ids[n:n+samples_per_batch])//templates_per_file)
	#template_idx = np.ravel(template_ids[n:n+samples_per_batch]) % templates_per_file

	t_ids = np.ravel(template_ids[n:n+samples_per_batch])

	t_ids = [int(i) for i in t_ids]
	batch_template_params = templates[t_ids]

	t_templates = np.zeros((n_templates * samples_per_batch, kmax-kmin), dtype=np.complex128)
	#start = time.time()

	# print(f'delta f: {delta_f} , f_final: {f_final} , kmin: {kmin}, kmax: {kmax}')
	
	for i in range(n_templates * samples_per_batch):
		temp_fd_approximant = select_approximant(batch_template_params[i,1], batch_template_params[i,2], fd_approximant, domain='frequency')
		t_templates[i] = get_fd_waveform(mass1 = batch_template_params[i,1], mass2 = batch_template_params[i,2], 
						spin1z = batch_template_params[i,3], spin2z = batch_template_params[i,4],
						approximant = temp_fd_approximant, f_lower = f_lower, delta_f = delta_f, f_final = f_final)[0].data[kmin:kmax]

	#create this batch's strains
	#TODO: optimise memory usage. we're creating the waveform and strain arrays separately, which is inefficient.
	strains = {}
	for ifo in ifos:
		strains[ifo] = np.zeros((samples_per_batch, duration*sample_rate))
		#reset the PSDs
		if noise_segments is not None and noise_type != "Gaussian":
			psds[ifo] = []
	print(psds[ifos[0]])
	for i in range(samples_per_batch):
		#print("sample:", n+i, "template", t_ids[i])
		if noise_type == "Gaussian":
			noise = np.zeros((len(ifos), duration*sample_rate))
			for j in range(len(ifos)):
				noise[j] = pycbc.noise.gaussian.noise_from_psd(duration*int(1/delta_t),delta_t,psds_full[ifos[j]])
				#seed=seed+n+i*len(ifos)+j
		else:
			if noise_segments is None:
				noise = fetch_noise_loaded(segments,duration,gps[n+i],sample_rate,paths)
			else:
				noise = []
				for ifo in range(len(ifos)):
					dat = get_data_from_OzStar(gps[n+i][ifo], duration, ifos[ifo])
					noise.append(dat.data)
					psds[ifos[ifo]].append(psd_preceding(dat, sample_rate, f_lower, delta_f).data[kmin:kmax])
				noise = np.array(noise)
				
		if params["injection"][n+i]:
			
			args = {'mass1': params['mass1'][n+i], 'mass2': params['mass2'][n+i],
					'spin1x': params['spin1x'][n+i], 'spin2x': params['spin2x'][n+i],
					'spin1y': params['spin1y'][n+i], 'spin2y': params['spin2y'][n+i],
					'spin1z': params['spin1z'][n+i], 'spin2z': params['spin2z'][n+i],
					'eccentricity': params['eccentricity'][n+i],
					'inclination': params['i'][n+i], 'distance': params['d'][n+i],
					'ra': params['ra'][n+i], 'dec': params['dec'][n+i],
					'pol': params['pol'][n+i], 'gps': params['gps'][n+i][0], "f_lower": f_lower, 
					"f_final": f_final, "delta_t": delta_t, "delta_f": delta_f, "td_approximant": td_approximant, "ifos": ifos}

			temp, merger_offset = get_projected_waveform_mp(args)
			#merger_offset = int(merger_offset*sample_rate)
			#print("merger offset", merger_offset)
						
			for ifo in ifos:
				#relevant quantities: 
				#1. how long the waveform currently is. should be no greater than 512 seconds
				#2. where the merger is. IF the merger is before the end of the strain AND the wform is 512 seconds long, we need to trim more
				#this shouldn't be used, waveforms should be shorter than the noise.
				w_len = np.min([len(temp[ifos.index(ifo)]), duration*sample_rate//2 + merger_offset])
				excess_waveform = len(temp[ifos.index(ifo)]) - duration*sample_rate//2 - merger_offset
				if excess_waveform < 0:
					excess_waveform = 0
				#else:
				#	print("trimming excess waveform, ", excess_waveform, "samples")
				#print("start, end: ", duration*sample_rate//2 - w_len + offset + merger_offset, duration*sample_rate//2 + offset + merger_offset)
				strains[ifo][i,duration*sample_rate//2 - w_len + offset + merger_offset: duration*sample_rate//2 + offset + merger_offset] = temp[ifos.index(ifo)][excess_waveform:]
				delta_t_h1 = all_detectors[ifo].time_delay_from_detector(other_detector=all_detectors[ifos[0]],
													right_ascension=params['ra'][n+i],
													declination=params['dec'][n+i],
													t_gps=params['gps'][n+i][0])

				strains[ifo][i] = np.roll(strains[ifo][i], round(delta_t_h1*sample_rate))

				strains[ifo][i] += noise[ifos.index(ifo)]
		else:
			#print("no injection in sample ", n+i)
			for ifo in ifos:
				strains[ifo][i] = noise[ifos.index(ifo)]
	#waveform_time += time.time() - start
	ret = {}

	for ifo in ifos:
		if noise_segments is not None or noise_type == "Gaussian":
			psds[ifo] = np.repeat(np.array(psds[ifo]), n_templates, axis=0)

		strain = [TimeSeries(strains[ifo][i], delta_t=delta_t) for i in range(samples_per_batch)]
		if save_strain:
			fp_s[ifos.index(ifo)][n - index * samples_per_file:samples_per_batch + n- index*samples_per_file] = strains[ifo].astype(np.float32)
			#(i)*n_templates*samples_per_batch + n_templates*n:(i+1)*n_templates*samples_per_batch + n_templates*n
			#fp_s[ifo][(i)*n_templates + n_templates*n:(i+1)*n_templates + n_templates*n] = strains[ifo][i]

		if save_spectrograms:
			for i in range(len(strains[ifo])):
				#compute spectrograms
				spec, re, im = process_single_series_complex(strains[ifo][i].astype(np.float32))
				q, vitmap, lineaware = process_single_series_qtransform(strains[ifo][i].astype(np.float32))
				fp_spec[ifos.index(ifo)][n + i - index*samples_per_file] = spec.astype(np.float32)
				fp_re[ifos.index(ifo)][n + i - index*samples_per_file] = re.astype(np.float32)
				fp_im[ifos.index(ifo)][n + i - index*samples_per_file] = im.astype(np.float32)
				fp_q[ifos.index(ifo)][n + i - index*samples_per_file] = q.astype(np.float32)
				fp_vitmap[ifos.index(ifo)][n + i - index*samples_per_file] = vitmap.astype(np.float32)
				fp_lineaware[ifos.index(ifo)][n + i - index*samples_per_file] = lineaware.astype(np.float32)

		strain = [highpass(i,f_lower).to_frequencyseries(delta_f=delta_f).data for i in strain]

		strain = np.array(strain)[:,kmin:kmax]
		#strain = tf.convert_to_tensor(strain, dtype=tf.complex128)
		#strains[ifo] = strain

		#strain = np.array([strain])[:,kmin:kmax]
		strain_np = np.repeat(strain, n_templates, axis=0)

		x = numpy_matched_filter(strain_np, t_templates, psds[ifo], N, kmin, kmax, duration, delta_t = delta_t, flow = f_lower)

		if reduced_chisq:
			#multiply by the reduced chisquared time series
			x *= reduced_chisquared_precomputed_SNR(x, t_templates, strain_np, psds[ifo], kmin, kmax, delta_f, num_bins = n_chisq_bins)

		ret[ifo] = x[:,len(x[0])//2-seconds_before*sample_rate+offset:len(x[0])//2+seconds_after*sample_rate+offset]
		del strain_np, x
		gc.collect()
	return ret

def create_memmap_file(file_path, shape, dtype=np.float32, data_type='SNR'):
	"""
	Create a memory-mapped file with the specified shape and data type.
	
	Args:
		file_path (str): Path to the memmap file.
		shape (tuple): Shape of the array.
		dtype (str): Data type of the array.
		data_type (str): Description of data being stored (e.g., 'SNR', 'strain').
		
	Returns:
		np.memmap: Memory-mapped array.
	"""
	while True:
		try:
			if not os.path.exists(file_path):
				# Create a new memmap file
				fp = np.memmap(file_path, dtype=dtype, mode='w+', shape=shape, offset = 128)
				print("Process creating {} memmap file".format(data_type))
			else:
				# Open existing memmap file
				fp = np.memmap(file_path, dtype=dtype, mode='r+', shape=shape, offset = 128)
				print("Process opened {} memmap file".format(data_type))
			return fp
		except:
			print(f"Error accessing {file_path}. Retrying in 1 second...")
			time.sleep(1)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--index', type=int)
	parser.add_argument('--totaljobs', type=int, default=1)
	parser.add_argument('--config-file', type=str, default=None)
	args = parser.parse_args()

	total_jobs = args.totaljobs
	index = args.index
	config_file = args.config_file

	if config_file:
		print("loading args from a config file")
		with open(config_file) as json_file:
			config = json.load(json_file)
			project_dir = config['project_dir']
			#noise_dir = config['noise_dir']
			seed = config['seed']
			fd_approximant = config['fd_approximant']
			td_approximant = config['td_approximant']
			noise_type = config['noise_type']
			n_signal_samples = config['n_signal_samples']
			n_noise_samples = config['n_noise_samples']
			ifos = config['detectors']
			seconds_before = config['seconds_before']
			seconds_after = config['seconds_after']
			f_lower = config['f_lower']
			duration = config['duration']
			delta_t = config['delta_t']
			project_dir = config['project_dir']
			if "noise_dir" in config:
				noise_dir = config['noise_dir']
			else:
				noise_dir = None
				print("no noise dir provided, assuming noise segments are provided instead")
			if "noise_segments" in config:
				noise_segments = config['noise_segments']
			else:
				noise_segments = None
				if noise_dir is None:
					raise ValueError("no noise dir or noise segments provided, cannot proceed")

			if 'chisq' in config:
				reduced_chisq = config['chisq']
				n_chisq_bins = config['n_chisq_bins']
				print("reduced chisq is", reduced_chisq)
				print("n_chisq_bins is", n_chisq_bins)
			else:
				reduced_chisq = False
			if "save_complex" in config.keys():
				save_complex = config['save_complex']
			else:
				save_complex = False
			if "save_strain" in config:
				save_strain = config['save_strain']
			else:
				save_strain = False
			if "save_spectrograms" in config:
				save_spectrograms = config['save_spectrograms']
			else:
				save_spectrograms = False
			
		for key, value in config.items():
			print(key, value)

	print("NOW STARTING JOB",index,"OF",total_jobs)


	#defining some configs. some of these need to come from config files in the future.
	sample_rate = int(1/delta_t)
	delta_f = 1/duration
	f_final = duration

	#ifos = ['H1', 'L1']

	offset = 0

	#fname = 'SNR.npy'
	#TODO: make it so we can choose between saving complex vs abs SNR
	#fname = 'SNR_abs.npy'

	#template_dir = "./template_banks/BNS_lowspin_freqseries"

	#waveforms_per_file = 100
	#templates_per_file = 1000

	#samples_per_batch is limited by how many samples we can fit into a 2 Gb tensor
	#mp_batch is limited by the amount of memory available.

	samples_per_batch = 10
	#samples_per_file = 10000


	#number of batches to process in parallel. determined by the available cores and memory.
	mp_batch = 10

	#n_cpus = 10
	#set n_cpus from os
	try:
		n_cpus = int(os.environ['SLURM_CPUS_PER_TASK'])
	except:
		print("Failed to get SLURM_CPUS_PER_TASK from environment, defaulting to 4")
		n_cpus = 4
	print("n_cpus:",n_cpus)
	mp_batch = n_cpus


	offset = np.min((offset*sample_rate, duration//2))

	samples_per_batch = min(100//(config['templates_per_waveform']),40)
	print("SAMPLES_PER_BATCH:",samples_per_batch)

	###################################################load noise segments
	if noise_segments is None:
		_, paths, _ = get_valid_noise_times(noise_dir,0)
		segments = load_noise(noise_dir)
	else:
		print("loading noise segments from", noise_segments)

	params = np.load(project_dir + "/params.npy", allow_pickle=True).item(0)
	template_ids = np.array(params['template_waveforms'])
	gps = params['gps']
	n_templates = len(template_ids[0])

	templates = np.load(project_dir + "/template_params.npy")

	#Damon's definition of N. from testing, it's just the total length of the segment in samples
	#N = (len(sample1)-1) * 2
	N = int(duration/delta_t)
	# print(f'N: {N}, delta_f: {delta_f}, f_lower: {f_lower}')
	kmin, kmax = np_get_cutoff_indices(f_lower, f_final, delta_f, N)
	# print(f'kmin: {kmin}, kmax: {kmax}')


	##CLEAN UP: JOB ARRAY STUFF GOING HERE FOR NOW

	samples_per_file = len(params['mass1'])//total_jobs
	print("samples per file is",samples_per_file)


	##################################################load PSD
	#psd = np.load(noise_dir + "/psd.npy")

	#since psd[0] is the sample frequencies, and the first frequency is always 0 Hz, psd[0][1] is sample frequency
	psds = {}
	t_psds = {}

	if noise_type == "Gaussian" or noise_segments is None:
		psds_full = {}
		#check if noise_dir is a valid directory
		try:
			if os.path.isdir(noise_dir):
				psds = load_psd(noise_dir, duration, ifos, f_lower, int(1/delta_t))
				#make a copy of the psds for later use
				psds_full = psds.copy()
				for psd in psds:
					psds[psd] = psds[psd][kmin:kmax]
			else:
				raise ValueError("noise_dir is not a valid directory. If you are attempting to pass a text file, it should be in a list")
		except:
			print("Reading PSDs from multiple text files")
			psds = load_psds_from_txt(noise_dir, duration, ifos, f_lower, int(1/delta_t))
			#make a copy of the psds for later use
			psds_full = psds.copy()
			for psd in psds:
				psds[psd] = psds[psd][kmin:kmax]

	else:
		print("computing PSDs from noise segments")
		for ifo in ifos:
			psds[ifo] = []
	if save_complex:
		fp = np.zeros((len(ifos),n_templates*len(params['mass1'])//total_jobs, (seconds_before + seconds_after)*sample_rate), dtype=np.complex64)
		#fp = create_memmap_file(project_dir + "/SNR_{}.npy".format(index), 
		#				shape=(len(ifos),n_templates*len(params['mass1'])//total_jobs, (seconds_before + seconds_after)*sample_rate), 
		#				dtype=np.complex64)

	else:
		fp = np.zeros((len(ifos),n_templates*len(params['mass1'])//total_jobs, (seconds_before + seconds_after)*sample_rate), dtype=np.float32)
		# fp = create_memmap_file(project_dir + "/SNR_abs_{}.npy".format(index), 
		# 				shape=(len(ifos),n_templates*len(params['mass1'])//total_jobs, (seconds_before + seconds_after)*sample_rate))

	if save_strain:
		fp_s = create_memmap_file(project_dir + "/strain_{}.npy".format(index), 
						shape=(len(ifos),len(params['mass1'])//total_jobs, duration*sample_rate), 
						data_type='strain')

	if save_spectrograms:
		# fp_spec = np.zeros((len(ifos), len(params['mass1'])//total_jobs, 224, 224), dtype=np.float32)
		# fp_re = np.zeros((len(ifos), len(params['mass1'])//total_jobs, 224, 224), dtype=np.float32)
		# fp_im = np.zeros((len(ifos), len(params['mass1'])//total_jobs, 224, 224), dtype=np.float32)
		# fp_q = np.zeros((len(ifos), len(params['mass1'])//total_jobs, 224, 224), dtype=np.float32)
		# fp_vitmap = np.zeros((len(ifos), len(params['mass1'])//total_jobs, 224, 224), dtype=np.float32)
		# fp_lineaware = np.zeros((len(ifos), len(params['mass1'])//total_jobs, 224, 224), dtype=np.float32)

		fp_spec = create_memmap_file(project_dir + "/spectrogram_{}.npy".format(index), 
						shape=(len(ifos), len(params['mass1'])//total_jobs, 224, 224), 
						dtype=np.float32, data_type='spectrogram')
		fp_re = create_memmap_file(project_dir + "/spectrogram_re_{}.npy".format(index), 
						shape=(len(ifos), len(params['mass1'])//total_jobs, 224, 224), 
						dtype=np.float32, data_type='spectrogram_re')
		fp_im = create_memmap_file(project_dir + "/spectrogram_im_{}.npy".format(index), 
						shape=(len(ifos), len(params['mass1'])//total_jobs, 224, 224), 
						dtype=np.float32, data_type='spectrogram_im')
		fp_q = create_memmap_file(project_dir + "/spectrogram_q_{}.npy".format(index), 
						shape=(len(ifos), len(params['mass1'])//total_jobs, 224, 224), 
						dtype=np.float32, data_type='spectrogram_q')
		fp_vitmap = create_memmap_file(project_dir + "/spectrogram_vitmap_{}.npy".format(index), 
						shape=(len(ifos), len(params['mass1'])//total_jobs, 224, 224), 
						dtype=np.float32, data_type='spectrogram_vitmap')
		fp_lineaware = create_memmap_file(project_dir + "/spectrogram_lineaware_{}.npy".format(index), 
						shape=(len(ifos), len(params['mass1'])//total_jobs, 224, 224), 
						dtype=np.float32, data_type='spectrogram_lineaware')

	#detectors = {'H1': Detector('H1'), 'L1': Detector('L1'), 'V1': Detector('V1'), 'K1': Detector('K1')}
	print("file will have shape ", fp.shape)

	##################################################calculate the SNR

	print("finished loading data, starting SNR calculation")

	allstart = time.time()

	template_time = 0
	template_load_time = 0
	waveform_time = 0
	SNR_time = 0
	convert_time = 0
	repeat_time = 0

	import gc

	from pycbc.waveform import get_td_waveform
	all_detectors = {'H1': Detector('H1'), 'L1': Detector('L1'), 'V1': Detector('V1'), 'K1': Detector('K1')}

	for n in range(index*samples_per_file,(index+1)*samples_per_file,samples_per_batch*mp_batch):
		#print("batch:", n//samples_per_batch)
		#print(n)
		#n is the index of the first batch to be processed
		end = min(n+mp_batch*samples_per_batch, (index+1)*samples_per_file)

		print("starting batches",[j for j in range(n,end,samples_per_batch)])

		start = time.time()
		with mp.Pool(n_cpus) as p:
			results = p.map(run_batch, [j for j in range(n,end,samples_per_batch)], chunksize=1)
			#results = p.map(run_batch, [j for j in range(n,min(n+mp_batch*samples_per_batch, samples_per_file),samples_per_batch)])
		template_time += time.time() - start

		#TODO: ensure we can handle the case where the number of samples is not divisible by samples_per_batch,
		#and where samples_per_batch*mp_batch is not divisible by samples_per_file

		#for i in range(mp_batch):
		for i in range(len(results)):
			#t_templates, strains = results[i]
			s_idx = n_templates * (i*samples_per_batch + n - index*samples_per_file)
			e_idx = n_templates * ((i+1)*samples_per_batch + n - index*samples_per_file)
			for ifo in ifos:
				if save_complex:
					fp[ifos.index(ifo)][s_idx:e_idx] = results[i][ifo].astype(np.complex64)
				else:
					fp[ifos.index(ifo)][s_idx:e_idx] = np.abs(results[i][ifo])
		#fp.flush()
		#garbage collect
		del results
		gc.collect()


	print("template time + waveform load + convert:", template_time)
	print("SNR time:", SNR_time)
	print("convert time:", convert_time)
	print("repeat time:", repeat_time)
	print("total time:", time.time() - allstart)


	t_time = time.time() - allstart
	print("it would take ", (25000 * t_time/(samples_per_file*total_jobs))/3600, "hours to process 25000 samples.")


	#fp.flush()

	#memmap'd files don't have a header describing the shape of the array, so we add one here

	#header = np.lib.format.header_data_from_array_1_0(fp)

	# if save_complex:
	# 	with open(project_dir + "/SNR_{}.npy".format(index), 'r+b') as f:
	# 		np.lib.format.write_array_header_1_0(f, header)
	# else:
	# 	with open(project_dir + "/SNR_abs_{}.npy".format(index), 'r+b') as f:
	# 		np.lib.format.write_array_header_1_0(f, header)

	# if save_strain:
	# 	fp_s.flush()
	# 	header = np.lib.format.header_data_from_array_1_0(fp_s)
	# 	with open(project_dir + "/strain_{}.npy".format(index), 'r+b') as f:
	# 		np.lib.format.write_array_header_1_0(f, header)

	if save_spectrograms:
		fp_spec.flush()
		header = np.lib.format.header_data_from_array_1_0(fp_spec)
		with open(project_dir + "/spectrogram_{}.npy".format(index), 'r+b') as f:
			np.lib.format.write_array_header_1_0(f, header)

		fp_re.flush()
		header = np.lib.format.header_data_from_array_1_0(fp_re)
		with open(project_dir + "/spectrogram_re_{}.npy".format(index), 'r+b') as f:
			np.lib.format.write_array_header_1_0(f, header)

		fp_im.flush()
		header = np.lib.format.header_data_from_array_1_0(fp_im)
		with open(project_dir + "/spectrogram_im_{}.npy".format(index), 'r+b') as f:
			np.lib.format.write_array_header_1_0(f, header)

		fp_q.flush()
		header = np.lib.format.header_data_from_array_1_0(fp_q)
		with open(project_dir + "/spectrogram_q_{}.npy".format(index), 'r+b') as f:
			np.lib.format.write_array_header_1_0(f, header)

		fp_vitmap.flush()
		header = np.lib.format.header_data_from_array_1_0(fp_vitmap)
		with open(project_dir + "/spectrogram_vitmap_{}.npy".format(index), 'r+b') as f:
			np.lib.format.write_array_header_1_0(f, header)

		fp_lineaware.flush()
		header = np.lib.format.header_data_from_array_1_0(fp_lineaware)
		with open(project_dir + "/spectrogram_lineaware_{}.npy".format(index), 'r+b') as f:
			np.lib.format.write_array_header_1_0(f, header)

	#save to disk
	if save_complex:
		np.save(project_dir + "/SNR_{}.npy".format(index), fp)
	else:
		np.save(project_dir + "/SNR_abs_{}.npy".format(index), fp)

	if save_strain:
		np.save(project_dir + "/strain_{}.npy".format(index), fp_s)
		
	# if save_spectrograms:
	# 	np.save(project_dir + "/spectrogram_{}.npy".format(index), fp_spec)
	# 	np.save(project_dir + "/spectrogram_re_{}.npy".format(index), fp_re)
	# 	np.save(project_dir + "/spectrogram_im_{}.npy".format(index), fp_im)
	# 	np.save(project_dir + "/spectrogram_q_{}.npy".format(index), fp_q)
	# 	np.save(project_dir + "/spectrogram_vitmap_{}.npy".format(index), fp_vitmap)
	# 	np.save(project_dir + "/spectrogram_lineaware_{}.npy".format(index), fp_lineaware)
	#pool.close()

	print("done!")
	#sanity check on the SNR values

	"""
	import matplotlib.pyplot as plt

	for i in range(len(fp)):
		if np.min(np.max(np.abs(fp[i]), axis = 1)) < 0.1:
			print("SNR is too low. not all jobs have necessarily finished.")


		plt.plot(np.max(np.abs(fp[i]), axis = 1), alpha=0.5)

	plt.savefig(project_dir + "/max_SNRs.png")

	plt.clf()


	if np.min(np.max(np.abs(fp[0]), axis = 1)[:n_signal_samples * n_templates]) > 0.1:
		print("all jobs should have finished. ")		
		for ifo in ifos:
			plt.hist(np.max(np.abs(fp[ifos.index(ifo)]), axis = 1)[:n_signal_samples *n_templates: n_templates]/ params[ifo+'_snr'][:n_signal_samples], bins=30, alpha=0.5)
		plt.xlabel("recovered/injected SNR")
		plt.savefig(project_dir + "/recovered_SNR.png")
	"""
