import os
import numpy as np
from pycbc.filter import highpass
from pycbc.types.timeseries import TimeSeries
from gwpy.timeseries import TimeSeries as GWPYTimeSeries
from scipy.interpolate import interp1d
#import soapcw as soap
import logging
from scipy.special import logsumexp
import pickle
import scipy.stats as st
from scipy.integrate import quad

#copied from soap.tools so we don't have to import the whole soap package
#Source: https://pypi.org/project/soapcw/
def transition_matrix(par,log=True):
    '''
    generate a symmetric transition matrix for 0 memory and 1 detector
    args
    -------
    par: float
        ratio of two outer elements to the centre element of the transition matrix
    kwargs
    ------
    log: bool
        return the log of the transition matrix if True, just probabilities if false
    returns 
    -------
    tr: array
        3x1 array of transition probabilities 
    '''
    v1 = 1.
    v2 = v1/par
    tr = [v2,v1,v2]
    if log:
        return np.log(tr/np.sum(tr))
    if not log:
        return tr/np.sum(tr)
    

#NOTE: this function is from soap's cython codebase. It will not run as quickly as the cython version.

class single_detector(object):

	def __init__(self, tr, obs, prog = False, lookup_table=None, make_vitmap = True):
		'''
		initialising viterbi class
		'''
		self.prog = prog
		
		# make sure input data of right type
		tr = np.array(tr).astype('double')
		obs = np.array(obs).astype('double')

		# make sure input data of right shape
		if len(np.shape(tr)) != 1:
			raise Exception("transition matrix wrong shape, should be Kx1 array")
		if len(np.shape(obs)) != 2:
			raise Exception("observation wrong shape, should be NxM array")
		
		self.run(tr, obs, lookup_table)
		if make_vitmap:
			self.get_vitmap()

		
	#------------------------------------------------------------------------------------------------
	# Get Viterbi map
	#------------------------------------------------------------------------------------------------

	def get_vitmap(self,log=False):
		"""
		normalise the viterbi output such that the sum of eah column is 1
		"""
		path_m = []
		for i in self.V:
			sump = logsumexp(i)
			val = np.exp(i-sump)
			path_m.append(val)
		path_m = np.array(path_m)
		if log:
			path_m = np.log(path_m)

		self.vitmap = path_m

	#------------------------
	# Read lookup table
	#--------------------------

	def read_lookup(self,lookup_table):
		"""
		Read the lookup table for chosen statistic
		"""
		if isinstance(lookup_table, str):
			with open(lookup_table,'rb') as f1:
				likelihood, params = pickle.load(f1)
			ranges = np.linspace(params["power_ranges"][0],params["power_ranges"][1],int(params["power_ranges"][2]))
			fact = abs(1./(ranges[0] - ranges[1]))
		else:
			# if it is a line aware stat object then creat ranges from the object
			# get the log odds ratio as the statitsic to use 
			likelihood = np.log(lookup_table.signoiseline)
			ranges = lookup_table.powers
			fact = abs(1./(ranges[0] - ranges[1]))
		
		return likelihood, ranges, fact


	#-------------------------------------------------------------------------------------------------
	# Basic Viterbi Algorithm
	#-------------------------------------------------------------------------------------------------


	def run(self, tr, obs, lookup_table = None):
		'''
		Run the viterbi algorithm for given single set of data and 3x1 transition matrix.
		This returns the track through the data which gives the largest sum of power.
		Args
		------
		tr: array
			transition matrix of sixe Kx1
		obs: array
			observation data of size NxM
		Returns
		-------
		vit_track: array
			index of the viterbi path found in data
		max_end_prob: float
			maximum probability at end of path
		V: array
			viterbi matrix
		prev: array
			previous positions of paths for each bin
		'''
		
		# find shape of the observation and create empty array for citerbi matrix and previous track positions
		shape = np.shape(obs)
		V = np.zeros(shape, dtype=np.double)
		prev = np.zeros(shape,dtype=np.int32)
		
		# defining variables
		#cdef int t,i,j         # indexes
		#cdef double pbar = 0   # progress bar
		#cdef double pt
		
		# finding length and width of observation
		length = shape[0]#len(obs)
		width = shape[1]#len(obs[0])
		
		# find size of the transition matrix
		tr_length = len(tr)
		# find half width of transntion, i.e. number of bins up and down it can move
		tr_width = int((tr_length-1)/2 )

		# vectors to store lookup tables
		#cdef double[:] logarr
		#cdef double fact
		#cdef int[:,:] obs_ind
		if lookup_table is not None:
			# get log statistic, the ranges of lookuptable and spacing factor
			logarr, ranges, fact = self.read_lookup(lookup_table)
			# shift observation to index for array
			obs_ind_arr = (obs - ranges[0])*fact
			obs_ind_arr[obs_ind_arr >= len(ranges)] = len(ranges)-1
			obs_ind_arr[obs_ind_arr < 0] = 0
			# save array to vector
			obs_ind = np.array(obs_ind_arr).astype(np.int32)
			del obs_ind_arr
		else:
			# if using the sum of power, set element of index to 2d array
			obs_ind = np.arange(np.prod(shape)).reshape(shape).astype(np.int32)
			# flatten input spectrogram for access in 1d
			logarr = np.ravel(obs)
		
		# run for first time index, i.e. fill with observation
		for i in range(width):
			V[0][i] = logarr[obs_ind[0][i]]
			prev[0][i] = 1
		
		# run iterative part of algorithm
		for t in range(1,length):
			pt = t/(length)*100
			for i in range(width):
				temp = -1e6
				for j in range(tr_length):
					if i+j-tr_width>=0 and i+j-tr_width<=width-1:
						o = obs_ind[t][i]
						value = tr[j] + logarr[o] + V[t-1][i+j-tr_width]

						if value>temp:
							temp = value
							V[t][i] = temp
							prev[t][i] = i+j-tr_width
						elif value == temp and j == int(tr_length/2.):
							temp = value
							V[t][i] = temp
							prev[t][i] = i+j-tr_width
							

			if self.prog == True:
				if pt>pbar:
					print('\r{} %'.format(round(pt)))
					#stdout.flush()
					pbar+=100./len(obs)

		max_end_prob = max(V[length-1][i] for i in range(width))
		#cdef int previous
		vit_track = np.zeros(length,dtype=np.int32)
		
						
		for i in range(width):
			if V[length-1][i] == max_end_prob:
				vit_track[length-1] = i # appends maximum path value from final step
				previous = prev[length-1][i]
				break

		for t in range(len(V)-2,-1,-1):
			vit_track[t] = previous # insert previous step
			previous = prev[t][previous]

		self.vit_track = np.array(vit_track)
		self.max_end_prob = max_end_prob
		self.V = np.array(V)
		self.prev = np.array(prev)

class LineAwareStatistic:

	def __init__(self, powers, ndet=2 ,k=2, N=48, signal_prior_width=1, line_prior_width=5, noise_line_model_ratio=1,approx=True):

		self.ndet = ndet
		self.signal_prior_width = signal_prior_width
		self.line_prior_width = line_prior_width
		self.noise_line_model_ratio = noise_line_model_ratio
		self.powers = powers
		self.k = k
		self.N = N
		self.kN = self.k*self.N
		
		if ndet == 1:
			self.signoiseline,self.signoise,self.sigline,self.sig,self.noise,self.line = self.gen_lookup_one_det(powers,approx=approx,signal_prior_width=signal_prior_width,line_prior_width=line_prior_width,noise_line_model_ratio=noise_line_model_ratio)
		elif ndet == 2:
			self.signoiseline,self.signoise,self.sigline,self.sig,self.noise,self.line = self.gen_lookup_two_det(powers,powers,approx=approx,signal_prior_width=signal_prior_width,line_prior_width=line_prior_width,noise_line_model_ratio=noise_line_model_ratio)
		else:
			raise Exception("This currently only works for 1 or 2 detectors")    
		
	def chi2_sig(self,lamb: np.array,gs: float,pv: float) -> np.array:
		"""
		returns the likelihood of two powers multiplied by the prior on the snr**2
		args
		-------
		lamb: float
			snr^2
		g1: float
			SFT power in detector 1
		g2: float
			SFT power in detector 2
		k: int
			number of degrees of freedom
		N: int
			Number of summed SFTS
		pv: float
			width of exponetial distribution of prior
		returns
		---------
		func*prior: float
		likelihood multiplied by prior
		"""
		if len(np.shape(gs)) == 0:
			func = st.ncx2.pdf(gs,df=self.kN,nc=lamb,loc=0,scale=1)
		else:
			func = np.prod([st.ncx2.pdf(i,df=self.kN,nc=lamb,loc=0,scale=1) for i in gs], axis=0)
		wid = 1./pv
		return func*wid*np.exp(-wid*lamb)


	def chi2_line(self,lamb,gs,pv):
		"""
		returns the likelihood of two powers multiplied by the prior on the snr**2
		works for 1 or 2 detectors only

		args
		-------
		lamb: float
		snr^2
		g1: float
		SFT power in detector 1
		g2: float
		SFT power in detector 2
		k: int
		number of degrees of freedom
		N: int
		Number of summed SFTS
		pv: float
		width of exponetial distribution of prior
		returns
		---------
		func*prior: float
		likelihood multiplied by prior
		"""
		if len(np.shape(gs)) == 0:
			func = st.ncx2.pdf(gs,df=self.kN,nc=lamb,loc=0,scale=1)
		else:
			func = 1./len(gs)*(st.chi2.pdf(gs[0],df=self.kN,loc=0,scale=1)*st.ncx2.pdf(gs[1],df=self.kN,nc=lamb,loc=0,scale=1) + st.ncx2.pdf(gs[0],df=self.kN,nc=lamb,loc=0,scale=1)*st.chi2.pdf(gs[1],df=self.kN,loc=0,scale=1))
		wid = 1./pv
		return func*wid*np.exp(-wid*lamb)


	def chi2_noise(self,gs):
		if len(np.shape(gs)) == 0:
			func = st.chi2.pdf(gs,df=self.kN,loc=0,scale=1)
		else:
			func = np.prod([st.chi2.pdf(i,df=self.kN,loc=0,scale=1) for i in gs],axis=0)
		return func


	def two_det(self,g1,g2,signal_prior_width=10,line_prior_width=10,noise_line_model_ratio=1., approx=True):
		"""
		integrate likelihood and prior to get evidence for powers g1 and g2
		args
		-------
		g1: float
		g2: float
		k: int
		N: int
		signal_prior_width: float
		line_prior_width: float
		noise_line_model_ratio: float
		approx: bool
			choose to approximate the integral with trapz or full integnoise_line_model_ration
		"""
		if approx:
			l = np.linspace(0,100,500)
			sig_int = np.trapz(np.nan_to_num(self.chi2_sig(l,[g1,g2],signal_prior_width)),l)
			line_int = np.trapz(np.nan_to_num(self.chi2_line(l,[g1,g2],line_prior_width)),l)
		else:
			sig_int, sig_err = quad(self.chi2_sig,0,np.inf, args = ([g1,g2],signal_prior_width))
			line_int, line_err = quad(self.chi2_line,0,np.inf, args = ([g1,g2],line_prior_width))

		noise = self.chi2_noise([g1,g2])
		siglinenoise = sig_int/(noise_line_model_ratio*line_int + noise )
		signoise = sig_int/noise
		sigline = sig_int/line_int
		
		return siglinenoise,signoise,sigline,sig_int,noise,line_int

	def one_det(self,g1,signal_prior_width=10,line_prior_width=10,noise_line_model_ratio=1., approx=True):
		"""
		integrate likelihood and prior to get evidence for powers g1 and g2
		args
		-------
		approx: bool
			choose to approximate the integral with trapz or full integnoise_line_model_ration
		"""

		if approx:
			l = np.linspace(0,100,500)
			sig_int = np.trapz(np.nan_to_num(self.chi2_sig(l,g1,signal_prior_width)),l)
			line_int = np.trapz(np.nan_to_num(self.chi2_line(l,g1,line_prior_width)),l)
		else:
			sig_int, sig_err = quad(self.chi2_sig,0,np.inf, args = (g1,signal_prior_width))
			line_int, line_err = quad(self.chi2_line,0,np.inf, args = (g1,line_prior_width))

		noise = self.chi2_noise(g1)
		siglinenoise = sig_int/(noise_line_model_ratio*line_int + noise )
		signoise = sig_int/noise
		sigline = sig_int/line_int
		
		return siglinenoise,signoise,sigline,sig_int,noise,line_int


	def gen_lookup_one_det(self,powers,approx=True,signal_prior_width=10,line_prior_width=10,noise_line_model_ratio=1):
		"""
		calculate lookup table for values of x in the detector
		args
		-----------
		powers: array, list
			list of spectrogram powers to calcualte statistic at
		returns
		---------
		signoiseline: log(sig/(noise+line))
		signoise: log(sig/(noise))
		sigline: log(sig/(line))
		sig: log(sig)
		noise: log(noise)
		line: log(line)
		"""

		signoiseline = np.zeros((len(powers)))
		signoise = np.zeros((len(powers)))
		sigline = np.zeros((len(powers)))
		sig = np.zeros((len(powers)))
		noise = np.zeros((len(powers)))
		line = np.zeros((len(powers)))
		
		for i in range(len(powers)):
			ig = self.one_det(powers[i],signal_prior_width,line_prior_width,noise_line_model_ratio,approx=approx)
			signoiseline[i] = ig[0]
			signoise[i] = ig[1]
			sigline[i] = ig[2]
			sig[i] = ig[3]
			noise[i] = ig[4]
			line[i] = ig[4]
					
		return signoiseline,signoise,sigline,sig,noise,line

	def gen_lookup_two_det(self,powers1,powers2,approx=True,signal_prior_width=10,line_prior_width=10,noise_line_model_ratio=1):
		"""
		calculate lookup table for values of x and y in each detector
		args
		----------
		powers1: array, list
			list of spectrogram powers to calcualte statistic at in det1
		powers2: array, list
			list of spectrogram powers to calcualte statistic at in det2


		returns
		---------
		signoiseline: log(sig/(noise+line))
		signoise: log(sig/(noise))
		sigline: log(sig/(line))
		sig: log(sig)
		noise: log(noise)
		line: log(line)
		"""

		signoiseline = np.zeros((len(powers1),len(powers2)))
		signoise = np.zeros((len(powers1),len(powers2)))
		sigline = np.zeros((len(powers1),len(powers2)))
		sig = np.zeros((len(powers1),len(powers2)))
		noise = np.zeros((len(powers1),len(powers2)))
		line = np.zeros((len(powers1),len(powers2)))
		
		for i in range(len(powers1)):
			for j in range(len(powers2)):
				if j > i:
					continue
				ig = self.two_det(powers1[i],powers2[j],signal_prior_width,line_prior_width,noise_line_model_ratio,approx=approx)
				# symmetric matrix so set opposite elements to same, format it [(x1,x2),(y1,y2)]
				signoiseline[(i,j),(j,i)] = ig[0]
				signoise[(i,j),(j,i)] = ig[1]
				sigline[(i,j),(j,i)] = ig[2]
				sig[(i,j),(j,i)] = ig[3]
				noise[(i,j),(j,i)] = ig[4]
				line[(i,j),(j,i)] = ig[4]
					
		return signoiseline,signoise,sigline,sig,noise,line


	def save_lookup(self,outdir,log=True, stat_type = "signoiseline"):
		"""
		save the lookup table for two detectors with the line aware statistic
		
		Args
		--------------
		outdir: string
		directory to save lookup table file
		pow_range: tuple
		ranges for the spectrogram power (lower, upper, number), default (1,400,500)
		
		"""
		minimum,maximum,num = min(self.powers),max(self.powers),len(self.powers)
		log_str = "log_" if log else ""
		fname = os.path.join(outdir,"{}{}_{}det_{}degfree_{}_{}_{}.pkl".format(log_str,stat_type,self.ndet,self.kN,self.signal_prior_width,self.line_prior_width,self.noise_line_model_ratio))
		
		if os.path.isfile(fname):
			pass
		else:
			with open(fname,'wb') as f:
				header = {"power_ranges":(minimum,maximum,num), "fraction_ranges": (1, 1, 1), "signal_prior":self.signal_prior_width, "line_prior_width":self.line_prior_width, "noise_line_model_ratio":self.noise_line_model_ratio}
				if log:
					pickle.dump([np.log(getattr(self,stat_type)), header],f)
				elif not log:
					pickle.dump([getattr(self,stat_type), header],f)


	def save_multiple(self,powerrange,signal_prior_widthrange,line_prior_widthrange,noise_line_model_ratiorange):

		pass

#TODO: generalise this code to work with any spectrogram shape and sample rate,f_lower etc.
def resize_spectrogram(freqs, power, target_size=224):
    """
    1) Interpolate the frequency axis from N_freq→target_size
    2) Bin-sum the time axis from N_time→target_size

    Inputs:
      freqs: 1D array length N_freq
      power: 2D array shape (N_time, N_freq)
    Output:
      resized: 2D array shape (target_size, target_size)
    """
    N_time, N_freq = power.shape
    if freqs.shape[0] != N_freq:
        raise ValueError(f"len(freqs)={len(freqs)} must match power.shape[1]={N_freq}.")

    # Interpolate frequency axis
    f_min, f_max = freqs[0], freqs[-1]
    new_freqs = np.linspace(f_min, f_max, target_size)
    interp_fn = interp1d(freqs, power, axis=1, kind='linear', fill_value='extrapolate')
    power_freq_resized = interp_fn(new_freqs)  # shape = (N_time, target_size)

    # Bin-sum time axis into target_size chunks
    chunks = np.array_split(power_freq_resized, target_size, axis=0)
    binned = np.stack([c.sum(axis=0) for c in chunks], axis=0)  # shape = (target_size, target_size)

    return binned

def process_single_series_complex(timeseries):
    """
    args = (detector_index, series_index)
    Return (amp_resized, re_resized, im_resized) or None if error.
    """
    sample_rate = 2048
    low_freq_cutoff = 30
    try:
        # Load raw data and preprocess
        d = timeseries.astype(np.float64)
        data = TimeSeries(d, delta_t=1/sample_rate)
        zoom = highpass(data, 30.0).whiten(
            4, 4, remove_corrupted=False, low_frequency_cutoff=low_freq_cutoff
        )
        zoom = zoom.crop(489, 505) #30 seconds timeseries

        # Compute complex spectrogram via GWPy
        gw_data = GWPYTimeSeries(
            zoom.numpy(),
            dt=zoom.delta_t,
            t0=float(zoom.start_time)
        )
        complex_spec = gw_data.fftgram(
            fftlength=0.1,
            overlap=0.05,
            window='hann'
        )

        # Extract amplitude, real, imaginary arrays
        spec_vals = complex_spec.value  # 2D complex array
        amp = np.abs(spec_vals)
        re = np.real(spec_vals)
        im = np.imag(spec_vals)

        # Obtain the frequency vector
        frecs = complex_spec.frequencies.value 

        # Resize each to (224,224)
        amp_resized = resize_spectrogram(frecs, amp, target_size=224)
        re_resized = resize_spectrogram(frecs, re, target_size=224)
        im_resized = resize_spectrogram(frecs, im, target_size=224)

        return amp_resized.T, re_resized.T, im_resized.T

    except Exception:
        logging.exception(f"Error processing detector timeseries")
        return None

def process_single_series_qtransform(timeseries):
	"""
	args = (detector_index, series_index)
	We read all_data[detector_index, series_index, :], do the highpass + whiten,
	compute qtransform to get (times, freqs, power), then resize power (timexfreq→224x224).
	Then compute vitmap and line_aware on the resized power.
	Return:
		- power_resized (shape 224x224)
		- vitmap_ng (shape 224x224)
		- vitmap_line (shape 224x224)
	"""
	sample_rate = 2048
	low_freq_cutoff = 30
	try:
		# Convert raw data to float64
		d = timeseries.astype(np.float64)

		# 2.1) Convert to TimeSeries and filter/whiten
		data = TimeSeries(d, delta_t=1/sample_rate)
		zoom = highpass(data, 30.0).whiten(
			4, 4, remove_corrupted=False, low_frequency_cutoff=low_freq_cutoff
		)
		zoom = zoom.crop(489, 505) #30 seconds timeseries

		# 2.2) Q-transform
		times, freqs, power = zoom.qtransform(
			0.01,
			logfsteps=200,
			qrange=(32, 128),
			frange=(30, 512),
		)
		# `power` has shape (N_freq, N_time)

		# 2.3) Resize this spectrogram: first transpose to (N_time, N_freq)
		power_time_freq = power.T
		power_resized = resize_spectrogram(freqs, power_time_freq, target_size=224)
		# `power_resized` shape = (224, 224) where rows=time-bin, cols=freq-bin

		# 2.4) Compute Viterbi‐"ng" on resized power
		tr_1 = transition_matrix(1.0)
		one_tracks_ng = single_detector(tr_1, power_resized)
		vitmap_ng = one_tracks_ng.vitmap.T  # transpose to (224,224)

		# 2.5) Line‐aware statistic on resized power
		pow_grid = np.linspace(1, 400, 40)
		
		sn_prior_width = 1.0
		ln_prior_width = 1.0
		ns_line_model_ratio = 0.1
		lineaware_1d = LineAwareStatistic(
			pow_grid,
			ndet=1,
			signal_prior_width=sn_prior_width,
			line_prior_width=ln_prior_width,
			noise_line_model_ratio=ns_line_model_ratio,
		)
		current_directory = os.getcwd()
		lookup_path = f"{current_directory}/log_signoiseline_1det_96degfree_{sn_prior_width}_{ln_prior_width}_{ns_line_model_ratio}.pkl"
		if not os.path.exists(lookup_path):
			lineaware_1d.save_lookup(current_directory)

		tr_1_line = transition_matrix(1.0)
		one_det_line_aware = single_detector(
			tr_1_line,
			power_resized,
			lookup_table=lookup_path
		)
		vitmap_line = one_det_line_aware.vitmap.T  # shape = (224,224)

		power_log = np.log2(power_resized + 1e-12)
		mu, sigma = power_log.mean(), power_log.std()
		vitmap_scaled = (vitmap_ng.astype(float) - vitmap_ng.mean()) / (vitmap_ng.std() + 1e-6)
		vitmap_scaled = vitmap_scaled*sigma + mu  # now vitmap and power_log live in comparable ranges
		lin_aware = (vitmap_line.astype(float) - vitmap_line.mean()) / (vitmap_line.std() + 1e-6)
		lin_aware = lin_aware*sigma + mu

		return power_log.T, vitmap_scaled, lin_aware

	except Exception:
		logging.exception(f"Error processing timeseries. No qtransform output")
		return None