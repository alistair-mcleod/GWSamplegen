"""A numpy-based implementation of PyCBC's chi-squared calculation."""

import numpy as np
import scipy
from GWSamplegen.snr_utils_np import np_correlate, np_sigmasq

def sigmasq_series_np(htilde, psd_np, delta_f):
	t_vec = np.copy(htilde)
	norm = 4.0 * delta_f
		
	t_vec = np.abs(t_vec)**2
	t_vec/= psd_np
	t_vec = np.cumsum(t_vec,axis =-1) * norm
	return t_vec

def power_chisq_bins_from_sigmasq_series_np(sigmasq_series, num_bins, kmin, kmax):
	sigmasq = sigmasq_series[:,-1]
	edge_vec = np.tile(np.arange(0, num_bins),(sigmasq.shape[0],1)) * sigmasq[:,None] / num_bins
	bins = np.array([np.searchsorted(sigmasq_series[i], edge_vec[i], side='right') for i in range(len(sigmasq_series))])
	bins += kmin
	return np.append(bins, np.expand_dims(np.repeat(kmax,bins.shape[0]),1), axis = -1)

def power_chisq_bins_np(htilde, psd_np, num_bins, delta_f,kmin,kmax):
	sigma_vec = sigmasq_series_np(htilde, psd_np, delta_f)
	return power_chisq_bins_from_sigmasq_series_np(sigma_vec, num_bins,kmin,kmax)

def chisq_accum_bin_np(chisq, q):
	chisq += np.abs(q) ** 2

def power_chisq_from_precomputed_np(SNR, bins, template, strain, psd, delta_f, kmin, kmax):
	#Bins is the output of power_chisq_bins
	#corr is np_correlate(tt,ss)/pp
	#snr * snr_norm is SNR_ret
	#snr_norm is np_sigmasq(tt, pp, delta_f)

	#q = np.zeros_like(SNR)
	#qtilde = np.zeros_like(SNR)
	#chisq = np.zeros(SNR.shape[1])

	corr = np_correlate(template, strain)/psd
	snr_norm = np_sigmasq(template, psd, delta_f)
	#SNR_copy = np.copy(SNR)/ snr_norm
	chisq = np.zeros(SNR.shape)

	#corr_fft = np.zeros_like(SNR)
	#corr_fft[:,kmin:kmax] = corr
	qtilde = np.zeros_like(SNR)
	num_bins = bins.shape[1] - 1
	s = 0
	for j in range(num_bins):
		for i in range(bins.shape[0]):

			k_min = int(bins[i,j])
			k_max = int(bins[i,j+1])
			#qtilde[i,k_min:k_max] = corr_fft[i,k_min:k_max]
			#qtilde[i,k_min:k_max] = corr[i,k_min-kmin:k_max-kmin]
			qtilde[i,k_min:k_max] = corr[i,k_min-kmin:k_max-kmin]
		#qtilde = np.fft.ifft(qtilde, axis = -1) * len(qtilde[0])
		qtilde = scipy.fftpack.ifft(qtilde, axis = -1, overwrite_x=True) * len(qtilde[0])

		chisq_accum_bin_np(chisq, qtilde)
		qtilde[:] = 0
	chisq = (chisq * num_bins - np.abs(SNR/snr_norm) ** 2) * (snr_norm ** 2.0)

	return np.real(chisq).astype(np.float32)

def power_chisq_np(SNR_in, template_in, strain_in, psd_in, num_bins, delta_f, kmin, kmax):

	bins = power_chisq_bins_np(template_in, psd_in, num_bins, delta_f, kmin, kmax)
	return power_chisq_from_precomputed_np(SNR_in, bins, template_in, strain_in, psd_in, delta_f, kmin, kmax)


def reduced_chisquared_precomputed_SNR(SNR, template, strain, psd, kmin, kmax, delta_f, num_bins = 16):
	#array-wise reduced chi-squared calculation
	#by using precomputed SNR, we avoid doubling up on the most expensive part of the calculation
	chisq = power_chisq_np(SNR, template, strain, psd, num_bins, delta_f, kmin, kmax)

	#divide by the number of degrees of freedom 
	chisq /= (num_bins*2 - 2)

	#perform PyCBC's transformation (TODO: check why they do it this way)
	chisq = np.clip(chisq, 1,np.inf)
	chisq = (0.5 * (1 + (chisq**3)))** (-1/6)
	
	return chisq