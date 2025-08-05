import os
import numpy as np
from pycbc.filter import highpass
from pycbc.types.timeseries import TimeSeries
from gwpy.timeseries import TimeSeries as GWPYTimeSeries
from scipy.interpolate import interp1d
import soapcw as soap
import logging


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
		tr_1 = soap.tools.transition_matrix(1.0)
		one_tracks_ng = soap.single_detector(tr_1, power_resized)
		vitmap_ng = one_tracks_ng.vitmap.T  # transpose to (224,224)

		# 2.5) Line‐aware statistic on resized power
		pow_grid = np.linspace(1, 400, 40)
		
		sn_prior_width = 1.0
		ln_prior_width = 1.0
		ns_line_model_ratio = 0.1
		lineaware_1d = soap.line_aware_stat.gen_lookup_python.LineAwareStatistic(
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

		tr_1_line = soap.tools.transition_matrix(1.0)
		one_det_line_aware = soap.single_detector(
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