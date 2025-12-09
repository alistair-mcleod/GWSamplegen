"""
Matched filtering functions based on PyCBC's implementation of matched filtering, but optimised for 
performing array-wise matched filtering on batches of samples and templates.
"""


import numpy as np
from typing import Tuple

def np_weighted_inner(
    one: np.ndarray, 
    two: np.ndarray,
    weight: np.ndarray
) -> np.ndarray:
    """
    Compute weighted inner product of two frequency series, for the sigmasq rescaling.

    Parameters
    ----------

    one: 
        batch array of templates in frequency domain
    two: 
        batch array of templates in frequency domain
    weight: 
        noise PSD in frequency domain
        
    Returns
    -------

    out: 
        batch array of weighted inner products
    """
    return np.sum(np.conjugate(one) * two / weight, axis=1)


def np_correlate(
    templates: np.ndarray, 
    samples: np.ndarray
) -> np.ndarray:
    """
    Compute correlation of templates and samples.

    Parameters
    ----------
    templates: 
        batch array of templates in frequency domain
    samples: 
        batch array of samples in frequency domain
        
    Returns
    -------
    out: 
        batch array of correlated samples & templates
    """
    temp = np.conjugate(templates) * samples
    return temp


def np_get_cutoff_indices(
    flow: int, 
    fhigh: int, 
    delta_f: int, 
    N: int
) -> Tuple[int, int]:
    
    """
    Compute indexes of low and high frequency cutoffs.

    Parameters
    ----------
    flow: 
        low frequency cutoff
    fhigh: 
        high frequency cutoff
    delta_f: 
        frequency series sampling rate
    N: 
        length of the samples
        
    Returns
    -------
    kmin: 
        index of low frequency cutoff
    kmax: 
        index of high frequency cutoff
    """
    if flow:
        kmin = int(flow / delta_f)
        if kmin < 0:
            err_msg = "Start frequency cannot be negative. "
            err_msg += "Supplied value and kmin {} and {}".format(flow, kmin)
            raise ValueError(err_msg)
    else:
        kmin = 1
    if fhigh:
        kmax = int(fhigh / delta_f)
        if kmax > int((N + 1)/2.):
            kmax = int((N + 1)/2.)
    else:
        # int() truncates towards 0, so this is
        # equivalent to the floor of the float
        kmax = int((N + 1)/2.)

    if kmax <= kmin:
        err_msg = "Kmax cannot be less than or equal to kmin. "
        err_msg += "Provided values of freqencies (min,max) were "
        err_msg += "{} and {} ".format(flow, fhigh)
        err_msg += "corresponding to (kmin, kmax) of "
        err_msg += "{} and {}.".format(kmin, kmax)
        raise ValueError(err_msg)

    return kmin,kmax


def np_sigmasq(template, psd, delta_f):
    """
    Batch compute template normalisation to rescale SNR time series.

    Parameters
    ----------
    template: 
        batch array of templates in frequency domain
    psd: 
        array of PSD
    delta_f: 
        frequency series sampling rate (default 1.0/2048)
        
    Returns
    -------
    template_norm: 
        batch array of normalisation constants
    """
    norm = 4.0 * delta_f
    
    sq = np_weighted_inner(template, template, psd)
    h_norm = np.real(sq) * norm
    
    template_norm = ((4.0 * delta_f) / np.sqrt(h_norm)).astype(np.complex128)
    template_norm = np.reshape(template_norm, (template_norm.shape[0], 1))

    return template_norm


def numpy_matched_filter(
    sample: np.ndarray, 
    template: np.ndarray, 
    psd: np.ndarray, 
    N: int, 
    kmin: int, 
    kmax: int, 
    duration: int, 
    delta_t: float = 1.0/2048, 
    flow: float = 30
) -> np.ndarray:    
    """Batch compute SNR time series using matched filtering. 
        Matched filtering is done row-wise for sample and template arrays.

    Parameters
    ----------

    sample: 
        A 2d array of strain samples in frequency domain. Can be the same strain sample multiple times,
        i.e. with np.repeat for matching with multiple templates.
    template: 
        A 2d array of templates in frequency domain
    psd: 
        PSD array
    N: 
        length of the samples 
    kmin: 
        index of low frequency cutoff
    kmax: 
        index of high frequency cutoff
    duration: 
        strain time series duration (seconds)
    delta_t: 
        strain time series sampling rate (default 1.0/2048)
    flow: 
        low frequency cutoff
        
    Returns
    -------
        snr_ts: 2d array of SNR time series
    """
    delta_f = 1.0 / duration
    tsamples = int(duration / delta_t)
    flen = int(2048 / delta_f) + 1

    # Correlate waveform with template
    sample_template_correlated = np_correlate(template, sample)

    # Divide by PSD
    sample_template_correlated /= psd

    # Pad row to desired length
    paddings = [[0, 0], [kmin, N-kmax]]
    sample_template_correlated = np.pad(sample_template_correlated, paddings, 'constant')

    sample_template_correlated = np.fft.ifft(sample_template_correlated)
    sample_template_correlated *= len(sample_template_correlated[0])

    # Calculate normalisation constant
    template_norm = np_sigmasq(template, psd, delta_f)

    # Rescale SNR timeseries
    snr_ts = sample_template_correlated * template_norm

    snr_ts = snr_ts.astype(np.complex64)

    return snr_ts


def mf_in_place(
    sample: np.ndarray, 
    psd: np.ndarray, 
    N: int, 
    kmin: int, 
    kmax: int, 
    template_conj: np.ndarray, 
    template_norm: np.ndarray
)-> np.ndarray:
    """A more optimised version of numpy_matched_filter. By precomputing template_conj with np.conjugate(template), 
    and template_norm with np_sigmasq we can speed up matched filtering by about 20% and reduce memory usage by half.
    This is only useful if you are matching with the SAME template array with a different sample array multiple times, 
    such as in a background run or injection run.

    Parameters
    ----------

    sample:
        A 2d array of strain samples in frequency domain. Can be the same strain sample multiple times,
        i.e. with np.repeat for matching with multiple templates.
    psd:
        PSD array
    N:
        length of the samples
    kmin:
        index of low frequency cutoff
    kmax:
        index of high frequency cutoff
    template_conj:
        A 2d array of templates in frequency domain
    template_norm:
        Normalisation constants for the templates
    
    Returns
    -------
    sample * template_norm: 
        2d array of SNR time series, in the same memory location as sample.
    """

    sample *= template_conj

    sample /= psd

    # Pad row to desired length
    paddings = [[0, 0], [kmin, N-kmax]]
    sample = np.pad(sample, paddings, 'constant')

    sample = np.fft.ifft(sample)
    sample *= len(sample[0])


    return sample * template_norm