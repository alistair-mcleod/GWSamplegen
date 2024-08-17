"""
================
Utilities for selecting templates for matching with data.
================

"""

import numpy as np
import scipy.stats as st
import json
from pycbc.tmpltbank.option_utils import metricParameters
from pycbc.tmpltbank.coord_utils import get_point_distance, get_cov_params
from pathlib import Path
import h5py
from typing import Tuple


def chirp_mass(m1,m2):
    return ((m1 * m2)**0.6)/ (m1 + m2)**0.2

def t_at_f(
    m1: float,
    m2: float,
    f: float
) -> float:
    """Compute the time at which a binary system with given masses will reach a given frequency, to the first order.
    
    Parameters
    ----------

    m1: float
        Primary mass of the binary system, in solar masses.
    m2: float
        Secondary mass of the binary system, in solar masses.
    f: float
        Frequency at which to compute the time, in Hz.

    Returns
    -------

    t_at_f: float
        Time at which the binary system will reach the given frequency, in seconds.
    """
    top = 5 * ((3e8)**5) * (((m1+m2)*1.99e30)**(1/3))
    bottom = (f**(8/3))*256*(np.pi**(8/3)) * ((6.67e-11)**(5/3)) *m1*m2 * 1.99e30 * 1.99e30
    #adding a 1% fudge factor as it's better to overestimate the time than underestimate it for the purpose of avoiding glitches
    return 1.01*(top/bottom) 


def f_at_t(
    m1: float,
    m2: float,
    t: float
) -> float:
    """Compute the frequency of a binary system with given masses at a given time, to the first order.
    
    Parameters
    ----------

    m1: float
        Primary mass of the binary system, in solar masses.
    m2: float
        Secondary mass of the binary system, in solar masses.
    t: float
        Time at which to compute the frequency, in seconds.

    Returns
    -------

    f_at_t: float
        Frequency of the binary system at the given time, in Hz.
    """

    top = 5 * ((3e8)**5) * (((m1+m2)*2e30)**(1/3))
    bottom = t*256*(np.pi**(8/3)) * ((6.67e-11)**(5/3)) *m1*m2 * 2e30 * 2e30

    return (top/bottom)**(3/8)



def load_pycbc_templates(
    bank_name: str, 
    template_dir: Path = "template_banks", 
    pnOrder: str = "threePointFivePN", 
    f_lower: float = 30,
    f_upper: float = 1024,
    deltaF: float = 0.01
) -> Tuple[np.ndarray, metricParameters, np.ndarray]:
    """Load a PyCBC template bank file, as well as the metric used to generate it.
    This function should be used in conjunction with a template bank generated by pycbc's pycbc_geom_aligned_bank

    Parameters
    ----------

    bank_name : str
        Name of the template bank file and the 'intermediate' file containing the associated metricParams. 
        The template bank file should be a text file named 
        bank_name.txt with each line structured as chirp mass,m1,m2,spin1z,spin2z. 
        The intermediate file is generated by PyCBC and should be called bank_name.hdf

    template_dir : Path
        Directory containing the template bank and intermediate file.

    pnOrder : str
        Post-Newtonian order used to generate the template bank.

    f_lower : float
        Lower frequency cutoff of the template bank. TODO: might not actually be necessary.

    f_upper : float
        Upper frequency cutoff of the template bank. Typically 1024 Hz.

    deltaF : float
        Frequency resolution of the template bank. Typically 0.01 Hz. 
        Note this delta F is not necessarily the same as the delta F used in the SNR time series.

    Returns
    -------
    templates : np.ndarray
        Array of templates, with each row structured as chirp mass,m1,m2,spin1z,spin2z.
    
    metricParams : metricParameters
        the pycbc.tmpltbank.metricParameters for the bank

    aXis : np.ndarray
        Pre-computed xi parameters for the templates. Use in choose_templates_new for faster template selection.
    """

    templates = np.loadtxt(template_dir + "/" + bank_name + ".txt", delimiter=",")

    #sort templates by chirp mass
    templates = templates[np.argsort(templates[:,0])]

    f = h5py.File(template_dir + "/" + bank_name + "_intermediate.hdf", "r")

    metricParams = metricParameters(pnOrder=pnOrder, fLow=f_lower, fUpper=f_upper, deltaF=deltaF)

    metricParams.evals = {metricParams.fUpper: f["metric_evals"][()]}
    metricParams.evecs = {metricParams.fUpper: f["metric_evecs"][()]}
    metricParams.evecsCV = {metricParams.fUpper: f["cov_evecs"][()]}

    #also pre-load the templates in the xi parameter space for faster template selection later. use in choose_templates_new
    aXis = get_cov_params(templates[:,1],templates[:,2],templates[:,3],templates[:,4],metricParams,metricParams.fUpper)
    
    return templates, metricParams, aXis

def fast_point_distance(
    aXis: np.ndarray, 
    point2: np.ndarray, 
    metricParameters: metricParameters
) -> np.ndarray:
    """Compute the distance between a point and a set of points in the xi parameter space, 
    using the metric defined by metricParameters.
    
    Parameters
    ----------

    aXis : np.ndarray
        Array of xi parameters for the set of points to compare to. Should be loaded by load_pycbc_templates.

    point2 : np.ndarray
        Point to compare to the set of points. Should be structured as [mass1,mass2,spin1z,spin2z].
    
    metricParameters : metricParameters
        the pycbc.tmpltbank.metricParameters for the bank. Should be loaded by load_pycbc_templates.
    
    Returns
    -------

    dist : np.ndarray
        Array of distances between the point and the set of points in the xi parameter space.
    """
    bMass1 = point2[0]
    bMass2 = point2[1]
    bSpin1 = point2[2]
    bSpin2 = point2[3]

    bXis = get_cov_params(bMass1, bMass2, bSpin1, bSpin2, metricParameters, metricParameters.fUpper)

    dist = (aXis[0] - bXis[0])**2
    for i in range(1,len(aXis)):
        dist += (aXis[i] - bXis[i])**2

    return dist

def choose_templates_new(
    templates: np.ndarray, 
    metricParams: metricParameters, 
    n_templates: int, 
    mass1: float, 
    mass2: float, 
    spin1z: float = 0, 
    spin2z: float = 0, 
    limit: int = 100, 
    aXis: np.ndarray = None
) -> np.ndarray:
    
    """ Choose a set of templates from a PyCBC template bank using the template's metric.
    The templates are chosen to be close to the given waveform parameters, 
    using the distance in xi space as an approximation of the overlap between the template and the waveform.

    Parameters
    ---------- 
    templates: array_like
        A list of templates to choose from. Columns should be [chirp mass, mass1, mass2, spin1z, spin2z]

    metricParams: pycbc.tmpltbank.metricParameters that were generated using the same metric as the template bank

    n_templates: int
        Number of templates to return.

    limit: int
        Maximum template index (sorted by distance) to consider. Templates are selected randomly up to this limit.

    aXis: array_like
        Pre-computed xi parameters for the templates. If not provided, they will be computed here. 
        Use the xi parameters provided by load_pycbc_templates to significantly speed up template selection."""
    
    if aXis is None:
        mismatches = get_point_distance(templates[:,1:5].T,[mass1,mass2,spin1z,spin2z],metricParams, metricParams.fUpper)[0]
    else:
        mismatches = fast_point_distance(aXis, [mass1,mass2,spin1z,spin2z], metricParams)

    #get the template indexes sorted by distance
    mismatches = np.argsort(mismatches)

    #np.argsort(mismatches)[::skip][:n_templates]

    #always return the best template first
    ret = [mismatches[0]]
    #append a random selection of the rest up to the limit
    ret.extend(np.random.choice(mismatches[1:limit], size = n_templates - 1, replace = False))

    return ret




def errfunc(mass1,mass2,m1true,m2true):
    #function for choosing a template which will produce a good match between the template and true waveform.
    #this function can produce templates with a resonable match, provided the template and true waveform are nonspinning.
    #for BNS systems, this method has been superseded by the use of the metric space.

    return np.abs(mass2/mass1 - m2true/m1true) + 1000*np.abs(chirp_mass(mass1,mass2) - chirp_mass(m1true,m2true))


def choose_templates(
    template_bank_params: np.ndarray, 
    waveform_params: dict, 
    templates_per_waveform: int,
    template_selection_width: float
) -> np.ndarray:
    
    """Choose a set of templates from a template bank using the waveform's chirp mass.
    The templates are chosen to be close to the given waveform parameters,
    using the distance in chirp mass as an approximation of the overlap between the template and the waveform.
    
    Parameters
    ----------

    template_bank_params : np.ndarray
        Array of template parameters, with each row structured as chirp mass,mass1,mass2,spin1z,spin2z.

    waveform_params : dict
        Dictionary of waveform parameters, with keys mass1 and mass2.
    
    templates_per_waveform : int
        Number of templates to return.
    
    template_selection_width : float
        Fraction of the chirp mass to consider when selecting templates. 
        For example, if the chirp mass is 30, and the template_selection_width is 0.1, 
        the templates will be selected from the range 27 to 33. A small value is recommended for BNS/NSBH systems,
        i.e. around 0.01.

    Returns
    -------

    template_indexes : np.ndarray
        Array of indexes of the chosen templates.    
    """

    mass1,mass2 = waveform_params['mass1'],waveform_params['mass2']
    cm = chirp_mass(mass1,mass2)

    t_mass1,t_mass2 = template_bank_params[:,1], template_bank_params[:,2]

    #given a loaded array of template params and a waveform param, return the closest template.

    best_template =  np.argsort(errfunc(mass1,mass2,t_mass1,t_mass2))[0]

    #selecting a template range
    low_idx = np.searchsorted(template_bank_params[:,0],cm*(1-template_selection_width/2))
    high_idx = np.searchsorted(template_bank_params[:,0],cm*(1+template_selection_width/2))

    #choosing some suboptimal templates from a normal distribution, and 1 optimal template.

    x = np.arange(low_idx, high_idx)

    # Sigma values for the two sides of the split distribution
    # Here, the 2 refers to the number of standard deviations either side of the peak

    sigma_low = int((best_template - low_idx) / 2)
    sigma_high = int((high_idx - best_template) / 2)

    diff = best_template - low_idx
    # Calculate separate PDFs for below and above the best template

    pdf1 = st.truncnorm.pdf(x, (low_idx - best_template) / sigma_low, (high_idx - best_template) / sigma_low, loc=best_template, scale=sigma_low)
    pdf2 = st.truncnorm.pdf(x, (low_idx - best_template) / sigma_high, (high_idx - best_template) / sigma_high, loc=best_template, scale=sigma_high)

    # Rescale each pdf to 50% and concatenate them together

    scale1 = 0.5 / pdf1[:diff].sum()
    scale2 = 0.5 / pdf2[diff:].sum()
    pdf3 = np.concatenate((pdf1[:diff]*scale1, scale2*pdf2[diff:]))

    #This is where we sample the number of templates we want
    #if there are nans, we are near the edge of our template bank.
    #we instead use the pdf from only one side.

    if scale1 == np.inf or np.isnan(scale1):
        print("Left PDF is nan, using right PDF")
        if len(x) < templates_per_waveform:
            x = np.arange(len(template_bank_params)-templates_per_waveform,len(template_bank_params)-1)
        else:
            x = np.random.choice(x[diff:], size=templates_per_waveform-1, p=pdf2[diff:]*scale2*2, replace=False)
    elif scale2 == np.inf or np.isnan(scale2):
        print("Right PDF is nan, using left PDF")
        x = np.random.choice(x[:diff], size=templates_per_waveform-1, p=pdf1[:diff]*scale1*2, replace=False)
    else:
        if len(x) < templates_per_waveform:
            x = np.arange(len(template_bank_params)-templates_per_waveform,len(template_bank_params)-1)
        else:
            pdf3 = np.concatenate((pdf1[:diff]*scale1, scale2*pdf2[diff:]))
            x = np.random.choice(x, size=templates_per_waveform-1, p=pdf3, replace=False)
    #x = np.random.choice(x, size=templates_per_waveform-1, p=pdf3, replace=False)

    x = np.sort(x)
    x = np.insert(x,0,best_template)

    #making sure the templates are all unique
    for i in range(1,len(x)-1):
        if x[i] >= x[i+1]:
            x[i+1] = x[i] + 1
    
    #making sure the templates are all within the template bank
    if x[-1] >= len(template_bank_params):
        x -= x[-1] - len(template_bank_params) -1

    return x