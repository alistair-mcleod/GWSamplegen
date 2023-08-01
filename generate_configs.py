#this file will generate the config files for a project. 

#NEW PLAN: each 'config' directory will instead be a `project` directory containing:
#1. an args file, which defines the waveform args to use, as well as the noise and template bank directories
#2. directories for  associated SNR series + signal/noise files, and a params file containing all that directory's parameters
# This way we can generate multiple files with the same parameters without having to create an entirely new config each time.
# There should also be some params that can be changed on a per-file basis, such as seconds of EW


#to note: we can have multiple noise datasets, and these can be used by different 'projects'
#we can also have multiple template banks, which are also not necessarily unique to a project
#we therefore need to store metadata in the noise and template bank directories, and will need to do some form of
#checking to determine if the noise and template bank are compatible with the project


#workflow of this project:
#1. generate param file(s) inc. template param file.
#2. generate template waveforms
#3. generate noise files
#4. for now: generate strain etc and save
#in future: step 4. will be done on the fly




#TODO params to add:
#SEEDS
#waveform length?

#Other params that probably should be added in a DIFFERENT file:
#seconds before + after merger to slice SNR timeseries
#seconds of early warning
#duration of noise to fetch?

import numpy as np
from pycbc.filter import sigma
from pycbc.detector import Detector
from pycbc.waveform import get_td_waveform
from pycbc.psd import interpolate
from utils.waveform_utils import choose_templates
import multiprocessing as mp
import h5py
import json
import os

import time

from bilby.core.prior import (
    ConditionalPowerLaw,
    ConditionalPriorDict,
    Constraint,
    Cosine,
    Gaussian,
    LogNormal,
    PowerLaw,
    PriorDict,
    Sine,
    Uniform,
)

from bilby.gw.prior import UniformComovingVolume, UniformSourceFrame

from typing import TYPE_CHECKING, Optional, Union


#SHIELD: SNR-based Highly Intelligent Early Latencyless Detection network

#start of user-defined params and param ranges. 

#seed for reproducibility

seed = 8810237

#number of CPUS to use
n_cpus = 20

project_dir = "./configs/rectest"
noise_dir = './noise/test'
#template_bank_dir = './template_banks/BNS_lowspin_freqseries'

#number of samples to generate
n_signal_samples = 50000
n_noise_samples = 50000

#n_samples = 10000
#noise_frac = 0.5
glitch_frac = 0


approximant = "SpinTaylorT4"
f_lower = 18
delta_t = 1/2048

#If possible, make waveform_lenth a power of 2. This reduces error between pycbc and tensorflow in the SNR calculation.
waveform_length = 1024

waveforms_per_file = 1000

detectors = ['H1','L1']

network_snr_threshold = 0
detector_snr_threshold = 6

#number of waveform templates to match with each waveform. These templates are taken from a distribution,
#but the first template chosen will be guaranteed to have a high overlap with the waveform.
templates_per_waveform = 1

template_approximant = "TaylorF2"
#when we select templates to match with the waveform, we select from a distribution of templates, but have to ensure
#that the template has at least some overlap with the waveform. for BBH signals you can use a width up to 0.05-0.1,
#but for BNS signals you should use a width of 0.02 or less as they are more sensitive to chirp mass.
template_selection_width = 0.01

################################################
#---------------INTRINSIC PARAMS---------------#
################################################


#alpha value to be used for power law priors
powerlaw_alpha = -3.0

#Prior functions to use for component masses.
#For an astrophysical BBH distribution, use Powerlaw
#For a BNS distribution, use Uniform (TODO implement Gaussian)
mass1prior = Uniform
mass2prior = Uniform

mass1_min = 1.0
mass1_max = 2.6

mass2_min = 1.0
mass2_max = 2.6

#prior functions to use for spins. 
spin1zprior = Uniform
spin2zprior = Uniform

spin1z_min = 0.0
spin1z_max = 0.0

spin2z_min = 0.0
spin2z_max = 0.0


################################################
#---------------EXTRINSIC PARAMS---------------#
################################################


#prior functions to use for right ascension and declination.
#RA is transformed from 0 <= x <= 1 to 0 <= x <= 2pi
#DEC is transformed from 0 <= x <= 1 to -pi/2 <= x <= pi/2

ra_prior = Uniform #RA should always be uniform
dec_prior = Cosine

ra_min = 0.0
ra_max = 1.0

dec_min = 0.0
dec_max = 1.0

#prior function for distance. Should be either Uniform, UniformSourceFrame or UniformComovingVolume.

d_prior = UniformSourceFrame

d_min = 10.0
d_max = 100.0

#prior function for inclination. Should be Sine.
#should be 0 <= inc_min <= inc_max <= 1.

inc_prior = Sine

inc_min = 0.0
inc_max = 1.0

#prior function for polarization. Should be Uniform.
#Polarization is transformed from 0 <= x <= 1 to 0 <= x <= 2pi
#should be 0 <= pol_min <= pol_max <= 1.

pol_prior = Uniform

pol_min = 0.0
pol_max = 1.0




if not os.path.exists(noise_dir):
    raise ValueError("Noise directory does not exist. Generate a directory of noise to use with this dataset.")

#check that these parameters are compatible with those from the noise directory
with open(noise_dir + '/args.json') as f:
    noise_args = json.load(f)
    for ifo in detectors:
        if ifo not in noise_args['detectors']:
            raise ValueError("""Noise directory does not contain all the specified detectors.
                             Check noise directory and config file.""")
   
    if noise_args['delta_t'] != delta_t:
        raise ValueError("""Noise delta_t does not match specified delta_t.
                             Check noise directory and config file.""")

#check that the template bank directory exists

#if not os.path.exists(template_bank_dir):
#    raise ValueError("Template bank directory does not exist. Generate a template bank to use with this dataset.")

#check that these parameters are compatible with those from the template bank directory
#TODO: instead of checking for compatibility, generate the template param file here! add it to the config dir
#since we're not saving templates any more.

#with open(template_bank_dir + '/args.json') as f:
#    template_bank_args = json.load(f)
#    if template_bank_args['approximant'] != template_approximant:

#        print("""Warning: template bank approximant is {} but specified approximant is {}"""\
#                         .format(template_bank_args['approximant'], approximant))

#    if template_bank_args['f_lower'] != f_lower:
#        print("""Warning: template bank f_lower is {} but specified f_lower is {}"""\
#                         .format(template_bank_args['f_lower'], f_lower))
    
#    if template_bank_args['delta_t'] != delta_t:
#        raise ValueError("""Fatal Error: template bank delta_t is {} but specified delta_t is {}"""\
#                         .format(template_bank_args['delta_t'], delta_t))


if not os.path.exists(project_dir):
    os.mkdir(project_dir)

#ADDING TEMPLATE BANK GENERATION HERE

tmass1_min = 1
tmass1_max = 3

tmass2_min = 1
tmass2_max = 3

q_min = 0.0
q_max = 1.0

#rather than selecting for spins, we scale the template spins by a factor. set to 0 for no spins, 1 for full spins
spin_scale = 1

#args = {'approximant': approximant,'f_lower': f_lower,'delta_t': delta_t, 'delta_f': delta_f, 'templates_per_file': templates_per_file,
#        'mass1_min': mass1_min,'mass1_max': mass1_max,'mass2_min': mass2_min,'mass2_max': mass2_max,
#        'q_min': q_min,'q_max': q_max}


#templates are stored in the form: chirp mass, mass 1, mass 2, spin 1z, spin 2z
templates = np.load("template_banks/GSTLal_templates.npy")

#select only templates that are within the specified range

templates = templates[(templates[:,1] >= tmass1_min) & (templates[:,1] <= tmass1_max) & (templates[:,2] >= tmass2_min) & 
    (templates[:,2] <= tmass2_max) & (templates[:,2]/templates[:,1] >= q_min) & (templates[:,2]/templates[:,1] <= q_max)]

templates[:,3] *= spin_scale
templates[:,4] *= spin_scale

#sort the templates by chirp mass
templates = templates[templates[:,0].argsort()]

#save the template waveform params and waveform generation args for future reference
np.save(project_dir+"/template_params.npy",templates)


#with open(main_dir+bank_dir+"/template_args.json", 'w') as fp:
#    json.dump(args, fp, sort_keys=False, indent=4)

print("Number of templates: ", len(templates))	


template_bank_params = np.load(project_dir+"/template_params.npy")


def constructPrior(
    prior: Union[Uniform, Cosine, UniformComovingVolume,PowerLaw,UniformSourceFrame], 
    min: float, 
    max: float,
    **kwargs
) -> PriorDict:
    #generic constructor for bilby priors. 
    
    if prior == PowerLaw:
        kwargs['alpha'] = powerlaw_alpha

    if max <= min:
        return max
    else:
        return prior(minimum = min, maximum = max, **kwargs)



#set a seed to ensure reproducibility
np.random.seed(seed)


prior = PriorDict()

prior['mass1'] = constructPrior(mass1prior, mass1_min, mass1_max)
prior['mass2'] = constructPrior(mass2prior, mass2_min, mass2_max)
prior['spin1z'] = constructPrior(spin1zprior, spin1z_min, spin1z_max)
prior['spin2z'] = constructPrior(spin2zprior, spin2z_min, spin2z_max)

prior['ra'] = constructPrior(Uniform, ra_min * 2 * np.pi, ra_max * 2 * np.pi, boundary = 'periodic')
prior['dec'] = constructPrior(dec_prior, np.pi * ra_min - np.pi/2, np.pi * ra_max - np.pi/2)

prior['d'] = constructPrior(d_prior, d_min, d_max, name = 'luminosity_distance')
prior['i'] = constructPrior(inc_prior, inc_min * np.pi, inc_max * np.pi)
prior['pol'] = constructPrior(pol_prior, pol_min * np.pi *2, pol_max * np.pi *2)



from utils.noise_utils import get_valid_noise_times, load_psd
valid_times, _, _ = get_valid_noise_times(noise_dir,waveform_length)
#gps = np.random.permutation(gps)

#gps = np.repeat(gps,len(detectors),axis=0).reshape((len(gps),len(detectors)))
#eventually handle timeslides, for now we just use the same GPS time for each ifo

print(len(valid_times), "GPS times available")

#if len(gps) < (n_signal_samples + n_noise_samples):
#    raise ValueError("Not enough noise samples in noise directory. Generate more noise, or reduce the number of samples.")

#load PSD from noise_dir

psd = np.load(noise_dir + "/psd.npy")


from pycbc.types import FrequencySeries

#psds = {}
#psds["H1"] = FrequencySeries(psd[1], delta_f = 1.0/psd[0][1], dtype = np.complex128)
#psds["L1"] = FrequencySeries(psd[2], delta_f = 1.0/psd[0][1], dtype = np.complex128)
psds = load_psd(noise_dir, waveform_length, detectors, f_lower, int(1/delta_t))

all_detectors = {'H1': Detector('H1'), 'L1': Detector('L1'), 'V1': Detector('V1'), 'K1': Detector('K1')}

def get_projected_waveform_mp(args):
    
    hp, hc = get_td_waveform(mass1 = args['mass1'], mass2 = args['mass2'], 
                             spin1z = args['spin1z'], spin2z = args['spin2z'],
                             inclination = args['i'], distance = args['d'],
                             approximant = approximant, f_lower = f_lower, delta_t = delta_t)
    
    snrs = {}
    #waveforms = np.empty(shape=(len(detectors), len(hp)))

    for detector in detectors:
        f_plus, f_cross = all_detectors[detector].antenna_pattern(
            right_ascension=args['ra'], declination=args['dec'],
            polarization=args['pol'],
            t_gps=args['gps'][0])
        
        detector_signal = f_plus * hp + f_cross * hc

        snr = sigma(htilde=detector_signal,
                    psd=interpolate(psds[detector], delta_f=detector_signal.delta_f),
                    low_frequency_cutoff=f_lower)
        
        snrs[detector] = snr

        #detector_index = detectors.index(detector)
        #waveforms[detector_index] = detector_signal

    return snrs #waveforms, snrs

#good_waveforms = []

good_params = []

generated_samples = 0
iteration = 0

wavetime = 0

#injection_no_glitch = n_samples * (1-noise_frac) * (1-glitch_frac)
#injection_plus_glitch = n_samples * glitch_frac * (1-noise_frac) /len(detectors)

#noise_no_glitch = n_samples * noise_frac * (1-glitch_frac)
#noise_plus_glitch = n_samples * glitch_frac * noise_frac /len(detectors)

#glitches_per_ifo = n_samples * (1 - noise_frac) * glitch_frac / len(detectors)

#params = {}

#for ifo in detectors:
#	params[ifo] = np.zeros(n_samples)
#	glitch_start = int(n_samples * (1 - noise_frac) * (1-glitch_frac) + detectors.index(ifo) * glitches_per_ifo)
#	glitch_end = int(glitch_start + glitches_per_ifo)
#	params[ifo][glitch_start:glitch_end] = 1
#	print(glitch_start, glitch_end)
        

from utils.glitch_utils import get_glitchy_times, get_glitchy_gps_time
from utils.noise_utils import generate_time_slides

max_waveform = 200
SNR_thresh = 6
f_thresh = 18


glitchless_times = {}
glitchy_times = {}
glitchy_freqs = {}

for ifo in detectors:
    glitchy_times[ifo] = []
    glitchless_times[ifo] = []
    glitchy_freqs[ifo] = []

for ifo in detectors:
    
    glitchy, glitchless, freq = get_glitchy_times("noise/test/{}_glitches.npy".format(ifo),
                                                  waveform_length, valid_times, max_waveform, SNR_thresh, f_thresh)
    
    glitchless_times[ifo] = glitchless
    glitchy_times[ifo] = glitchy
    glitchy_freqs[ifo] = freq

#create timeslide generators
min_separation = 3

glitchless_generator = generate_time_slides([glitchless_times[ifo] for ifo in detectors], min_separation)
one_glitch_generator = {}

for ifo in detectors:
    time_list = [glitchy_times[i] if i == ifo else glitchless_times[i] for i in detectors]
    one_glitch_generator[ifo] = generate_time_slides(time_list, min_separation)



while generated_samples < n_signal_samples:

    #generate waveforms_per_file samples at a time, to avoid memory issues.

    p = prior.sample(waveforms_per_file)

    #adding non-sampled args to the parameters
    #p['gps'] = gps[:waveforms_per_file]
    #gps = gps[waveforms_per_file:]
    p['gps'] = []

    for detector in detectors:
        p[detector + '_glitch'] = np.zeros(waveforms_per_file, dtype = bool)
    for i in range(waveforms_per_file):
        if np.random.uniform(0,1) < glitch_frac:
            glitchy_ifo = np.random.choice(detectors)
            p[glitchy_ifo + '_glitch'][i] = True
            glitch_time = list(next(one_glitch_generator[glitchy_ifo]))
            glitch_idx = np.where(glitch_time[detectors.index(glitchy_ifo)] == glitchy_times[glitchy_ifo])[0][0]
            #TODO: shift glitch based on template masses, rather than true masses?
            glitch_time[detectors.index(glitchy_ifo)] = get_glitchy_gps_time(valid_times, p['mass1'][i], p['mass2'][i], 
                                                                             glitch_time[detectors.index(glitchy_ifo)], glitchy_freqs[glitchy_ifo][glitch_idx])
            p['gps'].append(glitch_time)            
            #gps.append(next(one_glitch_generator[glitchy_ifo]))
        else:
            p['gps'].append(list(next(glitchless_generator)))

    p['injection'] = np.ones(waveforms_per_file, dtype = bool)

    
    #turn dict of lists into a list of dicts (for multiprocessing)
    params = [{key: p[key][i] for key in p.keys()} for i in range(len(p['mass1']))]

    
    #generate the waveforms
    start = time.time()
    with mp.Pool(processes = n_cpus) as pool:

        #snrs is a list of dicts, where each dict is {detector: snr}
        snrs = pool.map(get_projected_waveform_mp, params)
        #mp_waveforms is a list of lists, where each list is [waveform, snrs]
        
        #waveforms, snrs = zip(*mp_waveforms)
    
    wavetime += time.time() - start
    
    #save only the waveforms with network SNR above threshold.
    #save in numpy files with associated parameters.

    for i in range(len(snrs)):

        network_snr = np.sqrt(sum([snrs[i][detector]**2 for detector in snrs[i]]))

        if network_snr > network_snr_threshold and all([snr > detector_snr_threshold for snr in snrs[i].values()]):
            #this sample is suitable, get it ready for saving
            #good_waveforms.append(waveforms[i])

            #add the detector SNRs and network SNR as keys in params[i]
            params[i]['network_snr'] = network_snr
            for detector in detectors:
                params[i][detector + '_snr'] = snrs[i][detector]

            #ensure that mass2 < mass1
            if params[i]['mass2'] > params[i]['mass1']:
                params[i]['mass1'], params[i]['mass2'] = params[i]['mass2'], params[i]['mass1']

            #choose template waveform(s) for this sample, and add them to params[i]
            params[i]['template_waveforms'] = choose_templates(template_bank_params, params[i], 
                                                               templates_per_waveform, template_selection_width)

            good_params.append(params[i])
        #else:
        #    #print("discarding waveform with SNR " + str(network_snr))
        #    #recycle gps time
        #    #gps = np.append(gps, [params[i]['gps']], axis = 0)

    #if iteration == 0 and len(good_waveforms)/waveforms_per_file < 0.5:
    #    print("WARNING: check your distance prior and SNR threshold! Only " + str(len(good_waveforms)/waveforms_per_file) + 
    #          " of the samples meet the SNR threshold.")
    #elif iteration == 0:
    #    print("SNR threshold looks good, {}% of samples meet the threshold.".format(len(good_waveforms)/waveforms_per_file*100))
    
    #now that we only have the good waveforms, we can save them to file.
    #we don't necessarily have waveforms_per_file samples in good_waveforms, so we need to check that.

    #if len(good_waveforms) > waveforms_per_file:
    #    fname = project_dir+"/"+str(iteration*waveforms_per_file)+".npz"

    #    print(fname)

    #    temp = good_waveforms[:waveforms_per_file]
    #    good_waveforms = good_waveforms[waveforms_per_file:]
    #    np.savez(fname, *temp)

    #    generated_samples += waveforms_per_file
    #    iteration +=1
    generated_samples = len(good_params)
    if generated_samples <= waveforms_per_file:
        if generated_samples/waveforms_per_file < 0.5:
            print("WARNING: check your distance prior and SNR threshold! Only {}% of the samples meet the SNR threshold."\
                  .format(round(generated_samples/waveforms_per_file*100)))
        else:
            print("SNR threshold looks good, {}% of samples meet the threshold.".format(round(generated_samples/waveforms_per_file*100)))



#save the injection parameters to a file
#convert from a list of dictionaries to a dictionary of lists

good_params_dict = {key: np.array([good_params[i][key] for i in range(len(good_params))][:n_signal_samples]) for key in good_params[0].keys()}



#generate noise samples. most of the parameters aren't used, but the masses are used to choose the templates.

if n_noise_samples > 0:
    noise_p = prior.sample(n_noise_samples)
    noise_p['gps'] = []#gps[:n_noise_samples]
    noise_p['injection'] = np.zeros(n_noise_samples, dtype = bool)

    templates = []

    #TODO: get the actual max SNR for the noise segment maybe?
    noise_p['network_snr'] = np.zeros(n_noise_samples)
    for detector in detectors:
        noise_p[detector + '_snr'] = np.zeros(n_noise_samples)
        noise_p[detector + '_glitch'] = np.zeros(n_noise_samples, dtype = bool)

    for i in range(n_noise_samples):

        if noise_p['mass2'][i] > noise_p['mass1'][i]:
            noise_p['mass1'][i], noise_p['mass2'][i] = noise_p['mass2'][i], noise_p['mass1'][i]

        if np.random.uniform(0,1) < glitch_frac:
            glitchy_ifo = np.random.choice(detectors)
            noise_p[glitchy_ifo + '_glitch'][i] = True
            glitch_time = list(next(one_glitch_generator[glitchy_ifo]))
            glitch_idx = np.where(glitch_time[detectors.index(glitchy_ifo)] == glitchy_times[glitchy_ifo])[0][0]
            glitch_time[detectors.index(glitchy_ifo)] = get_glitchy_gps_time(valid_times, noise_p['mass1'][i], noise_p['mass2'][i], 
                                                                                glitch_time[detectors.index(glitchy_ifo)], glitchy_freqs[glitchy_ifo][glitch_idx])
            noise_p['gps'].append(glitch_time)            
            #noise_p['gps'].append(next(one_glitch_generator[glitchy_ifo]))
        else:
            noise_p['gps'].append(list(next(glitchless_generator)))

        params = {key: noise_p[key][i] for key in noise_p.keys()}
        templates.append(choose_templates(template_bank_params, params, templates_per_waveform, template_selection_width))

    noise_p['template_waveforms'] = np.array(templates)

    for key in good_params_dict.keys():
        good_params_dict[key] = np.append(good_params_dict[key], noise_p[key], axis = 0)

#np.save(project_dir+"/"+"noise_params.npy", noise_p)
np.save(project_dir+"/"+"params.npy", good_params_dict)

#save the arguments used to generate the parameters to a file

args = {"n_signal_samples": n_signal_samples,
            "n_noise_samples": n_noise_samples,
            "approximant": approximant,
            "f_lower": f_lower,
            "delta_t": delta_t,
            "detectors": detectors,
            "network_snr_threshold": network_snr_threshold,
            "detector_snr_threshold": detector_snr_threshold,
            "powerlaw_alpha": powerlaw_alpha,
            "mass1prior": mass1prior.__name__,
            "mass2prior": mass2prior.__name__,
            "mass1_min": mass1_min,
            "mass1_max": mass1_max,
            "mass2_min": mass2_min,
            "mass2_max": mass2_max,
            "spin1zprior": spin1zprior.__name__,
            "spin2zprior": spin2zprior.__name__,
            "spin1z_min": spin1z_min,
            "spin1z_max": spin1z_max,
            "spin2z_min": spin2z_min,
            "spin2z_max": spin2z_max,
            "ra_prior": ra_prior.__name__,
            "dec_prior": dec_prior.__name__,
            "ra_min": ra_min,
            "ra_max": ra_max,
            "dec_min": dec_min,
            "dec_max": dec_max,
            "d_prior": d_prior.__name__,
            "d_min": d_min,
            "d_max": d_max,
            "inc_prior": inc_prior.__name__,
            "inc_min": inc_min,
            "inc_max": inc_max,
            "pol_prior": pol_prior.__name__,
            "pol_min": pol_min,
            "pol_max": pol_max,
            "seed":seed}

#save args
with open(project_dir+"/"+"args.json", 'w') as f:
    json.dump(args, f, sort_keys=False, indent=4)

print("finished generating waveforms. time taken: " + str(wavetime/60) + " minutes")