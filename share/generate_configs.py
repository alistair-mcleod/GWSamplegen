#this file will generate the config files for a project. 


import argparse
import multiprocessing as mp
import json
import os
import time
#from typing import Union
import gc

import numpy as np
from pycbc.filter import sigma, match, matchedfilter
from pycbc.detector import Detector
from pycbc.waveform import get_td_waveform, get_fd_waveform, td_approximants, fd_approximants
from pycbc.types import FrequencySeries, TimeSeries
from pycbc.psd import interpolate
from pycbc.inject.inject import legacy_approximant_name
import bilby
from bilby.core.prior import (
    Cosine,
    PowerLaw,
    PriorDict,
    Sine,
    Uniform,
    Triangular,
)
from bilby.gw.prior import UniformComovingVolume, UniformSourceFrame

from GWSamplegen.waveform_utils import load_pycbc_templates, choose_templates_new, chirp_mass, maximum_f_lower, select_approximant, t_at_f,fast_point_distance
from GWSamplegen.glitch_utils import get_glitchy_times, get_glitchy_gps_time
from GWSamplegen.noise_utils import two_det_timeslide, get_valid_noise_times, load_psd
from GWSamplegen.prior_utils import constructPrior, TriUniform, draw_mass_pair_power, draw_spin_isotropic, sample_masses_from_cm_q
from GWSamplegen.template_utils import find_templates
#from asyncSNR_np import get_projected_waveform_mp
from GWSamplegen.waveform_utils import get_projected_waveform_mp

import astropy.units as u
import astropy.cosmology as cosmo
from astropy.cosmology import FlatwCDM
from astropy.utils import iers
iers.conf.auto_download = False




def get_snr(args):
    
    temp_td = select_approximant(args['mass1'], args['mass2'], td_approximant, domain='time')
    temp_f_lower = min(f_lower, maximum_f_lower(args['mass1'], args['mass2']))
    if temp_td in ["TaylorF2Ecc", "EccentricFD", "EccentricTD"]:
        temp_f_lower = 20

    if temp_td in td_approximants():
        
        try:
            hp, hc = get_td_waveform(
                mass1 = args['mass1'], mass2 = args['mass2'],
                spin1x = args['spin1x'], spin2x = args['spin2x'],
                spin1y = args['spin1y'], spin2y = args['spin2y'], 
                spin1z = args['spin1z'], spin2z = args['spin2z'],
                inclination = args['i'], distance = args['d'],
                approximant = temp_td, f_lower = temp_f_lower, delta_t = delta_t/4
            )
        except:
            hp, hc = get_td_waveform(
                mass1 = args['mass1'], mass2 = args['mass2'], 
                spin1x = args['spin1x'], spin2x = args['spin2x'],
                spin1y = args['spin1y'], spin2y = args['spin2y'],
                spin1z = args['spin1z'], spin2z = args['spin2z'],
                inclination = args['i'], distance = args['d'],
                approximant = temp_td, f_lower = temp_f_lower, delta_t = delta_t / 8, f_final = 8/delta_t
            )
        #downsample to the correct delta_t
        hp = hp.resample(delta_t)
        hc = hc.resample(delta_t)

    elif temp_td not in td_approximants() and temp_td in fd_approximants():
        hp_f, hc_f = get_fd_waveform(
            args, inclination=args['i'], distance=args['d'],
            approximant = temp_td, f_lower = temp_f_lower, delta_f = 1/duration, f_final = 1024)
        hp = hp_f.to_timeseries(delta_t=delta_t)
        hc = hc_f.to_timeseries(delta_t=delta_t)
        
    snrs = {}

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

    return snrs



# def get_snr(args):
#     if td_approximant == "SEOBNRv4PHM" and args['mass1'] + args['mass2'] < 9:
#         temp_td = "SEOBNRv4P"
#     elif td_approximant == "SEOBNRv4P" and args['mass1'] + args['mass2'] > 9:
#         temp_td = "SEOBNRv4PHM"
#     else:
#         temp_td = td_approximant
#     temp_td = waveform_approximant
#     if temp_td == "TaylorF2":
#         temp_td = "SpinTaylorT4"
#     try:
#         hp, hc = get_td_waveform(
#             mass1 = args['mass1'], mass2 = args['mass2'], 
#             spin1z = args['spin1z'], spin2z = args['spin2z'],
#             inclination = args['i'], distance = args['d'],
#             approximant = temp_td, f_lower = f_lower, delta_t = delta_t
#         )
#     except:
#         try:
#             hp, hc = get_td_waveform(
#                 mass1 = args['mass1'], mass2 = args['mass2'], 
#                 spin1z = args['spin1z'], spin2z = args['spin2z'],
#                 inclination = args['i'], distance = args['d'],
#                 approximant = temp_td, f_lower = 6, delta_t = delta_t
#             )
#         except:
#             hp, hc = get_td_waveform(
#                 mass1 = args['mass1'], mass2 = args['mass2'], 
#                 spin1z = args['spin1z'], spin2z = args['spin2z'],
#                 inclination = args['i'], distance = args['d'],
#                 approximant = temp_td, f_lower = 10, delta_t = delta_t / 8, f_final = 8/delta_t
#             )
#             #downsample to the correct delta_t
#         hp = hp.resample(delta_t)
    
#     snrs = {}

#     for detector in detectors:
#         f_plus, f_cross = all_detectors[detector].antenna_pattern(
#             right_ascension=args['ra'], declination=args['dec'],
#             polarization=args['pol'],
#             t_gps=args['gps'][0])
        
#         detector_signal = f_plus * hp + f_cross * hc

#         snr = sigma(htilde=detector_signal,
#                     psd=interpolate(psds[detector], delta_f=detector_signal.delta_f),
#                     low_frequency_cutoff=f_lower)
        
#         snrs[detector] = snr

#     return snrs


def get_template(task):
    if task[0][1] + task[0][2] <= 4:  # This is the number quoted for PyCBC in GWTC-3
        approx = "TaylorF2"
    else:
        approx = "SEOBNRv4_ROM"

    # if chirp_mass(task[0][1], task[0][2]) < 1.73:  # This is the number quoted for GstLAL in GWTC-3
    #     approx = 'TaylorF2'
    # else:
    #     approx = "SEOBNRv4_ROM"
        
    hp, _ = get_td_waveform(
        mass1=task[0][1],
        mass2=task[0][2],
        spin1z=task[0][3],
        spin2z=task[0][4],
        distance=100,
        f_lower=task[2],
        approximant=approx,
        delta_t=task[1]
    )
    hp.prepend_zeros(int(task[3]/task[1] - len(hp.data)))
    hp_fs = hp.to_frequencyseries()
    
    return hp_fs.data


def get_match(task):
    args = task[0]
    h1 = get_projected_waveform_mp(args)[0]
    h1_fs = TimeSeries(h1, delta_t=args["delta_t"]).to_frequencyseries()
    
    overlaps = []
    for key in range(task[2],task[3]):
        x = match(h1_fs, FrequencySeries(template_waveforms[key], delta_f=h1_fs.delta_f), psd=psds['H1'])  # Currently using H1 psd cause it is less sensitive
        overlaps.append([x[0], key])
        
    overlaps = sorted(overlaps, key=lambda x:x[0], reverse=True)
    
    return overlaps


def get_overlaps_for_template(x, y, z):
    temp = []
    for i in list(z):
        temp.append(x[list(y).index(int(i))])
    return temp


def template_choice(chosen_templates, chosen_overlaps, n_templates, inj_index, all_indexes, all_overlaps, choice_indexes, choice_overlaps, skewed=False):
    if len(choice_indexes) < n_templates:
        print(f'For injection {inj_index}, there are not sufficient templates with net network SNR > 3 to sample from, so taking the best {n_templates} templates.')
        chosen_templates.append(all_indexes[:n_templates])
        chosen_overlaps.append(get_overlaps_for_template(all_overlaps, all_indexes, all_indexes[:n_templates]))
    else:
        temp = [choice_indexes[0]]
        if skewed:
            p = np.copy([len(choice_indexes[1:])-j for j in range(len(choice_indexes[1:]))])
            p = p / np.sum(p)
            temp.extend(np.random.choice(choice_indexes[1:], size=n_templates-1, replace=False, p=p))
        else:
            temp.extend(np.random.choice(choice_indexes[1:], size=n_templates-1, replace=False))
        chosen_templates.append(temp)
        chosen_overlaps.append(get_overlaps_for_template(choice_overlaps, choice_indexes, temp))
    return chosen_templates, chosen_overlaps


def choose_templates_match(overlaps, n_templates, all_network_snrs):
    """ Choose a set of templates from a bank that uses the PyCBC `match` function to calculate template overlaps.
    Currently only selecting the highest overlap template and the rest are randomly sampled.
    
    Parameters
    ----------
    overlaps: array_like
        A list of lists, which one list per injection samples containing the template indexes and respective overlaps with the injection.
    n_templates: int
        Number of templates to choose for each injection."""
    
    chosen_templates = []
    chosen_overlaps = []
    for key,i in enumerate(overlaps):
        i = [[j[0],j[1],all_network_snrs[key]*j[0]] for j in i]
        all_overlaps = np.copy(i)[:,0]
        indexes = np.copy(i)[:,1]
        all_net_snrs = np.copy(i)[:,2]
        
        temp_overlaps_gt0_5 = [[j[0],j[1]] for j in i if j[0]>0.5]
        overlaps_gt0_5 = np.copy(temp_overlaps_gt0_5)[:,0]
        indexes_gt0_5 = np.copy(temp_overlaps_gt0_5)[:,1]

        temp_overlaps_netsnr_gt3 = [[j[0],j[1],j[2]] for j in i if j[2]>3]
        overlaps_snr_gt3 = np.copy(temp_overlaps_netsnr_gt3)[:,0]
        indexes_snr_gt3 = np.copy(temp_overlaps_netsnr_gt3)[:,1]
        net_snr_gt3 = np.copy(temp_overlaps_netsnr_gt3)[:,2]

        if template_selection == "overlap":
            chosen_templates, chosen_overlaps = template_choice(
                chosen_templates, chosen_overlaps,
                n_templates, key,
                indexes, all_overlaps,
                indexes_gt0_5, overlaps_gt0_5,
                skewed=template_selection_skewed
            )
        elif template_selection == "snr":
            chosen_templates, chosen_overlaps = template_choice(
                chosen_templates, chosen_overlaps,
                n_templates, key,
                indexes, all_overlaps,
                indexes_snr_gt3, overlaps_snr_gt3,
                skewed=template_selection_skewed
            )
        elif template_selection == "random":
            temp = [indexes[0]]
            if template_selection_skewed:
                p = np.copy([len(indexes[1:])-j for j in range(len(indexes[1:]))])
                p = p / np.sum(p)
                temp.extend(np.random.choice(indexes[1:], size=n_templates-1, replace=False, p=p))
            else:
                temp.extend(np.random.choice(indexes[1:], size=n_templates-1, replace=False))
            chosen_templates.append(temp)
            chosen_overlaps.append(get_overlaps_for_template(all_overlaps, indexes, temp))
        elif template_selection == "best":
            chosen_templates.append(indexes[:n_templates])
            chosen_overlaps.append(get_overlaps_for_template(all_overlaps, indexes, indexes[:n_templates]))
        else:
            print("Invalid `template_selection` method. Please choose either 'overlap', 'snr', 'random', or 'best'.")
            exit()
        
    return chosen_templates, chosen_overlaps




def get_overlaps(args):
    # if td_approximant == "SEOBNRv4PHM" and args['mass1'] + args['mass2'] < 9:
    #     temp_td_approximant = "SEOBNRv4P"
    # elif td_approximant == "SEOBNRv4P" and args['mass1'] + args['mass2'] > 9:
    #     temp_td_approximant = "SEOBNRv4PHM"
    # else:
    #     temp_td_approximant = td_approximant
    temp_td_approximant = select_approximant(args['mass1'], args['mass2'], td_approximant, domain='time')

    if temp_td_approximant in td_approximants():
        hp,_ = get_td_waveform(approximant=temp_td_approximant, mass1=args['mass1'], mass2=args['mass2'],
                            spin1x=args['spin1x'], spin2x=args['spin2x'],
                            spin1y=args['spin1y'], spin2y=args['spin2y'],
                            spin1z=args['spin1z'], spin2z=args['spin2z'],
                            f_lower=10, delta_t=1/(8*2048))

        hp = hp.resample(delta_t)
        if -hp.sample_times[0] > duration:
            print("shortening sample")
            chop = np.argmin(np.abs(hp.sample_times + duration-10))
            hp.start_time = hp.sample_times[chop]
            hp.data = hp.data[chop:]
            print("New sample delta f is ", 1/hp.delta_f)
        try:
            #print("trying to convert to frequency series")
            hp_f = hp.to_frequencyseries(delta_f=1/duration)
        except:
            print("failed to convert to freq. series on sample ", args)
            print("masses were ", args['mass1'], args['mass2'])
        del hp
    elif temp_td_approximant not in td_approximants() and temp_td_approximant in fd_approximants():
        print("TD approximant is FD only. No need to convert.")
        hp_f, _ = get_fd_waveform(args, approximant=temp_td_approximant, f_lower=20, delta_f=1/duration)


    olaps = []
    for j in range(len(args['template_waveforms'])):
        idx = int(args['template_waveforms'][j])
        temp_fd_approximant = select_approximant(template_bank_params[idx][1], template_bank_params[idx][2], fd_approximant, domain='frequency')
        hpt_f, _ = get_fd_waveform(mass1 = template_bank_params[idx][1], mass2 = template_bank_params[idx][2],
                                spin1z = template_bank_params[idx][3], spin2z = template_bank_params[idx][4],
                                f_lower = 10, delta_f = 1/1024, f_final = 2048*4, approximant = temp_fd_approximant)
        hpt_f.resize(len(hp_f))
        olap = matchedfilter.match(hp_f, hpt_f, psd=psds['L1'], low_frequency_cutoff=f_lower, high_frequency_cutoff=1024)[0]
        olaps.append(olap)

        del hpt_f
    del hp_f
    gc.collect()
    print("overlaps for injection ", args['mass1'], args['mass2'], " are ", olaps)
    return olaps


#import args from a config file if it exists

parser = argparse.ArgumentParser()
parser.add_argument('--configfile', type=str, default=None)
parser.add_argument('--jobid', type=int, default=None)
parser.add_argument('--njobs', type=int, default=None)
args = parser.parse_args()
config_file = args.configfile
job_id = args.jobid
n_jobs = args.njobs


#start of user-defined params and param ranges. 

#seed for reproducibility
seed = 88102

#number of CPUS to use
n_cpus = 20

project_dir = "./configs/gaussian_test"
noise_dir = './noise/test'
#TODO: add template_bank to args file and make it compatible with BBH
template_bank = "PyCBC_98_aligned_spin"
bank_type = "pycbc"

template_bank = "./template_banks/bank_4-200.npy"
bank_type = "spiir"
template_range = 100
template_selection = "overlap"
template_selection_skewed = False

#noise_type should be either Gaussian or Real. If Gaussian, it will use the PSD saved from the noise directory.
noise_type = "Gaussian"

#number of samples to generate
n_signal_samples = 10000
n_noise_samples = 0

#n_samples = 10000
#noise_frac = 0.5
glitch_frac = 0.0

td_approximant = "SpinTaylorT4"
f_lower = 18
delta_t = 1/2048

#If possible, make waveform_length a power of 2. This reduces error between pycbc and tensorflow in the SNR calculation.
duration = 1024

seconds_before = 1
seconds_after = 1

detectors = ['H1','L1']

network_snr_threshold = 0
detector_snr_threshold = 4

#number of waveform templates to match with each waveform. These templates are taken from a distribution,
#but the first template chosen will be guaranteed to have a high overlap with the waveform.
templates_per_waveform = 1

fd_approximant = "TaylorF2"
#when we select templates to match with the waveform, we select from a distribution of templates, but have to ensure
#that the template has at least some overlap with the waveform. for BBH signals you can use a width up to 0.05-0.1,
#but for BNS signals you should use a width of 0.02 or less as they are more sensitive to chirp mass.
template_selection_width = 0.01

################################################
#---------------INTRINSIC PARAMS---------------#
################################################


#alpha value to be used for power law priors
# powerlaw_alpha = -3.0

#Prior functions to use for component masses.
#For an astrophysical BBH distribution, use Powerlaw
#For a BNS distribution, use Uniform (TODO implement Gaussian)
mass1prior = Uniform
mass2prior = Uniform
mass1_power = -2.35
mass2_power = 1

mass1_min = 1.4#1.0
mass1_max = 1.4#2.6

mass2_min = 1.4
mass2_max = 1.4

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

#d_prior = UniformSourceFrame
d_prior = UniformComovingVolume

d_min = 10.0
d_max = 100.0

#if you want to force samples to have a certain effective distance, set d_eff_scaling to True and set the prior as desired.
#This overrides the network_snr_threshold and detector_snr_threshold parameters.
#If d_eff_scaling is set to False, the d_eff prior will be ignored and the d_eff will be based on the sample's extrinsic params.

d_eff_scaling = False
d_eff_target = "L1"

#d_eff_prior = UniformSourceFrame
d_eff_prior = UniformComovingVolume

d_eff_min = 250.0
d_eff_max = 250.0

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

#if there is a config file specified, overwrite the above params with those from the config file
if config_file:
    print("loading args from a config file")
    with open(config_file) as json_file:
        config = json.load(json_file)

        seed = config['seed']
        n_signal_samples = config['n_signal_samples']
        n_noise_samples = config['n_noise_samples']
        glitch_frac = config['glitch_frac']
        project_dir = config['project_dir']
        template_bank = config['template_bank']
        bank_type = config['bank_type']
        template_range = config['template_range']
        template_selection = config['template_selection']
        template_selection_skewed = config['template_selection_skewed']
        noise_dir = config['noise_dir']
        noise_type = config['noise_type']
        templates_per_waveform = config['templates_per_waveform']
        td_approximant = config['td_approximant']
        fd_approximant = config['fd_approximant']
        f_lower = config['f_lower']
        delta_t = config['delta_t']
        duration = config['duration']
        seconds_before = config['seconds_before']
        seconds_after = config['seconds_after']
        detectors = config['detectors']
        network_snr_threshold = config['network_snr_threshold']
        detector_snr_threshold = config['detector_snr_threshold']
        # powerlaw_alpha = config['powerlaw_alpha']
        mass1prior = eval(config['mass1prior'])
        mass2prior = eval(config['mass2prior'])
        mass1_power = config['mass1_power']
        mass2_power = config['mass2_power']
        mass1_min = config['mass1_min']
        mass1_max = config['mass1_max']
        mass2_min = config['mass2_min']
        mass2_max = config['mass2_max']
        if "chirp_mass_prior" in config:
            chirp_mass_prior = eval(config['chirp_mass_prior'])
            chirp_mass_power = config['chirp_mass_power'] 
            chirp_mass_min = config['chirp_mass_min']
            chirp_mass_max = config['chirp_mass_max']
            mass_ratio_prior = eval(config['mass_ratio_prior'])
            mass_ratio_power = config['mass_ratio_power']
        else:
            chirp_mass_prior = None
        if "SNR_prior" in config:
            SNR_prior = eval(config['SNR_prior'])
            SNR_power = config['SNR_power']
            SNR_min = config['SNR_min']
            SNR_max = config['SNR_max']
        else:
            SNR_prior = None
        spin1zprior = eval(config['spin1zprior'])
        spin2zprior = eval(config['spin2zprior'])
        spin1z_min = config['spin1z_min']
        spin1z_max = config['spin1z_max']
        spin2z_min = config['spin2z_min']
        spin2z_max = config['spin2z_max']
        ra_prior = eval(config['ra_prior'])
        dec_prior = eval(config['dec_prior'])
        ra_min = config['ra_min']
        ra_max = config['ra_max']
        dec_min = config['dec_min']
        dec_max = config['dec_max']
        d_prior = eval(config['d_prior'])
        d_min = config['d_min']
        d_max = config['d_max']
        d_eff_scaling = config['d_eff_scaling']
        d_eff_target = config['d_eff_target']
        d_eff_min = config['d_eff_min']
        d_eff_max = config['d_eff_max']
        inc_prior = eval(config['inc_prior'])
        inc_min = config['inc_min']
        inc_max = config['inc_max']
        pol_prior = eval(config['pol_prior'])
        pol_min = config['pol_min']
        pol_max = config['pol_max']

        if "spin_type" in config:
            spin_type = config['spin_type']
        else:
            spin_type = "Aligned"
            print("Set spin_type in your params file! defaulting to Aligned")

        if "eccentricity_prior" in config:
            eccentricity_prior = eval(config['eccentricity_prior'])
            eccentricity_power = config['eccentricity_power']
            eccentricity_min = config['eccentricity_min']
            eccentricity_max = config['eccentricity_max']
        else:
            eccentricity_prior = None

        if bank_type == "pycbc_smart_match":
            metricParam_source = config['metricParam_source']
            if "smart_match_limit" in config:
                smart_match_limit = config['smart_match_limit']
            else:
                smart_match_limit = 100
            print("using smart match limit: ", smart_match_limit)
        
        if bank_type == "pycbc":
            if "pycbc_match_limit" in config:
                pycbc_match_limit = config['pycbc_match_limit']
            else:
                pycbc_match_limit = 100
            print("using pycbc match limit: ", pycbc_match_limit)

        if "template_dir" in config:
            template_dir = config['template_dir']
        else:
            template_dir = "template_banks"

        if "noise_segments" in config:
            #NOTE: this way of loading noise segments is preferred, as it does not require that
            #all noise and associated glitch files are saved in the same directory.
            #However, using this requires that you are running this code on the OzStar cluster.
            #This will eventually work elsewhere but for now it is OzStar only.
            noise_segments = config['noise_segments']
        else:
            noise_segments = None

        if "template_mass1_min" in config:
            template_mass1_min = config['template_mass1_min']
            template_mass1_max = config['template_mass1_max']
            template_mass2_min = config['template_mass2_min']
            template_mass2_max = config['template_mass2_max']
            print("using template mass constraints", template_mass1_min, template_mass1_max, 
                  template_mass2_min, template_mass2_max)

        if not os.path.exists(project_dir):
            os.makedirs(project_dir, exist_ok=True)
        with open(os.path.join(project_dir, "args.json"), 'w') as f:
            #save a copy of the args to the project directory
            json.dump(config, f, sort_keys=False, indent=4)
    for key, value in config.items():
        print(key, value)

if template_selection not in ["overlap", "snr", "random", "best"]:
    print("Invalid `template_selection` method. Please choose either 'overlap' or 'snr'.")
    exit()

n_signal_samples_total = n_signal_samples
n_noise_samples_total = n_noise_samples

if job_id is not None:
    print("job id: ", job_id, "of ", n_jobs)
    #need to divide the number of signal and noise samples by the number of jobs
    if job_id == n_jobs-1:
        n_signal_samples = int(n_signal_samples / n_jobs) +  n_signal_samples % n_jobs
        n_noise_samples = int(n_noise_samples / n_jobs) +  n_noise_samples % n_jobs
    else:
        n_signal_samples = int(n_signal_samples / n_jobs)
        n_noise_samples = int(n_noise_samples / n_jobs)
    print("n_signal_samples: ", n_signal_samples)
    print("n_noise_samples: ", n_noise_samples)
else:
    print("running as standalone job")
    job_id = 0

waveforms_per_batch = max(min(n_signal_samples//5, 1000), 100)

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

import h5py
from pycbc.tmpltbank.coord_utils import get_cov_params
def load_pycbc_templates_from_hdf(hdf_file):
    f = h5py.File(hdf_file, 'r')
    templates = np.zeros((len(f['mass1']),6))
    templates[:,1] = f['mass1'][()]
    templates[:,2] = f['mass2'][()]
    templates[:,3] = f['spin1z'][()]
    templates[:,4] = f['spin2z'][()]
    templates[:,5] = f['f_lower'][()]
    templates[:,0] = chirp_mass(templates[:,1], templates[:,2])

    #sort by chirp mass
    templates = templates[templates[:,0].argsort()]
    print("Number of templates: ", len(templates))
    #for i in range(len(f['mass1'])):
    #	templates.append(f_at_t(f['mass1'][i], f['mass2'][i], f['f_low'][i], f['f_high'][i], f['duration'][i], f['delta_t'][i]))
    return templates



#loading a bank of pre-generated templates. TODO: handle multiple ways of selecting templates.
#For BNS templates, PyCBC's geom_aligned_spin is a good choice as it produces transformation matrices for template selection,
#but requires the TaylorF2 metric which isn't accurate for BBH. 

if bank_type == "pycbc":
    print("Note: this bank matching method only works with BNS templates.")
    template_bank_params, metricParams, aXis = load_pycbc_templates(template_bank, template_dir=template_dir)
    np.save(project_dir+"/template_params.npy",template_bank_params)
    print("Number of templates: ", len(template_bank_params))	
elif bank_type == "spiir":
    template_bank_params = np.load(template_bank)
    np.save(project_dir+"/template_params.npy", arr=template_bank_params)
    print("Number of templates: ", len(template_bank_params))	

elif bank_type == "pycbc_smart_match":
    #if template bank is an hdf file, load the templates from the hdf file
    #and generate the metricParams and aXis from the metricParam_source file
    #else load the templates from the txt file (same as the pycbc option)

    if template_bank.endswith(".hdf"):
        print("loading templates from hdf file")
        _, metricParams, _ = load_pycbc_templates(metricParam_source, template_dir=template_dir)
        template_bank_params = load_pycbc_templates_from_hdf(template_bank)
        aXis = get_cov_params(template_bank_params[:,1],template_bank_params[:,2],template_bank_params[:,3],template_bank_params[:,4],metricParams,metricParams.fUpper)
    else:
        print("loading templates from txt file")
        template_bank_params, metricParams, aXis = load_pycbc_templates(template_bank, template_dir=template_dir)
    #TODO: eventually switch distance to redshift, then we can work with detector-frame masses and automatically trim templates
    #nsbh = ((template_bank_params[:,1] > 3) & (template_bank_params[:,2] < 3.75) & (template_bank_params[:,1] < 100))
    
    #the 3.125 is chosen as max m2 as 2.5 * 1.25 is the maximum m2 detector frame mass in GWTC-3
    #aXis = aXis[:,nsbh]
    #template_bank_params = template_bank_params[nsbh]    
    #for matching, waveform is generated in the time domain with SEOBNRv4_PHM, then converted to frequency domain
    #templates are always generated in frequency domain with SEOBNRV4_ROM
    if template_mass1_min is not None:
        cut = ((template_bank_params[:,1] > template_mass1_min) & (template_bank_params[:,1] < template_mass1_max) &
                (template_bank_params[:,2] > template_mass2_min) & (template_bank_params[:,2] < template_mass2_max))
    else:
        cut = np.ones(len(template_bank_params), dtype=bool)
    aXis = aXis[:,cut]
    template_bank_params = template_bank_params[cut]
    print("Number of templates after cut: ", len(template_bank_params))
    np.save(project_dir+"/template_params.npy", arr=template_bank_params)

else:
    print(f"Invalid template bank type: {bank_type}")
    print("Program exiting...")
    exit()


global psds


#set a seed to ensure reproducibility
np.random.seed(seed)



if chirp_mass_prior is not None:
    prior = PriorDict(conversion_function=sample_masses_from_cm_q)
else:
    prior = PriorDict()

# IF YOU ARE USING DIFFERENT PRIOR DISTRIBUTIONS FOR M1 AND M2, MAKE SURE YOU KNOW WHAT YOU ARE DOING.
# CURRENTLY, M1 AND M2 ARE SWAPPED IF M2 IS SAMPLED ABOVE M1, WHICH ONLY WORKS WHEN THEY USE THE SAME
# DISTRIBUTION. IF USING DIFFERENT DISTRIBUTIONS (EG. POWER LAW), BE CAREFUL AND TEST THIS CODE FIRST.

if chirp_mass_prior is not None:
    print("sampling m1 and m2 using chirp mass prior")
    if chirp_mass_prior == TriUniform:
        prior['chirp_mass'] = constructPrior(chirp_mass_prior, chirp_mass(mass1_min, mass2_min), 
                                            chirp_mass(mass1_max, mass2_max), 
                                            mode = chirp_mass(mass1_min, mass2_min),
                                            r = chirp_mass_power)
    else:
        prior['chirp_mass'] = constructPrior(chirp_mass_prior, chirp_mass(mass1_min, mass2_min), 
                                            chirp_mass(mass1_max, mass2_max), 
                                            alpha = chirp_mass_power)
    prior['mass_ratio'] = constructPrior(mass_ratio_prior, mass2_min/mass1_max, min(mass2_max/mass1_min,1), alpha = mass_ratio_power)
    prior['mass1_source'] = bilby.core.prior.Constraint(minimum=mass1_min, maximum=mass1_max, name='mass1_source')
    prior['mass2_source'] = bilby.core.prior.Constraint(minimum=mass2_min, maximum=mass2_max, name='mass2_source')

else:

    if mass1prior != PowerLaw:
        #Primarily used for BNS
        prior['mass1_source'] = constructPrior(mass1prior, mass1_min, mass1_max)
        prior['mass2_source'] = constructPrior(mass2prior, mass2_min, mass2_max)
    elif mass1prior == PowerLaw and mass2prior != PowerLaw:
        #Primarily used for NSBH
        print("mass1 is a power law, mass2 is not")
        prior['mass1_source'] = constructPrior(mass1prior, mass1_min, mass1_max, alpha = mass1_power)
        prior['mass2_source'] = constructPrior(mass2prior, mass2_min, mass2_max)

prior['spin1z'] = constructPrior(spin1zprior, spin1z_min, spin1z_max)
prior['spin2z'] = constructPrior(spin2zprior, spin2z_min, spin2z_max)

prior['ra'] = constructPrior(Uniform, ra_min * 2 * np.pi, ra_max * 2 * np.pi, boundary = 'periodic')
prior['dec'] = constructPrior(dec_prior, np.pi * ra_min - np.pi/2, np.pi * ra_max - np.pi/2)

if d_prior == Triangular:
    print("triangular distance prior")
    prior['d'] = constructPrior(d_prior, d_min, d_max, mode = d_max)
else:
    prior['d'] = constructPrior(d_prior, d_min, d_max, name = 'luminosity_distance')

prior['i'] = constructPrior(inc_prior, inc_min * np.pi, inc_max * np.pi)
prior['pol'] = constructPrior(pol_prior, pol_min * np.pi *2, pol_max * np.pi *2)

if eccentricity_prior is not None:
    prior['eccentricity'] = constructPrior(eccentricity_prior, eccentricity_min, eccentricity_max, alpha = eccentricity_power)

if d_eff_scaling:
    prior['d_eff'] = constructPrior(d_eff_prior, d_eff_min, d_eff_max, name = 'luminosity_distance')

if SNR_prior is not None:
    prior['network_snr'] = constructPrior(SNR_prior, SNR_min, SNR_max, alpha = SNR_power)

from GWSamplegen.noise_utils import get_valid_noise_times_from_segments #, get_data_from_OzStar
if noise_segments is None:
    valid_times, _, _ = get_valid_noise_times(noise_dir,duration)
else:
    print("Noise GPS times are now:",noise_segments)
    valid_times = get_valid_noise_times_from_segments(noise_segments,duration)

print(len(valid_times), "GPS times available")

#load PSD 


psds = load_psd(noise_dir, duration, detectors, f_lower, int(1/delta_t))

all_detectors = {'H1': Detector('H1'), 'L1': Detector('L1'), 'V1': Detector('V1'), 'K1': Detector('K1')}

good_params = []

generated_samples = 0
iteration = 0

wavetime = 0

# get correct PyCBC waveform
if bank_type == "spiir":
    approx_name, approx_phase_order = legacy_approximant_name(td_approximant)

#get longest waveform in template bank. As the templates are sorted by chirp mass, this will be the first template.
#hp, _ = get_td_waveform(mass1=template_bank_params[0,1], mass2=template_bank_params[0,2], 
#						delta_t=delta_t, f_lower=f_lower, approximant=approx_name, phase_order=approx_phase_order)
#TODO: maybe replace with the t_at_f function 
#from GWSamplegen.waveform_utils import t_at_f

#max_waveform_length = len(hp) * delta_t + 1 #adding a safety factor of 1 second
#max_waveform_length = max(32, int(np.ceil(max_waveform_length/10)*10)) #rounding up to the nearest 10 seconds / setting to 12 for BBHs
#NOTE: for now, setting to max BNS waveform length.
max_waveform_length = 100
print("max waveform length: ", max_waveform_length)
if duration < 2*max_waveform_length:
    print(f"Desired duration of {duration} seconds for generating samples is less than two times the max_waveform_length of {max_waveform_length} seconds of your injection parameter space. This will mean that samples positioned at the middle of your sample will encounter issues due to filter wrap around in matched filtering.")
    print("Please fix the duration input parameter.")
    exit()

#TODO: add glitch SNR threshold as a config parameter
SNR_thresh = 7

if noise_type == "Real":

    glitchless_times = {}
    glitchy_times = {}
    glitchy_freqs = {}
    glitchy_SNRs = {}

    for ifo in detectors:
        glitchy_times[ifo] = []
        glitchless_times[ifo] = []
        glitchy_freqs[ifo] = []

    for ifo in detectors:
        if noise_segments is None:
            glitchy, glitchless, freq, glitch_snr = get_glitchy_times(noise_dir+"/{}_glitches.npy".format(ifo),
                                            duration, valid_times, max_waveform_length, SNR_thresh, f_lower, seconds_before, seconds_after)
        else:
            glitchy, glitchless, freq, glitch_snr = get_glitchy_times("/fred/oz016/alistair/Omicron_all/{}_glitches.npy".format(ifo),
                                            duration, valid_times, max_waveform_length, SNR_thresh, f_lower, seconds_before, seconds_after)
        glitchless_times[ifo] = glitchless
        glitchy_times[ifo] = glitchy
        glitchy_freqs[ifo] = freq
        glitchy_SNRs[ifo] = glitch_snr


    #create timeslide generators
    min_separation = 3

    glitchless_generator = two_det_timeslide([glitchless_times[ifo] for ifo in detectors], min_separation)
    one_glitch_generator = {}

    for ifo in detectors:
        time_list = [glitchy_times[i] if i == ifo else glitchless_times[i] for i in detectors]
        one_glitch_generator[ifo] = two_det_timeslide(time_list, min_separation)
    
    if len(detectors) == 2:
        two_glitch_generator = two_det_timeslide([glitchy_times[ifo] for ifo in detectors], min_separation)

if bank_type == "spiir":
    # organise tasks for getting template waveforms
    print(f"Shape of template_bank_params: {np.shape(template_bank_params)}")
    print(f"Generating tasks array required to load template waveforms.")
    template_tasks = []
    for i in range(len(template_bank_params)):
        template_tasks.append([template_bank_params[i, :], delta_t, f_lower, duration])

    print("Starting generation of template waveforms")
    # load template bank waveforms to memory
    with mp.Pool(processes=n_cpus) as pool:
        template_waveforms = pool.map(get_template, template_tasks)
        pool.close()
        pool.join()
    print("Successfully finished generation of template waveforms")


previous_good_params_length = 0
while generated_samples < n_signal_samples:

    #generate waveforms_per_file samples at a time, to avoid memory issues.

    p = prior.sample(waveforms_per_batch)
    if chirp_mass_prior is not None:
        p['mass1_source'], p['mass2_source'] = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses(p['chirp_mass'], p['mass_ratio'])
    else:
        
        if mass1prior == PowerLaw and mass2prior == PowerLaw:
            m1 = []
            m2 = []
            for i in range(waveforms_per_batch):
                pair = next(iter(draw_mass_pair_power(mass1_min, mass1_max, mass2_min, mass2_max, mass1_power, mass2_power)))
                m1.append(pair[0])
                m2.append(pair[1])
            p['mass1_source'] = m1
            p['mass2_source'] = m2

    cosmol = FlatwCDM(H0=67.9, Om0=0.3065, w0=-1)
    m1_df = []
    m2_df = []
    z = []
    for key in range(len(p['mass1_source'])):
        z.append(cosmo.z_at_value(cosmol.luminosity_distance, u.Quantity(p['d'][key], u.Mpc)))
        m1_df.append(p['mass1_source'][key] * (1 + z[key]))
        m2_df.append(p['mass2_source'][key] * (1 + z[key]))
    p['z'] = z
    p['mass1'] = m1_df
    p['mass2'] = m2_df

    if spin_type == "Aligned":
        p['spin1x'] = np.zeros(waveforms_per_batch)
        p['spin1y'] = np.zeros(waveforms_per_batch)
        p['spin2x'] = np.zeros(waveforms_per_batch)
        p['spin2y'] = np.zeros(waveforms_per_batch)
    
    elif spin_type == "Isotropic":
        #TODO: rename spin1z and spin2z to spin1 and spin2
        p['spin1x'], p['spin1y'], p['spin1z'] = zip(*[draw_spin_isotropic(spin1z_max) for _ in range(waveforms_per_batch)])
        p['spin2x'], p['spin2y'], p['spin2z'] = zip(*[draw_spin_isotropic(spin2z_max) for _ in range(waveforms_per_batch)]) 
        p['spin1x'] = np.array(p['spin1x'])
        p['spin1y'] = np.array(p['spin1y'])
        p['spin1z'] = np.array(p['spin1z'])
        p['spin2x'] = np.array(p['spin2x'])
        p['spin2y'] = np.array(p['spin2y'])
        p['spin2z'] = np.array(p['spin2z'])
    else:
        raise ValueError("Invalid spin type. Please choose either 'Aligned' or 'Isotropic'.")

    #adding non-sampled args to the parameters
    p['gps'] = []

    #if we're using real noise, deal with glitches and timeslides
    
    for detector in detectors:
        p[detector + '_glitch'] = np.zeros(waveforms_per_batch, dtype = bool)

    if noise_type == "Real":
        for i in range(waveforms_per_batch):

            #TODO: generalise to 3+ detectors. For now, works fine for one or two detectors.
            if isinstance(glitch_frac, list):
                for j in range(len(glitch_frac)):
                    if np.random.uniform(0,1) < glitch_frac[j]:
                        p[detectors[j] + '_glitch'][i] = True
                    
                if np.all([p[det + "_glitch"][i] for det in detectors]):
                    #2 detector glitches
                    glitch_time = list(next(two_glitch_generator))
                    for j in range(len(detectors)):
                        glitch_idx = np.where(glitch_time[j] == glitchy_times[detectors[j]])[0][0]
                        glitch_time[j] = get_glitchy_gps_time(valid_times, p['mass1'][i], p['mass2'][i], 
                                                        glitch_time[j], glitchy_freqs[detectors[j]][glitch_idx],
                                                        glitchy_SNRs[detectors[j]][glitch_idx])
                
                elif np.any([p[det + "_glitch"][i] for det in detectors]):
                    #1 detector glitch
                    glitchy_ifo = np.where([p[det + "_glitch"][i] for det in detectors])[0][0]
                    glitch_time = list(next(one_glitch_generator[detectors[glitchy_ifo]]))
                    glitch_idx = np.where(glitch_time[glitchy_ifo] == glitchy_times[detectors[glitchy_ifo]])[0][0]
                    glitch_time[glitchy_ifo] = get_glitchy_gps_time(valid_times, p['mass1'][i], p['mass2'][i],
                                                                    glitch_time[glitchy_ifo], glitchy_freqs[detectors[glitchy_ifo]][glitch_idx],
                                                                    glitchy_SNRs[detectors[glitchy_ifo]][glitch_idx])
                
                else:
                    #no glitches
                    glitch_time = list(next(glitchless_generator))

                p['gps'].append(glitch_time)

            else:
                #TODO: this is legacy code, and should be removed once it won't cause issues
                #This code assumes the glitch fraction is a float, 
                #and that each detector has an equal chance of having a glitch.
                #glitches are now a list of floats, one for each ifo.
                if np.random.uniform(0,1) < glitch_frac:
                    #if the glitch fraction is a float, each detector has an equal chance of having the glitch
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
    
    #otherwise, just choose some random GPS times from valid_times
    #TODO: adapt code to work with Gaussian noise without requiring a directory of real noise
    else:
        p['gps'] = np.random.choice(valid_times, size = (waveforms_per_batch, len(detectors)))

    p['injection'] = np.ones(waveforms_per_batch, dtype = bool)

    if d_eff_scaling:
        #adjust distance to reach the desired effective distance
        d_effs = Detector(d_eff_target).effective_distance(p['d'], p['ra'], p['dec'], p['pol'], np.array(p['gps'])[:,0], p['i'])
        p['d'] = p['d']/(d_effs / p['d_eff'])

    else:
        #TODO: add effective distance for each detector
        p['d_eff'] = np.zeros(waveforms_per_batch)

    if eccentricity_prior is None:
        p['eccentricity'] = np.zeros(waveforms_per_batch)

    
    #turn dict of lists into a list of dicts (for multiprocessing)
    params = [{key: p[key][i] for key in p.keys()} for i in range(len(p['mass1']))]

    #get the SNRs of the samples
    start = time.time()
    with mp.Pool(processes=n_cpus) as pool:

        #snrs is a list of dicts, where each dict is {detector: snr}
        snrs = pool.map(get_snr, params, chunksize = 1)
        pool.close()
        pool.join()
        #mp_waveforms is a list of lists, where each list is [waveform, snrs]
        
    wavetime += time.time() - start
    
    #save only the waveforms with network SNR and detector SNRs above threshold.

    template_tasks = []
    for i in range(len(snrs)):

        network_snr = np.sqrt(sum([snrs[i][detector]**2 for detector in snrs[i]]))

        if SNR_prior is not None:
            #we simply rescale the network SNR to match the target SNR

            network_SNR_scale = params[i]['network_snr'] / network_snr 
            network_snr *= network_SNR_scale
            #params[i]['network_snr'] = network_snr 
            for detector in detectors:
                snrs[i][detector] *= network_SNR_scale
            #[snrs[i][detector] for detector in snrs[i]] = [snr * network_SNR_scale for snr in snrs[i].values()]
            #scale the distance of the sample, as this is how we change the SNR
            params[i]['d'] = params[i]['d'] / network_SNR_scale

            #if np.min([snrs[i][detector] for detector in snrs[i]]) < detector_snr_threshold * network_SNR_scale:
            #    #if after scaling the network SNR one or both of the detector SNRs are below the threshold,
            #    #we instead scale by the minimum of the two detector SNRs
            #    network_SNR_scale = np.min([snrs[i][detector] for detector in snrs[i]]) / params[i]['network_snr']

        if (network_snr > network_snr_threshold and all([snr > detector_snr_threshold for snr in snrs[i].values()])) or d_eff_scaling:
            #this sample is suitable, get it ready for saving

            #add the detector SNRs and network SNR as keys in params[i]
            params[i]['network_snr'] = network_snr
            for detector in detectors:
                params[i][detector + '_snr'] = snrs[i][detector]

            #ensure that mass2 < mass1
            if params[i]['mass2_source'] > params[i]['mass1_source']:
                params[i]['mass1_source'], params[i]['mass2_source'] = params[i]['mass2_source'], params[i]['mass1_source']
                params[i]['mass1'], params[i]['mass2'] = params[i]['mass2'], params[i]['mass1']

            #choose template waveform(s) for this sample, and add them to params[i]
            if bank_type == "pycbc":
                #params[i]['template_waveforms'] = choose_templates(template_bank_params, params[i], 
                #                                                   templates_per_waveform, template_selection_width)
                params[i]['template_waveforms'] = choose_templates_new(template_bank_params, metricParams, 
                                                                       templates_per_waveform, params[i]['mass1'], params[i]['mass2'], 
                                                                       params[i]['spin1z'], params[i]['spin2z'], limit = pycbc_match_limit, aXis = aXis)
            #elif bank_type == "pycbc_smart_match":
            #    print("bing")

            elif bank_type == "spiir":
                params[i]['template_waveforms'] = []
                cm = chirp_mass(params[i]['mass1'], params[i]['mass2'])
                inj_args = {
                    "mchirp": cm, "mass1": params[i]["mass1"], "mass2": params[i]["mass2"],
                    "spin1z": params[i]["spin1z"], "spin2z": params[i]["spin2z"],
                    "i": params[i]["i"], "ra": params[i]["ra"], "dec": params[i]["dec"],
                    "pol": params[i]["pol"], "approx": approx_name, "d": 100,
                    "gps": params[i]["gps"], "f_low": f_lower, "delta_t": delta_t,
                    "duration": duration, "phase_order": approx_phase_order
                }
                arg_closest = (np.abs(template_bank_params[:,0] - cm)).argmin()
                minimum = max(arg_closest - template_range, 0)
                maximum = min(arg_closest + template_range, len(template_bank_params[:,0]))
                
                template_tasks.append([inj_args, 1, minimum, maximum, i])
                
            good_params.append(params[i])
    
    # Sample templates for non-pycbc bank (ie. using match)
    if bank_type == "spiir":
        
        # get overlap of bank subsets with each injection
        print("Running template selection multiprocessing to sample templates for each injection")
        t_task = time.time()
        with mp.Pool(processes=n_cpus) as pool:
            overlaps = pool.map(get_match, template_tasks)
            pool.close()
            pool.join()
        print(f"Time for template selection multiprocessing of all tasks: {time.time() - t_task} seconds")
        
        all_network_snrs = [i['network_snr'] for i in good_params]
        templates, chosen_overlaps = choose_templates_match(overlaps, templates_per_waveform, all_network_snrs)
        
        # I want functionality to do the following eventually:
        # 1. Sample templates that return any level of overlap from the set of templates
        # 2. Sample templates such that they all have a net network SNR > 6 or some threshold based on overlap * optimal network_snr
        # 3. Sample templates with any overlap, but use that to label ones with net network snr < 3? as noise instead of injection samples
        
        for i in range(len(templates)):
            good_params[i+previous_good_params_length]['template_waveforms'] = templates[i]
            good_params[i+previous_good_params_length]['overlaps'] = chosen_overlaps[i]

    elif bank_type == "pycbc_smart_match":
        print("running smart match")
        t_task = time.time()
        print(len(good_params[previous_good_params_length: ]), "tasks to run")
        t_args = {"aXis": aXis, "metricParams": metricParams, "template_bank_params":template_bank_params,
                   "td_approximant": td_approximant, "fd_approximant": fd_approximant,
                  "f_lower": f_lower, "psds": psds , "duration": duration}
        with mp.Pool(processes=n_cpus) as pool:
            olaps = pool.starmap(find_templates, [(good_params[previous_good_params_length: ][n], t_args, smart_match_limit, 0.95, templates_per_waveform, 20) for n in range(len(good_params[previous_good_params_length: ]))], chunksize = 1)
            pool.close()
            pool.join()
        print(f"Time for template selection multiprocessing of all tasks: {time.time() - t_task} seconds")
        olaps = np.array(olaps)
        for i in range(len(olaps)):
            print(olaps[i])
            good_params[i+previous_good_params_length]['overlaps'] = olaps[i, 0]
            good_params[i+previous_good_params_length]['template_waveforms'] = olaps[i, 1]
            
    elif bank_type == "pycbc":
        #we still need to get the proper overlaps for the chosen templates
        print("getting overlaps for chosen templates")
        t_task = time.time()
        with mp.Pool(processes=n_cpus) as pool:
            overlaps = pool.map(get_overlaps, [good_params[i+previous_good_params_length] for i in range(len(good_params[previous_good_params_length:]))], chunksize = 1)
            pool.close()
            pool.join()
        print(f"Time for overlap calculation multiprocessing of all tasks: {time.time() - t_task} seconds")
        for i in range(len(overlaps)):
            good_params[i+previous_good_params_length]['overlaps'] = overlaps[i]


    generated_samples = len(good_params)
    if generated_samples <= waveforms_per_batch:
        if generated_samples/waveforms_per_batch < 0.5:
            print(f"WARNING: check your distance prior and SNR threshold! Only {generated_samples/waveforms_per_batch*100:.2f}% of the samples meet the SNR threshold.")
        else:
            print("SNR threshold looks good, {}% of samples meet the threshold.".format(round(generated_samples/waveforms_per_batch*100)))

    print(f"Number of good injections so far: {len(good_params)}")
    print(f"Number of good injections in this batch: {len(good_params) - previous_good_params_length}")
    print(f"Number of bad injections in this batch: {waveforms_per_batch - (len(good_params) - previous_good_params_length)}")

    previous_good_params_length = len(good_params)


print('done samples with injections')

if not d_eff_scaling and np.max([i['network_snr'] for i in good_params]) < 500:
    #rescale the distance to ensure at least one sample has a network SNR of 500
    # this is to ensure that the distance prior is well sampled.
    # TODO: will need a flag for this in the future, as it's not always necessary.

    print("Rescaling a sample to ensure at least one sample has a network SNR of 500")
    max_snr_idx = np.argmax([i['network_snr'] for i in good_params])
    max_snr = good_params[max_snr_idx]['network_snr']
    max_snr_distance = good_params[max_snr_idx]['d']
    
    rescale = 500/max_snr

    rescale *= np.random.uniform(1, 1.2)# + np.random.uniform(0, 0.2)

    good_params[max_snr_idx]['d'] = good_params[max_snr_idx]['d']/rescale
    good_params[max_snr_idx]['H1_snr'] = good_params[max_snr_idx]['H1_snr']*rescale
    good_params[max_snr_idx]['L1_snr'] = good_params[max_snr_idx]['L1_snr']*rescale
    good_params[max_snr_idx]['network_snr'] = good_params[max_snr_idx]['network_snr']*rescale

#save the injection parameters to a file
#convert from a list of dictionaries to a dictionary of lists
if n_signal_samples > 0:
    good_params_dict = {key: np.array([good_params[i][key] for i in range(len(good_params))][:n_signal_samples]) for key in good_params[0].keys()}

np.save(project_dir+"/"+"params_{}.npy".format(job_id), good_params_dict)


#generate noise samples. most of the parameters aren't used, but the masses are used to choose the templates.

if n_noise_samples > 0:
    noise_p = prior.sample(n_noise_samples)

    if chirp_mass_prior is not None:
        noise_p['mass1_source'], noise_p['mass2_source'] = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses(noise_p['chirp_mass'], noise_p['mass_ratio'])
    else:
        if mass1prior == PowerLaw and mass2prior == PowerLaw:
            m1 = []
            m2 = []
            for i in range(n_noise_samples):
                pair = next(iter(draw_mass_pair_power(mass1_min, mass1_max, mass2_min, mass2_max, mass1_power, mass2_power)))
                m1.append(pair[0])
                m2.append(pair[1])
            noise_p['mass1_source'] = m1
            noise_p['mass2_source'] = m2

    cosmol = FlatwCDM(H0=67.9, Om0=0.3065, w0=-1)
    m1_df = []
    m2_df = []
    z = []
    for key in range(len(noise_p['mass1_source'])):
        z.append(cosmo.z_at_value(cosmol.luminosity_distance, u.Quantity(noise_p['d'][key], u.Mpc)))
        m1_df.append(noise_p['mass1_source'][key] * (1 + z[key]))
        m2_df.append(noise_p['mass2_source'][key] * (1 + z[key]))
    noise_p['z'] = z
    noise_p['mass1'] = m1_df
    noise_p['mass2'] = m2_df

    if spin_type == "Aligned":
        noise_p['spin1x'] = np.zeros(waveforms_per_batch)
        noise_p['spin1y'] = np.zeros(waveforms_per_batch)
        noise_p['spin2x'] = np.zeros(waveforms_per_batch)
        noise_p['spin2y'] = np.zeros(waveforms_per_batch)
    
    elif spin_type == "Isotropic":
        #TODO: rename spin1z and spin2z to spin1 and spin2
        noise_p['spin1x'], noise_p['spin1y'], noise_p['spin1z'] = zip(*[draw_spin_isotropic(spin1z_max) for _ in range(n_noise_samples)])
        noise_p['spin2x'], noise_p['spin2y'], noise_p['spin2z'] = zip(*[draw_spin_isotropic(spin2z_max) for _ in range(n_noise_samples)]) 
        noise_p['spin1x'] = np.array(noise_p['spin1x'])
        noise_p['spin1y'] = np.array(noise_p['spin1y'])
        noise_p['spin1z'] = np.array(noise_p['spin1z'])
        noise_p['spin2x'] = np.array(noise_p['spin2x'])
        noise_p['spin2y'] = np.array(noise_p['spin2y'])
        noise_p['spin2z'] = np.array(noise_p['spin2z'])
    else:
        raise ValueError("Invalid spin type. Please choose either 'Aligned' or 'Isotropic'.")

    noise_p['gps'] = []
    noise_p['overlaps'] = np.zeros(shape=(n_signal_samples, templates_per_waveform))  # This is dummy data so it saves correctly
    noise_p['injection'] = np.zeros(n_noise_samples, dtype = bool)
    noise_p['template_waveforms'] = np.random.randint(0, len(template_bank_params), size=(n_noise_samples,templates_per_waveform))
    templates = []
    noise_p['d_eff'] = np.zeros(n_noise_samples)
    noise_p['eccentricity'] = np.zeros(n_noise_samples)

    #TODO: get the actual max SNR for the noise segment maybe?
    noise_p['network_snr'] = np.zeros(n_noise_samples)
    for detector in detectors:
        noise_p[detector + '_snr'] = np.zeros(n_noise_samples)
        noise_p[detector + '_glitch'] = np.zeros(n_noise_samples, dtype = bool)

    for i in range(n_noise_samples):

        if noise_p['mass2_source'][i] > noise_p['mass1_source'][i]:
            noise_p['mass1_source'][i], noise_p['mass2_source'][i] = noise_p['mass2_source'][i], noise_p['mass1_source'][i]
            noise_p['mass1'][i], noise_p['mass2'][i] = noise_p['mass2'][i], noise_p['mass1'][i]


        if noise_type == "Real":

            if isinstance(glitch_frac, list):
                for j in range(len(glitch_frac)):
                    if np.random.uniform(0,1) < glitch_frac[j]:
                        noise_p[detectors[j] + '_glitch'][i] = True
                    
                if np.all([noise_p[det + "_glitch"][i] for det in detectors]):
                    #2 detector glitches
                    glitch_time = list(next(two_glitch_generator))
                    for j in range(len(detectors)):
                        glitch_idx = np.where(glitch_time[j] == glitchy_times[detectors[j]])[0][0]
                        glitch_time[j] = get_glitchy_gps_time(valid_times, noise_p['mass1'][i], noise_p['mass2'][i], 
                                                        glitch_time[j], glitchy_freqs[detectors[j]][glitch_idx],
                                                        glitchy_SNRs[detectors[j]][glitch_idx])
                
                elif np.any([noise_p[det + "_glitch"][i] for det in detectors]):
                    #1 detector glitch
                    glitchy_ifo = np.where([noise_p[det + "_glitch"][i] for det in detectors])[0][0]
                    glitch_time = list(next(one_glitch_generator[detectors[glitchy_ifo]]))
                    glitch_idx = np.where(glitch_time[glitchy_ifo] == glitchy_times[detectors[glitchy_ifo]])[0][0]
                    glitch_time[glitchy_ifo] = get_glitchy_gps_time(valid_times, noise_p['mass1'][i], noise_p['mass2'][i],
                                                                    glitch_time[glitchy_ifo], glitchy_freqs[detectors[glitchy_ifo]][glitch_idx],
                                                                    glitchy_SNRs[detectors[glitchy_ifo]][glitch_idx])
                
                else:
                    #no glitches
                    glitch_time = list(next(glitchless_generator))

                noise_p['gps'].append(glitch_time)
            
            else:
                #TODO: this is also legacy code. remove.
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
        
        else:
            noise_p['gps'] = np.random.choice(valid_times, size = (n_noise_samples, len(detectors)))

        #params = {key: noise_p[key][i] for key in noise_p.keys()}
        #templates.append(choose_templates(template_bank_params, params, templates_per_waveform, template_selection_width))

    #noise_p['template_waveforms'] = np.array(templates)

    if n_signal_samples > 0:
        print(good_params_dict.keys())
        for key in good_params_dict.keys():
            good_params_dict[key] = np.append(good_params_dict[key], noise_p[key], axis = 0)
    else:
        good_params_dict = noise_p

#np.save(project_dir+"/"+"noise_params.npy", noise_p)
np.save(project_dir+"/"+"params_{}.npy".format(job_id), good_params_dict)

print("finished generating waveforms. time taken: " + str(wavetime/60) + " minutes")

import matplotlib.pyplot as plt

if n_signal_samples > 0:
    for detector in detectors:
        plt.hist(good_params_dict[detector+"_snr"][:n_signal_samples], bins = 100, alpha = 0.5)

    plt.xlabel("Injected SNR")
    plt.yscale('log')
    #plt.xlim(0,30)
    plt.savefig(project_dir+"/injected_SNR.png")

    sigs = np.where(good_params_dict['overlaps'][:,0] ==0)[0][0]

    fiftyp = np.sort(good_params_dict['overlaps'][:sigs,0])[len(good_params_dict['overlaps'][:sigs])//2]
    ninetyp = np.sort(good_params_dict['overlaps'][:sigs,0])[len(good_params_dict['overlaps'][:sigs])//10]
    ninetyninep = np.sort(good_params_dict['overlaps'][:sigs,0])[len(good_params_dict['overlaps'][:sigs])//100]


    f = plt.figure(figsize=(4,3), dpi=200)
    plt.hist(good_params_dict['overlaps'][:sigs,0], bins=100, alpha=0.5)
    plt.axvline(fiftyp, color='b', linestyle='dashed', linewidth=1, label = '50%')
    plt.axvline(ninetyp, color='r', linestyle='dashed', linewidth=1, label = '90%')
    plt.axvline(ninetyninep, color='g', linestyle='dashed', linewidth=1, label = '99%')
    plt.legend(loc='upper left')
    plt.xlabel('Max. overlap')
    plt.ylabel('Count')
    plt.yscale('log')
    #make sure axis labels are visible
    plt.tight_layout()
    plt.savefig(project_dir+"/overlaps.png")