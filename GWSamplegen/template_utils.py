from GWSamplegen.waveform_utils import maximum_f_lower, t_at_f, select_approximant, fast_point_distance, chirp_mass
from pycbc.waveform import get_td_waveform, get_fd_waveform, td_approximants, fd_approximants
import numpy as np
from pycbc.filter import matchedfilter
import gc
import h5py

def cfac(e):
    #a correction factor: when searching for circular templates for an eccentric waveform,
    #adjust the template mass by this factor. Tested on BNS signals.
    return e/20 + 1


def find_templates(waveform, args, futile_limit = 10, match_target = 0.9, required_templates = 10, acceptable_thresh = 10):
    #waveform is a dictionary containing the waveform parameters

    #print("f_lower: ", f_lower)
    #used to be parameters, templates, psd, futile_limit = 10, match_target = 0.9, required_templates = 10, acceptable_thresh = 10
    olaps = []

    if "eccentricity" in waveform:
        #a = (eccentric_masses_to_circular(waveform['eccentricity']) -1) /8 + 1
        a = cfac(waveform['eccentricity'])
    else: 
        a = 1

    idx = np.argsort(fast_point_distance(args["aXis"], (waveform['mass1'] * a, waveform['mass2'] * a, 
                                            waveform['spin1z'], waveform['spin2z']), args["metricParams"]))
    
    temp_td_approximant = select_approximant(waveform['mass1'], waveform['mass2'], args["td_approximant"], domain='time')

    #note: approximant for hp_f was IMRPhenomPv2
    #this waveform should be fast FD waveform that's accurate (for BBH that's IMRPhenomPv2)
    #hp_f, _ = get_fd_waveform(mass1 = waveform['mass1'], mass2 = waveform['mass2'],
    #        spin1z = waveform['spin1z'], spin2z = waveform['spin2z'],
    #        f_lower = 10, delta_f = 1/1024, f_final = 2048*4, approximant = waveform_approximant)

    #temp_f_lower = min(10, maximum_f_lower(waveform['mass1'], waveform['mass2']))
    if temp_td_approximant != "SpinTaylorT4" and t_at_f(waveform['mass1'], waveform['mass2'], 10) > 100:
        temp_f_lower = 20
    else:
        temp_f_lower = 10
    temp_f_lower = min(temp_f_lower, maximum_f_lower(waveform['mass1'], waveform['mass2']))

    if temp_td_approximant in ["TaylorF2Ecc", "EccentricFD", "EccentricTD"]:
        print("Eccentric approximant detected. Using f_lower = 20 Hz.")
        temp_f_lower = 20
    temp_delta_t = 1/(8*2048)
    if temp_td_approximant == "IMRPhenomXPHM":
        if waveform['mass1'] + waveform['mass2'] < 4:
            temp_f_lower = 25
            temp_delta_t = 1/2048
        else:
            temp_f_lower = 15
    hp = None
    if temp_td_approximant in td_approximants():
        try:
            hp,_ = get_td_waveform(approximant=temp_td_approximant, mass1=waveform['mass1'], mass2=waveform['mass2'],
                                spin1x=waveform['spin1x'], spin2x=waveform['spin2x'],
                                spin1y=waveform['spin1y'], spin2y=waveform['spin2y'],
                                spin1z=waveform['spin1z'], spin2z=waveform['spin2z'],
                                f_lower=temp_f_lower, delta_t=temp_delta_t)
        except:
            print("Failed to generate waveform for injection. Parameters:", waveform, flush=True)
            hp,_ = get_td_waveform(approximant=temp_td_approximant, mass1=waveform['mass1'], mass2=waveform['mass2'],
                    f_lower=temp_f_lower, delta_t=temp_delta_t)
        hp = hp.resample(1/2048)

        if -hp.sample_times[0] > args["duration"]:
            print("shortening sample")
            chop = np.argmin(np.abs(hp.sample_times + args["duration"]-10))
            hp.start_time = hp.sample_times[chop]
            hp.data = hp.data[chop:]
            print("New sample delta f is ", 1/hp.delta_f)

        #print("trying to convert to frequency series")
        hp_f = hp.to_frequencyseries(delta_f=1/args["duration"])

    elif args["td_approximant"] not in td_approximants() and args["td_approximant"] in fd_approximants():
        print("TD approximant is FD only. No need to convert.")
        hp_f, _ = get_fd_waveform(waveform, approximant = args["td_approximant"], f_lower = temp_f_lower, delta_f = 1/args["duration"])

    for j in range(futile_limit):
        #note: approximant for hpt_f was SEOBNRv4_ROM
        #this waveform should be the template bank waveform. Should NOT include precession etc. 
        #unless the bank was generated with it.
        temp_fd_approximant = select_approximant(args["template_bank_params"][idx[j]][1], args["template_bank_params"][idx[j]][2], args["fd_approximant"], domain='frequency')
        hpt_f, _ = get_fd_waveform(mass1 = args["template_bank_params"][idx[j]][1], mass2 = args["template_bank_params"][idx[j]][2],
                                spin1z = args["template_bank_params"][idx[j]][3], spin2z = args["template_bank_params"][idx[j]][4],
                                f_lower = temp_f_lower, delta_f = 1/1024, f_final = 2048*4, approximant = temp_fd_approximant)
        
        hpt_f.resize(len(hp_f))
        olap = matchedfilter.match(hp_f, hpt_f, psd=args["psds"]['L1'], low_frequency_cutoff=args["f_lower"], high_frequency_cutoff=1024)[0]
        #olap = get_overlap_nsbh_mp((hp_f, args["template_bank_params"][idx[j]], args["template_bank_params"][idx[j],5]))
        olaps.append(olap)
        #if olap > match_target:
        #    print("found a match for injection ", n)
        del hpt_f

        if np.max(olaps) > match_target and j == acceptable_thresh:
            #print("found good enough matches for injection ", n, "best was ", np.max(olaps))
            break
    olaps = np.array(olaps)
    if np.max(olaps) < match_target:
        print("failed to find a good match for injection. best was ", np.max(olaps), ". m1, m2 were: ", waveform['mass1'], waveform['mass2'])
    else:
        print("found a good match for injection. best was ", np.max(olaps))
    #argsort the olaps and return the template ids
    del hp_f, hp
    #run garbage collection
    gc.collect()
    return olaps[np.argsort(olaps)[::-1][:required_templates]], idx[np.argsort(olaps)[::-1][:required_templates]] 


def load_pycbc_templates_from_hdf(hdf_file):
    #load templates from a pycbc hdf file produced by brute_bank, uberbank etc.

    f = h5py.File(hdf_file, 'r')
    templates = np.zeros((len(f['mass1']),6))
    templates[:,1] = f['mass1'][()]
    templates[:,2] = f['mass2'][()]
    templates[:,3] = f['spin1z'][()]
    templates[:,4] = f['spin2z'][()]
    templates[:,5] = f['f_lower'][()]
    templates[:,0] = chirp_mass(templates[:,1], templates[:,2])
    
    templates = templates[templates[:,0].argsort()]
    return templates