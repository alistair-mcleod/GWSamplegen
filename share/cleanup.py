import numpy as np
import json
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('--projectdir', type=str, default=None)
parser.add_argument('--njobs', type=int, default=None)
args = parser.parse_args()
project_dir = args.projectdir
n_jobs = args.njobs

config_file = os.path.join(project_dir, "args.json")

with open(config_file) as f:
	config = json.load(f)

	#project_dir = config['project_dir']
	#n_signal_samples = config['n_signal_samples']


x = np.load(os.path.join(project_dir, "params_0.npy"), allow_pickle=True).item()


#need to load in this order to ensure samples remain saved as signals first, then noise

z = {}

def get_signal_end(injection_array):
	#find where the last signal sample is in the file. 
	#This ensures that files with unequal numbers of samples are handled correctly.
	indices = np.where(np.diff(injection_array))[0]
	if indices.size > 0:
		return indices[0] + 1
	else:
		return len(injection_array)

end = get_signal_end(x['injection'])
#print("end", end)

for key in x.keys():
	z[key] = x[key][:end]

for i in range(1, n_jobs):
	try:
		y = np.load(os.path.join(project_dir, "params_" + str(i) + ".npy"), allow_pickle=True).item()
	except:
		print("Could not load params_" + str(i) + ".npy")
		continue
	end = get_signal_end(y['injection'])
	print("end", end)
	for key in x.keys():
		z[key] = np.concatenate((z[key], y[key][:end]), axis=0)


if get_signal_end(x['injection']) != len(x['injection']):
	end = get_signal_end(x['injection'])
	for key in x.keys():
		z[key] = np.concatenate((z[key], x[key][end:]), axis=0)

	for i in range(1, n_jobs):
		try:
			y = np.load(os.path.join(project_dir, "params_" + str(i) + ".npy"), allow_pickle=True).item()
		except:
			print("Could not load params_" + str(i) + ".npy")
			continue
		end = get_signal_end(y['injection'])
		print("end", end)
		for key in x.keys():
			z[key] = np.concatenate((z[key], y[key][end:]), axis=0)

#now save the new dictionary

np.save(os.path.join(project_dir, "params.npy"), z)
print("Saved new params.npy")