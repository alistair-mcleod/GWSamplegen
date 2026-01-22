# GWSamplegen

A library of tools and scripts for generating compact binary merger samples in real LIGO noise. Specialised for creating SNR time series datasets for use in deep learning.


## Installation

First, ensure you have a python virtual environment you want to install GWSamplegen into (version 3.10 or above). Next, clone this repository with:

```
git clone https://github.com/alistair-mcleod/GWSamplegen.git
```
then `cd` into GWSamplegen and run

```
bash install.sh
```

and the repository should be installed into your virtual environment.

## Usage instructions

To use GWsamplegen, you must first create an arguments file which specifies the parameter distributions of the dataset you wish to create. An example arguments file is provided in `configs/args.json`. You can then run `bash ${GWSAMPLEGEN_DIR}/share/generate_configs_workflow.sh /path/to/args.json` to build the workflow, which will first generate the parameter file, then generate the SNR time series dataset. The generated dataset and parameter file will be saved in the `project_dir` specified in the arguments file.

## Usage notes

#### Condor compatibility
Currently, the job submission scripts are currently written with Slurm syntax, and most of the repository is built around the Slurm job scheduling system. Adapting this code to a Condor-based cluster (i.e. CIT) is currently under development. 

#### Noise availability

While this repository does work Gaussian noise, it is primarily designed to work with real LIGO noise. If you do not have LIGO noise already available, use the `fetch_noise.py` script to download desired segments of LIGO data. 