#!/bin/bash
#SBATCH --output=./logs/%x_%a.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --time=40:00:00
#SBATCH --mem=65G

#this script should be run using the generate_configs_workflow.sh script

n_jobs=${SLURM_ARRAY_TASK_COUNT}
#if there's a second argument, use it as n_jobs
if [ ! -z "$2" ]; then
    n_jobs=$2
    echo "Using n_jobs from argument: $n_jobs"
fi
job_id=${SLURM_ARRAY_TASK_ID}

#if there's a third argument, use it as cpus
if [ ! -z "$3" ]; then
    cpus=$3
    echo "Using cpus from argument: $cpus"
else
    cpus=28
    echo "Using default cpus: $cpus"
fi

config_file=$1

python ${GWSAMPLEGEN_DIR}/share/generate_configs.py --configfile=${config_file} --njobs=${n_jobs} --jobid=${job_id} --cpus=${cpus}