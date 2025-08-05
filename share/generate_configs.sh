#!/bin/bash
#SBATCH --output=./logs/%x_%a.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --time=40:00:00
#SBATCH --mem=100G

#this script should be run using the generate_configs_workflow.sh script

n_jobs=${SLURM_ARRAY_TASK_COUNT}
#if there's a second argument, use it as n_jobs
if [ ! -z "$2" ]; then
    n_jobs=$2
    echo "Using n_jobs from argument: $n_jobs"
fi
job_id=${SLURM_ARRAY_TASK_ID}

config_file=$1

python ${GWSAMPLEGEN_DIR}/share/generate_configs.py --configfile=${config_file} --njobs=${n_jobs} --jobid=${job_id}