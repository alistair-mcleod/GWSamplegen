#!/bin/bash
#SBATCH --output=./logs/%x_%a.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=50:00:00
#SBATCH --mem=80gb
#SBATCH --array=0-39

#this script should be run using the generate_configs_workflow.sh script

config_file=$1
#check if there's a second argument
if [ -z "$2" ]; then
	n_jobs=$SLURM_ARRAY_TASK_COUNT
else
	n_jobs=$2
fi
#${SLURM_ARRAY_TASK_COUNT}
#TODO: make the number of jobs a parameter
python ${GWSAMPLEGEN_DIR}/share/asyncSNR_np.py --index=$SLURM_ARRAY_TASK_ID --totaljobs=$n_jobs --config-file=$config_file