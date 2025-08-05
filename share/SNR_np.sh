#!/bin/bash
#SBATCH --output=./logs/%x_%a.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=20:00:00
#SBATCH --mem=160gb
#SBATCH --array=0-19

#this script should be run using the generate_configs_workflow.sh script

config_file=$1

#$SLURM_ARRAY_TASK_COUNT
python ${GWSAMPLEGEN_DIR}/share/asyncSNR_np.py --index=$SLURM_ARRAY_TASK_ID --totaljobs=$SLURM_ARRAY_TASK_COUNT --config-file=$config_file