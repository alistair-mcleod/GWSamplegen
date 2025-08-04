#! /bin/bash
#SBATCH --output=./logs/%x.log
#SBATCH --time=02:00:00



#ml python/3.10.4

#source /fred/oz016/alistair/nt_310/bin/activate

project_dir=$1
n_jobs=$2

echo "Cleaning up temporary files"

python ${GWSAMPLEGEN_DIR}/share/cleanup.py --projectdir=$project_dir --njobs=$n_jobs
#get the exit code of the last command
exit_code=$?
if [ $exit_code -ne 0 ]; then
	echo "Error during cleanup, exit code: $exit_code"
	exit $exit_code
fi
echo "Finished cleaning up temporary files"