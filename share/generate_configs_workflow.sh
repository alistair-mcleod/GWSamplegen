#!/bin/bash

cd ${GWSAMPLEGEN_DIR}

config_file=$1
dependency=$2

n_jobs=20

#if config_file is not a valid file, exit

if [ ! -f $config_file ]; then
	echo "Config file not found! Check the path and try again."
	exit 1
fi
#TODO: change to jq
jobname=$(cat $config_file | python3 -c "import sys, json; print(json.load(sys.stdin)['jobname'])")

echo "Job name: $jobname"

if [ -z "$jobname" ]; then
	#exit
	echo "Job name not found in config file!"
	exit 1
fi

if [ -z "$dependency" ]; then
	dependency=-1
else
	echo "Dependency: $dependency"
fi


main_name="gen_"$jobname
save_dir=$(cat $config_file | python3 -c "import sys, json; print(json.load(sys.stdin)['project_dir'])")
echo "Save directory: $save_dir"

#comment out the block of lines below if you don't need to generate new configs
main=$(ssh farnarkle2 "sbatch --job-name=${main_name} --array=0-$(($n_jobs-1)) --output=${GWSAMPLEGEN_DIR}/logs/%x_%a.log --dependency=after:$dependency --parsable ${GWSAMPLEGEN_DIR}/share/generate_configs.sh $config_file $n_jobs")
echo "Submitted config generation job $main"
cleanup_name="cleanup_"$jobname
cleanup=$(sbatch --job-name=${cleanup_name} --mem=1G --dependency=afterok:$main --parsable ${GWSAMPLEGEN_DIR}/share/cleanup.sh $save_dir $n_jobs)


SNR_jobname="SNR_"$jobname

if [ -z "$cleanup" ]; then
	SNR_job=$(sbatch --job-name=${SNR_jobname} --parsable ${GWSAMPLEGEN_DIR}/share/SNR_np.sh $config_file)
	
else
	SNR_job=$(sbatch --job-name=${SNR_jobname} --dependency=afterok:$cleanup --parsable ${GWSAMPLEGEN_DIR}/share/SNR_np.sh $config_file)
fi


final_cleanup_name="SNR_cleanup_"$jobname

#this final job is necessary to ensure the SNR_abs.npy is not over-allocated space.
#numpy's memmap function will allocate ~3x more disk space than the actual file size.
#copying the file ensures it is allocated the correct amount of space.
final=$(sbatch --job-name=${final_cleanup_name} --ntasks=2 --output=${GWSAMPLEGEN_DIR}/logs/%x.log --time=1:00:00 --mem=100G --dependency=afterok:$SNR_job --parsable --wrap  "cd ${save_dir}; cp SNR_abs.npy SNR2.npy; rm SNR_abs.npy; mv SNR2.npy SNR_abs.npy")

#this echo ensures the final job's ID is printed to the terminal for dependency chaining
echo "Final job: "
echo $final