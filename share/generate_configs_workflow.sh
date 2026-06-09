#!/bin/bash

cd ${GWSAMPLEGEN_DIR}

config_file=$1
dependency=$2

n_jobs=20
n_SNR_jobs=40

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

configs_cpus=28

#create a string of jobs for easy cancellation
cancel_string=""

#check if save_dir/params.npy already exists, if it does, skip config generation
if [ -f $save_dir/params.npy ]; then
	echo "Params file already exists, skipping config generation"
	dep=""
else
	#if the file has the key "params_memory" use that value for --mem
	if jq -e '.params_mem' $config_file > /dev/null; then
		configs_mem="--mem=$(jq -r '.params_mem' ${config_file})G"
		echo "Using params_mem from config file: ${configs_mem}"
	else
		configs_mem=""
	fi

	main=$(ssh farnarkle2 "sbatch --job-name=${main_name} --array=0-$(($n_jobs-1)) $configs_mem --output=${GWSAMPLEGEN_DIR}/logs/%x_%a.log --dependency=after:$dependency --cpus-per-task=${configs_cpus} --parsable ${GWSAMPLEGEN_DIR}/share/generate_configs.sh $config_file $n_jobs $configs_cpus")
	echo "Submitted config generation job $main"
	cleanup_name="cleanup_"$jobname
	cleanup=$(sbatch --job-name=${cleanup_name} --mem=1G --dependency=afterok:$main --parsable ${GWSAMPLEGEN_DIR}/share/cleanup.sh $save_dir $n_jobs)
	dep="--dependency=afterok:$cleanup"
	cancel_string="$main $cleanup"
fi


SNR_jobname="SNR_"$jobname

if jq -e '.SNR_mem' $config_file > /dev/null; then
	SNR_mem="--mem=$(jq -r '.SNR_mem' $config_file)G"
	echo "Using SNR_mem from config file: ${SNR_mem}"
else
	SNR_mem=""
fi
SNR_job=$(sbatch --job-name=${SNR_jobname} --array=0-$(($n_SNR_jobs-1)) $dep $SNR_mem --parsable ${GWSAMPLEGEN_DIR}/share/SNR_np.sh $config_file $n_SNR_jobs)
# if [ -z "$cleanup" ]; then
# 	SNR_job=$(sbatch --job-name=${SNR_jobname} --parsable ${GWSAMPLEGEN_DIR}/share/SNR_np.sh $config_file)
	
# else
# 	SNR_job=$(sbatch --job-name=${SNR_jobname} $dep --parsable ${GWSAMPLEGEN_DIR}/share/SNR_np.sh $config_file)
# fi
cancel_string="$cancel_string $SNR_job"

final_cleanup_name="SNR_cleanup_"$jobname

#this final job is necessary to ensure the SNR_abs.npy is not over-allocated space.
#numpy's memmap function will allocate ~3x more disk space than the actual file size.
#copying the file ensures it is allocated the correct amount of space.
#final=$(sbatch --job-name=${final_cleanup_name} --ntasks=2 --output=${GWSAMPLEGEN_DIR}/logs/%x.log --time=2:00:00 --mem=10G --dependency=afterok:$SNR_job --parsable --wrap  "cd '${save_dir}'; ls; echo '${save_dir}'; for f in *.npy; do echo fixing file \${f}; cp \${f} \${f}.temp && rm \${f} && mv \${f}.temp \${f}; done")
final=$(sbatch --job-name=${final_cleanup_name} --mem=180G --ntasks=2 --dependency=afterok:$SNR_job --output=${GWSAMPLEGEN_DIR}/logs/%x.log  --parsable ${GWSAMPLEGEN_DIR}/share/cleanup.sh $save_dir $n_SNR_jobs "data")
cancel_string="$cancel_string $final"

#this echo ensures the final job's ID is printed to the terminal for dependency chaining. Make sure this is te last line of the script.
echo "Final job: "
echo $final
echo "To cancel all jobs, run: scancel $cancel_string"
