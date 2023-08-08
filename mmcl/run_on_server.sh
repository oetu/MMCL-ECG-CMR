#!/usr/bin/bash
#SimCLR

#SBATCH --job-name=mm_ecg
#SBATCH --output=/vol/aimspace/users/tuo/ECGMultimodalContrastiveLearning/slurm_output/mm_ecg-%A.out  # Standard output of the script (Can be absolute or relative path). %A adds the job id to the file name so you can launch the same script multiple times and get different logging files
#SBATCH --error=/vol/aimspace/users/tuo/ECGMultimodalContrastiveLearning/slurm_output/mm_ecg-%A.err  # Standard error of the script
#SBATCH --time=0-23:59:59  # Limit on the total run time (format: days-hours:minutes:seconds)
#SBATCH --gres=gpu:1  # Number of GPUs if needed
#SBATCH --cpus-per-task=24  # Number of CPUs (Don't use more than 24 per GPU)
#SBATCH --mem=128G  # Memory in GB (Don't use more than 126G per GPU)

# load python module
module load python/anaconda3
source /opt/anaconda3/etc/profile.d/conda.sh

# activate corresponding environment
conda deactivate # If you launch your script from a terminal where your environment is already loaded, conda won't activate the environment. This guards against that. Not necessary if you always run this script from a clean terminal
conda activate mm

cmd="./run.sh"
echo $cmd && $cmd