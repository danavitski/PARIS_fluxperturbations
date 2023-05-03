#!/bin/bash
#SBATCH -p thin
#SBATCH -t 4:00:00 
#SBATCH -n 64
#SBATCH -N 1
#SBATCH --mail-user=daan.kivits@wur.nl
#SBATCH --mail-type=FAIL,END

#source /home/dkivits/.bashrc 

# Only if using virtual environment!
module purge
module load 2021
module load CDO/1.9.10-gompi-2021a
module load NCO/5.0.1-foss-2021a
module load Anaconda3/2021.05

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/sw/arch/Centos8/EB_production/2021/software/Anaconda3/2021.05/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/sw/arch/Centos8/EB_production/2021/software/Anaconda3/2021.05/etc/profile.d/conda.sh" ]; then
        . "/sw/arch/Centos8/EB_production/2021/software/Anaconda3/2021.05/etc/profile.d/conda.sh"
    else
        export PATH="/sw/arch/Centos8/EB_production/2021/software/Anaconda3/2021.05/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate cte-hr-env

python extract_countrymasks_Europe.py 
