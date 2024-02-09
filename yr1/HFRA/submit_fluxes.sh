#!/bin/bash
#SBATCH -p genoa
#SBATCH -t 04:00:00 
#SBATCH -n 16
#SBATCH -N 1
#SBATCH --mail-user=daan.kivits@wur.nl
#SBATCH --mail-type=FAIL,END
#SBATCH --job-name=PARIS_HFRA 

module load 2022
module load Anaconda3/2022.05
module load CDO/2.0.6-gompi-2022a
module load NCO/5.1.0-foss-2022a

source activate cte-hr-env

python /projects/0/ctdas/PARIS/Experiments/scripts/yr1/HFRA/paris_HFRA.py > /projects/0/ctdas/PARIS/Experiments/scripts/yr1/HFRA/submit_fluxes_HFRA.log

echo "Job finished"
