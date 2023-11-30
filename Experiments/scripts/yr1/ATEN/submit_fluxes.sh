#!/bin/bash
#SBATCH -p genoa
#SBATCH -t 04:00:00 
#SBATCH -n 16
#SBATCH -N 1
#SBATCH --mail-user=daan.kivits@wur.nl
#SBATCH --mail-type=FAIL,END
#SBATCH --job-name=PARIS_ATEN

#module purge
module load 2022
module load Anaconda3/2022.05
module load CDO/2.0.6-gompi-2022a
module load NCO/5.1.0-foss-2022a

source activate cte-hr-env

cd "$(dirname "$1")"
echo "Current working directory: $(pwd)" 

#python $1 > submit_fluxes.log
#python /projects/0/ctdas/PARIS/Experiments/scripts/yr1/BASE/combine_for_paris.py 2021 > submit_fluxes.log
python /projects/0/ctdas/PARIS/Experiments/scripts/yr1/ATEN/paris_ATEN.py > /projects/0/ctdas/PARIS/Experiments/scripts/yr1/ATEN/submit_fluxes_ATEN.log

echo "Job finished"
