#!/bin/bash
#SBATCH -p thin
#SBATCH -t 04:00:00 
#SBATCH -n 64
#SBATCH -N 1
#SBATCH --mail-user=daan.kivits@wur.nl
#SBATCH --mail-type=FAIL,END
#SBATCH --job-name=PARIS_ATEN

cd "$(dirname "$1")"
echo "Current working directory: $(pwd)"

source /home/dkivits/.bashrc 

# Only if using virtual environment!
module purge
module load 2021
module load CDO/1.9.10-gompi-2021a
module load NCO/5.0.1-foss-2021a
module load Anaconda3/2021.05
module load netCDF/4.8.0-gompi-2021a 

source activate cte-hr-env

#python $1 > submit_fluxes.log
#python /projects/0/ctdas/PARIS/Experiments/scripts/yr1/BASE/combine_for_paris.py 2021 > submit_fluxes.log
python /projects/0/ctdas/PARIS/Experiments/scripts/yr1/ATEN/paris_ATEN.py > /projects/0/ctdas/PARIS/Experiments/scripts/yr1/ATEN/submit_fluxes_ATEN.log

echo "Job finished"
