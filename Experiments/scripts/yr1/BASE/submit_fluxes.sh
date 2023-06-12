#!/bin/bash
#SBATCH -p thin
#SBATCH -t 02:00:00 
#SBATCH -n 64
#SBATCH -N 1
#SBATCH --mail-user=daan.kivits@wur.nl
#SBATCH --mail-type=FAIL,END

source /home/dkivits/.bashrc 

# Only if using virtual environment!
module purge
module load 2021
module load CDO/1.9.10-gompi-2021a
module load NCO/5.0.1-foss-2021a
module load Anaconda3/2021.05
module load netCDF/4.8.0-gompi-2021a 

source activate cte-hr-env

python /projects/0/ctdas/PARIS/Experiments/scripts/yr1/BASE/combine_for_paris.py 2021 > submit_fluxes.log

#nccopy -u -d9 /projects/0/ctdas/PARIS/CTE-HR/PARIS_OUTPUT/paris_input.nc /projects/0/ctdas/PARIS/CTE-HR/PARIS_OUTPUT/paris_input_u_d9.nc
