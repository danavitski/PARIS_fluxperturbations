#!/bin/bash
#SBATCH -p genoa
#SBATCH -t 04:00:00 
#SBATCH -n 16
#SBATCH -N 1
#SBATCH --mail-user=daan.kivits@wur.nl
#SBATCH --mail-type=FAIL,END

module purge
module load 2022
module load CDO/2.0.6-gompi-2022a
module load Anaconda3/2022.05
module load netCDF/4.9.0-gompi-2022a

#python /projects/0/ctdas/PARIS/Experiments/scripts/yr1/BASE/daily_to_single_ctehr.py >> submit_fluxes.log 2>&1 
python /projects/0/ctdas/PARIS/Experiments/scripts/yr1/BASE/combine_for_paris.py >> submit_fluxes.log 2>&1 

#nccopy -u -s -d9 /projects/0/ctdas/PARIS/CTE-HR/PARIS_OUTPUT/paris_input.nc /projects/0/ctdas/PARIS/CTE-HR/PARIS_OUTPUT/paris_input_s_u_d9.nc

echo "Job finished"
