#!/bin/bash
#SBATCH --job-name=STILT_fpflux_multiplication_test
#SBATCH -t 00:15:00
#SBATCH --ntasks=1
#SBATCH -p genoa
#SBATCH --mail-type=ALL
#SBATCH --mail-user=daan.kivits@wur.nl

cd /projects/0/ctdas/PARIS/STILT_scripts/

module load 2022
module load Anaconda3/2022.05
source activate cte-hr-env

python /projects/0/ctdas/PARIS/STILT_scripts/footprint_flux_multiplication_sparse_vs_dense_summed.py
python /projects/0/ctdas/PARIS/STILT_scripts/footprint_flux_multiplication_sparse_vs_dense_summed_non_ortho_stationloop.py