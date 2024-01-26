#!/bin/bash
#SBATCH --time 01:00:00                     # WALLTIME limit of 5 days (120 hours)
#SBATCH -n 1                              # ask for 52 tasks (amount of stations in stationfile_all.csv)
#SBATCH -N 1                                # ask for 1 node
#SBATCH -p genoa                            # run on the genoa partition
#SBATCH --mail-type=ALL
#SBATCH --mail-user=daan.kivits@wur.nl
#SBATCH --job-name=integrated_fp_plot
#SBATCH --output fp_plot.out                 
#SBATCH --error fp_plot.err

# Import statements
module load 2022
module load Anaconda3/2022.05

source activate cte-hr-env

python /projects/0/ctdas/PARIS/DATA/footprints/footprints_sparse_to_dense_Auke.py