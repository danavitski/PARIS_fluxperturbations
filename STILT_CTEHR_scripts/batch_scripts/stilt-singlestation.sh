#!/bin/bash
#SBATCH --job-name=STILT_PDM
#SBATCH --time 2-00:00:00                     # WALLTIME limit of 5 days (120 hours)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2G                    # max memory per CPU
#SBATCH -p genoa
#SBATCH --mail-type=ALL
#SBATCH --mail-user=daan.kivits@wur.nl
#SBATCH --output std.out                    # standard output file
#SBATCH --error std.err                     # standard error file

module load 2022
module load R/4.2.1-foss-2022a
module load Anaconda3/2022.05
module load NCO/5.1.0-foss-2022a

source activate cte-hr-env

# Set neccessary environment variables
export PATH="$PATH:/projects/0/ctdas/PARIS/STILT_Model/merged_stilt_hysplit"
export MKL_NUM_THREADS=1

# Declare STILT directories
basedir='/projects/0/ctdas/PARIS/STILT_Model/'
rundir="${basedir}STILT_Exe/"
bdyfiles_dir="${basedir}stilt_hysplit/bdyfiles/"
sourcepath="${basedir}stiltR/"
path="/projects/0/ctdas/PARIS/DATA/footprints/wur/PARIS_recompile/"
#path="/gpfs/scratch1/shared/dkivits/STILT/footprints/"

# go to STILT compilation directory
cd $sourcepath

# Declare station file location
FILENAME="/projects/0/ctdas/PARIS/DATA/stationfile_all.csv"

# Declare station to run STILT for
station='TOB'
subdir="${rundir}${station}/"

# First create subdirectory to store STILT output per station
# NOTE: This is done outside of the setup_auto.sh script, because the script
# we want to logfile to reside in the subdirectory. If the subdirectory does
# not exist, the script will not be able to write to the subdirectory.
if [ ! -d ${subdir}   ]
then
    echo "First create subdirectory: ${subdir}"
    mkdir ${subdir}
else
    echo "Subdirectory ${subdir} already exists."
fi

# Run setup_auto.sh script
bash /projects/0/ctdas/PARIS/transport_models/STILT_Model/setup_multi.sh $rundir $path $bdyfiles_dir $subdir > "${subdir}STILT_log" 

# Single-cycle multi-station run:
Rscript stilt_loop_dense_and_sparse.r --sparse --filter-times --overwrite-localization --station $station --stationfile $FILENAME --nhrs 240 --npars 250 --path $path --rundir $rundir --sourcepath $sourcepath >> "${subdir}STILT_log" &  # Run your executable, note the "&"

# Multi-cycle multi-station runs:
#for i in {1..3}
#do
#    Rscript stilt_loop_dense_and_sparse.r --sparse --filter-times --cycle-num $i --nhrs 240 --npars 250
#done

# Multi-station runs:
#Rscript stilt_loop_dense_and_sparse.r --station 'CBW' --sparse --filter-times --ens-members 10 --nhrs 240 --npars 200
#Rscript stilt_loop_dense_and_sparse.r --station 'CBW' --sparse --filter-times --ens-members 10 --nhrs 240 --npars 150
#Rscript stilt_loop_dense_and_sparse.r --station 'CBW' --sparse --filter-times --ens-members 10 --nhrs 240 --npars 1000

# Station-specific runs:
#Rscript stilt.r --station 'TST'
#Rscript stilt.r --station 'HEI' 
#Rscript stilt.r --station 'GAT' 
#Rscript stilt.r --station 'HPB'
#Rscript stilt.r --station 'KRE'
#Rscript stilt.r --station 'LIN'
#Rscript stilt.r --station 'KIT'
#Rscript stilt.r --station 'STK'
#Rscript stilt.r --station 'CES'
#Rscript stilt.r --station 'LUT'
#Rscript stilt.r --station 'SAC'
#Rscript stilt.r --station 'IPR'
#Rscript stilt.r --station 'FRE'
#Rscript stilt.r --station 'GNS'
#Rscript stilt.r --station 'COU'
#Rscript stilt.r --station 'AND'
#Rscript stilt.r --station 'OVS'
#Rscript stilt.r --station '2MV'
#Rscript stilt.r --station 'ROC'
#Rscript stilt.r --station 'ZWT'
#Rscript stilt.r --station 'WMS'

#Rscript stilt.r --station 'RHI_UW' 
#Rscript stilt.r --station 'RHI_DW'
#Rscript stilt.r --station 'BOR_UW' 
#Rscript stilt.r --station 'BOR_DW' 
#Rscript stilt.r --station 'LYO_UW' 
#Rscript stilt.r --station 'LYO_DW' 
#Rscript stilt.r --station 'LIL_UW' 
#Rscript stilt.r --station 'LIL_DW' 
#Rscript stilt.r --station 'LUX_UW' 
#Rscript stilt.r --station 'LUX_DW' 
#Rscript stilt.r --station 'RUR_UW' 
#Rscript stilt.r --station 'RUR_DW' 
#Rscript stilt.r --station 'BER_UW' 
#Rscript stilt.r --station 'BER_DW' 
#Rscript stilt.r --station 'MUN_UW' 
#Rscript stilt.r --station 'MUN_DW'

echo
echo "STILT simulation for station ${station} is done."