#!/bin/bash
#SBATCH --time 00:15:00                     # WALLTIME limit of 5 days (120 hours)
#SBATCH -n 16                              # ask for 52 tasks (amount of stations in stationfile_all.csv)
#SBATCH -N 1                                # ask for 1 node
##SBATCH --mem-per-cpu=2G                   # max memory per CPU
#SBATCH -p genoa                            # run on the genoa partition
#SBATCH --mail-type=ALL
#SBATCH --mail-user=daan.kivits@wur.nl
#SBATCH --output fp_mult.out                    
#SBATCH --error fp_mult.err
##SBATCH --job-name=STILT_fpflux_multiplication

# Import statements
#module purge
module load 2022
module load Anaconda3/2022.05
source activate cte-hr-env

# Declare fp, flux, obspack, bg, and STILT directories
fpdir='/projects/0/ctdas/PARIS/DATA/footprints/wur/PARIS_recompile/'
#fpdir='/projects/0/ctdas/PARIS/DATA/footprints/wur/multi-batch-test/'
bgdir='/projects/0/ctdas/PARIS/DATA/background/STILT/'
outdir='/projects/0/ctdas/PARIS/DATA/obspacks/'
obspack_path='/projects/0/ctdas/PARIS/DATA/obs/obspack_co2_466_GVeu_20230913.zip'
stiltdir='/projects/0/ctdas/PARIS/STILT_Model/STILT_Exe/'

# Define for which flux perturbation experiment the flux multiplication is being done
perturbation=$1
#perturbation="BASE"

# Define output directory based on perturbation experiment
outdir="/projects/0/ctdas/PARIS/DATA/obspacks/${perturbation}/"

station="MLH"
stationname="Malin Head"

# Set bool variables for verbose and sum-variables
verbose='True'

sum='True'

# Define ObsPack subset range (format: YYYY-MM-DD)
start_date="2021-02-01"
end_date="2021-02-03"
#start_date="2021-01-01"
#end_date="2022-01-01"

# Define grid
lat_ll=33.0
lat_ur=72.0
lon_ll=-15.0
lon_ur=35.0
lon_step=0.2
lat_step=0.1

# Define STILT run-specific variables
sim_len=240

# Define in how many parts the run should be split over time
nmonths_split=2

if [ -v perturbation ]; then
    echo 'Using PARIS fluxes'
    fluxdir='/projects/0/ctdas/PARIS/CTE-HR/PARIS_OUTPUT/'
else
    echo 'Using CTE-HR fluxes'
    fluxdir='/projects/0/ctdas/awoude/NRT/ICOS_OUTPUT/'
fi

# Define logfile directory
LOGDIR="${outdir}logs/"
  
#if LOGDIR does not exist, create it
if [ ! -d "${LOGDIR}" ]; then
    mkdir -p ${LOGDIR}
fi

echo "Starting single station FP-FLUX multiplication calculation as a test!"

# Define logfile
LOGFILE="${LOGDIR}${station}_fp_mult.log"

# Remove logfile if it already exists
if [ -f "$LOGFILE" ]; then
    rm $LOGFILE
fi

echo "Starting script for station ${stationname}"

# Run the STILT model for the current station and with the other user-specified arguments
if [ $verbose = 'True' ]; then
        if [ $sum = 'True' ]; then
                python -m cProfile -o profile.out /projects/0/ctdas/PARIS/STILT_scripts/fp_flux_mult/fp_flux_mult_MP_multiworkers.py --verbose --sum-variables --station $station --fluxdir $fluxdir --fpdir $fpdir --bgdir $bgdir --outdir $outdir --stiltdir $stiltdir --obspack_path $obspack_path --start_date $start_date --end_date $end_date --lat_ll $lat_ll --lat_ur $lat_ur --lon_ll $lon_ll --lon_ur $lon_ur --lat_step $lat_step --lon_step $lon_step --sim_len $sim_len --nmonths_split $nmonths_split --perturbation $perturbation >> ${LOGFILE} 2>&1 # Run your executable, note the "&"
        else
                python -m cProfile -o profile.out /projects/0/ctdas/PARIS/STILT_scripts/fp_flux_mult/fp_flux_mult_MP_multiworkers.py --verbose --station $station --fluxdir $fluxdir --fpdir $fpdir --bgdir $bgdir --outdir $outdir --stiltdir $stiltdir --obspack_path $obspack_path --start_date $start_date --end_date $end_date --lat_ll $lat_ll --lat_ur $lat_ur --lon_ll $lon_ll --lon_ur $lon_ur --lat_step $lat_step --lon_step $lon_step --sim_len $sim_len --nmonths_split $nmonths_split --perturbation $perturbation >> ${LOGFILE} 2>&1 # Run your executable, note the "&"
        fi
else
        if [ $sum = 'True' ]; then
                python -m cProfile -o profile.out /projects/0/ctdas/PARIS/STILT_scripts/fp_flux_mult/fp_flux_mult_MP_multiworkers.py --sum-variables --station $station --fluxdir $fluxdir --fpdir $fpdir --bgdir $bgdir --outdir $outdir --stiltdir $stiltdir --obspack_path $obspack_path --start_date $start_date --end_date $end_date --lat_ll $lat_ll --lat_ur $lat_ur --lon_ll $lon_ll --lon_ur $lon_ur --lat_step $lat_step --lon_step $lon_step --sim_len $sim_len --nmonths_split $nmonths_split --perturbation $perturbation >> ${LOGFILE} 2>&1 # Run your executable, note the "&"
        else
                python -m cProfile -o profile.out /projects/0/ctdas/PARIS/STILT_scripts/fp_flux_mult/fp_flux_mult_MP_multiworkers.py --station $station --fluxdir $fluxdir --fpdir $fpdir --bgdir $bgdir --outdir $outdir --stiltdir $stiltdir --obspack_path $obspack_path --start_date $start_date --end_date $end_date --lat_ll $lat_ll --lat_ur $lat_ur --lon_ll $lon_ll --lon_ur $lon_ur --lat_step $lat_step --lon_step $lon_step --sim_len $sim_len --nmonths_split $nmonths_split --perturbation $perturbation >> ${LOGFILE} 2>&1 # Run your executable, note the "&"
        fi
fi

wait 

echo
echo "All footprint-flux multiplications are done."