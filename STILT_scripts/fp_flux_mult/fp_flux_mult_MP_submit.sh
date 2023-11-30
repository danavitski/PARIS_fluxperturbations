#!/bin/bash
#SBATCH --time 00-00:30:00                      # WALLTIME limit of 5 days (120 hours)
#SBATCH -n 52                                  # ask for 52 tasks (amount of stations in stationfile_all.csv)
##SBATCH -N 1                                    # ask for 1 node
#SBATCH --cpus-per-task=1                       # ask for 1 core per task (52 cores in total)
#SBATCH --mem-per-cpu=2G                        # max memory per node
##SBATCH -p fat                                 # run on the fat partition
#SBATCH -p fat
#SBATCH --mail-type=ALL
#SBATCH --mail-user=daan.kivits@wur.nl
#SBATCH --output fp_mult.out                    
#SBATCH --error fp_mult.err
#SBATCH --job-name=STILT_fpflux

# Limit number of threads used to 1
#export OPENBLAS_NUM_TRHEADS=1
#export OMP_NUM_THREADS=1

# Import statements
module load 2022
module load Anaconda3/2022.05
module load netCDF/4.9.0-gompi-2022a
source activate cte-hr-env

# Define for which flux perturbation experiment the flux multiplication is being done
perturbation=$1

# Define whether to aggregate the flux variables in the ObsPack or leave them as separate variables
sum=$2

# Declare fp, flux, obspack, bg, and STILT directories
fpdir='/projects/0/ctdas/PARIS/DATA/footprints/wur/PARIS_recompile/'
#fpdir='/projects/0/ctdas/PARIS/DATA/footprints/wur/multi-batch-test/'
bgdir='/projects/0/ctdas/PARIS/DATA/background/STILT/'
obspack_path='/projects/0/ctdas/PARIS/DATA/obs/obspack_co2_466_GVeu_20230913.zip'
stiltdir='/projects/0/ctdas/PARIS/STILT_Model/STILT_Exe/'

# Define output directory based on perturbation experiment
#outdir="/projects/0/ctdas/PARIS/DATA/obspacks/${perturbation}/"
outdir="/projects/0/ctdas/PARIS/DATA/obspacks/test/"

# If outdir does not exist, create it
if [ ! -d "${outdir}" ]; then
    mkdir -p ${outdir}
fi

# Define ObsPack subset range
start_date="2021-01-01"
#end_date="2021-01-03"
end_date="2022-01-01"

# Define grid
lat_ll=33.0
lat_ur=72.0
lon_ll=-15.0
lon_ur=35.0
lon_step=0.2
lat_step=0.1

# Define STILT run-specific variables
sim_len=240
npars=250

# Define in how many parts the run should be split over time
nmonths_split=1

# Define whether to print verbose output
verbose='False'

# Define whether to use the 'regular' NRT CTE-HR or PARIS-specific fluxes. Choose between "CTEHR" or "PARIS".
fluxtype="PARIS"

if [ ${fluxtype} = "PARIS" ]; then
    echo 'Using PARIS fluxes'
    fluxdir='/projects/0/ctdas/PARIS/CTE-HR/PARIS_OUTPUT/'
elif [ ${fluxtype} = "CTEHR" ]; then
    echo 'Using CTE-HR fluxes'
    fluxdir='/projects/0/ctdas/awoude/NRT/ICOS_OUTPUT/'
else
    echo "Please specify a valid fluxtype. Choose between "CTEHR" or "PARIS"."
fi

# Define logfile directory
LOGDIR="${outdir}logs/"
  
#if LOGDIR does not exist, create it
if [ ! -d "${LOGDIR}" ]; then
    mkdir -p ${LOGDIR}
fi

# Declare station file location
FILENAME="/projects/0/ctdas/PARIS/DATA/stationfile_all.csv"

echo "Starting multiple FP-FLUX multiplication calculations in parallel!"

# Calculate amount of rows in FILENAME
N_LINES=$(tail -n +2 < $FILENAME | wc -l)        # get number of lines in FILENAME
echo "Number of stations in  station file: $N_LINES"

# Run 1 job per task
N_JOBS=$N_LINES+1                     # get number of jobs
#N_JOBS=3                     # get number of jobs

for ((i=2;i<=$N_JOBS;i++))
#for ((i=50;i<=$N_JOBS;i++))
do
    # Extract station name from FILENAME
    # NOTE: It extracts the station_name from the second column of FILENAME. 
    # If the stationcode is in another column, specify that column here.
    station=$(awk -F',' -v row="$i" 'NR==row {print $2}' "$FILENAME")
    stationname=$(awk -F',' -v row="$i" 'NR==row {print $1}' "$FILENAME")

    # Define logfile
    LOGFILE="${LOGDIR}${station}_fp_mult.log"

    # Remove logfile if it already exists
    if [ -f "$LOGFILE" ]; then
        rm $LOGFILE
    fi

    echo "Starting script for station ${stationname} (code: ${station})"

    # Run the STILT model for the current station and with the other user-specified arguments
    if [ $verbose = 'True' ]; then
            if [ $sum = 'True' ]; then
                    python /projects/0/ctdas/PARIS/STILT_scripts/fp_flux_mult/fp_flux_mult_MP_multiworkers.py --verbose --sum-variables --station $station --fluxdir $fluxdir --fpdir $fpdir --bgdir $bgdir --outdir $outdir --stiltdir $stiltdir --obspack_path $obspack_path --start_date $start_date --end_date $end_date --lat_ll $lat_ll --lat_ur $lat_ur --lon_ll $lon_ll --lon_ur $lon_ur --lat_step $lat_step --lon_step $lon_step --sim_len $sim_len --npars $npars --nmonths_split $nmonths_split --perturbation $perturbation >> ${LOGFILE} 2>&1 & # Run your executable, note the "&"
                    echo "option 1"
            else
                    python /projects/0/ctdas/PARIS/STILT_scripts/fp_flux_mult/fp_flux_mult_MP_multiworkers.py --verbose --station $station --fluxdir $fluxdir --fpdir $fpdir --bgdir $bgdir --outdir $outdir --stiltdir $stiltdir --obspack_path $obspack_path --start_date $start_date --end_date $end_date --lat_ll $lat_ll --lat_ur $lat_ur --lon_ll $lon_ll --lon_ur $lon_ur --lat_step $lat_step --lon_step $lon_step --sim_len $sim_len --npars $npars --nmonths_split $nmonths_split --perturbation $perturbation >> ${LOGFILE} 2>&1 & # Run your executable, note the "&"
                    echo "option 2"
            fi
    else
            if [ $sum = 'True' ]; then
                    python /projects/0/ctdas/PARIS/STILT_scripts/fp_flux_mult/fp_flux_mult_MP_multiworkers.py --sum-variables --station $station --fluxdir $fluxdir --fpdir $fpdir --bgdir $bgdir --outdir $outdir --stiltdir $stiltdir --obspack_path $obspack_path --start_date $start_date --end_date $end_date --lat_ll $lat_ll --lat_ur $lat_ur --lon_ll $lon_ll --lon_ur $lon_ur --lat_step $lat_step --lon_step $lon_step --sim_len $sim_len --npars $npars --nmonths_split $nmonths_split --perturbation $perturbation >> ${LOGFILE} 2>&1 & # Run your executable, note the "&"
                    echo "option 3"
            else
                    #python -m cProfile -o profile.out /projects/0/ctdas/PARIS/STILT_scripts/fp_flux_mult/fp_flux_mult_MP_multiworkers.py --station $station --fluxdir $fluxdir --fpdir $fpdir --bgdir $bgdir --outdir $outdir --stiltdir $stiltdir --obspack_path $obspack_path --start_date $start_date --end_date $end_date --lat_ll $lat_ll --lat_ur $lat_ur --lon_ll $lon_ll --lon_ur $lon_ur --lat_step $lat_step --lon_step $lon_step --sim_len $sim_len --npars $npars --nmonths_split $nmonths_split --perturbation $perturbation >> ${LOGFILE} 2>&1 & # Run your executable, note the "&"
                    python /projects/0/ctdas/PARIS/STILT_scripts/fp_flux_mult/fp_flux_mult_MP_multiworkers.py --station $station --fluxdir $fluxdir --fpdir $fpdir --bgdir $bgdir --outdir $outdir --stiltdir $stiltdir --obspack_path $obspack_path --start_date $start_date --end_date $end_date --lat_ll $lat_ll --lat_ur $lat_ur --lon_ll $lon_ll --lon_ur $lon_ur --lat_step $lat_step --lon_step $lon_step --sim_len $sim_len --npars $npars --nmonths_split $nmonths_split --perturbation $perturbation >> ${LOGFILE} 2>&1 & # Run your executable, note the "&"
                    echo "option 4"
            fi
    fi

done

#Wait for all
wait $!

echo
echo "All footprint-flux multiplications are done."

# Test if all calcultions have been completed
grep -r "Flux multiplication finished for station*" ${LOGFILE} | wc -l
echo "Total: {$N_JOBS}"
