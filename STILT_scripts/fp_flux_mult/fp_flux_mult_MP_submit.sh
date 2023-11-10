#!/bin/bash
#SBATCH --time 01:00:00                     # WALLTIME limit of 5 days (120 hours)
#SBATCH -n 312                              # ask for 52 tasks (amount of stations in stationfile_all.csv)
#SBATCH -N 2                                # ask for 1 node
#SBATCH --mem-per-cpu=1750                  # max memory per node
#SBATCH -p genoa                            # run on the genoa partition
#SBATCH --mail-type=ALL
#SBATCH --mail-user=daan.kivits@wur.nl
#SBATCH --output fp_mult.out                    
#SBATCH --error fp_mult.err
#SBATCH --job-name=STILT_fpflux_multiplication

# Import statements
module load 2022
module load Anaconda3/2022.05
module load netCDF/4.9.0-gompi-2022a
source activate cte-hr-env

# Declare fp, flux, obspack, bg, and STILT directories
fluxdir='/projects/0/ctdas/awoude/NRT/ICOS_OUTPUT/'
fpdir='/projects/0/ctdas/PARIS/DATA/footprints/wur/PARIS/'
bgdir='/projects/0/ctdas/PARIS/DATA/background/STILT/'
outdir='/projects/0/ctdas/PARIS/DATA/obspacks/'
obspack_path='/projects/0/ctdas/PARIS/DATA/obs/obspack_co2_466_GVeu_20230913.zip'
stiltdir='/projects/0/ctdas/PARIS/STILT_Model/STILT_Exe/'

# Define ObsPack subset range
start_date="2021-01-01"
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
nmonths_split=6

# Define whether to use the 'regular' NRT CTE-HR or PARIS-specific fluxes
fluxtype='CTEHR'

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

for ((i=2;i<=$N_JOBS;i++))
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
  python /projects/0/ctdas/PARIS/STILT_scripts/fp_flux_mult/fp_flux_mult_MP_multiworkers.py --station $station --fluxdir $fluxdir --fpdir $fpdir --bgdir $bgdir --outdir $outdir --stiltdir $stiltdir --obspack_path $obspack_path --start_date $start_date --end_date $end_date --lat_ll $lat_ll --lat_ur $lat_ur --lon_ll $lon_ll --lon_ur $lon_ur --lat_step $lat_step --lon_step $lon_step --sim_len $sim_len --npars $npars --nmonths_split $nmonths_split --fluxtype $fluxtype >> ${LOGFILE} 2>&1 & # Run your executable, note the "&"
  
done

#Wait for all
wait $!

echo
echo "All footprint-flux multiplications are done."

# Test if all calcultions have been completed
grep -r "Flux multiplication finished for station*" ${LOGFILE} | wc -l
echo "Total: {$N_LINES}"
