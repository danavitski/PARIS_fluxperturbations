#!/bin/bash
#SBATCH --time 01-00:00:00                  # WALLTIME limit
#SBATCH -n 1                                # ask for 52 tasks (amount of stations in stationfile_all.csv)
#SBATCH -N 1                                # ask for 1 node
#SBATCH --mem-per-cpu=3G                    # max memory per CPU
#SBATCH -p genoa                            # run on the genoa partition
#SBATCH --job-name multi_batch_STILT        # name to display in queue
#SBATCH --output std.out                    # standard output file
#SBATCH --error std.err                     # standard error file
#SBATCH --mail-type=ALL                     # send email on job start, end and fail
#SBATCH --mail-user=daan.kivits@wur.nl      # email address

# Import statements
module load 2022
module load R/4.2.1-foss-2022a
module load Anaconda3/2022.05
module load NCO/5.1.0-foss-2022a

# Declare STILT directories
basedir='/projects/0/ctdas/PARIS/transport_models/STILT_Model/'
rundir="${basedir}STILT_Exe/"
bdyfiles_dir="${basedir}stilt_hysplit/bdyfiles/"
sourcepath="${basedir}stiltR/"
fpdir='/projects/0/ctdas/PARIS/DATA/footprints/wur/'
#fpdir='/gpfs/scratch1/shared/dkivits/STILT/footprints/'

# Declare alternative (project-specific) directory for STILT output
altdir="for_Ruben"

if [ ! -v $altdir ]
then
    rundir="${rundir}${altdir}/"

    if [ ! -d ${rundir}   ]
    then
        mkdir $rundir
    fi

  path="${fpdir}${altdir}/"
fi

# Set neccessary environment variables
export PATH="$PATH:${basedir}/merged_stilt_hysplit"
export MKL_NUM_THREADS=1

# Define grid
numpix_x=140                            # number of pixels in x directions in grid
numpix_y=90                             # number of pixels in y directions in grid
lon_ll=-25                              # lower left corner of grid
lat_ll=30                               # lower left corner of grid
lon_ur=45                               # lower left corner of grid
lat_ur=75                               # lower left corner of grid
lon_res=0.5                             # resolution in degrees longitude
lat_res=0.5                             # resolution in degrees latitude

# go to STILT compilation directory
cd $sourcepath

# Declare station file location
FILENAME="/projects/0/ctdas/PARIS/DATA/obs/stations/stationfile_Ruben.csv"

echo "Starting multiple STILT simulations in parallel!"

# Calculate amount of rows in FILENAME
#N_LINES=$(tail -n +2 < $FILENAME | wc -l)
N_LINES=$(tail -n +1 < $FILENAME | wc -l)          # get number of lines in FILENAME
echo "Number of lines in station file is: $N_LINES"

# Run 1 job per task
N_JOBS=$N_LINES+1                     # get number of jobs

for ((i=2;i<=$N_JOBS;i++))
do

  # Extract station name from FILENAME
  # NOTE: It extracts the station_name from the second column of FILENAME. 
  # If the stationcode is in another column, specify that column here.
  station=$(awk -F',' -v row="$i" 'NR==row {print $2}' "$FILENAME")
  stationname=$(awk -F',' -v row="$i" 'NR==row {print $1}' "$FILENAME")
  echo "${stationname}"

  # Define subdirectory for station in rundir
  subdir="${rundir}${station}/"

  # First create subdirectory to store STILT output per station
  # NOTE: This is done outside of the setup_auto.sh script, because the script
  # we want to logfile to reside in the subdirectory. If the subdirectory does
  # not exist, the script will not be able to write to the subdirectory.
  if [ ! -d ${subdir}   ]
  then
    echo "First create subdirectory: ${subdir}"
    mkdir ${subdir}
  fi

  # Create neccessary STILT directories using the setup_auto.sh script. This is
  # done for each job, because each job needs its own directory. The script is 
  # adapted from the original setup.sh script provided in the STILT package.
  bash /projects/0/ctdas/PARIS/transport_models/STILT_Model/setup_multi.sh $rundir $path $bdyfiles_dir $subdir > "${subdir}STILT_log" &


  echo "Running STILT for station ${stationname} (code: ${station})"
  
  # Run the STILT model for the current station and with the other user-specified arguments
  Rscript stilt_loop_dense_and_sparse.r --sparse --overwrite-localization --filter-times --station $station --stationfile $FILENAME --nhrs 240 --npars 250 --numpix_x $numpix_x --numpix_y $numpix_y --lon_ll $lon_ll --lon_ur $lon_ur --lat_ll $lat_ll --lat_ur $lat_ur --lon_res $lon_res --lat_res $lat_res --path $path --rundir $rundir --sourcepath $sourcepath >> "${subdir}STILT_log" &  # Run your executable, note the "&"
  #Rscript stilt_loop_dense_and_sparse.r --sparse --overwrite-localization --station $station --stationfile $FILENAME --nhrs 240 --npars 250 --numpix_x $numpix_x --numpix_y $numpix_y --lon_ll $lon_ll --lon_ur $lon_ur --lat_ll $lat_ll --lat_ur $lat_ur --lon_res $lon_res --lat_res $lat_res --path $path --rundir $rundir --sourcepath $sourcepath >> "${subdir}STILT_log" &  # Run your executable, note the "&"
  
done

#Wait for all
wait
 
echo
echo "All STILT simulations are done."

# Test if all STILT simulations are done
echo "Finished succesfully: "
grep -r "Runs.done" ${rundir} | wc -l
echo "Total: {$N_LINES}"
