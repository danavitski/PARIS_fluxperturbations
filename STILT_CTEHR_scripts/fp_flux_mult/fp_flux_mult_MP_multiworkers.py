"""
This script performs the footprint multiplication process using multiple workers.

The script takes input files containing footprints and fluxes, and multiplies them together to calculate the CO2
concentration at receptor locations. It uses multiple workers to parallelize the computation and improve performance.

For this workflow, STILT (Stochastic Time-Inverted Lagrangian Transport) influence fields stored in a sparse format are expected as input,
as well as CTE-HR flux fields or PARIS-specific flux fields (which are modified CTE-HR fluxes). 

Usage:
    python fp_flux_mult_MP_multiworkers.py --station <stationcode> --fluxdir <fluxdir> --fpdir <fpdir> --bgdir <bgdir> --outdir <outdir> 
    --stiltdir <stiltdir> --obspack_path <obspack_path> --start_date <start_date> --end_date <end_date> --lat_ll <lat_ll> --lat_ur <lat_ur> 
    --lon_ll <lon_ll> --lon_ur <lon_ur> --lat_step <lat_step> --lon_step <lon_step> --sim_len <sim_len> --npars <npars> --nmonths_split <nmonths_split> 
    --perturbation <perturbation> --verbose --sum-variables

Arguments:

    --station <stationcode>: Code of station to run the script for (str)
    --fluxdir <fluxdir>: Directory where flux files are stored (str)
    --fpdir <fpdir>: Directory where footprint files are stored (str)
    --bgdir <bgdir>: Directory where (TM3) background files are stored (str)
    --outdir <outdir>: Directory where output files are stored (str)
    --stiltdir <stiltdir>: Directory where STILT is executed (str)
    --obspack_path <obspack_path>: Path to the ObsPack collection zip file (str)
    --start_date <start_date>: Date from when to subset the ObsPacks (str)
    --end_date <end_date>: Date up to where to subset the ObsPacks (str)
    --lat_ll <lat_ll>: Latitude of ll corner to define the grid extent (float)
    --lat_ur <lat_ur>: Latitude of ur corner to define the grid extent (float)
    --lon_ll <lon_ll>: Longitude of ll corner to define the grid extent (float)
    --lon_ur <lon_ur>: Longitude of ur corner to define the grid extent (float)
    --lat_step <lat_step>: Latitude cell size step (float)
    --lon_step <lon_step>: Longitude cell size step (float)
    --sim_len <sim_len>: Simulation length in hours (int)
    --npars <npars>: Number of particles (int)
    --nmonths_split <nmonths_split>: Controls in how many pieces the run should be split into multiple multiprocessing pool tasks (int)
    --perturbation <perturbation>: Controls for which perturbation experiment the fp-flux multiplication script is ran (str)
    --verbose: Controls whether the script should do logging (bool)
    --sum-variables: Controls whether the script should sum the fluxvariables or not. TRUE: only produce a "mixed" molefraction field. FALSE: transport all fluxvariables individually and store them in ObsPack (bool)

Note:
    This script requires the following modules to be loaded onto the HPC or locally:

        - netCDF4
        - pandas
        - numpy
        - xarray

    The rest of the modules should be available in the standard Python 3.11.0 environment.

"""

# Import necessary modules
import pandas as pd
import numpy as np
import glob
import netCDF4 as nc
from datetime import datetime, timedelta
import pandas as pd
import xarray as xr
import os
import time
from functions.fluxfile_functions import *
from functions.background_functions import *
from functions.obspack_identification_col import *
from functions.obspack_identification_url import *
import tqdm
from multiprocessing import Pool
from functools import partial
import argparse
import logging

# Set logging options
#logging.basicConfig(level=logging.info, format=' [%(levelname)-7s] (%(asctime)s) py-%(module)-20s : %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Load command-line arguments
parser = argparse.ArgumentParser(description='Flux - footprint multiplication script')

######################
##### USER INPUT #####
######################
# Define command-line arguments
parser.add_argument('--station', type=str, help='Code of station to run the script for')
parser.add_argument('--stationsfile', type=str, help='Path to the (.csv) file that contains all station metadata')
parser.add_argument('--fluxdir', type=str, help='Directory where flux files are stored')
parser.add_argument('--fpdir', type=str, help='Directory where footprint files are stored')
parser.add_argument('--bgdir', type=str, help='Directory where (TM3) background files are stored')
parser.add_argument('--outdir', type=str, help='Directory where output files are stored')
parser.add_argument('--stiltdir', type=str, help='Directory where STILT is executed')
parser.add_argument('--obspack_path', type=str, help='Path to the ObsPack collection zip file')
parser.add_argument('--start_date', type=str, help='Date from when to subset the ObsPacks')
parser.add_argument('--end_date', type=str, help='Date up to where to subset the ObsPacks')
parser.add_argument('--lat_ll', type=float, help='Latitude of ll corner to define the grid extent (float)')
parser.add_argument('--lat_ur', type=float, help='Latitude of ur corner to define the grid extent (float)')
parser.add_argument('--lon_ll', type=float, help='Longitude of ll corner to define the grid extent (float)')
parser.add_argument('--lon_ur', type=float, help='Longitude of ur corner to define the grid extent (float)')
parser.add_argument('--lat_step', type=float, help='Latitude cell size step (float)')
parser.add_argument('--lon_step', type=float, help='Longitude cell size step (float)')
parser.add_argument('--sim_len', type=int, help='Simulation length in hours (int)')
parser.add_argument('--npars', type=int, help='Number of particles (int)')
parser.add_argument('--nmonths_split', type=int, help='Controls in how many pieces the run should be split into multiple multiprocessing pool tasks (int)')
parser.add_argument('--perturbation', type=str, help='Controls for which perturbation experiment the fp-flux multiplication script is ran (str)')
parser.add_argument('--verbose', action="store_true", help='Controls whether the script should do logging (bool)')
parser.add_argument('--sum-variables', action="store_true", help='Controls whether the script should sum the fluxvariables or not. TRUE: only produce a "mixed" molefraction field. FALSE: transport all fluxvariables individually and store them in ObsPack (bool)')

# Parse the command-line arguments
args = parser.parse_args()

# Access the values of the arguments
stationcode = args.station
stationsfile = args.stationsfile
fluxdir = args.fluxdir
filepath = args.fpdir
bgfilepath = args.bgdir
outpath = args.outdir
stilt_rundir = args.stiltdir
obspack_filepath = args.obspack_path
start_date = args.start_date
end_date = args.end_date
nmonths_split = args.nmonths_split
perturbation=args.perturbation
verbose=args.verbose
sum_vars=args.sum_variables

## Define ObsPack DOI
#DOI = '10.18160/PEKQ-M4T1'

# Provide extent
ll_lat = args.lat_ll
ur_lat = args.lat_ur
ll_lon = args.lon_ll
ur_lon = args.lon_ur

# Provide lat/lon step
lat_step = args.lat_step
lon_step = args.lon_step

# Define simulation length
sim_length = args.sim_len

# Define number of particles
npars = args.npars

# Define stationlist
stationsfile = pd.read_csv(stationsfile, header=0)
stationslist = stationsfile['code']

# CTE-HR variable list to loop over later
if perturbation != None:
    fluxtype='PARIS'
    fluxvarnamelist = ['flux_ff_exchange_prior', 'flux_ocean_exchange_prior', 'flux_fire_exchange_prior',
            'flux_bio_exchange_prior']
else:
    fluxtype='CTEHR'
    fluxvarnamelist = ['nep', 'fire', 'ocean', 'combustion']

## Load ObsPack
#coll = collection.get(DOI)
#obspack_name = coll.title.split(';')[1].strip()
obspack_name = obspack_filepath.split('/')[-1]

########################
##### DON'T CHANGE #####
########################
# Define lat/lon grid
# here, the extent is given in corner coordinates, but the lats and lons are defined in the center of the grid cells!
ctehr_bbox = [[ll_lon, ur_lon], [ll_lat, ur_lat]]
lats, lons = coordinate_list(ll_lat+0.5*lat_step, ur_lat+0.5*lat_step, ll_lon+0.5*lon_step, ur_lon+0.5*lon_step, lat_step, lon_step)

# Create directory if it does not exist
if not os.path.exists(outpath):
    os.makedirs(outpath)

# Get dict of ObsPack files that are within the CTE-HR domain, have the highest inlet height, and contain data within the given time range.
# This contains both the filenames of the ObsPacks as well as the Dobj objects.
obspack_list = filter_obspacks(zip_filepath = obspack_filepath, extent = ctehr_bbox, start_date = start_date, end_date = end_date)

def process_station(date_pairs):
    return fun(date_pair=date_pairs)

if __name__ == "__main__":
    if ((nmonths_split != None and nmonths_split > 1) and (start_date != None and end_date != None)):
        print("Multiprocessing, running script for the whole time period in " + str(nmonths_split) + " pieces")

        # Create list of dates
        datelist = list(date_range(start_date, end_date, nmonths_split))
        
        # Create pairs of consecutive dates
        date_pairs = [(datelist[i], datelist[i + 1]) for i in range(len(datelist) - 1)]
        print("All Date pairs: " + str(date_pairs))

        fun = partial(
            main,
            filepath=filepath,
            fluxdir=fluxdir,
            bgfilepath=bgfilepath,
            outpath=outpath,
            stilt_rundir=stilt_rundir,
            perturbationcode = perturbation,
            fluxvarnamelist=fluxvarnamelist,
            lats=lats,
            lons=lons,
            sim_length=sim_length,
            npars=npars,
            obspack_list=obspack_list,
            stationsfile=stationsfile,
            stationcode=stationcode,
            fluxtype=fluxtype,
            sum_vars=sum_vars,
            start_date=start_date,
            end_date=end_date,
            verbose=verbose
            )

        # Use the separate function in the map call
        with Pool(processes=nmonths_split) as p:
            p.map(process_station, date_pairs, chunksize=1)
            p.close()
            p.join()
        
        logging.info('!!! Check if all intermediate files have been created !!!')
        
        # Combine all partial netCDF ObsPack files and delete the intermediate files
        obspack_orig = [s for s in obspack_list if stationcode.lower() in s.lower()][0]
        partial_files= glob.glob(outpath + '/pseudo_co2_' + stationcode.lower() + '*.nc')
        filelist = [xr.open_dataset(file) for file in partial_files]

        logging.info('Combining the following files: ' + str(partial_files))

        # Define output filestring
        complete_filestr = outpath + 'pseudo_' + obspack_orig.split('/')[-1]

        # Merge all files
        combined_dataset = xr.merge(filelist, join='outer')
        combined_dataset.to_netcdf(complete_filestr)
        
        # Delete intermediate files
        for file in partial_files:
            os.remove(file)

    elif ((nmonths_split == None or nmonths_split <= 1) and (start_date != None and end_date != None)):
        print("No multiprocessing, running script for the whole time period at once")

        # Create pairs of consecutive dates
        date_pair = [start_date, end_date]
        print("All Date pairs: " + str(date_pair))

        main(
            filepath=filepath,
            fluxdir=fluxdir,
            bgfilepath=bgfilepath,
            outpath=outpath,
            stilt_rundir=stilt_rundir,
            perturbationcode = perturbation,
            fluxvarnamelist=fluxvarnamelist,
            lats=lats,
            lons=lons,
            sim_length=sim_length,
            npars=npars,
            obspack_list=obspack_list,
            stationsfile=stationsfile,
            stationcode=stationcode,
            fluxtype=fluxtype,
            sum_vars=sum_vars,
            start_date=start_date,
            end_date=end_date,
            date_pair=date_pair,
            verbose=verbose
            )
        
    else:
        print("Please provide a start and end date in the format 'YYYY-MM-DD'")

    # Log total time
    print("Flux multiplication finished for station " + stationcode + "!")