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

# Load command-line arguments
parser = argparse.ArgumentParser(description='Flux - footprint multiplication script')

######################
##### USER INPUT #####
######################
# Define command-line arguments
parser.add_argument('--station', type=str, help='Code of station to run the script for')
parser.add_argument('--fluxtype', type=str, help='Switch to control what fluxes are used to calculate the pseudo observations. Current options are "CTE-HR" or "PARIS"')
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
parser.add_argument('--verbose', action="store_true", help='Controls whether the script should do logging (str)')

# Parse the command-line arguments
args = parser.parse_args()

# Access the values of the arguments
stationcode = args.station
fluxtype = args.fluxtype
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

# Time the script
start_time = time.time()

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
stationsfile = pd.read_csv('/projects/0/ctdas/PARIS/DATA/stationfile_all.csv', header=0)
stationslist = stationsfile['code']

# CTE-HR variable list to loop over later
if fluxtype=='PARIS':
    #fluxvarnamelist = ['A_Public_power','B_Industry','C_Other_stationary_combustion_consumer',
    #    'F_On-road','H_Aviation','I_Off-road','G_Shipping', 'cement', 'combustion',
    #    'flux_ff_exchange_prior', 'flux_ocean_exchange_prior', 'flux_fire_exchange_prior',
    #    'flux_bio_exchange_prior']
    fluxvarnamelist = ['flux_bio_exchange_prior']
elif fluxtype=='CTEHR':
    fluxvarnamelist = ['nep', 'fire', 'ocean', 'combustion']
else:
    print('Input fluxtype not recognized, please select one that is defined in the find_fluxfiles() function')


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
    
    if nmonths_split != None:

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
            stationsfile=stationsfile,
            stationcode=stationcode,
            fluxvarnamelist=fluxvarnamelist,
            perturbationcode = perturbation,
            lats=lats,
            lons=lons,
            sim_length=sim_length,
            npars=npars,
            obspack_list=obspack_list,
            fluxtype=fluxtype,
            verbose=verbose,
            )
        
        # Use the separate function in the map call
        with Pool(processes=nmonths_split) as p:
            p.map(process_station, date_pairs)
    else:    
        main(fluxdir = fluxdir, filepath = filepath, bgfilepath = bgfilepath, 
                outpath = outpath, stilt_rundir = stilt_rundir, perturbationcode = perturbation,
                fluxvarnamelist = fluxvarnamelist, lats = lats, lons = lons, start_date = start_date,
                end_date = end_date, sim_length = sim_length, npars = npars, obspack_list = obspack_list, 
                stationsfile = stationsfile, stationcode = stationcode, fluxtype = fluxtype, verbose = verbose)

        # Print total time
        print("Flux multiplication finished for station " + stationcode + "!")
        print("--- %s seconds ---" % (time.time() - start_time))