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

######################
##### USER INPUT #####
######################
## Define ObsPack DOI
#DOI = '10.18160/PEKQ-M4T1'

# Provide extent
ll_lon = -15
ll_lat = 33.0
ur_lon = 35.0
ur_lat = 72.0

# Provide lat/lon step
lon_step, lat_step = 0.2, 0.1

# Provide start date and end date
start_date = "2021-01-01"
end_date = "2022-01-01"

# Define simulation length
sim_length = 240

# Define stationlist
stationsfile = pd.read_csv('/projects/0/ctdas/PARIS/DATA/stationfile_all.csv', header=0)
stationslist = stationsfile['code']

# Define footprint, background, flux and output filepaths
fluxdir = '/projects/0/ctdas/awoude/NRT/ICOS_OUTPUT/'
#fluxdir = '/gpfs/scratch1/shared/dkivits/fluxes/CTE-HR/modified/'
filepath = '/gpfs/scratch1/shared/dkivits/STILT/particle_sensitivity/'
bgfilepath = '/projects/0/ctdas/PARIS/DATA/background/STILT/'
outpath = '/projects/0/ctdas/PARIS/DATA/obspacks/'
obspack_filepath = '/projects/0/ctdas/PARIS/DATA/obs/obspack_co2_466_GVeu_20230913.zip'

#################
##### TEST ######
#################
# Select station
stationslist = ['CBW']

#####################
##### END TEST ######
#####################

########################
##### DON'T CHANGE #####
########################
# Define lat/lon grid
# here, the extent is given in corner coordinates, but the lats and lons are defined in the center of the grid cells!
ctehr_bbox = [[ll_lon, ur_lon], [ll_lat, ur_lat]]
lats, lons = coordinate_list(ll_lat+0.5*lat_step, ur_lat+0.5*lat_step, ll_lon+0.5*lon_step, ur_lon+0.5*lon_step, lat_step, lon_step)

# CTE-HR variable list to loop over later
fluxfilenamelist = ['nep', 'fire', 'ocean', 'anthropogenic']
fluxvarnamelist = ['nep', 'fire', 'ocean', 'combustion']

# Create directory if it does not exist
if not os.path.exists(outpath):
    os.makedirs(outpath)

# Loop over all stations
for stationcode in stationslist:
    pseudo_df, mixed_df = compare_ens_members(fluxdir = fluxdir, filepath = filepath, bgfilepath = bgfilepath, 
                    fluxfilenamelist = fluxfilenamelist, 
                    fluxvarnamelist = fluxvarnamelist, lats = lats, lons = lons, sim_length = sim_length,
                    stationsfile = stationsfile, stationcode = stationcode)

    #pseudo_df.to_csv('/projects/0/ctdas/PARIS/DATA/pseudo_ens' + stationcode +'.csv')
    #mixed_df.to_csv('/projects/0/ctdas/PARIS/DATA/mixed_ens' + stationcode +'.csv')

#plot_ens_members_fromcsv(pseudo_csv = '/projects/0/ctdas/PARIS/DATA/pseudo_ensCBW.csv',
#                         mixed_csv = '/projects/0/ctdas/PARIS/DATA/mixed_ensCBW.csv',
#                         save = True)

# fun = partial(main, fluxdir = fluxdir, filepath = filepath, bgfilepath = bgfilepath, 
#             outpath = outpath, fluxfilenamelist = fluxfilenamelist, 
#             fluxvarnamelist = fluxvarnamelist, lats = lats, lons = lons, sim_length = sim_length,
#             npars = npars, obspack_dict = obspack_list, obspack_name = obspack_name)

# with Pool(16) as p:
#     p.map(fun, stationslist)
