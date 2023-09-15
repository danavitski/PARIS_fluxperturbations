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
import tqdm

# Define variable list to loop over later
fluxfilenamelist = ['nep', 'fire', 'ocean', 'anthropogenic']
fluxvarnamelist = ['nep', 'fire', 'ocean', 'combustion']
fluxdir = '/projects/0/ctdas/awoude/NRT/ICOS_OUTPUT/'
#fluxdir = '/gpfs/scratch1/shared/dkivits/fluxes/CTE-HR/modified/'

# Open arbitrary ObsPack file to insert results into
ObsPack_file = '/projects/0/ctdas/PARIS/DATA/obs/co2_bik_tower-insitu_45_allvalid-180magl.nc'
ObsPack_outputfile = os.path.dirname(ObsPack_file) + '/pseudo_' + os.path.basename(ObsPack_file)

#if not os.path.exists(ObsPack_outputfile):
shutil.copyfile(ObsPack_file, ObsPack_outputfile)

# Open ObsPack file and create necessary variables
ObsPack = nc.Dataset(ObsPack_outputfile, 'a', )
ObsPack.createVariable('cte_contribution', 'f4', ('time',))
ObsPack.createVariable('background', 'f4', ('time',))
ObsPack.createVariable('pseudo_observation', 'f4', ('time',))

# Define start time of ObsPack file
obspack_basetime = datetime(1970,1,1,0,0,0)
obspack_starttime = obspack_basetime + timedelta(seconds=int(ObsPack.variables['time'][0]))

# Define stationlist
stationsfile = pd.read_csv('/projects/0/ctdas/PARIS/DATA/stationlist_highestalt.csv',header=0)
stationslist = stationsfile['code']

# Define simulation length
sim_length = 240

# Define number of particles
npars = 100

# Define lat/lon grid
lats, lons = coordinate_list(33.05, 72.05, -14.9, 35.1, 0.1, 0.2)

# Define footprint file list
filepath = '/gpfs/scratch1/shared/dkivits/STILT/footprints/'
bgfilepath = '/projects/0/ctdas/PARIS/DATA/background/STILT/'
outpath = '/gpfs/scratch1/shared/dkivits/STILT/footprints_multiplied/'

# Create directory if it does not exist
if not os.path.exists(outpath):
    os.makedirs(outpath)

# Select station
stationslist = ['CBW']
#station_loc = [13.4189, 56.0976]

# Time the script
start_time = time.time()

# Loop over all stations
for station in stationslist:

    # Get list of footprint files for station
    sparse_files = sorted(glob.glob(filepath + 'footprint_' + station + '*.nc'))

    # Extract all unique months between start and end time
    mons = footprint_unique_months(sparse_files, sim_length)

    # Extract all times in footprint files
    # times = footprint_hours(sparse_files, sim_length)

    # Only once per station, create list of all CTE-HR flux files
    fluxstring = find_fluxfiles(fluxdir = fluxdir, variablelist_files = fluxfilenamelist, months = mons)

    # Check if selection went right
    print(fluxstring)

    # Open all files in fluxstring as xr_mfdataset, and add variables in variablelist
    cte_ds = open_multiple_fluxfiles(fluxstring, variablelist_vars = fluxvarnamelist)

    # Get 3D station location
    lat,lon,agl = get_3d_station_location(station, stationsfile)    
 
    # Loop over all footprints files
    create_obs_sim_dict(fp_filelist = sparse_files, flux_dataset = cte_ds,
                        lats = lats, lons = lons, list_of_mons = mons, RDatapath = filepath, 
                        bgpath = bgfilepath, npars = npars, station_lat = lat, station_lon = lon, 
                        station_agl = agl, stationname = station)
    
# Print total time
print("--- %s seconds ---" % (time.time() - start_time))

"""
start_df = pd.read_csv(sparse_files[0], header=0)
timelist = start_df['times'].unique()

fp_curtime_str = fp_curtime.strftime("%Yx%mx%dx%H")
fp_file = glob.glob(filepath + 'footprint_' + station + '_' + fp_curtime_str + '*.csv')[0]

# Loop over footprint files
for i in range(0,len(sparse_files)):
    # Get start and end time of footprint
    file = sparse_files[i]
    timestr = file[-38:-25]

    df = pd.read_csv(file, header=0)
    timelist = df['times'].unique()

    fp_starttime = datetime.strptime(timestr, '%Yx%mx%dx%H')
    fp_endtime = fp_starttime - timedelta(hours=len(timelist))

    
        fluxfile = [s for s in fluxstring if var in s and curmon in s][0]
        print(fluxfile)

        # Open CTE-HR flux file and extract time
        #CTEHR_nc = nc.Dataset(file, 'r')
        #timevar = CTEHR_nc.variables['Time'][:]

        #curtime = basedate + timedelta(seconds=int(timevar[0]))

    # Loop over all times in footprint file
    for y in range(0,len(timelist)):
        curtime = timelist[y]
        subdf = df[df['times'] == curtime]
        
        # Create datetime object of current time
        fp_curtime = fp_starttime - timedelta(hours=y)
        curmon = fp_curtime.strftime("%Y%m")
"""


"""
print(np.shape(array))
print(array[:,:,20])

# test plot footprint
fig = plt.figure(1, figsize=(16,9))
ax = plt.axes(projection = ccrs.PlateCarree())
ax.add_feature(cf.COASTLINE)
ax.add_feature(cf.BORDERS)

im = ax.imshow(array[:,:,47], cmap='Reds', extent=[-15,35,33,72])

# Open CTE-HR flux file, extract time to loop over at later stage
    CTEHR_nc = nc.Dataset(file, 'r')
    timevar = CTEHR_nc.variables['Time'][:]
    curtime = basedate + timedelta(seconds=int(timevar[0]))
"""