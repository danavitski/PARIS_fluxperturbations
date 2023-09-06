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
fluxfilelist = ['nep', 'fire', 'ocean', 'anthropogenic']
fluxnamelist = ['nep', 'fire', 'ocean', 'combustion']
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
lats = list(np.round(np.arange(33.05, 72.05, 0.1),4))
lons = list(np.round(np.arange(-14.9, 35.1, 0.2),4))

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

    # Define time range and list of times
    timestr_start = sparse_files[0][-37:-24]
    timestr_end = sparse_files[-1][-37:-24]
    fp_range_start = datetime.strptime(timestr_start, '%Yx%mx%dx%H')- timedelta(hours=sim_length)
    fp_range_end = datetime.strptime(timestr_end, '%Yx%mx%dx%H') 
    times = pd.date_range(start=fp_range_start, end=fp_range_end, freq='H')

    # Extract all unique months between start and end time
    mons = pd.date_range(fp_range_start, fp_range_end, freq='D').strftime("%Y%m").unique().tolist()

    if stationslist.index(station) == 0:
        # Loop over all CTE-HR flux variables
        fluxstring = []
        for var in fluxfilelist:
            for mon in mons:
                # Define which CTE-HR flux files to loop over
                fluxstring += sorted(glob.glob(fluxdir + var + '.' + mon + '.nc'))

        # Check if selection went right
        print(fluxstring)

        # Open all files in fluxstring as xr_mfdataset
        cte_ds = xr.open_mfdataset(fluxstring, combine='by_coords')
        cte_ds = add_variables(cte_ds, fluxnamelist)

    # Get 3D station location
    lat = stationsfile[stationsfile['code']==station]['lat'].values[0]
    lon = stationsfile[stationsfile['code']==station]['lon'].values[0]
    agl = stationsfile[stationsfile['code']==station]['alt'].values[0]
    
    # Drop times that are not in the range of the footprint files
    for datetime in times:
        if datetime.hour not in range(fp_range_start.hour, (fp_range_end + timedelta(hours=1)).hour):
            times = times.drop(datetime)

    # Loop over all footprints files
    for i in tqdm.tqdm(range(0,len(sparse_files))):
           
        # Get start and end time of footprint
        sparse_file = sparse_files[i]
        fp_df_sparse = nc.Dataset(sparse_file, 'r')

        # Get start time of footprint
        timestr = sparse_file[-37:-24]
        fp_starttime = pd.to_datetime(timestr, format="%Yx%mx%dx%H")
        obspack_starttime = int((fp_starttime - obspack_basetime).total_seconds())

        # Calculate hours since start of CTE-HR flux file to use later for indexing in the for-loop
        time_diff = (fp_starttime - pd.to_datetime(mons[0], format="%Y%m"))
        hours_since_start_of_ds = time_diff.seconds // 3600 + time_diff.days * 24

        # Extract latitude indices from sparse footprint file using list comprehension
        lat_indices = [lats.index(i) for i in list(np.round(fp_df_sparse.variables['Latitude'][:].tolist(), 4))]
        lon_indices = [lons.index(i) for i in list(np.round(fp_df_sparse.variables['Longitude'][:].tolist(), 4))]
        
        # Select current footprint time
        hours_into_file = (hours_since_start_of_ds - fp_df_sparse.variables['Time'][:]).astype(int)

        # Convert the list of indices to xr.DataArrays, so that they can be used for efficient non-ortagonal indexing
        lat_indices = xr.DataArray(lat_indices, dims=['pos'])
        lon_indices = xr.DataArray(lon_indices, dims=['pos'])
        hours_into_file = xr.DataArray(hours_into_file, dims=['pos'])

        # Get last times of all particles
        last_times = get_last_times(t = fp_starttime, path = filepath, npars = npars, lat = lat, lon = lon, agl = agl)
        
        # Calculate background concentration
        bg = []
        for y, row in last_times.iterrows():
            bg.append(get_bg(fp_starttime, row, bgdir = bgfilepath))

        ObsPack.variables['cte_contribution'][i] = (cte_ds[hours_into_file,
                                                        lat_indices,
                                                        lon_indices] * fp_df_sparse.variables['Influence'][:]).sum()

        ObsPack.variables['background'][i] = np.mean(bg)
        ObsPack.variables['pseudo_observation'][i] = np.mean(bg)

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