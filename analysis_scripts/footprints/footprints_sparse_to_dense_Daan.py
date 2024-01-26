# DK, 09-01-2024
# Adapted from Auke van der Woude

# Import necessary modules
import pandas as pd
import numpy as np
import glob
import netCDF4 as nc
from datetime import datetime
import pandas as pd
import logging
from plot_2D_funcs import plot_on_worldmap_v2
import os


def filter_files_by_date(file_list, start_date, end_date):
    """ Function to filter a list of files by a certain date range. 
    """
    filtered_files = []

    date_format = "%Y-%m-%d"  # adjust the format based on your actual filenames

    start_datetime = datetime.strptime(start_date, date_format)
    end_datetime = datetime.strptime(end_date, date_format)

    for file_name in file_list:
        date_str = file_name.split("_")[3][:13]  # adjust the index based on your actual filenames
        file_datetime = datetime.strptime(date_str, "%Yx%mx%dx%H")

        if start_datetime <= file_datetime <= end_datetime:
            filtered_files.append(file_name)

    return filtered_files

# Set logging options
logging.basicConfig(level=logging.INFO, format=' [%(levelname)-7s] (%(asctime)s) py-%(module)-20s : %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Define the stations
stationlist = ['RGL']

# Define the time range
timerange = [datetime(2021, 4, 1, 0, 0), datetime(2021, 10, 1, 0, 0)]
timelabels = [timerange[0].strftime('%Y-%m-%d'), timerange[1].strftime('%Y-%m-%d')]

# Define plotting extent
lat_ll = 35
lat_ur = 75
lon_ll = -15
lon_ur = 30
plotting_extent = [lon_ll, lon_ur, lat_ll, lat_ur]

# Define colormap
cmap='Reds'

# Define filepaths
inpath = '/projects/0/ctdas/PARIS/DATA/footprints/wur/PARIS_recompile/'
stilt_rundir = '/projects/0/ctdas/PARIS/STILT_Model/STILT_Exe/'   
stationfile= '/projects/0/ctdas/PARIS/DATA/stationfile_uob_wur_ganesan_manning.csv'

# Initialise the lats and lons of the 3d grid
lats = np.arange(33.05, 71.95, 0.1, dtype=np.float64)
lons = np.arange(-14.9, 35.1, 0.2, dtype=np.float64)

lats_plot = np.arange(33, 72, 0.1, dtype=np.float64)
lons_plot = np.arange(-15, 35, 0.2, dtype=np.float64)

# Define functions to convert lat and lon to index (speedy)
lon_to_idx = lambda lons_: np.round((lons_ - lons.min()) / int(round(len(lons) * np.diff(lons).mean(), 0)) * len(lons), 0).astype(int)
lat_to_idx = lambda lats_: np.round((lats_ - lats.min()) / int(round(len(lats) * np.diff(lats).mean(), 0)) * len(lats), 0).astype(int)

sum_array = np.zeros((len(lats), len(lons)))

for station in stationlist:
    # Print status
    logging.info(f'Processing station {station}')

    full_filelist = glob.glob(inpath + 'footprint_' + station + '*.nc')
    filtered_files = filter_files_by_date(full_filelist, timelabels[0], timelabels[1])

    # Create empty sum_array
    sum_array_perstation = np.zeros((len(lats), len(lons)))

    # Import list of missing footprint files
    missing_fp_filestr = stilt_rundir + station + '/missing.footprints'
    if os.path.isfile(missing_fp_filestr):
        missing_fp_file = pd.read_csv(missing_fp_filestr)
        missing_fp_filelist = list(missing_fp_file['ident'])

    # Loop over files
    for file in sorted(filtered_files)[0:10]:
        # Print status
        logging.info(f'Processing {file}')
        
        # Get the ident from the filename
        ident = os.path.splitext(os.path.basename(file).split(sep='_')[-1])[0]

        # Check if the file is in the missing footprint file list
        if os.path.exists(missing_fp_filestr)==False or ident not in missing_fp_filelist:
            
            # Read in dataset
            with nc.Dataset(file) as ds:
                times = np.array(ds['Time'][:])

                # We need to know how many times we have, because that is how much influence-times we have
                timelist = np.unique(times)

                # Read the footprint
                fp = np.array(ds['Influence'][:])
                
                # Initialise the dense grid.
                array = np.zeros((len(lats), len(lons), len(timelist)))

                # Convert the lats and lons to indices
                lons_ds = np.array(ds['Longitude'][:])
                lats_ds = np.array(ds['Latitude'][:])
                lons_idx = lon_to_idx(lons_ds)
                lats_idx = lat_to_idx(lats_ds)

                # Loop over all times (this could probably be more efficient)
                for y in range(0,len(timelist)):

                    # Select the lats, lons and sensitivity for this time
                    curtime = timelist[y]
                    sublons = lons_idx[times == curtime]
                    sublats = lats_idx[times == curtime]
                    subfp = fp[times == curtime]

                    # Fill the dense matrix with the correct influences.
                    array[sublats, sublons, y] = subfp
                
                # Sum the array over the time dimension
                array = array.sum(axis=2)

            # Add the array to the sum_array_perstation
            sum_array_perstation += array

        else:
            logging.info(f'Footprint is empty and no surface exchange with particles is recorded, taking background concentration at station location as pseudo observation.')

    # Add the sum_array_perstation to the sum_array
    sum_array += sum_array_perstation

    # Change zeroes in array to NaNs
    sum_array[sum_array == 0] = np.nan
    sum_array[sum_array <= 2e-5] = np.nan

# Plot sum_array for current station
#im = ax.imshow(sum_array, cmap=cmap, extent=plotting_extent, norm=LogNorm(vmin=0.01, vmax=1))
    
#fpath = f'/projects/0/ctdas/PARIS/DATA/footprints/plots/integrated_fps/integratedfp_{stationlist[0]}_{stationlist[1]}_{timelabels[0]}_{timelabels[1]}.png'
fpath = f'/projects/0/ctdas/PARIS/DATA/footprints/plots/integrated_fps/integratedfp_{stationlist[0]}_{timelabels[0]}_{timelabels[1]}_wide.png'

plot_on_worldmap_v2(sum_array, lons_plot, lats_plot, unit='Integrated surface influence\n [$ppm μmol^{-1} ms^{-2} s^{-1}$]', minmax=[1e-5, 1e-2], extent=plotting_extent, save_plot=True, stations = stationlist, stationfile = stationfile, fpath=fpath)