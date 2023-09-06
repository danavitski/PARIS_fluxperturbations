from datetime import datetime
from datetime import timedelta
from functions.background_functions import *
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import glob

# Define stationlist
stationfile = pd.read_csv('/projects/0/ctdas/PARIS/DATA/stationlist_highestalt.csv', header=0)

# Define number of particles
npars = 100
    
# Define basepath and get list of background and footprint files
basepath = '/gpfs/scratch1/shared/dkivits/STILT/footprints/'

# Loop over all stations
#stationslist = stationfile['code']
stationslist = ['CBW']

# Time the duration of the script
start_time = datetime.now()
for station in stationslist:
    
    # Get list of footprint files for station
    fpfiles = sorted(glob.glob(basepath + 'footprint_' + station + '_*.nc'))
    
    # Define time range and list of times
    timestr_start = fpfiles[0][-37:-24]
    timestr_end = fpfiles[-1][-37:-24]

    lat = stationfile[stationfile['code']==station]['lat'].values[0]
    lon = stationfile[stationfile['code']==station]['lon'].values[0]
    agl = stationfile[stationfile['code']==station]['alt'].values[0]

    fp_range_start = datetime.strptime(timestr_start, '%Yx%mx%dx%H')
    fp_range_end = datetime.strptime(timestr_end, '%Yx%mx%dx%H') 
    times = pd.date_range(start=fp_range_start, end=fp_range_end, freq='H')

    # Drop times that are not in the range of the footprint files
    for datetime in times:
        if datetime.hour not in range(fp_range_start.hour, (fp_range_end + timedelta(hours=1)).hour):
            times = times.drop(datetime)

    # Create dataframe to store background concentrations later
    bg_df = pd.DataFrame(index=times, columns=['Background'])

    # Loop over all times in timelist
    for t in tqdm.tqdm(times):
        
        # Get last times of all particles
        last_times = get_last_times(t, path = basepath, npars = npars, lat = lat, lon = lon, agl = agl)
        
        # Calculate background concentration
        bg = []
        for i, row in last_times.iterrows():
            bg.append(get_bg(t, row))
        bg_df.loc[t, 'Background'] = np.mean(bg)

        # Plot histogram of background concentrations
        #print(np.mean(bg))
        #plt.hist(np.squeeze(bg))   

    bg_df.to_csv('/projects/0/ctdas/PARIS/DATA/background/STILT/bg_extracted/' + station + '.csv')

# Print duration of script
print('Duration: {}'.format(datetime.now() - start_time))