import netCDF4 as nc
import numpy as np
#import sys
#from Experiments.scripts.functions.functions import get_lu
#sys.path.insert(0, '/projects/0/ctdas/PARIS/Experiments/scripts/functions/')
from functions.funs import *
import os
import shutil
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt

os.chdir('/projects/0/ctdas/PARIS/Experiments/scripts/yr1/DFIN/')

experimentcode = 'DFIN'
scriptpath = '/projects/0/ctdas/PARIS/Experiments/scripts/yr1/' + experimentcode
plotpath = '/projects/0/ctdas/PARIS/CTE-HR/analysis/plots/' + experimentcode + '/'

inpath = '/projects/0/ctdas/PARIS/CTE-HR/PARIS_OUTPUT/'
paris_perturbation_path = inpath + experimentcode + '/'
paris_perturbation_file = paris_perturbation_path + 'paris_ctehr_perturbedflux_yr1_' + experimentcode + '.nc'
paris_base_path = inpath + 'paris_input_s_u_d9.nc'

# If the target directory does not yet exist, create it
if not os.path.exists(paris_perturbation_path):
    os.mkdir(paris_perturbation_path)

shutil.copyfile(paris_base_path, paris_perturbation_file)

paris_base = nc.Dataset(paris_base_path, 'r', format='NETCDF3_CLASSSIC')
paris_perturbation = nc.Dataset(paris_perturbation_file, 'r+', format='NETCDF3_CLASSSIC')
mask_01_02 = nc.Dataset('/projects/0/ctdas/PARIS/Experiments/landmask/paris_countrymask_0.2x0.1deg_2D.nc', 'r', format='NETCDF3_CLASSIC')
#mask_005 = nc.Dataset('/projects/0/ctdas/PARIS/Experiments/landmask/paris_countrymask_0.05deg_2D.nc', 'r', format='NETCDF3_CLASSIC')

lon_bounds = [paris_base.variables['longitude'][0], paris_base.variables['longitude'][-1]]
lat_bounds = [paris_base.variables['latitude'][0], paris_base.variables['latitude'][-1]]
res_lon, res_lat = round((paris_base.variables['longitude'][1]-paris_base.variables['longitude'][0]),1), round((paris_base.variables['latitude'][1] - paris_base.variables['latitude'][0]),1)
nx = int((lon_bounds[1] + res_lon - lon_bounds[0])/res_lon)
ny = int((lat_bounds[1] + res_lat - lat_bounds[0])/res_lat)
#nx = paris_base.variables['F_On-road'][:,:,:].shape[-1]
#ny = paris_base.variables['F_On-road'][:,:,:].shape[-2]
#time_list = pd.date_range(datetime.datetime(2021,1,1,0,0,0), datetime.datetime(2021,1,2,0,0,0), freq='1H') 
#np.arange(datetime.datetime(2021,1,1,0,0,0), datetime.datetime(2022,1,1,0,0,0), datetime.timedelta(months=+1))
#lon_list = np.arange(lon_bounds[0],lon_bounds[1], res_lon)
#lat_list = np.arange(lat_bounds[0],lat_bounds[1], res_lat)
#month_list = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

## EXPERIMENT-SPECIFIC PART
# EXTRACT FINLAND FROM COUNTRY MASK
FIN_mask = mask_01_02.variables['FIN'][:,:].data

# EXTRACT TRANSPORT EMISSIONS
nep = paris_base.variables['flux_bio_exchange_prior'][:,:,:]

# GET LAND USE
lu = get_lu(nep)

# CREATE A FOREST FILTER
filter = ((lu == 2) | (lu == 5) | (lu == 8))


# LOOP OVER TIME
for time in range(0, len(paris_base.variables['time'])):
    print('Busy with ... ' + str(time))
    nep_array = nep[time,:,:]
    masked_array = nep_array * FIN_mask * 2
    masked_array[~filter] = 0
    nep_array_FIN_forest_scaled = nep_array + masked_array

    # SAVE FLUX PERTRUBATION TO NEWLY COPIED FLUX SET
    paris_perturbation.variables['flux_bio_exchange_prior'][time,:,:] = nep_array_FIN_forest_scaled

paris_base.close()
paris_perturbation.close()

"""

# PLOT
# If the target directory does not yet exist, create it
if not os.path.exists(plotpath):
    os.mkdir(plotpath)

side_by_side = False
time_list = pd.date_range(dt.datetime(2021,1,1,0,0,0), dt.datetime(2021,1,2,0,0,0), freq='1H') 
for time in range(0, len(time_list)):
        print('Busy with ... ' + str(time))
        
        nep_array = nep[time,:,:]
        masked_array = nep_array * FIN_mask * 2
        masked_array[~filter] = 0
        nep_array_FIN_forest_scaled = nep_array + masked_array
        
        dif = nep_array_FIN_forest_scaled - nep_array
        
        if np.array_equal(nep_array_FIN_forest_scaled, nep_array):
            print('Arrays are equal')
        else:
            print('Arrays are not equal')

        if side_by_side == True:
            fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols = 3, figsize = (16, 9))
            base = ax1.imshow(nep_array, origin='lower')#, vmin = 0, vmax = 5e-6)
            half = ax2.imshow(nep_array_FIN_forest_scaled, origin='lower')#, vmin = 0, vmax = 5e-6)
            dif = ax3.imshow(nep_array - nep_array_FIN_forest_scaled, origin='lower')#, vmin = 0, vmax = 5e-6)
            fig.colorbar(dif, orientation='horizontal', pad = 0.05, ax = [ax1, ax2, ax3])
            
        else:
            fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (16, 9))
            dif = ax.imshow(nep_array - nep_array_FIN_forest_scaled, origin='lower')#, vmin = 0, vmax = 1e-6)
            fig.colorbar(dif, location='right', pad = 0.05, ax = ax)

        fig.suptitle(str(time_list[time]))
        plt.savefig(plotpath + experimentcode + '_' + str(time) + '.png', bbox_inches = 'tight')
        plt.show()

"""