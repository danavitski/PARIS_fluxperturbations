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
import datetime as datetime

experimentcode = 'DFIN'
scriptpath = '/projects/0/ctdas/PARIS/Experiments/scripts/yr1/' + experimentcode
plotpath = scriptpath + '/plots/'

inpath = '/projects/0/ctdas/PARIS/CTE-HR/PARIS_OUTPUT/'
paris_perturbation_path = inpath + experimentcode + '/'
paris_perturbation_file = paris_perturbation_path + 'paris_ctehr_perturbedflux_yr1_' + experimentcode + '.nc'
paris_base_path = inpath + 'paris_input_u_d9.nc'

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
res_lon, res_lat = paris_base.variables['longitude'][1]-paris_base.variables['longitude'][0], paris_base.variables['latitude'][1] - paris_base.variables['latitude'][0]
nx = int((lon_bounds[1] - lon_bounds[0])/res_lon)
ny = int((lat_bounds[1] - lat_bounds[0])/res_lat)
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
#FIN_mask_bool = np.where(FIN_mask != 0, 1, 0)

# EXTRACT TRANSPORT EMISSIONS
nep = paris_base.variables['flux_bio_exchange_prior'][:,:,:]
lu = get_lu(nep)

filter = ((lu == 2) | (lu == 5) | (lu == 8))
for time in range(0, len(paris_base.variables['time'])):
    print('Busy with ... ' + str(time))
    nep_array = nep[time,:,:]
    masked_array = nep_array * FIN_mask * 2
    masked_array[~filter] = 0
    nep_FIN_forest_scaled = nep_array + masked_array

    # SAVE FLUX PERTRUBATION TO NEWLY COPIED FLUX SET
    paris_perturbation.variables['flux_bio_exchange_prior'][time,:,:] = nep_FIN_forest_scaled

paris_base.close()
paris_perturbation.close()

"""
for time in range(0, len(time_list)):
        print('Busy with ... ' + str(time))
        fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (16, 9))
        #fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (16, 9))
        base = ax1.imshow(nep[time,:,:], origin='lower', vmin = 0, vmax = 5e-6)
        half = ax2.imshow(nep_FIN_forest_scaled, origin='lower', vmin = 0, vmax = 5e-6)
        #dif = ax.imshow(ff_emis_trans[time,:,:] - GER_ff_emis_trans_scaled, origin='lower', vmin = 0, vmax = 5e-6)
        fig.suptitle(str(time_list[time]))
        fig.colorbar(half, location='right')
        #fig.colorbar(dif, location='right')
        plt.savefig(plotpath + experimentcode + '_' + str(time) + '.png', bbox_inches = 'tight')
"""