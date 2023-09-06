# IMPORT NECESSARY PACKAGES
import netCDF4 as nc
import numpy as np
import pandas as pd
import datetime as dt
#import sys
#from Experiments.scripts.functions.functions import get_lu
#sys.path.insert(0, '/projects/0/ctdas/PARIS/Experiments/scripts/functions/')
from functions.funs import *
import os
import shutil

os.chdir('/projects/0/ctdas/PARIS/Experiments/scripts/yr1/ATEN/')

experimentcode = 'ATEN'
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
#lon_list = np.arange(lon_bounds[0],lon_bounds[1], res_lon)
#lat_list = np.arange(lat_bounds[0],lat_bounds[1], res_lat)
#month_list = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

## EXPERIMENT-SPECIFIC PART
ff_list = [
    'A_Public_power',
    'B_Industry',
    'C_Other_stationary_combustion_consumer',
    'F_On-road',
    'H_Aviation',
    'I_Off-road',
    'G_Shipping'
]

# EXTRACT TRANSPORT EMISSIONS FROM BASE FILE
ff_emis_total = paris_base.variables['flux_ff_exchange_prior'][:,:,:]

# SCALE DOWN TOTAL EMISSIONS BY 10%
ff_emis_total_scaled = ff_emis_total * 0.9

# SAVE FLUX PERTRUBATION TO NEWLY COPIED FLUX SET
paris_perturbation.variables['flux_ff_exchange_prior'][:] = ff_emis_total_scaled

# RE-CALCULATE TOTAL EMISSIONS
dummy = np.ones((len(paris_base.variables['time']),ny,nx))
for var in ff_list:
    dummy = dummy + paris_perturbation.variables[var][:,:,:]

# SAVE TOTAL EMISSIONS TO NEWLY COPIED FLUX SET
paris_perturbation.variables['combustion'][:,:,:] = dummy

# RE-CALCULATE TOTAL EMISSIONS INCLUDING CEMENT PRODUCTION
paris_perturbation.variables['flux_ff_exchange_prior'][:] = paris_perturbation.variables['combustion'][:] + paris_perturbation.variables['cement'][:]

# CLOSE FILES
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
        
        ff_emis_total_scaled = ff_emis_total[time] * 0.9
        dif = ff_emis_total[time] - ff_emis_total_scaled
        
        if np.array_equal(ff_emis_total_scaled, ff_emis_total):
            print('Arrays are equal')
        else:
            print('Arrays are not equal')

        if side_by_side == True:
            fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols = 3, figsize = (16, 9))
            base = ax1.imshow(ff_emis_total[time,:,:], origin='lower')#, vmin = 0, vmax = 5e-6)
            half = ax2.imshow(ff_emis_total_scaled, origin='lower')#, vmin = 0, vmax = 5e-6)
            dif = ax3.imshow(ff_emis_total[time,:,:] - ff_emis_total_scaled, origin='lower')#, vmin = 0, vmax = 5e-6)
            fig.colorbar(dif, orientation='horizontal', pad = 0.05, ax = [ax1, ax2, ax3])
            
        else:
            fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (16, 9))
            dif = ax.imshow(ff_emis_total[time,:,:] - ff_emis_total_scaled, origin='lower', vmin = 0, vmax = 5e-6)
            fig.colorbar(dif, location='right', pad = 0.05, ax = ax)

        fig.suptitle(str(time_list[time]))
        plt.savefig(plotpath + experimentcode + '_' + str(time) + '.png', bbox_inches = 'tight')
        plt.show()
"""