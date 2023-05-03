import netCDF4 as nc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import datetime

#fluxpath = '/projects/0/ctdas/PARIS/CTE-HR/output/20210201/'
fluxpath = '/home/dkivits/PARIS/CTE-HR/output/'
fluxpath_2021 = '/projects/0/ctdas/PARIS/CTE-HR/output/2021/'
cdlpath = '/home/dkivits/PARIS/NetCDF_template/paris_ctehr_template.cdl'

ff_emis = nc.Dataset(fluxpath + 'ff_emissions_CO2.nc','r')
ff_reg = nc.Dataset(fluxpath + 'regional.anthropogenic.nc','r')
nep_reg = nc.Dataset(fluxpath + 'regional.nep.nc','r')
#fire_reg = nc.Dataset(fluxpath + 'regional.fire.nc','r')
#ocean_reg = nc.Dataset(fluxpath + 'regional.ocean.nc','r')

ff_emis_2021 = nc.Dataset(fluxpath_2021 + 'ff_emissions_CO2.nc','r')
ff_reg_2021 = nc.Dataset(fluxpath_2021 + 'regional.anthropogenic.nc','r')
nep_reg_2021 = nc.Dataset(fluxpath_2021 + 'regional.nep.nc','r')
#fire_reg_2021 = nc.Dataset(fluxpath_2021 + 'regional.fire.nc','r')
#ocean_reg_2021 = nc.Dataset(fluxpath_2021 + 'regional.ocean.nc','r')

#mask_01_02 = nc.Dataset('/projects/0/ctdas/NRT/data/europe_countrymask2.nc', 'r', format='NETCDF3_CLASSIC')
#mask_005 = nc.Dataset('/projects/0/ctdas/PARIS/landmask/europe_countrymasks_0.05deg_2D.nc', 'r', format='NETCDF3_CLASSIC')
mask_01_02 = nc.Dataset('/home/dkivits/PARIS/Experiments/landmask/europe_countrymasks_0.2x0.1deg_2D.nc', 'r', format='NETCDF3_CLASSIC')
mask_005 = nc.Dataset('/home/dkivits/PARIS/Experiments/landmask/europe_countrymasks_0.05deg_2D.nc', 'r', format='NETCDF3_CLASSIC')

lon_bounds = [-15,35]
lat_bounds = [33,72]
res_lon, res_lat = 0.2, 0.1
nx = int((lon_bounds[1] - lon_bounds[0])/res_lon)
ny = int((lat_bounds[1] - lat_bounds[0])/res_lat)
time_list = np.arange(datetime.datetime(2021,1,1,0,0,0), datetime.datetime(2022,1,1,0,0,0), datetime.timedelta(months=1))
lon_list = np.arange(lon_bounds[0],lon_bounds[1], res_lon)
lat_list = np.arange(lat_bounds[0],lat_bounds[1], res_lat)

## LOAD TEMPLATE FILE
template = nc.fromcdl(cdlfilename = cdlpath, filename = 'paris_ctehr_HGER.nc', mode = 'a', format = 'NETCDF3_CLASSIC')

## BASIC VARIABLES (TIME, LAT, LON)
for time in time_list:
        template.variables['time'][:] = time_list[time]

for lon in lon_list:
        template.variables['longitude'][:] = lon_list[time]

for lat in lat_list:
        template.variables['latitude'][:] = lat_list[time]

## COPY EXISTING CTE-HR FLUXES
template.variables['flux_ff_exchange'][:,:,:] = ff_emis_2021.variables['anthropogenic'][:,:,:]
#template.variables['flux_ocean_exchange'][:,:,:] = ocean_reg_2021.variables['ocean'][:,:,:]
#template.variables['flux_fire_exchange'][:,:,:] = fire_reg_2021.variables['fire'][:,:,:]
template.variables['flux_bio_exchange_prior'][:,:,:] = nep_reg_2021.variables['nep'][:,:,:]

## EXPERIMENT-SPECIFIC PART
# EXTRACT GERMANY FROM COUNTRY MASK
GER_mask = mask_01_02.variables['DEU'][:,:].data
GER_mask_bool = np.where(GER_mask != 0, 1, 0)
ones_array = np.ones((ny,nx))
ones_array[GER_mask_bool == 1] = 0.5 * GER_mask[GER_mask_bool == 1]

# EXTRACT TRANSPORT EMISSIONS
ff_emis_trans = ff_emis.variables['F_On-road'][:,:,:]
for time in range(0, template.variables['time'][:]):
        print('Busy with ... ' + str(time))
        GER_ff_emis_trans_scaled = ff_emis_trans[time] * ones_array
        template.variables['flux_ff_exchange'][time,:,:] = GER_ff_emis_trans_scaled

plt.figure(figsize = (16,9))
plt.imshow(GER_ff_emis_trans_scaled, origin='lower')
plt.colorbar()