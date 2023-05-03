import netCDF4 as nc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob

fluxpath = '/projects/0/ctdas/PARIS/CTE-HR/output/'
folderstr = sorted(glob.glob(fluxpath + '*/', recursive=True))
print(folderstr)

for folder in folderstr:
        print(folder)
        filestr = sorted(glob.glob(str(folder) + '*/'))
        print(filestr)
        for file in filestr:
                print(file)

ff_emis = nc.Dataset(fluxpath + 'ff_emissions_CO2.nc','r')
ff_reg = nc.Dataset(fluxpath + 'regional.anthropogenic.nc','r')
bio_reg = nc.Dataset(fluxpath + 'regional.bio.nc','r')
nep_reg = nc.Dataset(fluxpath + 'regional.nep.nc','r')
mask_01_02 = nc.Dataset('/projects/0/ctdas/NRT/data/europe_countrymask2.nc', 'r', format='NETCDF3_CLASSIC')
mask_005 = nc.Dataset('/projects/0/ctdas/PARIS/landmask/europe_countrymasks_0.05deg_2D.nc', 'r', format='NETCDF3_CLASSIC')

ff_emis_trans = ff_emis.variables['F_On-road'][:,:,:]
nx = ff_emis.variables['F_On-road'][:,:,:].shape[-1]
ny = ff_emis.variables['F_On-road'][:,:,:].shape[-2]

GER_mask = mask_01_02.variables['DEU'][:,:].data
GER_mask_bool = np.where(GER_mask != 0, 1, 0)

ones_array = np.ones((ny,nx))
ones_array[GER_mask_bool == 1] = 0.5 * GER_mask[GER_mask_bool == 1]

new_nc = nc.Dataset('germany_2x_ff_trans.nc', 'w', format = 'NETCDF3_CLASSIC')
new_nc.createDimension('longitude', nx)
new_nc.createDimension('latitude', ny)
new_nc.createDimension('time', None)

ff_emis_trans_scaled = new_nc.createVariable('flux_ff_exchange','f4',('time', 'latitude', 'longitude'), zlib=True)

for time in range(0, ff_emis_trans.shape[0]):
        print('Busy with ... ' + str(time))
        print(ff_emis_trans[time].shape)
        print(GER_mask.shape)

        GER_ff_emis_trans_scaled = ff_emis_trans[time] * ones_array
        ff_emis_trans_scaled[time,:,:] = GER_ff_emis_trans_scaled

plt.figure(figsize = (16,9))
plt.imshow(GER_ff_emis_trans_scaled, origin='lower')
plt.colorbar()