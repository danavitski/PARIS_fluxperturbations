import netCDF4 as nc
import matplotlib.pyplot as plt
import pandas as pd

fluxpath = '/projects/0/ctdas/PARIS/CTE-HR/output/20210201/'

ff_emis = nc.Dataset(fluxpath + 'ff_emissions_CO2.nc','r')
ff_reg = nc.Dataset(fluxpath + 'regional.anthropogenic.nc','r')
bio_reg = nc.Dataset(fluxpath + 'regional.bio.nc','r')
nep_reg = nc.Dataset(fluxpath + 'regional.nep.nc','r')
mask = nc.Dataset('/projects/0/ctdas/NRT/data/europe_countrymask2.nc', 'r')

ff_emis_trans = ff_emis.variables['F_On-road'][0,:,:].data
GER_mask = mask.variables['DEU'][:,:].data

GER_ff_emis_trans = GER_mask * ff_emis_trans * 0.5
plt.figure(figsize = (16,9))
plt.imshow(GER_ff_emis_trans)
plt.colorbar()
