import netCDF4 as nc
import numpy as np
from plot_2D_funcs import plot_on_worldmap_v2

filepath = '/gpfs/scratch1/shared/dkivits/STILT/footprints/footprint_BIK_2023x01x19x10x53.23Nx023.01Ex00300.nc'
footname = filepath.split('/')[-1]
title = 'STILT ' + str(footname[0:-3])
foot = nc.Dataset(filepath)

lon = np.arange(-15,35,0.2)
lat = np.arange(33,72,0.1)
extent = [-15,35,33,72]
unit = 'ppm/micromol/m2s'

for i in range(0,len(foot['foot'])):
    fpath = '/projects/0/ctdas/PARIS/DATA/footprints/' + footname[0:-3] + '_' + str(i) + '.png'
    plot_on_worldmap_v2(foot['foot'][i], lon, lat, title, unit, extent, minmax=[1e-7, 1e-3], save_plot=True, fpath=fpath)