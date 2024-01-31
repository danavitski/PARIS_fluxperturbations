import sys
import netCDF4 as nc
import datetime
import numpy as np
import subprocess
from glob import glob
import platform
import os
import pandas as pd

os.chdir('/projects/0/ctdas/PARIS/Experiments/scripts/yr1/BASE/')

OUTPATH = '/projects/0/ctdas/PARIS/CTE-HR/PARIS_OUTPUT'
cdlpath = '/projects/0/ctdas/PARIS/cdl_template/paris_input.cdl'

countrylist = pd.read_csv('/projects/0/ctdas/PARIS/Experiments/landmask/country_list.csv', sep = ',')
outname = 'paris_input'
innames = [
            'regional.nep' ,
            'ff_emissions_CO2', 
            'regional.ocean', 
            'regional.fire'
            ]

varnames = [
            'nep', 
            'anthropogenic', 
            'ocean', 
            'fire'
            ]
outnames = [
            'flux_bio_exchange_prior',
            'flux_ff_exchange_prior',
            'flux_ocean_exchange_prior',
            'flux_fire_exchange_prior'
]

lon_bounds = [-14.9, 35.1]
lat_bounds = [33.05, 72.05]
res_lon, res_lat = 0.2, 0.1
nx = int((lon_bounds[1] + res_lon - lon_bounds[0])/res_lon)
ny = int((lat_bounds[1] + res_lat - lat_bounds[0])/res_lat)
lon_list = np.arange(lon_bounds[0],lon_bounds[1], res_lon)
lat_list = np.arange(lat_bounds[0],lat_bounds[1], res_lat)

paris_files=[]
for inname in innames:
    paris_files += sorted(glob(f'{OUTPATH}/*{inname}*.nc'))

print(paris_files)

## LOAD TEMPLATE FILE
template = nc.Dataset.fromcdl(cdlfilename = cdlpath, mode = 'a', ncfilename = OUTPATH + '/paris_input.nc')

## BASIC VARIABLES (TIME, LAT, LON) + FOSSIL FUEL EMISSIONS
for file in paris_files:
    print('Working on ... ' + file)
    paris_file = nc.Dataset(file)
    
    sectorname = list(paris_file.variables.keys())[-1]
    sectorindex = varnames.index(sectorname)
    new_varname = outnames[sectorindex]

    if template.variables[new_varname] == None:
        print('Requested variables already present, skipping variable extraction!')
    else:
        print('Extracting variable information from template file to output file')
        if 'ff_emissions_CO2' in file:
            for name, var in paris_file.variables.items():
                print('Working on: ' + str(name))
                if name != 'anthropogenic':
                    template.variables[name][:] = paris_file.variables[name][:]
        
        print('Working on: ' + str(new_varname))
        template.variables[new_varname][:] = paris_file.variables[sectorname][:]

## FILL IN COUNTRY MASK DATA
for i in range(0,len(countrylist)):
    template.variables['country_name'][i] = countrylist['name'][i]
    template.variables['country_abbrev'][i] = countrylist['code'][i]

template.close()
#os.remove(paris_files)