import sys
import netCDF4 as nc
import datetime
import numpy as np
import subprocess
from glob import glob
import platform
import os
import pandas as pd
from cdo import Cdo

cdo = Cdo()

paris_dir = '/projects/0/ctdas/PARIS/'
cte_dir = paris_dir + 'CTE-HR/'
template_dir = paris_dir + 'templates/'

inpath = cte_dir + 'output/'
outpath =  cte_dir + 'PARIS_OUTPUT/'
outname = outpath + 'paris_input.nc'

cdlpath = template_dir + 'cdl_template/paris_input.cdl'
landmaskpath = template_dir + 'landmask/country_list.csv'
landmaskfile = pd.read_csv(landmaskpath, sep = ',')

now = datetime.datetime.now()
DOI = ' https://doi.org/10.5281/zenodo.6477331'

comments_PARIS = { # Flux-specific comments for different files
            'flux_bio_exchange_prior': f'Net ecosystem productivity (gross primary production minus respiration). Positive fluxes are emissions, negative mean uptake. These fluxes are the result of the SiB4 (Version 4.2-COS, hash 1e29b25, https://doi.org/10.1029/2018MS001540) biosphere model, driven by ERA5 reanalysis data at a 0.5x0.5 degree resolution. The NEP per plant functional type are distributed according to the high resolution CORINE land-use map (https://land.copernicus.eu/pan-european/corine-land-cover), and aggregated to CTE-HR resolution. For more information, see {DOI}\n',

            'flux_ff_exchange_prior': f'Hourly estimates of fossil fuel  (including biofuel) emission, based on a range of sources. They include emissions from public power, industry, households, ground transport, aviation, shipping, and calcination of cement. Our product does not include carbonation of cement and human respiration. Public power is based on ENTSO-E data (https://transparency.entsoe.eu/), Industry, Ground transport, Aviation, and Shipping is based on Eurostat data (https://ec.europa.eu/eurostat/databrowser/). Household emissions are based on a degree-day model, driven by ERA5 reanalysis data. Spatial distributions of the emissions are based on CAMS data (https://doi.org/10.5194/essd-14-491-2022). Cement emissions are taken from GridFED V.2021.3 (https://zenodo.org/record/5956612#.YoTmvZNBy9F). For more information, see {DOI}\n',

            'flux_ocean_exchange_prior': f'Hourly ocean fluxes, based on a climatology of Jena CarboScope fluxes (https://doi.org/10.17871/CarboScope-oc_v2020, https://doi.org/10.5194/os-9-193-2013). An adjustment, based on windspeed and temperature, is applied to obtain hourly fluxes at the CTE-HR resolution. Positive fluxes are emissions and negative fluxes indicate uptake. Please always cite the original Jena CarboScope data when using this file, and use the original data when only low resolution ocean fluxes are required. For more information, see {DOI}\n',

            'flux_fire_exchange_prior': f'This is a version of the GFAS fire emissions (https://doi.org/10.5194/acp-18-5359-2018), re-gridded to match the resolution of the biosphere, fossil fuel, and ocean fluxes of the CTE-HR product. Please always cite the original GFAS data when using this file, and use the original data when only fire emissions are required. For more information, see {DOI}\n Contains modified Copernicus Atmosphere Monitoring Service Information [2020].'
            }

innames_PARIS = [
            'regional.nep' ,
            'ff_emissions_CO2', 
            'regional.ocean', 
            'regional.fire'
            ]

varnames_PARIS = [
            'nep', 
            'anthropogenic', 
            'ocean', 
            'fire'
            ]
varnames_PARIS_out = [
            'flux_bio_exchange_prior',
            'flux_ff_exchange_prior',
            'flux_ocean_exchange_prior',
            'flux_fire_exchange_prior'
]

paris_files=[]
for inname in innames_PARIS:
    paris_files += sorted(glob(f'{outpath}/*{inname}*.nc'))

## LOAD TEMPLATE FILE
with nc.Dataset.fromcdl(cdlfilename = cdlpath, mode = 'a', ncfilename = outname) as template:

    ## BASIC VARIABLES (TIME, LAT, LON) + FOSSIL FUEL EMISSIONS
    for file in paris_files:
        print('Working on ... ' + file)
        paris_file = nc.Dataset(file)
        
        sectorname = list(paris_file.variables.keys())[-1]
        sectorindex = varnames_PARIS.index(sectorname)
        new_varname = varnames_PARIS_out[sectorindex]

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
            template.variables[new_varname].comment = comments_PARIS[new_varname]

    ## FILL IN COUNTRY MASK DATA
    for i in range(0,len(landmaskfile)):
        template.variables['country_name'][i] = landmaskfile['name'][i]
        template.variables['country_abbrev'][i] = landmaskfile['code'][i]

    # Update creation date of template file
    template.creation_date = now.strftime("%Y-%m-%d %H:%M")