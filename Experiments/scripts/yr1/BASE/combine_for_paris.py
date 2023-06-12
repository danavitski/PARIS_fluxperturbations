import sys
import netCDF4 as nc
import datetime
import numpy as np
import subprocess
from cdo import Cdo
from glob import glob
import platform
import os
import pandas as pd

cdo = Cdo()
INPATH = '/projects/0/ctdas/PARIS/CTE-HR/output'
OUTPATH = '/projects/0/ctdas/PARIS/CTE-HR/PARIS_OUTPUT'
cdlpath = '/projects/0/ctdas/PARIS/cdl_template/paris_input.cdl'

# If the target directory does not yet exist, create it
if not os.path.exists(OUTPATH):
    os.mkdir(OUTPATH)

DOI = ' https://doi.org/10.5281/zenodo.6477331'

now = datetime.datetime.now()
GIT_HASH = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
SIB_GIT_HASH = '1e29b25'

gridfile = '/projects/0/ctdas/PARIS/Experiments/scripts/yr1/BASE/europe.grid'
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

comments = { # Flux-specific comments for different files
            'nep': f'Net ecosystem productivity (gross primary production minus respiration). Positive fluxes are emissions, negative mean uptake. These fluxes are the result of the SiB4 (Version 4.2-COS, hash {SIB_GIT_HASH}, https://doi.org/10.1029/2018MS001540) biosphere model, driven by ERA5 reanalysis data at a 0.5x0.5 degree resolution. The NEP per plant functional type are distributed according to the high resolution CORINE land-use map (https://land.copernicus.eu/pan-european/corine-land-cover), and aggregated to CTE-HR resolution. For more information, see {DOI}\n',

            'anthropogenic': f'Hourly estimates of fossil fuel emission (including biofuel), based on a range of sources. They include emissions from public power, industry, households, ground transport, aviation, shipping, and calcination of cement. Our product does not include carbonation of cement and human respiration. Public power is based on ENTSO-E data (https://transparency.entsoe.eu/), Industry, Ground transport, Aviation, and Shipping is based on Eurostat data (https://ec.europa.eu/eurostat/databrowser/). Household emissions are based on a degree-day model, driven by ERA5 reanalysis data. Spatial distributions of the emissions are based on CAMS data (https://doi.org/10.5194/essd-14-491-2022). Cement emissions are taken from GridFED V.2021.3 (https://zenodo.org/record/5956612#.YoTmvZNBy9F). For more information, see {DOI}\n',

            'anthropogenic.persector': f'Hourly estimates of fossil fuel  (including biofuel) emission, based on a range of sources. They include emissions from public power, industry, households, ground transport, aviation, shipping, and calcination of cement. Our product does not include carbonation of cement and human respiration. Public power is based on ENTSO-E data (https://transparency.entsoe.eu/), Industry, Ground transport, Aviation, and Shipping is based on Eurostat data (https://ec.europa.eu/eurostat/databrowser/). Household emissions are based on a degree-day model, driven by ERA5 reanalysis data. Spatial distributions of the emissions are based on CAMS data (https://doi.org/10.5194/essd-14-491-2022). Cement emissions are taken from GridFED V.2021.3 (https://zenodo.org/record/5956612#.YoTmvZNBy9F). For more information, see {DOI}\n',

            'ocean': f'Hourly ocean fluxes, based on a climatology of Jena CarboScope fluxes (https://doi.org/10.17871/CarboScope-oc_v2020, https://doi.org/10.5194/os-9-193-2013). An adjustment, based on windspeed and temperature, is applied to obtain hourly fluxes at the CTE-HR resolution. Positive fluxes are emissions and negative fluxes indicate uptake. Please always cite the original Jena CarboScope data when using this file, and use the original data when only low resolution ocean fluxes are required. For more information, see {DOI}\n',

            'fire': f'This is a version of the GFAS fire emissions (https://doi.org/10.5194/acp-18-5359-2018), re-gridded to match the resolution of the biosphere, fossil fuel, and ocean fluxes of the CTE-HR product. Please always cite the original GFAS data when using this file, and use the original data when only fire emissions are required. For more information, see {DOI}\n Contains modified Copernicus Atmosphere Monitoring Service Information [2020].'
            }

#month = sys.argv[1] # Get the month for which to run the script
year = sys.argv[1]

attrs = {
'institution': 'Wageningen University, department of Meteorology and Air Quality, Wageningen, the Netherlands; \n \
Rijksuniversiteit Groningen, Groningen, the Netherlands; \n \
ICOS Carbon Portal, Lund, Sweden',
'contact': 'Daan Kivits; daan.kivits@wur.nl',
#'URL': 'carbontracker.eu/cte-hr#NOTYETVALID',
'Conventions': 'CF-1.8',
'creation_date': f'{now:%Y-%m-%d %H:%M}',
'crs': 'spherical earth with radius of 6370 km',
'disclaimer': 'This data belongs to the CarbonTracker project',
#'history': f'File created on {now:%Y-%m-%d %H:%M} by {os.getlogin()}, using the code on https://git.wageningenur.nl/ctdas/CTDAS/-/tree/near-real-time, with hash {GIT_HASH}. Platform: {platform.platform()}, Python version {sys.version}',
'history': f'File created on {now:%Y-%m-%d %H:%M} by dkivits, using the code on https://git.wageningenur.nl/ctdas/CTDAS/-/tree/near-real-time, with hash {GIT_HASH}. Platform: {platform.platform()}, Python version {sys.version}',
'creator': 'Auke van der Woude, https://orcid.org/0000-0002-6286-2621',
'frequency': '1h',
'geospatial_lat_resolution': '0.1 degree',
'geospatial_lon_resolution': '0.2 degree',
'keywords': 'carbon flux',
'license': 'CC-BY-4.0',
'nominal_resolution': '0.1x0.2 degree',
#'realm': 'Atmosphere',
#'references': '{TO BE ADDED}',
'source': f'CTE-HR 1.0. Created using the code from https://git.wageningenur.nl/ctdas/CTDAS/-/tree/near-real-time, hash {GIT_HASH}',
}


for inname, outname, varname in zip(innames, outnames, varnames):
    infiles = sorted(glob(f'{INPATH}/{year}*/{inname}*'))
    paris_files = sorted(glob(f'{OUTPATH}/*.{year}.nc'))
    outfile = f'{OUTPATH}/paris_input_{inname}.{year}.nc'

    if not any(inname in s for s in paris_files):
        print ('WARNING: No existing output files found. Creating new output files')
        print(f'Merging {len(infiles)} files for {inname}, year {year}')
        tmpfile = cdo.mergetime(input=infiles, output='tmp.nc')
        
        print(f'Setting grid on {outfile}')
        if os.path.exists(outfile):
            os.remove(outfile)
        outfile = cdo.setgrid(gridfile, input=tmpfile, output=outfile)
        os.remove(tmpfile)

        attrs['comment'] = comments[varname]
        with nc.Dataset(outfile, 'r+') as ds:
            print('Setting attributes')
            for name, value in attrs.items():
                setattr(ds, name, value)
            for k, v in ds.variables.items():
                if v.ndim == 3:
                    ds[k][:] = ds[k][:] * 1e-6
                    ds[k].units = 'mol m-2 s-1'
            if 'anthrop' in outname:
                print('Changing longname of cement')
                ds['cement'].long_name = 'Emissions from the calcination of cement'
    else:
        print ('WARNING: Found existing output files. Skipping file creation.')


#print(f'Merging {len(paris_files)} sector-specific yearly files into one multi-sector yearly file')
#outfile = cdo.merge(input=paris_files, output=f'{OUTPATH}{outname}.nc')
#os.remove(paris_files)

lon_bounds = [-14.9, 35.1]
lat_bounds = [33.05, 72.05]
res_lon, res_lat = 0.2, 0.1
nx = int((lon_bounds[1] - lon_bounds[0])/res_lon)
ny = int((lat_bounds[1] - lat_bounds[0])/res_lat)
lon_list = np.arange(lon_bounds[0],lon_bounds[1], res_lon)
lat_list = np.arange(lat_bounds[0],lat_bounds[1], res_lat)

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