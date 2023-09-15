# Import necessary modules
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf
import netCDF4 as nc
from datetime import datetime, timedelta
import pandas as pd

fluxfilelist = ['nep', 'fire', 'ocean', 'anthropogenic']
fluxdir = '/projects/0/ctdas/awoude/NRT/ICOS_OUTPUT/'

lats = np.arange(33, 72, 0.1)
lons = np.arange(-15, 35, 0.2)

filepath = '/gpfs/scratch1/shared/dkivits/STILT/footprints/'
csv_files = sorted(glob.glob(filepath+'footprint_HEI*.csv'))
for i in range(0,len(csv_files)):
    file = csv_files[i]
    df = pd.read_csv(file, header=0)
    timelist = df['times'].unique()
    array = np.zeros((len(lats), len(lons), len(timelist)))

    for y in range(0,len(timelist)):
        curtime = timelist[y]
        subdf = df[df['times'] == curtime]
        array[subdf['index_x'], subdf['index_y'], y] = subdf['values']


"""
print(np.shape(array))
print(array[:,:,20])

# test plot footprint
fig = plt.figure(1, figsize=(16,9))
ax = plt.axes(projection = ccrs.PlateCarree())
ax.add_feature(cf.COASTLINE)
ax.add_feature(cf.BORDERS)

im = ax.imshow(array[:,:,47], cmap='Reds', extent=[-15,35,33,72])
"""
