import netCDF4 as nc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import datetime
from dateutil.relativedelta import relativedelta

mask_01_02 = nc.Dataset('/projects/0/ctdas/PARIS/Experiments/landmask/paris_countrymask_0.2x0.1deg_2D.nc', 'r', format='NETCDF3_CLASSIC')
mask_005 = nc.Dataset('/projects/0/ctdas/PARIS/Experiments/landmask/paris_countrymask_0.05deg_2D.nc', 'r', format='NETCDF3_CLASSIC')

## EXPERIMENT-SPECIFIC PART
# EXTRACT GERMANY FROM COUNTRY MASK
GER_mask_02_01 = mask_01_02.variables['DEU'][:,:].data
GER_mask_005 = mask_005.variables['DEU'][:,:].data

#GER_mask_bool = np.where(GER_mask != 0, 1, 0)
#ones_array = np.ones((ny,nx))
#ones_array[GER_mask_bool == 1] = 0.5 * GER_mask[GER_mask_bool == 1]

fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (16, 9))
mask_02_01 = ax2.imshow(GER_mask_02_01)
mask_005 = ax1.imshow(GER_mask_005)
fig.suptitle('Germany', size = 20, weight='bold')
plt.colorbar(mask_02_01, location='right')
plt.savefig('/projects/0/ctdas/PARIS/Experiments/landmask/plots/mask_DEU.png', bbox_inches = 'tight')