# Daan Kivits, 2023

# A simple script that is used to calculate the monthly RMSE values between the observations and 2018 CTE-HR 
# flux sets, and plot the absolute timeseries of the transported 2018 CTE-HR fluxes compared to the observations in a timeseries
# plot that is subdivided into the Northern, Temperate, and Mediterranean climate regions (see thesis work for the included
# stations in this analysis). We included a TM5 background run as a reference, which is a 7-day mean of the TM5 simulation results.
# A 7-day running mean is also calculated and shown for both the simulated as well as the observed mixing ratios.

# This variation of the script returns a figure with a 16:9 aspect ratio, but this can be changed in line 33.

##############################################
########## LOAD NECCESSARY PACKAGES ##########
##############################################
from pandas import read_hdf, read_csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from matplotlib.dates import MO, TU, WE, TH, FR, SA, SU
import xarray as xr
import os
from scipy.stats import norm
from scipy import stats
import netCDF4 as nc
import glob
import seaborn as sns
from itertools import combinations
from functions.fluxfile_functions import *
from functions.obspack_identification_url import *

##############################################
########## DEFINE USER SETTINGS ##############
##############################################
# Define timerange and labels
timerange = [datetime(2021, 4, 1), datetime(2021, 10, 1)]
timelabels = [timerange[0].strftime('%Y-%m-%d'), timerange[1].strftime('%Y-%m-%d')]

# Define station
station = 'RGL'

# Define obsvarname based on whether station is ObsPack statin or not
non_obspack_filepath = '/projects/0/ctdas/PARIS/DATA/obs/non_obspacksites/'
non_obspack_sites = unique_stationcodes(non_obspack_filepath=non_obspack_filepath, case='upper')

if station in non_obspack_sites:
    obsvarname = 'co2'
else:
    obsvarname = 'value'

# Define experiment
#experimentlist = ['BASE', 'ATEN', 'PTEN', 'DFIN', 'HGER', 'HFRA']
experimentlist = ['BASE', 'ATEN']

# Define plotting mode
mode = 'scatter'
plt_rolling_mean = False

# Define interpolation settings
interp_interval = 3
interp_method = 'linear'

# Set grid as true
plt.rcParams['axes.grid'] = True

# Plot limits
plot_limits_absolute = [395, 435]
plot_limits_difference = [-15, 15]

# Define plot style
plot_style = 'poster'

# Define groupname for analysis
groupname1 = 'WUR'
groupname2 = 'UoB'

# Define parameters for group1
inpath= '/projects/0/ctdas/PARIS/DATA/obspacks/'
pseudo_obs_varname = 'pseudo_observation'
bckg_varname = 'background'

# Define histogram type
hist_type = 'percent'

##############################################
########## PLOTTING PART #####################
########## (DO NOT EDIT) #####################
##############################################
# Define outpath and create it 
outpath = '/projects/0/ctdas/PARIS/Experiments/plots/' + groupname1 + '/' + station + '/'

if not os.path.exists(outpath):
    os.makedirs(outpath)

if plot_style == 'poster':
    ms = 20
    labelsize = 22
    
    # Subplot spacing parameters
    left  = 0.15  # the left side of the subplots of the figure
    right = 0.95    # the right side of the subplots of the figure
    bottom = 0.2   # the bottom of the subplots of the figure
    top = 0.95      # the top of the subplots of the figure
    wspace = 0.1   # the amount of width reserved for blank space between subplots
    hspace = 0.1   # the amount of height reserved for white space between subplots    
    
    # Define the bin edges and calculate histogram
    bin_size = 2.5
    min_edge = plot_limits_difference[0] 
    max_edge = plot_limits_difference[1] 
    bin_edges = np.arange(min_edge, max_edge + bin_size, bin_size)

    mpl.rcParams['xtick.major.pad']=8
    mpl.rcParams['ytick.major.pad']=8
    mpl.rcParams['xtick.labelsize']=20
    mpl.rcParams['ytick.labelsize']=20
    mpl.rcParams['axes.labelpad']=10

if plot_style == 'report':
    ms = 10
    labelsize=labelsize

    # Subplot spacing parameters
    left  = 0.125  # the left side of the subplots of the figure
    right = 0.9    # the right side of the subplots of the figure
    bottom = 0.1   # the bottom of the subplots of the figure
    top = 0.9      # the top of the subplots of the figure
    wspace = 0.2   # the amount of width reserved for blank space between subplots
    hspace = 0.2   # the amount of height reserved for white space between subplots

    # Define the bin edges and calculate histogram
    bin_size = 2.5
    min_edge = plot_limits_difference[0] 
    max_edge = plot_limits_difference[1] 
    bin_edges = np.arange(min_edge, max_edge + bin_size, bin_size)

##########################
# ABSOLUTE PLOT ##########
##########################
imagepath = outpath + 'obs_sim_' + timelabels[0] + '_' + timelabels[1] + '.png'

# Initialize figure and axis
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16,9))
ax.set_xlabel('Time', size=labelsize)
ax.set_ylabel('Atmospheric CO$_{2}$ concentration (ppm)', size=labelsize)
        
# Make sure grid is below graph
ax.set_axisbelow(True)
    
# Set the ticks and ticklabels for all axes
plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

# Set the ticks and ticklabels for all axes
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%B'))

# Set the limits for all axes
ax.set_xlim([timerange[0], timerange[1]])        
ax.set_ylim([plot_limits_absolute[0], plot_limits_absolute[1]])

# Set subplot spacing
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

# Loop over experiments to add data
for experiment in experimentlist:

    # Update inpath to make experiment-specific
    inpath = inpath + experiment + '/'

    # Load resulting obspack
    obspack = glob.glob(inpath + station + '*.nc')
    
    if station in non_obspack_sites: 
        globstring = '/projects/0/ctdas/PARIS/DATA/obspacks/' + experiment + '/*' + station.upper() + '*.nc'
    else:
        globstring = '/projects/0/ctdas/PARIS/DATA/obspacks/' + experiment + '/*' + station.lower() + '*.nc'
        
    with xr.open_dataset(glob.glob(globstring)[0]) as ds:
        
        # Slice ds to adhere to timerange
        ds = ds.sel(time=slice(timerange[0], timerange[1]))

        # Load ObsPack results
        time_full = ds['time'][:]
        obs_full = ds[obsvarname][:]
        bckg_full = ds['background'][:]
        pseudo_obs_full = ds['pseudo_observation'][:]
        cont_full = ds['mixed'][:]

        # Mask results for availability of pseudo observations
        mask = np.isnan(ds['pseudo_observation'][:])
        time = ds['time'][~mask]
        obs = ds[obsvarname][~mask]
        bckg = ds['background'][~mask]
        cont = ds['mixed'][~mask]
        pseudo_obs = ds['pseudo_observation'][~mask]
        
        # Calculate difference 
        dif = pseudo_obs - obs

        # Interpolate data and calculate rolling mean
        """
        obs_int = obs.interp(method=interp_method)
        bckg_int = bckg.interp(method=interp_method)
        cont_int = cont.interp(method=interp_method)
        pseudo_obs_int = pseudo_obs.interp(method=interp_method)
        dif_int = dif.interp(method=interp_method)
        obs_rolling = obs_int.rolling(time=interp_interval, center=True).mean()
        bckg_rolling = bckg_int.rolling(time=interp_interval, center=True).mean()
        cont_rolling = cont_int.rolling(time=interp_interval, center=True).mean()
        pseudo_obs_rolling = pseudo_obs_int.rolling(time=interp_interval, center=True).mean()
        dif_rolling = dif_int.rolling(time=interp_interval, center=True).mean()
        """

        if mode == 'scatter':
            # Plot results
            pseudo_obs_pts = ax.scatter(time, pseudo_obs*1e6, label = f'model ({experiment})', alpha = 1, s = ms, zorder = 1)
            if experiment == experimentlist[-1]:
                obs_pts = ax.scatter(time, obs*1e6, label = 'observations', color = 'red', alpha = 1, s = ms, zorder = 1)

            # Plot rolling mean
            if plt_rolling_mean:
                obs_int = obs.interp(method=interp_method)
                pseudo_obs_int = pseudo_obs.interp(method=interp_method)
                obs_rolling = obs_int.rolling(time=interp_interval, center=True).mean()
                pseudo_obs_rolling = pseudo_obs_int.rolling(time=interp_interval, center=True).mean()

                obs_int_pts = ax.plot(time, obs_int*1e6, label = f'observations ({interp_interval}-day mean)', alpha = 1, zorder = 1)
                pseudo_obs_int_pts = ax.plot(time, pseudo_obs_int*1e6, label = f'model ({interp_interval}-day mean)', alpha = 1, zorder = 1)

        elif mode == 'line':
            # Plot results
            pseudo_obs_pts = ax.plot(time, pseudo_obs*1e6, label = f'model ({experiment})', alpha = 1, zorder = 1)
            if experiment == experimentlist[-1]:
                obs_pts = ax.scatter(time, obs*1e6, label = 'observations', color = 'red', alpha = 1, s = ms, zorder = 1)

            # Plot rolling mean
            if plt_rolling_mean:
                obs_int = obs.interp(method=interp_method)
                pseudo_obs_int = pseudo_obs.interp(method=interp_method)
                obs_rolling = obs_int.rolling(time=interp_interval, center=True).mean()
                pseudo_obs_rolling = pseudo_obs_int.rolling(time=interp_interval, center=True).mean()
            
                obs_int_pts = ax.plot(time, obs_int*1e6, label = f'observations ({interp_interval}-day mean)', alpha = 1, zorder = 1)
                pseudo_obs_int_pts = ax.plot(time, pseudo_obs_int*1e6, label = f'model ({interp_interval}-day mean)', alpha = 1, zorder = 1)

        else:
            raise ValueError('Invalid mode selected. Please select either "scatter" or "line".')

# Plot a simple legend
ax.legend(loc='upper right', fontsize=labelsize)

plt.savefig(imagepath, dpi=300, bbox_inches='tight')

########################################
# CTE-HR FLUX CONTRIBUTION PLOT ########
########################################
imagepath = outpath + 'obs_sim_' + experiment + '_' + timelabels[0] + '_' + timelabels[1] + '_fluxcontribution.png'

# Initialize figure and axis
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,9), gridspec_kw={'width_ratios': [4, 1]})

## First window (timeseries)
# Set labels
axes[0].set_xlabel('Time', size=labelsize)
axes[0].set_ylabel('Contribution to total atmospheric\n CO$_{2}$ concentration (ppm)', size=labelsize)

# Set the ticks and ticklabels for all axes
plt.setp(axes[0].get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

# Set the ticks and ticklabels for all axes
axes[0].xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%B'))
    
# Set the limits for all axes
axes[0].set_xlim([timerange[0], timerange[1]])        
axes[0].set_ylim([plot_limits_difference[0], plot_limits_difference[1]])

# Make sure grid is below graph
for i in range(0,len(axes)):
    axes[i].set_axisbelow(True)

# Set subplot spacing
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

# Loop over experiments to add data
for experiment in experimentlist:

    # Update inpath to make experiment-specific
    inpath = inpath + experiment + '/'

    # Load resulting obspack
    obspack = glob.glob(inpath + station + '*.nc')
    
    if station in non_obspack_sites: 
        globstring = '/projects/0/ctdas/PARIS/DATA/obspacks/' + experiment + '/*' + station.upper() + '*.nc'
    else:
        globstring = '/projects/0/ctdas/PARIS/DATA/obspacks/' + experiment + '/*' + station.lower() + '*.nc'
    
    with xr.open_dataset(glob.glob(globstring)[0]) as ds:
        
        # Slice ds to adhere to timerange
        ds = ds.sel(time=slice(timerange[0], timerange[1]))

        # Load ObsPack results
        time_full = ds['time'][:]
        obs_full = ds[obsvarname][:]
        bckg_full = ds['background'][:]
        pseudo_obs_full = ds['pseudo_observation'][:]
        cont_full = ds['mixed'][:]

        # Mask results for availability of pseudo observations
        mask = np.isnan(ds['pseudo_observation'][:])
        time = ds['time'][~mask]
        obs = ds[obsvarname][~mask]
        bckg = ds['background'][~mask]
        cont = ds['mixed'][~mask]
        pseudo_obs = ds['pseudo_observation'][~mask]
        
        # Calculate difference 
        dif = pseudo_obs - obs

        if mode == 'scatter':
            # Plot results
            cont_pts = axes[0].scatter(time, cont * 1e6 , label = f'CTE-HR contribution ({experiment})', alpha = 1, s = ms, zorder = 1)
            
           # Plot rolling mean
            if plt_rolling_mean:
                cont_int = cont.interp(method=interp_method)
                cont_rolling = cont_int.rolling(time=interp_interval, center=True).mean()

                cont_int_pts = axes[0].plot(time, cont_int*1e6, label = f'CTE-HR contribution ({experiment}; {interp_interval}-day mean)', alpha = 1, zorder = 1)

        elif mode == 'line':
            # Plot results
            cont_pts = axes[0].plot(time, cont * 1e6, label = f'CTE-HR contribution ({experiment})', alpha = 1, zorder = 1)

            # Plot rolling mean
            if plt_rolling_mean:
                cont_int = cont.interp(method=interp_method)
                cont_rolling = cont_int.rolling(time=interp_interval, center=True).mean()

                cont_int_pts = axes[0].plot(time, cont_int*1e6, label = f'CTE-HR contribution ({experiment}; {interp_interval}-day mean)', alpha = 1, zorder = 1)

        else:
            raise ValueError('Invalid mode selected. Please select either "scatter" or "line".')

        # Plot a histogram in second window
        sns.histplot(y=cont*1e6, bins = bin_edges, stat = hist_type, kde=True, ax = axes[1], common_norm=False, alpha=0.6, line_kws={'label': '', 'alpha': 1})

## Second window (histogram)
# Set labels and limits
axes[1].set_ylim([plot_limits_difference[0], plot_limits_difference[1]])
axes[1].set(xlabel='')
axes[1].set(ylabel='')
axes[1].set_yticklabels([])

# Plot a simple legend
axes[0].legend(loc='upper right', fontsize=labelsize)

plt.savefig(imagepath, dpi=300, bbox_inches='tight')

########################################################
# DIFFERENCE PLOT (BETWEEN SIMULATIONS AND OBS) ########
########################################################
# Define unique combinations of experiments for later plotting purposes
experimentcode_uniquepairs = list(combinations(set(experimentlist), 2))

for exp_pair in experimentcode_uniquepairs:
    imagepath = outpath + 'obs_sim_' + exp_pair[0] + '_' + exp_pair[1] + '_' + timelabels[0] + '_' + timelabels[1] + '_scenariodif.png'

    # Initialize figure and axis
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,9), gridspec_kw={'width_ratios': [4, 1]})

    ## First window (timeseries)
    # Set labels
    axes[0].set_xlabel('Time', size=labelsize)
    axes[0].set_ylabel('Atmospheric CO$_{2}$ concentration (ppm)', size=labelsize)

    # Set the ticks and ticklabels for all axes
    plt.setp(axes[0].get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Set the ticks and ticklabels for all axes
    axes[0].xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%B'))

    # Set the limits for all axes
    axes[0].set_xlim([timerange[0], timerange[1]])
    axes[0].set_ylim([-20, 20])
    
    # Make sure grid is below graph
    for i in range(0,len(axes)):
        axes[i].set_axisbelow(True)

    # Set subplot spacing
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

    for exp in exp_pair:
        # Update inpath to make experiment-specific
        inpath = inpath + exp + '/'

        # Load resulting obspack
        obspack = glob.glob(inpath + station + '*.nc')
        
        if station in non_obspack_sites: 
            globstring = '/projects/0/ctdas/PARIS/DATA/obspacks/' + exp + '/*' + station.upper() + '*.nc'
        else:
            globstring = '/projects/0/ctdas/PARIS/DATA/obspacks/' + exp + '/*' + station.lower() + '*.nc'
    
        with xr.open_dataset(glob.glob(globstring)[0]) as ds:
            # Slice ds to adhere to timerange
            ds = ds.sel(time=slice(timerange[0], timerange[1]))

            # Mask results for availability of pseudo observations
            mask = np.isnan(ds['pseudo_observation'][:])
            time = ds['time'][~mask]
            obs = ds[obsvarname][~mask]
            pseudo_obs = ds['pseudo_observation'][~mask]
            
            # Calculate difference 
            dif = pseudo_obs - obs

        if mode == 'scatter':
            # Plot results
            dif_plt = axes[0].scatter(time, dif*1e6, label = f'{exp} - obs', alpha = 1, s = ms, zorder = 1)

        elif mode == 'line':
            # Plot results
            dif_plt = axes[0].plot(time, dif*1e6, label = f'{exp} - obs', alpha = 1, zorder = 1)

        else:
            raise ValueError('Invalid mode selected. Please select either "scatter" or "line".')

        # Plot a histogram in second window
        sns.histplot(y=cont*1e6, bins = bin_edges, stat = hist_type, kde=True, ax = axes[1], common_norm=False, alpha=0.6, line_kws={'label': '', 'alpha': 1})

    ## Second window (histogram)
    # Set labels and limits
    axes[1].set_ylim([plot_limits_difference[0], plot_limits_difference[1]])
    axes[1].set(xlabel='')
    axes[1].set(ylabel='')
    axes[1].set_yticklabels([])

    # Plot a simple legend
    axes[0].legend(loc='upper right', fontsize=labelsize)

    plt.savefig(imagepath, dpi=300, bbox_inches='tight')


########################################################
# DIFFERENCE PLOT (BETWEEN SIMULATIONS AND OBS) ########
########################################################
# Define unique combinations of experiments for later plotting purposes
experimentcode_uniquepairs = list(combinations(set(experimentlist), 2))

for exp_pair in experimentcode_uniquepairs:
    imagepath = outpath + 'obs_sim_' + exp_pair[0] + '_' + exp_pair[1] + '_' + timelabels[0] + '_' + timelabels[1] + '_scenariodif.png'

    # Initialize figure and axis
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,9), gridspec_kw={'width_ratios': [4, 1]})

    ## First window (timeseries)
    # Set labels
    axes[0].set_xlabel('Time', size=labelsize)
    axes[0].set_ylabel('Atmospheric CO$_{2}$ concentration (ppm)', size=labelsize)

    # Set the ticks and ticklabels for all axes
    plt.setp(axes[0].get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Set the ticks and ticklabels for all axes
    axes[0].xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%B'))

    # Set the limits for all axes
    axes[0].set_xlim([timerange[0], timerange[1]])
    axes[0].set_ylim([-20, 20])
    
    # Make sure grid is below graph
    for i in range(0,len(axes)):
        axes[i].set_axisbelow(True)

    # Set subplot spacing
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

    for exp in exp_pair:
        # Update inpath to make experiment-specific
        inpath = inpath + exp + '/'

        # Load resulting obspack
        obspack = glob.glob(inpath + station + '*.nc')
        
        if station in non_obspack_sites: 
            globstring = '/projects/0/ctdas/PARIS/DATA/obspacks/' + exp + '/*' + station.upper() + '*.nc'
        else:
            globstring = '/projects/0/ctdas/PARIS/DATA/obspacks/' + exp + '/*' + station.lower() + '*.nc'
    
        with xr.open_dataset(glob.glob(globstring)[0]) as ds:
            # Slice ds to adhere to timerange
            ds = ds.sel(time=slice(timerange[0], timerange[1]))

            # Mask results for availability of pseudo observations
            mask = np.isnan(ds['pseudo_observation'][:])
            time = ds['time'][~mask]
            obs = ds[obsvarname][~mask]
            pseudo_obs = ds['pseudo_observation'][~mask]
            
            # Calculate difference 
            dif = pseudo_obs - obs

        if mode == 'scatter':
            # Plot results
            dif_plt = axes[0].scatter(time, dif*1e6, label = f'{exp} - obs', alpha = 1, s = ms, zorder = 1)

        elif mode == 'line':
            # Plot results
            dif_plt = axes[0].plot(time, dif*1e6, label = f'{exp} - obs', alpha = 1, zorder = 1)

        else:
            raise ValueError('Invalid mode selected. Please select either "scatter" or "line".')

        # Plot a histogram in second window
        sns.histplot(y=cont*1e6, bins = bin_edges, stat = hist_type, kde=True, ax = axes[1], common_norm=False, alpha=0.6, line_kws={'label': '', 'alpha': 1})

    ## Second window (histogram)
    # Set labels and limits
    axes[1].set_ylim([plot_limits_difference[0], plot_limits_difference[1]])
    axes[1].set(xlabel='')
    axes[1].set(ylabel='')
    axes[1].set_yticklabels([])

    # Plot a simple legend
    axes[0].legend(loc='upper right', fontsize=labelsize)

    plt.savefig(imagepath, dpi=300, bbox_inches='tight')


##########################
# CORRELATION PLOT #######
##########################
for exp_pair in experimentcode_uniquepairs:
    imagepath = outpath + 'obs_sim_' + exp_pair[0] + '_' + exp_pair[1] + '_' + timelabels[0] + '_' + timelabels[1] + '_corr.png'

    # Initialize figure and axis
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16,9))

    ## First window (timeseries)
    # Set labels
    ax.set_xlabel('Observed CO$_{2}$ concentration (ppm)', size=labelsize)
    ax.set_ylabel('Modelled CO$_{2}$ concentration (ppm)', size=labelsize)
    
    # Make sure grid is below graph
    ax.set_axisbelow(True)

    # Set the limits for all axes
    ax.set_xlim([plot_limits_absolute[0]+5, plot_limits_absolute[1]])
    ax.set_ylim([plot_limits_absolute[0]+5, plot_limits_absolute[1]])

    # Plot a 1:1 line
    zero_line = ax.plot([0, 500], [0, 500], color = 'black', linestyle = '--', alpha = 0.5, zorder = 0)

    # Set subplot spacing
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

    for exp in exp_pair:
        # Update inpath to make experiment-specific
        inpath = inpath + exp + '/'

        # Load resulting obspack
        obspack = glob.glob(inpath + station + '*.nc')
        
        if station in non_obspack_sites: 
            globstring = '/projects/0/ctdas/PARIS/DATA/obspacks/' + exp + '/*' + station.upper() + '*.nc'
        else:
            globstring = '/projects/0/ctdas/PARIS/DATA/obspacks/' + exp + '/*' + station.lower() + '*.nc'
    
        with xr.open_dataset(glob.glob(globstring)[0]) as ds:
            # Slice ds to adhere to timerange
            ds = ds.sel(time=slice(timerange[0], timerange[1]))

            # Mask results for availability of pseudo observations
            mask = np.isnan(ds['pseudo_observation'][:])
            time = ds['time'][~mask]
            obs = ds[obsvarname][~mask]
            pseudo_obs = ds['pseudo_observation'][~mask]

        if mode == 'scatter':
            # Plot results
            dif_plt = ax.scatter(obs*1e6, pseudo_obs*1e6, label = f'{exp}', alpha = 1, s = ms, zorder = 1)

        elif mode == 'line':
            # Continue, as line plots are not possible with correlation plots
            continue

        else:
            raise ValueError('Invalid mode selected. Please select either "scatter" or "line".')

    # Plot a simple legend
    ax.legend(loc='upper right', fontsize=labelsize)

    plt.savefig(imagepath, dpi=300, bbox_inches='tight')
