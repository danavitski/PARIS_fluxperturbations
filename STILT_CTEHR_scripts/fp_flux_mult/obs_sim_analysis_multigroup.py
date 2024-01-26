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
#plot_limits_difference = [-1.5, 1.5]
plot_limits_difference = [-15, 15]

# Define plot style
plot_style = 'poster'

# Define parameters for group1
inpath_group1= '/projects/0/ctdas/PARIS/DATA/obspacks/'
pseudo_obs_varname_group1 = 'pseudo_observation'
bckg_varname_group1 = 'background'

# Define parameters for group2
inpath_group2= inpath_group1 + 'uob/'
cont_varname_group2 = 'excess'
bckg_varname_group2 = 'baseline'

# Define groupnames to compare
groupname2 = 'UoB'
groupname1 = 'WUR'

# Define histogram type
hist_type = 'percent'

##############################################
########## PLOTTING PART #####################
########## (DO NOT EDIT) #####################
##############################################
# Define outpath and create it 
outpath = '/projects/0/ctdas/PARIS/Experiments/plots/' + groupname1 + '_' + groupname2 + '/' + station + '/'
if not os.path.exists(outpath):
    os.makedirs(outpath)

if plot_style == 'poster':
    ms = 30
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
# Loop over experiments to add data
for experiment in experimentlist:
    imagepath = outpath + 'obs_sim_' + experiment + '_' + timelabels[0] + '_' + timelabels[1] + '.png'

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

    if station in non_obspack_sites: 
        file_group1 = glob.glob(inpath_group1 + experiment + '/*' + station.upper() + '*.nc')[0]
        file_group2 = glob.glob(inpath_group2 + station.upper() + '_' + experiment + '*.nc')[0]

    else:
        file_group1 = glob.glob(inpath_group1 + experiment + '/*' + station.lower() + '*.nc')[0]
        file_group2 = glob.glob(inpath_group2 + station.upper() + '_' + experiment + '*.nc')[0]

    with xr.open_dataset(file_group1) as ds:
        # Slice ds to adhere to timerange
        ds = ds.sel(time=slice(timerange[0], timerange[1]))

        # Mask results for availability of pseudo observations
        mask = np.isnan(ds[pseudo_obs_varname_group1][:])
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
            pseudo_obs_pts = ax.scatter(time, pseudo_obs*1e6, label = f'{groupname1} model ({experiment})', alpha = 1, s = ms, zorder = 1)

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
            pseudo_obs_pts = ax.plot(time, pseudo_obs*1e6, label = f'{groupname1} model ({experiment})', alpha = 1, zorder = 1)

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

    with xr.open_dataset(file_group2) as ds:
        # Correct time variable in group 2 file
        ds['nmeasure'] = ds['time']
        ds = ds.drop(labels='time')
        ds = ds.rename({'nmeasure': 'time'})
        
        # Slice ds to adhere to timerange
        ds = ds.sel(time=slice(timerange[0], timerange[1]))

        # Mask results for availability of pseudo observations
        mask = np.isnan(ds[cont_varname_group2][:])
        time2 = ds['time'][~mask]
        cont = ds[cont_varname_group2][~mask]
        bckg = ds[bckg_varname_group2][~mask]
        pseudo_obs = cont + bckg
        
        # Calculate difference 
        dif = pseudo_obs - obs

        if mode == 'scatter':
            # Plot results
            pseudo_obs_pts = ax.scatter(time2, pseudo_obs, label = f'{groupname2} model ({experiment})', alpha = 1, s = ms, zorder = 1)
            obs_pts = ax.scatter(time, obs*1e6, label = 'observations', color = 'red', alpha = 1, s = ms, zorder = 1)

            # Plot rolling mean
            if plt_rolling_mean:
                obs_int = obs.interp(method=interp_method)
                pseudo_obs_int = pseudo_obs.interp(method=interp_method)
                obs_rolling = obs_int.rolling(time=interp_interval, center=True).mean()
                pseudo_obs_rolling = pseudo_obs_int.rolling(time=interp_interval, center=True).mean()

                obs_int_pts = ax.plot(time, obs_int, label = f'observations ({interp_interval}-day mean)', alpha = 1, zorder = 1)
                pseudo_obs_int_pts = ax.plot(time2, pseudo_obs_int*1e6, label = f'model ({interp_interval}-day mean)', alpha = 1, zorder = 1)

        elif mode == 'line':
            # Plot results
            pseudo_obs_pts = ax.plot(time2, pseudo_obs, label = f'{groupname2} model ({experiment})', alpha = 1, zorder = 1)
            obs_pts = ax.scatter(time, obs*1e6, label = 'observations', color = 'red', alpha = 1, s = ms, zorder = 1)

            # Plot rolling mean
            if plt_rolling_mean:
                obs_int = obs.interp(method=interp_method)
                pseudo_obs_int = pseudo_obs.interp(method=interp_method)
                obs_rolling = obs_int.rolling(time=interp_interval, center=True).mean()
                pseudo_obs_rolling = pseudo_obs_int.rolling(time=interp_interval, center=True).mean()
            
                obs_int_pts = ax.plot(time, obs_int*1e6, label = f'observations ({interp_interval}-day mean)', alpha = 1, zorder = 1)
                pseudo_obs_int_pts = ax.plot(time2, pseudo_obs_int*1e6, label = f'model ({interp_interval}-day mean)', alpha = 1, zorder = 1)

        else:
            raise ValueError('Invalid mode selected. Please select either "scatter" or "line".')

    # Plot a simple legend
    ax.legend(loc='upper right', fontsize=labelsize)

    plt.savefig(imagepath, dpi=300, bbox_inches='tight')

########################################
# CTE-HR FLUX CONTRIBUTION PLOT ########
########################################
# Loop over experiments to add data
for experiment in experimentlist:
    imagepath = outpath + 'obs_sim_' + experiment + '_' + timelabels[0] + '_' + timelabels[1] + '_fluxcontribution.png'

    # Initialize figure and axis
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,9), gridspec_kw={'width_ratios': [4, 1]})

    ## First window (timeseries)
    # Set labels
    axes[0].set_xlabel('Time', size=labelsize)
    axes[0].set_ylabel('Contribution to total atmospheric\n CO$_{2}$ concentration (ppm)', size=labelsize)

    # Set the ticks and ticklabels for all axes
    plt.setp(axes[0].get_xticklabels(), rotation=45, ha='right',rotation_mode='anchor')

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
    
    if station in non_obspack_sites: 
        file_group1 = glob.glob(inpath_group1 + experiment + '/*' + station.upper() + '*.nc')[0]
        file_group2 = glob.glob(inpath_group2 + station.upper() + '_' + experiment + '*.nc')[0]

    else:
        file_group1 = glob.glob(inpath_group1 + experiment + '/*' + station.lower() + '*.nc')[0]
        file_group2 = glob.glob(inpath_group2 + station.upper() + '_' + experiment + '*.nc')[0]

    with xr.open_dataset(file_group1) as ds:
        # Slice ds to adhere to timerange
        ds = ds.sel(time=slice(timerange[0], timerange[1]))

        # Mask results for availability of pseudo observations
        mask = np.isnan(ds[pseudo_obs_varname_group1][:])
        time = ds['time'][~mask]
        cont = ds['mixed'][~mask]

        if mode == 'scatter':
            # Plot results
            cont_pts = axes[0].scatter(time, cont * 1e6 , label = f'{groupname1} contribution ({experiment})', alpha = 1, s = ms, zorder = 1)
            
        # Plot rolling mean
            if plt_rolling_mean:
                cont_int = cont.interp(method=interp_method)
                cont_rolling = cont_int.rolling(time=interp_interval, center=True).mean()

                cont_int_pts = axes[0].plot(time, cont_int*1e6, label = f'{groupname1} contribution ({experiment}; {interp_interval}-day mean)', alpha = 1, zorder = 1)

        elif mode == 'line':
            # Plot results
            cont_pts = axes[0].plot(time, cont * 1e6, label = f'{groupname1} contribution ({experiment})', alpha = 1, zorder = 1)

            # Plot rolling mean
            if plt_rolling_mean:
                cont_int = cont.interp(method=interp_method)
                cont_rolling = cont_int.rolling(time=interp_interval, center=True).mean()

                cont_int_pts = axes[0].plot(time, cont_int*1e6, label = f'{groupname1} contribution ({experiment}; {interp_interval}-day mean)', alpha = 1, zorder = 1)

        else:
            raise ValueError('Invalid mode selected. Please select either "scatter" or "line".')

        # Plot a histogram in second window
        sns.histplot(y=cont*1e6, bins = bin_edges, stat = hist_type, kde=True, ax = axes[1], common_norm=False, alpha=0.6, line_kws={'label': '', 'alpha': 1})

    with xr.open_dataset(file_group2) as ds:
        # Correct time variable in group 2 file
        ds['nmeasure'] = ds['time']
        ds = ds.drop(labels='time')
        ds = ds.rename({'nmeasure': 'time'})
        
        # Slice ds to adhere to timerange
        ds = ds.sel(time=slice(timerange[0], timerange[1]))

        # Mask results for availability of pseudo observations
        mask = np.isnan(ds[cont_varname_group2][:])
        time = ds['time'][~mask]
        cont = ds[cont_varname_group2][~mask]

        if mode == 'scatter':
            # Plot results
            cont_pts = axes[0].scatter(time, cont, label = f'{groupname2} contribution ({experiment})', alpha = 1, s = ms, zorder = 1)
            
            # Plot rolling mean
            if plt_rolling_mean:
                cont_int = cont.interp(method=interp_method)
                cont_rolling = cont_int.rolling(time=interp_interval, center=True).mean()

                cont_int_pts = axes[0].plot(time, cont_int, label = f'{groupname2} contribution ({experiment}; {interp_interval}-day mean)', alpha = 1, zorder = 1)

        elif mode == 'line':
            # Plot results
            cont_pts = axes[0].plot(time, cont, label = f'{groupname2} contribution ({experiment})', alpha = 1, zorder = 1)

            # Plot rolling mean
            if plt_rolling_mean:
                cont_int = cont.interp(method=interp_method)
                cont_rolling = cont_int.rolling(time=interp_interval, center=True).mean()

                cont_int_pts = axes[0].plot(time, cont_int, label = f'{groupname2} contribution ({experiment}; {interp_interval}-day mean)', alpha = 1, zorder = 1)

        else:
            raise ValueError('Invalid mode selected. Please select either "scatter" or "line".')

        # Plot a histogram in second window
        sns.histplot(y=cont, bins = bin_edges, stat = hist_type, kde=True, ax = axes[1], common_norm=False, alpha=0.6, line_kws={'label': '', 'alpha': 1})

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
# DIFFERENCE PLOT ########
##########################
for experiment in experimentlist:
    imagepath = outpath + 'obs_sim_' + experiment + '_' + timelabels[0] + '_' + timelabels[1] + '_dif.png'

    # Initialize figure and axis
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,9), gridspec_kw={'width_ratios': [4, 1]})

    ## First window (timeseries)
    # Set labels
    axes[0].set_xlabel('Time', size=labelsize)
    axes[0].set_ylabel('CO$_{2}$ mole fraction difference (ppm)', size=labelsize)

    # Set the ticks and ticklabels for all axes
    plt.setp(axes[0].xaxis.get_ticklabels(), rotation=45, ha='right',rotation_mode='anchor')

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

    if station in non_obspack_sites: 
        file_group1 = glob.glob(inpath_group1 + experiment + '/*' + station.upper() + '*.nc')[0]
        file_group2 = glob.glob(inpath_group2 + station.upper() + '_' + experiment + '*.nc')[0]

    else:
        file_group1 = glob.glob(inpath_group1 + experiment + '/*' + station.lower() + '*.nc')[0]
        file_group2 = glob.glob(inpath_group2 + station.upper() + '_' + experiment + '*.nc')[0]

    with xr.open_dataset(file_group1) as ds1:
        with xr.open_dataset(file_group2) as ds2:
            # Correct time variable in group 2 file
            ds2['nmeasure'] = ds2['time']
            ds2 = ds2.drop(labels='time')
            ds2 = ds2.rename({'nmeasure': 'time'})

            # Slice ds to adhere to timerange
            ds1 = ds1.sel(time=slice(timerange[0], timerange[1]))
            ds2 = ds2.sel(time=slice(timerange[0], timerange[1]))
            
            # First create masks for both datasets and find the common time range
            mask1 = np.isnan(ds1[pseudo_obs_varname_group1][:])
            mask2 = np.isnan(ds2[cont_varname_group2][:])

            common_ds = xr.merge([ds1, ds2], join='inner')

            # Extract values from common_ds
            time = common_ds['time']
            obs = common_ds[obsvarname]
            pseudo_obs1 = common_ds[pseudo_obs_varname_group1]
            pseudo_obs2 = common_ds[cont_varname_group2] + common_ds[bckg_varname_group2]
            
            # Calculate difference 
            dif = pseudo_obs1*1e6 - pseudo_obs2

        if mode == 'scatter':
            # Plot results
            dif_plt = axes[0].scatter(time, dif, label = f'{groupname1} - {groupname2} ({experiment})', alpha = 1, s = ms, zorder = 1)

        elif mode == 'line':
            # Plot results
            dif_plt = axes[0].plot(time, dif, label = f'{groupname1} - {groupname2} ({experiment})', alpha = 1, zorder = 1)

        else:
            raise ValueError('Invalid mode selected. Please select either "scatter" or "line".')

        # Plot a histogram in second window
        sns.histplot(y=dif, bins = bin_edges, stat = hist_type, kde=True, ax = axes[1], common_norm=False, alpha=0.6, line_kws={'label': '', 'alpha': 1})

    ## Second window (histogram)
    # Set labels and limits
    axes[1].set_ylim([plot_limits_difference[0], plot_limits_difference[1]])
    axes[1].set(xlabel='')
    axes[1].set(ylabel='')
    axes[1].set_yticklabels([])

    # Plot a simple legend
    axes[0].legend(loc='upper right', fontsize=labelsize)

    plt.savefig(imagepath, dpi=300, bbox_inches='tight')


####################################################
# DIFFERENCE PLOT (BACKGROUNDS / BASELINES) ########
####################################################
for experiment in experimentlist:
    imagepath = outpath + 'obs_sim_' + timelabels[0] + '_' + timelabels[1] + '_backgrounddif.png'

    # Initialize figure and axis
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(16,9))

    ## First window (timeseries)
    # Set labels
    axes.set_xlabel('Time', size=labelsize)
    axes.set_ylabel('Atmospheric CO$_{2}$ concentration difference (ppm)', size=labelsize)

    # Set the ticks and ticklabels for all axes
    plt.setp(axes.xaxis.get_ticklabels(), rotation=45, ha='right',rotation_mode='anchor')

    # Set the ticks and ticklabels for all axes
    axes.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
    axes.xaxis.set_major_formatter(mdates.DateFormatter('%B'))

    # Set the limits for all axes
    axes.set_xlim([timerange[0], timerange[1]])
    axes.set_ylim([plot_limits_difference[0], plot_limits_difference[1]])
    
    # Make sure grid is below graph
    axes.set_axisbelow(True)

    # Set subplot spacing
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

    if station in non_obspack_sites: 
        file_group1 = glob.glob(inpath_group1 + experiment + '/*' + station.upper() + '*.nc')[0]
        file_group2 = glob.glob(inpath_group2 + station.upper() + '_' + experiment + '*.nc')[0]

    else:
        file_group1 = glob.glob(inpath_group1 + experiment + '/*' + station.lower() + '*.nc')[0]
        file_group2 = glob.glob(inpath_group2 + station.upper() + '_' + experiment + '*.nc')[0]

    with xr.open_dataset(file_group1) as ds1:
        with xr.open_dataset(file_group2) as ds2:
            # Correct time variable in group 2 file
            ds2['nmeasure'] = ds2['time']
            ds2 = ds2.drop(labels='time')
            ds2 = ds2.rename({'nmeasure': 'time'})

            # Slice ds to adhere to timerange
            ds1 = ds1.sel(time=slice(timerange[0], timerange[1]))
            ds2 = ds2.sel(time=slice(timerange[0], timerange[1]))
            
            # First create masks for both datasets and find the common time range
            mask1 = np.isnan(ds1[pseudo_obs_varname_group1][:])
            mask2 = np.isnan(ds2[cont_varname_group2][:])

            common_ds = xr.merge([ds1, ds2], join='inner')

            # Extract values from common_ds
            time = common_ds['time']
            obs = common_ds[obsvarname]
            bckg1 = common_ds[bckg_varname_group1]
            bckg2 = common_ds[bckg_varname_group2]
            
            # Calculate difference 
            dif = bckg1*1e6 - bckg2

        if mode == 'scatter':
            # Plot results
            dif_plt = axes.scatter(time, dif, label = f'Background concentration {groupname1} - {groupname2}', alpha = 1, s = ms, zorder = 1)

        elif mode == 'line':
            # Plot results
            dif_plt = axes.plot(time, dif, label = f'Background concentration {groupname1} - {groupname2}', alpha = 1, zorder = 1)

        else:
            raise ValueError('Invalid mode selected. Please select either "scatter" or "line".')

    # Plot a simple legend
    axes.legend(loc='upper right', fontsize=labelsize)

    plt.savefig(imagepath, dpi=300, bbox_inches='tight')


##############################################
# DIFFERENCE PLOT (BETWEEN SCENARIOS) ########
##############################################
# Define unique combinations of experiments for later plotting purposes
experimentcode_uniquepairs = sorted(map(sorted, combinations(set(experimentlist), 2)), reverse=True)

# Only retain combinations with BASE scenario since the others are not relevant
experimentcode_uniquepairs = [set for set in experimentcode_uniquepairs if 'BASE' in set]

for exp in experimentcode_uniquepairs:
    exp.remove('BASE')
    exp=exp[0]

    #imagepath = outpath + 'obs_sim_' + groupname1 + '-BASE_' + groupname2 + '-BASEvs' + exp + '_' + timelabels[0] + '_' + timelabels[1] + '_samebackground_.png'
    #imagepath = outpath + 'obs_sim_BASEvs' + exp + '_' + groupname1 + '_' + groupname2 + timelabels[0] + '_' + timelabels[1] + '_samebackground.png'
    #imagepath = outpath + 'obs_sim_BASE_' + exp + '_' + groupname1 + 'vs' + groupname2 + timelabels[0] + '_' + timelabels[1] + '_samebackground.png'
    imagepath = outpath + 'obs_sim_BASE_' + exp + '_' + groupname1 + 'vs' + groupname2 + timelabels[0] + '_' + timelabels[1] + '_samebackground_NOHIST.png'

    # Initialize figure and axis
    #fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,9), gridspec_kw={'width_ratios': [4, 1]})
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(16,10))

    ## First window (timeseries)
    # Set labels
    axes.set_ylabel('CO$_{2}$ mole fraction difference (ppm)', size=labelsize)

    # Set the ticks and ticklabels for all axes
    plt.setp(axes.xaxis.get_ticklabels(), rotation=45, ha='right',rotation_mode='anchor')

    # Set the ticks and ticklabels for all axes
    axes.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
    axes.xaxis.set_major_formatter(mdates.DateFormatter('%B'))

    # Set the limits for all axes
    axes.set_xlim([timerange[0], timerange[1]])
    axes.set_ylim([plot_limits_difference[0], plot_limits_difference[1]])
    
    # Make sure grid is below graph
    #for i in range(0,len(axes)):
    #    axes[i].set_axisbelow(True)
    
    axes.set_axisbelow(True)

    # Set subplot spacing
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

    if station in non_obspack_sites: 
        file_exp1 = glob.glob(inpath_group1 +'BASE' + '/*' + station.upper() + '*.nc')[0]
        file_exp2 = glob.glob(inpath_group1 + exp + '/*' + station.upper() + '*.nc')[0]
        
    else:
        file_exp1 = glob.glob(inpath_group1 +'BASE' + '/*' + station.lower() + '*.nc')[0]
        file_exp2 = glob.glob(inpath_group1 + exp + '/*' + station.lower() + '*.nc')[0]

    file_exp1_group2= glob.glob(inpath_group2 + station.upper() + '_' +'BASE' + '*.nc')[0]
    file_exp2_group2= glob.glob(inpath_group2 + station.upper() + '_' + exp + '*.nc')[0]

    with xr.open_dataset(file_exp1) as ds1:
        with xr.open_dataset(file_exp2) as ds2:
            with xr.open_dataset(file_exp1_group2) as ds3:
                with xr.open_dataset(file_exp2_group2) as ds4:
                    # Correct time variable in group 2 files
                    ds3['nmeasure'] = ds3['time']
                    ds3 = ds3.drop(labels='time')
                    ds3 = ds3.rename({'nmeasure': 'time'})

                    ds4['nmeasure'] = ds4['time']
                    ds4 = ds4.drop(labels='time')
                    ds4 = ds4.rename({'nmeasure': 'time'})

                    # Slice ds to adhere to timerange
                    ds1 = ds1.sel(time=slice(timerange[0], timerange[1]))
                    ds2 = ds2.sel(time=slice(timerange[0], timerange[1]))
                    ds3 = ds3.sel(time=slice(timerange[0], timerange[1]))
                    ds4 = ds4.sel(time=slice(timerange[0], timerange[1]))

                    common_ds13 = xr.merge([ds1,ds3], join='inner')
                    common_ds24 = xr.merge([ds2,ds4], join='inner')

                    # Extract values from datasets
                    time12 = ds1['time']
                    time13 = common_ds13['time']
                    pseudo_obs1 = ds1[pseudo_obs_varname_group1]
                    pseudo_obs2 = ds2[pseudo_obs_varname_group1]
                    pseudo_obs3 = common_ds13[cont_varname_group2] + common_ds13[bckg_varname_group1]*1e6
                    pseudo_obs4 = common_ds24[cont_varname_group2] + common_ds24[bckg_varname_group1]*1e6

                    # Calculate difference 
                    dif12 = pseudo_obs1*1e6 - pseudo_obs2*1e6
                    dif13 = pseudo_obs1*1e6 - pseudo_obs3
                    dif24 = pseudo_obs2*1e6 - pseudo_obs4
                    dif34 = pseudo_obs3 - pseudo_obs4

                if mode == 'scatter':
                    # Plot results
                    #dif_plt12 = axes.scatter(time12, dif12, label = f'{groupname1} model (BASE - {exp})', alpha = 1, s = ms, zorder = 1)
                    #dif_plt34 = axes.scatter(time13, dif34, label = f'{groupname2} model (BASE - {exp})', alpha = 1, s = ms, zorder = 1)
                    dif_plt13 = axes.scatter(time13, dif13, label = f'{groupname1} model - {groupname2} model (BASE)', alpha = 1, s = ms, zorder = 1)
                    dif_plt24 = axes.scatter(time13, dif24, label = f'{groupname1} model - {groupname2} model ({exp})', alpha = 1, s = ms, zorder = 1)
                    
                elif mode == 'line':
                    # Plot results
                    #dif_plt12 = axes.plot(time12, dif12, label = f'{groupname1} model (BASE - {exp})', alpha = 1, zorder = 1)
                    #dif_plt34 = axes.plot(time13, dif34, label = f'{groupname2} model (BASE - {exp})', alpha = 1, zorder = 1)
                    dif_plt13 = axes.plot(time13, dif13, label = f'{groupname1} model - {groupname2} model (BASE)', alpha = 1, zorder = 1)
                    dif_plt24 = axes.plot(time13, dif24, label = f'{groupname1} model - {groupname2} model ({exp})', alpha = 1, zorder = 1)

                else:
                    raise ValueError('Invalid mode selected. Please select either "scatter" or "line".')
                
                # Plot a histogram in second window
                #sns.histplot(y=dif12, bins = np.arange(-1.5, 1.5 + 1, 0.1), stat = hist_type, ax = axes[1], common_norm=False, alpha=0.6, line_kws={'label': '', 'alpha': 1})
                #sns.histplot(y=dif34, bins = np.arange(-1.5, 1.5 + 1, 0.1), stat = hist_type, ax = axes[1], common_norm=False, alpha=0.6, line_kws={'label': '', 'alpha': 1})
                #sns.histplot(y=dif13, bins = np.arange(-15, 15 + 1, 0.1), stat = hist_type, ax = axes[1], common_norm=False, alpha=0.6, line_kws={'label': '', 'alpha': 1})
                #sns.histplot(y=dif24, bins = np.arange(-15, 15 + 1, 0.1), stat = hist_type, ax = axes[1], common_norm=False, alpha=0.6, line_kws={'label': '', 'alpha': 1})

    ## Second window (histogram)
    # Set labels and limits
    #axes[1].set_ylim([plot_limits_difference[0], plot_limits_difference[1]])
    #axes[1].set(xlabel='')
    #axes[1].set(ylabel='')
    #axes[1].set_yticklabels([])

    # Plot a simple legend
    axes.legend(loc='lower right', fontsize=labelsize)

    plt.savefig(imagepath, dpi=300, bbox_inches='tight')


##########################
# CORRELATION PLOT #######
##########################
for experiment in experimentlist:
    imagepath = outpath + 'obs_sim_' + experiment + '_' + timelabels[0] + '_' + timelabels[1] + '_corr.png'

    # Initialize figure and axis
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16,9))

    ## First window (timeseries)
    # Set labels
    ax.set_xlabel(f'{groupname1} modelled CO$_{2}$ concentration (ppm)', size=labelsize)
    ax.set_ylabel(f'{groupname2} modelled CO$_{2}$ concentration (ppm)', size=labelsize)
    
    # Make sure grid is below graph
    ax.set_axisbelow(True)

    # Set the limits for all axes
    ax.set_xlim([plot_limits_absolute[0], plot_limits_absolute[1]-5])
    ax.set_ylim([plot_limits_absolute[0], plot_limits_absolute[1]-5])

    # Plot a 1:1 line
    zero_line = ax.plot([0, 500], [0, 500], color = 'black', linestyle = '--', alpha = 0.5, zorder = 0)

    # Set subplot spacing
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

    if station in non_obspack_sites: 
        file_group1 = glob.glob(inpath_group1 + experiment + '/*' + station.upper() + '*.nc')[0]
        file_group2 = glob.glob(inpath_group2 + station.upper() + '_' + experiment + '*.nc')[0]

    else:
        file_group1 = glob.glob(inpath_group1 + experiment + '/*' + station.lower() + '*.nc')[0]
        file_group2 = glob.glob(inpath_group2 + station.upper() + '_' + experiment + '*.nc')[0]

    with xr.open_dataset(file_group1) as ds1:
        with xr.open_dataset(file_group2) as ds2:
            # Correct time variable in group 2 file
            ds2['nmeasure'] = ds2['time']
            ds2 = ds2.drop(labels='time')
            ds2 = ds2.rename({'nmeasure': 'time'})

            # Slice ds to adhere to timerange
            ds1 = ds1.sel(time=slice(timerange[0], timerange[1]))
            ds2 = ds2.sel(time=slice(timerange[0], timerange[1]))
            
            # First create masks for both datasets and find the common time range
            mask1 = np.isnan(ds1[pseudo_obs_varname_group1][:])
            mask2 = np.isnan(ds2[cont_varname_group2][:])

            common_ds = xr.merge([ds1, ds2], join='inner')

            # Extract values from common_ds
            time = common_ds['time']
            obs = common_ds[obsvarname]
            pseudo_obs1 = common_ds[pseudo_obs_varname_group1]
            pseudo_obs2 = common_ds[cont_varname_group2] + common_ds[bckg_varname_group2]

            if mode == 'scatter':
                # Plot results
                dif_plt = ax.scatter(pseudo_obs1*1e6, pseudo_obs2, label = f'{experiment}', alpha = 1, s = ms, zorder = 1)

            elif mode == 'line':
                # Continue, as line plots are not possible with correlation plots
                continue

            else:
                raise ValueError('Invalid mode selected. Please select either "scatter" or "line".')

    # Plot a simple legend
    ax.legend(loc='lower right', fontsize=labelsize)

    plt.savefig(imagepath, dpi=300, bbox_inches='tight')
