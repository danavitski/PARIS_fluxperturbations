def write_dict_to_ObsPack(obspack_list, obs_sim_dict, obspack_name = 'obspack_co2_466_GVeu_20230913', stationcode = 'CBW',
                        outpath = '/projects/0/ctdas/PARIS/DATA/obspacks/'):
    """ Function to write dictionary (that contains the simulation results, the background
    and pseudo_observations) to a copy of the original ObsPacks in the ObsPack collection.
     
    Input variables:
    obspack_list: list of ObsPacks that meet the filtering criteria
    obs_sim_dict: dictionary that contains the simulation results, the background and observations
    obspack_name: name of the ObsPack
    stationcode: stationcode of the station
    outpath: path to the output directory
    """
    # Find ObsPack file in list of ObsPacks
    obspack_orig = [s for s in obspack_list if stationcode.lower() in s.lower()][0]
    obspack = outpath + 'pseudo_' + os.path.basename(obspack_orig)

    # Copy file to newfile location
    shutil.copyfile(obspack_orig, obspack)

    # Open newfile in 'append' mode
    ds = xr.open_dataset(obspack, decode_times = False)
    ds = ds.set_index({'time':'time'})
    print('Original index: ' + str(ds['time'].indexes['time']))

    ds['time'] = pd.to_datetime(ds['time'], unit='s')
    print('Changed index: ' + str(ds['time'].indexes['time']))
    
    # Define start time of ObsPack file
    obspack_basetime = np.datetime64('1970-01-01T00:00:00')

    # Define a list of total seconds of the ds.time variable
    ds_time_seconds = ds_time_seconds = ((ds.time.values.astype('datetime64[s]') - obspack_basetime.astype('datetime64[s]'))
                   .astype(np.int64))

    print(ds_time_seconds)

    # Loop over all keys and values in dictionary
    for key, values in obs_sim_dict.items():
        # Convert the object to datetime object
        key = datetime.strptime(key, "%Y-%m-%d %H:%M:%S")
        
        # Calculate time in seconds since start of ObsPack file
        obspack_time = int(((np.datetime64(key) - obspack_basetime).astype('timedelta64[s]')).item())


        # Calculate time difference between simulation time and ObsPack time
        diff_dt = obspack_time - ds_time_seconds
        diff_index = np.abs(diff_dt).argmin()
        diff_min = diff_dt[diff_index]
        
        # Resample to hourly data if time difference is larger than 1 minute
        if diff_min >= 60:
            ds = ds.resample({'time': 'H'}).mean()

        # Convert values to numpy array
        values = np.array(values)

        # Write values to new variables in new ObsPack file
        ds.variables['cte_contribution'][diff_index] = values[0]
        ds.variables['background'][diff_index] = values[1]
        ds.variables['pseudo_observation'][diff_index] = values[2]

    # Close newfile
    ds.close()