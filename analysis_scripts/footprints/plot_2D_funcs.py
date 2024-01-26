    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import cartopy
    import cartopy.crs as ccrs
    from cartopy.feature import ShapelyFeature
    from cartopy.io.shapereader import Reader
    import os
    import cartopy.feature as cfeature
    import pandas as pd 
    import matplotlib.patheffects as path_effects
    from matplotlib.ticker import LogFormatterMathtext

def plot_on_worldmap_v2(data2D, lon, lat, unit, extent, minmax=[], save_plot=False, fpath='', stations = None, stationfile = None, title=None):
    """ 
    UPDATED COLORBAR
    """
    # Worlmap - plotting
    land_50m = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='face',
                                        facecolor=cfeature.COLORS['land'])
    
    # Set the directory for the background image - only works in pyharp env for now
    os.environ["CARTOPY_USER_BACKGROUNDS"] = os.path.join('/projects/0/ctdas/annewil/visualisation/cartopy_user_backgrounds')
    """ AB 02/12/22 """

    # Determine plot width and height
    wi = 8
    dlon = extent[1] - extent[0]
    dlat = extent[3] - extent[2]
    midlon = extent[0] + 0.5*(extent[1]-extent[0])
    midlat = extent[2] + 0.5*(extent[3]-extent[2])
    hi =  ((dlat/dlon) * wi)
    
    # Create empty subplots, one with projection for a map, one for the colorbar
    fig,[ax,cax] = plt.subplots(1,2, gridspec_kw={"width_ratios":[100,1]}, figsize=(wi, hi))
    
    # Hide matplotlib auto tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

    plt.subplots_adjust(wspace=0.03)

    # Set projection on first plot
    ax = fig.add_subplot(1,2,1, projection=ccrs.PlateCarree(midlon), frameon=False)
    ax.set_extent(extent)
    
    # Setting the background image and creating the plot
    #ax.background_img(name='explorer', resolution='high') # world - gebco - world_topo_bath - explorer

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                    linewidth=1, color='lightgray', alpha=0.5, linestyle='-',zorder=0)
    gl.top_labels = gl.right_labels = gl.left_labels = gl.bottom_labels = False
    #ax.add_feature(cartopy.feature.RIVERS, alpha = .7)
    ax.add_feature(cartopy.feature.BORDERS, edgecolor='white',zorder=0)
    ax.add_feature(cartopy.feature.COASTLINE, edgecolor='white',zorder=0)
    ax.add_feature(land_50m, facecolor='lightgray', alpha = 1,zorder=0)

    if minmax == []:
        cmap = mpl.colormaps['Reds']
        cb1 = mpl.colorbar.ColorbarBase(cax, cmap='Reds',orientation='vertical')
        im = ax.pcolormesh(lon, lat, data2D,cmap='Reds',transform=ccrs.PlateCarree(), alpha=0.6, edgecolors=None, zorder=3)

    else:
        cmap = mpl.colormaps['Reds']
        norm = mpl.colors.Normalize(vmin=minmax[0], vmax=minmax[1])
        cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cmap,norm=norm,orientation='vertical')
        im = ax.pcolormesh(lon, lat, data2D, transform=ccrs.PlateCarree(), vmin=minmax[0], vmax=minmax[1], cmap=cmap, alpha=0.6, edgecolors=None, zorder=3)

    cax.tick_params(labelsize=16)
    cb1.ax.set_ylabel(unit, size=20, labelpad=15)

    # add stations
    # Code from Auke
    if stations != None and stationfile != None:
        df = pd.read_csv(stationfile, index_col=False)
        sites = df['code'].values

        for s in stations:
            if s in sites:
                lat = df['lat'][df.code==s].item()
                lon = df['lon'][df.code==s].item()
                ha = 'right' if s in {'BRM', 'PRS', 'RGL'} else 'left'
                va = 'top' if s in {'TRN', 'RGL'} else 'bottom'
                ax.scatter(lon, lat, c='k',s=30, alpha = 0.7, transform=ccrs.PlateCarree())
                txt = ax.annotate(s, (lon, lat), fontsize=13, ha=ha, va=va, fontweight='normal', transform=ccrs.PlateCarree())
                txt.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'), path_effects.Normal()])
                
            else:
                raise ValueError(f"Station {s} not in station list!")
                
    # add title
    if title != None:
        ax.set_title(f"{title}", fontsize=12, loc='left')

    if save_plot == True:
        # Save figure to file
        plt.savefig(f"{fpath}", bbox_inches='tight', dpi=300)
        print(f"Stored file to: {fpath}")