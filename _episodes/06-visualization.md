---
title: "Visualize and Publish with Python"
teaching: 0
exercises: 0
questions:
- "How to create animation plots and publish them on the web?"
objectives:
- "Learn how to create simple animations with python"
- "Learn to publish your python notebook on the web (gist and nbviewer)"
keypoints:
- "Get an overview of nbviewer"
---


{: .language-python}

We will need a number of the libraries introduced in the previous lesson.

import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
{: .language-python}

Since geographic data files can often be very large, when we first open our data file in xarray it simply loads the metadata associated with the file (this is known as "lazy loading"). We can then view summary information about the contents of the file before deciding whether we'd like to load some or all of the data into memory.

dset = xr.open_dataset(accesscm2_pr_file)
print(dset)
{: .language-python}

<xarray.Dataset>
Dimensions:    (bnds: 2, lat: 144, lon: 192, time: 60)
Coordinates:
  * time       (time) datetime64[ns] 2010-01-16T12:00:00 ... 2014-12-16T12:00:00
  * lon        (lon) float64 0.9375 2.812 4.688 6.562 ... 355.3 357.2 359.1
  * lat        (lat) float64 -89.38 -88.12 -86.88 -85.62 ... 86.88 88.12 89.38
Dimensions without coordinates: bnds
Data variables:
    time_bnds  (time, bnds) datetime64[ns] ...
    lon_bnds   (lon, bnds) float64 ...
    lat_bnds   (lat, bnds) float64 ...
    pr         (time, lat, lon) float32 ...
Attributes:
    CDI:                    Climate Data Interface version 1.9.8 (https://mpi...
    source:                 ACCESS-CM2 (2019): \naerosol: UKCA-GLOMAP-mode\na...
    institution:            CSIRO (Commonwealth Scientific and Industrial Res...
    Conventions:            CF-1.7 CMIP-6.2
    activity_id:            CMIP
    branch_method:          standard
    branch_time_in_child:   0.0
    branch_time_in_parent:  0.0
    creation_date:          2019-11-08T08:26:37Z
    data_specs_version:     01.00.30
    experiment:             all-forcing simulation of the recent past
    experiment_id:          historical
    external_variables:     areacella
    forcing_index:          1
    frequency:              mon
    further_info_url:       https://furtherinfo.es-doc.org/CMIP6.CSIRO-ARCCSS...
    grid:                   native atmosphere N96 grid (144x192 latxlon)
    grid_label:             gn
    initialization_index:   1
    institution_id:         CSIRO-ARCCSS
    mip_era:                CMIP6
    nominal_resolution:     250 km
    notes:                  Exp: CM2-historical; Local ID: bj594; Variable: p...
    parent_activity_id:     CMIP
    parent_experiment_id:   piControl
    parent_mip_era:         CMIP6
    parent_source_id:       ACCESS-CM2
    parent_time_units:      days since 0950-01-01
    parent_variant_label:   r1i1p1f1
    physics_index:          1
    product:                model-output
    realization_index:      1
    realm:                  atmos
    run_variant:            forcing: GHG, Oz, SA, Sl, Vl, BC, OC, (GHG = CO2,...
    source_id:              ACCESS-CM2
    source_type:            AOGCM
    sub_experiment:         none
    sub_experiment_id:      none
    table_id:               Amon
    table_info:             Creation Date:(30 April 2019) MD5:e14f55f257cceaf...
    title:                  ACCESS-CM2 output prepared for CMIP6
    variable_id:            pr
    variant_label:          r1i1p1f1
    version:                v20191108
    cmor_version:           3.4.0
    tracking_id:            hdl:21.14100/b4dd0f13-6073-4d10-b4e6-7d7a4401e37d
    license:                CMIP6 model data produced by CSIRO is licensed un...
    CDO:                    Climate Data Operators version 1.9.8 (https://mpi...
    history:                Tue Jan 12 14:50:25 2021: ncatted -O -a history,p...
    NCO:                    netCDF Operators version 4.9.2 (Homepage = http:/...
{: .output}

We can see that our dset object is an xarray.Dataset, which when printed shows all the metadata associated with our netCDF data file.

In this case, we are interested in the precipitation variable contained within that xarray Dataset:

print(dset['pr'])
{: .language-python}

<xarray.DataArray 'pr' (time: 60, lat: 144, lon: 192)>
[1658880 values with dtype=float32]
Coordinates:
  * time     (time) datetime64[ns] 2010-01-16T12:00:00 ... 2014-12-16T12:00:00
  * lon      (lon) float64 0.9375 2.812 4.688 6.562 ... 353.4 355.3 357.2 359.1
  * lat      (lat) float64 -89.38 -88.12 -86.88 -85.62 ... 86.88 88.12 89.38
Attributes:
    standard_name:  precipitation_flux
    long_name:      Precipitation
    units:          kg m-2 s-1
    comment:        includes both liquid and solid phases
    cell_methods:   area: time: mean
    cell_measures:  area: areacella
{: .output}

We can actually use either the dset['pr'] or dset.pr syntax to access the precipitation xarray.DataArray.

To calculate the precipitation climatology, we can make use of the fact that xarray DataArrays have built in functionality for averaging over their dimensions.

clim = dset['pr'].mean('time', keep_attrs=True)
print(clim)
{: .language-python}

<xarray.DataArray 'pr' (lat: 144, lon: 192)>
array([[1.8461452e-06, 1.9054805e-06, 1.9228980e-06, ..., 1.9869783e-06,
        2.0026005e-06, 1.9683730e-06],
       [1.9064508e-06, 1.9021350e-06, 1.8931637e-06, ..., 1.9433096e-06,
        1.9182237e-06, 1.9072245e-06],
       [2.1003202e-06, 2.0477617e-06, 2.0348527e-06, ..., 2.2391034e-06,
        2.1970161e-06, 2.1641599e-06],
       ...,
       [7.5109556e-06, 7.4777777e-06, 7.4689174e-06, ..., 7.3359679e-06,
        7.3987890e-06, 7.3978440e-06],
       [7.1837171e-06, 7.1722038e-06, 7.1926393e-06, ..., 7.1552149e-06,
        7.1576678e-06, 7.1592167e-06],
       [7.0353467e-06, 7.0403985e-06, 7.0326828e-06, ..., 7.0392648e-06,
        7.0387587e-06, 7.0304386e-06]], dtype=float32)
Coordinates:
  * lon      (lon) float64 0.9375 2.812 4.688 6.562 ... 353.4 355.3 357.2 359.1
  * lat      (lat) float64 -89.38 -88.12 -86.88 -85.62 ... 86.88 88.12 89.38
Attributes:
    standard_name:  precipitation_flux
    long_name:      Precipitation
    units:          kg m-2 s-1
    comment:        includes both liquid and solid phases
    cell_methods:   area: time: mean
    cell_measures:  area: areacella
{: output}

Now that we've calculated the climatology, we want to convert the units from kg m-2 s-1 to something that we are a little more familiar with like mm day-1.

To do this, consider that 1 kg of rain water spread over 1 m2 of surface is 1 mm in thickness and that there are 86400 seconds in one day. Therefore, 1 kg m-2 s-1 = 86400 mm day-1.

The data associated with our xarray DataArray is simply a numpy array,

type(clim.data)
{: .language-python}

numpy.ndarray
{: .output}

so we can go ahead and multiply that array by 86400 and update the units attribute accordingly:

clim.data = clim.data * 86400
clim.attrs['units'] = 'mm/day' 

print(clim)
{: .language-python}

<xarray.DataArray 'pr' (lat: 144, lon: 192)>
array([[0.15950695, 0.16463352, 0.16613839, ..., 0.17167493, 0.17302468,
        0.17006743],
       [0.16471735, 0.16434446, 0.16356934, ..., 0.16790195, 0.16573453,
        0.1647842 ],
       [0.18146767, 0.17692661, 0.17581128, ..., 0.19345854, 0.18982219,
        0.18698342],
       ...,
       [0.64894656, 0.64607999, 0.64531446, ..., 0.63382763, 0.63925537,
        0.63917372],
       [0.62067316, 0.61967841, 0.62144403, ..., 0.61821057, 0.6184225 ,
        0.61855632],
       [0.60785395, 0.60829043, 0.60762379, ..., 0.60819248, 0.60814875,
        0.6074299 ]])
Coordinates:
  * lon      (lon) float64 0.9375 2.812 4.688 6.562 ... 353.4 355.3 357.2 359.1
  * lat      (lat) float64 -89.38 -88.12 -86.88 -85.62 ... 86.88 88.12 89.38
Attributes:
    standard_name:  precipitation_flux
    long_name:      Precipitation
    units:          mm/day
    comment:        includes both liquid and solid phases
    cell_methods:   area: time: mean
    cell_measures:  area: areacella
{: .output}

We could now go ahead and plot our climatology using matplotlib, but it would take many lines of code to extract all the latitude and longitude information and to setup all the plot characteristics. Recognising this burden, the xarray developers have built on top of matplotlib.pyplot to make the visualisation of xarray DataArrays much easier.

fig = plt.figure(figsize=[12,5])

ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=180))

clim.plot.contourf(ax=ax,
                   levels=np.arange(0, 13.5, 1.5),
                   extend='max',
                   transform=ccrs.PlateCarree(),
                   cbar_kwargs={'label': clim.units})
ax.coastlines()

plt.show()
{: .language-python}

# Save your animations in `mp4`

We are taking one of our first example where we plot the ECMWF ERA-Interim Vorticity over a pre-defined geographical area.

To avoid downloading large datasets on your laptop, we use one frame only and randomly "perturb" the Vorticity field to demonstrate how
to create and save your animations in python:


<pre data-executable="true" data-language="python">%matplotlib inline
def drawmap(ax,map,x,y,VO, cmap, bounds, norm, title):
    
    ax.set_title(title, fontsize=14)

    map.drawcoastlines()
    map.fillcontinents(color='#ffe2ab')
# draw parallels and meridians.
    map.drawparallels(np.arange(-90.,91.,20.))
    map.drawmeridians(np.arange(-180.,181.,10.))
    map.drawparallels(np.arange(-90.,120.,30.),labels=[1,0,0,0])
    cs = map.contourf(x,y,VO, cmap=cmap, norm=norm, levels=bounds,shading='interp', zorder=1, ax=ax)

    return cs
    
def myanimate(i, ax, map, x,y,VO, cmap, bounds, norm):
    ax.clear()
    # change VO (randomly...)
    VO += 0.1 * np.random.randn()
    new_contour = drawmap(ax,map,x,y,VO, cmap, bounds, norm, 'ECMWF ERA-Interim VO at 850 hPa: Frame %03d'%(i) ) 
    return new_contour
	
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as c
import matplotlib.animation as animation
%matplotlib inline
from mpl_toolkits.basemap import Basemap, shiftgrid
import numpy as np
import netCDF4


FFMpegWriter = animation.writers['ffmpeg']
metadata = dict(title='ECMWF ERA-Interim VO at 850 hPa from 2001-06-01 00:00', artist='Carpentry@UIO',
                comment='Movie for ECMWF ERA-Interim VO at 850 hPa from 2001-06-01 00:00')
writer = FFMpegWriter(fps=20, metadata=metadata)


f = netCDF4.Dataset('EI_VO_850hPa_Summer2001.nc', 'r')
lats = f.variables['lat'][:]
lons = f.variables['lon'][:]
VO = f.variables['VO'][0,0,:,:]*100000  # read first time and unique level
fig = plt.figure(figsize=[12,15])  # a new figure window
ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)


map = Basemap(projection='merc',llcrnrlat=38,urcrnrlat=76,\
            llcrnrlon=-65,urcrnrlon=30, resolution='c', ax=ax)
    
map.drawmeridians(np.arange(-180.,180.,60.),labels=[0,0,0,1])
# make a color map of fixed colors
cmap = c.ListedColormap(['#00004c','#000080','#0000b3','#0000e6','#0026ff','#004cff',
                         '#0073ff','#0099ff','#00c0ff','#00d900','#33f3ff','#73ffff','#c0ffff', 
                         (0,0,0,0),
                         '#ffff00','#ffe600','#ffcc00','#ffb300','#ff9900','#ff8000','#ff6600',
                         '#ff4c00','#ff2600','#e60000','#b30000','#800000','#4c0000'])
bounds=[-200,-100,-75,-50,-30,-25,-20,-15,-13,-11,-9,-7,-5,-3,3,5,7,9,11,13,15,20,25,30,50,75,100,200]
norm = c.BoundaryNorm(bounds, ncolors=cmap.N) # cmap.N gives the number of colors of your palette
    

# shift data so lons go from -180 to 180 instead of 0 to 360.
VO,lons = shiftgrid(180.,VO,lons,start=False)
llons, llats = np.meshgrid(lons, lats)
x,y = map(llons,llats)

first_contour = drawmap(ax,map,x,y,VO,cmap, bounds, norm, 'ECMWF ERA-Interim VO at 850 hPa 2001-06-01 00:00' ) 

## make a color bar
fig.colorbar(first_contour, cmap=cmap, norm=norm, boundaries=bounds, ticks=bounds, ax=ax, orientation='horizontal')

ani = animation.FuncAnimation(fig, myanimate, frames=np.arange(50), 
    fargs=(ax, map, x,y,VO, cmap, bounds, norm), interval=50)
ani.save("writer_ECMWF_EI_VO_850hPa_2001060100.mp4")

f.close()
</pre>

# Embedded animations within your jupyter notebook

The main goal here is to create animations embedded within your jupyter notebook. This is fairly simple to plot your animation within your jupyter notebook.

Let's continue our previous example, and add the following:

<pre data-executable="true" data-language="python">
from IPython.display import HTML

HTML(ani.to_html5_video())
</pre>

<video src="{{ page.root }}/fig/writer_ECMWF_EI_VO_850hPa_2001060100.mp4" poster="{{ page.root }}/fig/EI_VO850hPa_2001060100.png" width="400" controls preload></video>

# Make your jupyter notebook interactive with Jupyter Widgets

Instead of creating a movie, you can allow users to select themselves which plots to show:

<pre data-executable="true" data-language="python">%matplotlib inline
def drawmap(llons,llats,VO, title):
    
    
    fig = plt.figure(figsize=[12,15])  # a new figure window
    ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)


    map = Basemap(projection='merc',llcrnrlat=38,urcrnrlat=76,\
            llcrnrlon=-65,urcrnrlon=30, resolution='c', ax=ax)
    
    ax.set_title(title, fontsize=14)

    map.drawcoastlines()
    map.fillcontinents(color='#ffe2ab')
# draw parallels and meridians.
    map.drawparallels(np.arange(-90.,91.,20.))
    map.drawmeridians(np.arange(-180.,181.,10.))
    map.drawparallels(np.arange(-90.,120.,30.),labels=[1,0,0,0])
# make a color map of fixed colors
    cmap = c.ListedColormap(['#00004c','#000080','#0000b3','#0000e6','#0026ff','#004cff',
                         '#0073ff','#0099ff','#00c0ff','#00d900','#33f3ff','#73ffff','#c0ffff', 
                         (0,0,0,0),
                         '#ffff00','#ffe600','#ffcc00','#ffb300','#ff9900','#ff8000','#ff6600',
                         '#ff4c00','#ff2600','#e60000','#b30000','#800000','#4c0000'])
    bounds=[-200,-100,-75,-50,-30,-25,-20,-15,-13,-11,-9,-7,-5,-3,3,5,7,9,11,13,15,20,25,30,50,75,100,200]
    norm = c.BoundaryNorm(bounds, ncolors=cmap.N) # cmap.N gives the number of colors of your palette
    
    x,y = map(llons,llats)

    cs = map.contourf(x,y,VO, cmap=cmap, norm=norm, levels=bounds,shading='interp', zorder=2, ax=ax)

    return cs
    
def myanimate(i, llons,llats,VO):
    #ax.clear()
    print(VO.min(),VO.max())
    # change VO (randomly...)
    VO += 0.1 * np.random.randn()
    new_contour = drawmap(llons,llats,VO, 'ECMWF ERA-Interim VO at 850 hPa: Frame %03d'%(i) ) 
    
	
    
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors as c
%matplotlib inline
from mpl_toolkits.basemap import Basemap, shiftgrid
import numpy as np
import netCDF4
from ipywidgets import interact



f = netCDF4.Dataset('EI_VO_850hPa_Summer2001.nc', 'r')
lats = f.variables['lat'][:]
lons = f.variables['lon'][:]
VO = f.variables['VO'][0,0,:,:]*100000  # read first time and unique level

# shift data so lons go from -180 to 180 instead of 0 to 360.
VO,lons = shiftgrid(180.,VO,lons,start=False)
llons, llats = np.meshgrid(lons, lats)

f.close()

@interact(time=(0,50))
def finteract(time):
     ca = myanimate(time, llons,llats,VO)

</pre>

# Share your jupyter notebooks (nbviewer)

To be able to share your jupyter notebook:

- Save your jupyter notebook on your local computer (rename it as `share_your_notebook_DC.ipynb`); 
the extension `ipynb` is the default extension for a jupyter notebook and you usually do not need to add it (added automatically)
- Open your saved notebook file (`share_your_notebook_DC.ipynb`) with your favorite editor and copy its content
- Open a new window in your web browser at [http://www.github.com](http://www.github.com) and login with your github username and password (you need to register 
beforehand if you don't have a github account yet).
- Open another window or tab in your web browser at [https://gist.github.com/](https://gist.github.com/) 
- Paste the content in the main window and add a title and a description
- Click on `Create public gist`
- Copy the `gist key` that appears in your url (it has been generated when you clicked on `create public key`)
- Go to  [http://nbviewer.jupyter.org](http://nbviewer.jupyter.org) and paste your `gist key` and click on `Go!`
- Then you can share the resulting url 

For instance, the jupyter notebook generated has been shared and can be viewed [here](http://nbviewer.jupyter.org/gist/annefou/5e5750b90a99b5d6b3de9f328a77dccc).

> ## Rendering jupyter notebook on github
> 
> You can also store your jupyter notebook on github (and you are strongly encouraged to use a version control to keep your programs...) and
> according there is no interactive features or any javascript embedded, github will automatically show your jupyter notebook.
>
{: .callout}

# GEOJSON 

A very simple way to visualize and explore GeoJSON files is to store them on github because gitHub supports 
[rendering geoJSON and topoJSON map files](http://jupyter-gmaps.readthedocs.io/en/latest/authentication.html) 
within GitHub repositories. 

Once available in your github repository, you can use your browser to visualize and share your GEOJSON plot. The final url depends on your github username:

~~~
<script src="https://embed.github.com/view/geojson/<username>/<repo>/<ref>/<path_to_file>"></script>
~~~
{: .bash}

For instance, the file `no-all-all.geojson` has been stored in the lesson repository at [https://github.com/annefou/metos_python/blob/gh-pages/data/no-all-all.geojson](https://github.com/annefou/metos_python/blob/gh-pages/data/no-all-all.geojson).

Then to visualize it, use:

~~~
<script src="https://embed.github.com/view/geojson/annefou/metos_python/gh-pages/data/no-all-all.geojson>"></script>
~~~
{: .bash} 


<script src="https://embed.github.com/view/geojson/annefou/metos_python/gh-pages/data/no-all-all.geojson"></script>

However, there is a number of limitations as described in the [documentation](http://jupyter-gmaps.readthedocs.io/en/latest/authentication.html).



> ## Other interesting python packages 
>
> Packages that are worth to mention for analyzing spatio-temporal data:
> 
> - "matplotlib, geopandas, pynio & pyngl, pyqgis, plotly, bokeh, gmaps, folium, cartopy, iris"
> - "nodebox-opengl - For playing around with animations"
> - "pandas, pytables, fiona, descartes, pyproj"
{: .callout}
