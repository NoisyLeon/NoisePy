import quakedbase
import numpy as np
import pyaftan
import obspy
from mpl_toolkits.basemap import Basemap, shiftgrid, cm
import numpy.ma as ma
import matplotlib.pyplot as plt
# dset=quakedbase.quakeASDF('/scratch/summit/life9360/WUS_quake.h5')
# dset.get_events(startdate='2005-1-01', enddate='2011-12-31', Mmin=5.5, magnitudetype='MS')
# dset.get_stations(startdate='2005-1-01', enddate='2011-12-31', channel='LHZ', network='TA,US,IU,CI,AZ,BK,NN,UU' ,
#         minlatitude=25.0, maxlatitude=50.0, minlongitude=-130.0, maxlongitude=-100.0)
# dset.get_surf_waveforms_mp(outdir='/work3/leon/downloaded_waveforms', subsize=1000, deletemseed=True, nprocess=6, snumb=1930)
# print dset.events[0]
# Retrieving earthquake catalog
# ISC catalog
# dset.get_events(startdate='1991-01-01', enddate='2015-02-01', Mmin=5.5, magnitudetype='mb', gcmt=True)
# gcmt catalog
# dset.get_events(startdate='1991-01-01', enddate='2017-08-31', Mmin=5.5, magnitudetype='mb', gcmt=True)
# # Getting station information
# dset.get_stations(channel='LHZ', minlatitude=52., maxlatitude=72.5, minlongitude=-172., maxlongitude=-122.)

# # Downloading data
# t1=timeit.default_timer()

# dset.read_surf_waveforms_DMT(datadir='/scratch/summit/life9360/ALASKA_work/surf_19950101_20170831', verbose=False)

# dset.quake_prephp(outdir='/scratch/summit/life9360/WUS_quake_working_dir/pre_disp')
# inftan      = pyaftan.InputFtanParam()
# inftan.tmax = 100.
# inftan.tmin = 5.
# dset.quake_aftan(prephdir='/scratch/summit/life9360/WUS_quake_working_dir/pre_disp_R', inftan=inftan)
# dset.interp_disp(verbose=True)
# dset.quake_get_field()



import eikonaltomo
# # # 
dset=eikonaltomo.EikonalTomoDataSet('/scratch/summit/life9360/eikonal_quake_debug.h5')
# # dset2=eikonaltomo.EikonalTomoDataSet('../eikonal_tomo_quake_mp.h5')
# dset.set_input_parameters(minlon=235., maxlon=260., minlat=31., maxlat=50., pers=np.array([60.]))
# dset.quake_eikonal(inasdffname='/scratch/summit/life9360/WUS_quake.h5', workingdir='/scratch/summit/life9360/eikonal_working', fieldtype='Tph', channel='Z',
#             data_type='FieldDISPpmf2interp', amplplc=True, cdist=None)

# dset=eikonaltomo.EikonalTomoDataSet('/scratch/summit/life9360/eikonal_quake_mp.h5')
# dset.set_input_parameters(minlon=235., maxlon=260., minlat=31., maxlat=50., pers=np.array([60.]))
# dset.quake_eikonal_mp(inasdffname='/scratch/summit/life9360/WUS_quake.h5', workingdir='/scratch/summit/life9360/eikonal_working_mp', fieldtype='Tph', channel='Z',
#             data_type='FieldDISPpmf2interp', amplplc=True, cdist=None)

# reason_n    = dset['Eikonal_run_0/60_sec/E00979/reason_n'].value
# reason_n_helm    = dset['Eikonal_run_0/60_sec/E00979/reason_n_helm'].value
# appV        = dset['Eikonal_run_0/60_sec/E00979/appV'].value
# corV        = dset['Eikonal_run_0/60_sec/E00979/corV'].value
# lplc_amp    = dset['Eikonal_run_0/60_sec/E00979/lplc_amp'].value
# amp         = dset['Eikonal_run_0/60_sec/E00979/amp'].value
# 
reason_n    = dset['Eikonal_run_0/60_sec/E01678/reason_n'].value
reason_n_helm    = dset['Eikonal_run_0/60_sec/E01678/reason_n_helm'].value
appV        = dset['Eikonal_run_0/60_sec/E01678/appV'].value
corV        = dset['Eikonal_run_0/60_sec/E01678/corV'].value
lplc_amp    = dset['Eikonal_run_0/60_sec/E01678/lplc_amp'].value
amp         = dset['Eikonal_run_0/60_sec/E01678/amp'].value
# 
# # 
# reason_n    = dset['Eikonal_run_0/60_sec/E01461/reason_n'].value
# reason_n_helm    = dset['Eikonal_run_0/60_sec/E01461/reason_n_helm'].value
# appV        = dset['Eikonal_run_0/60_sec/E01461/appV'].value
# corV        = dset['Eikonal_run_0/60_sec/E01461/corV'].value
# lplc_amp    = dset['Eikonal_run_0/60_sec/E01461/lplc_amp'].value
# amp         = dset['Eikonal_run_0/60_sec/E01461/amp'].value
# # 
# ##
# # header
# ##

nlat_grad       = dset.attrs['nlat_grad']
nlon_grad       = dset.attrs['nlon_grad']
nlat_lplc       = dset.attrs['nlat_lplc']
nlon_lplc       = dset.attrs['nlon_lplc']
minlon          = dset.attrs['minlon']
maxlon          = dset.attrs['maxlon']
minlat          = dset.attrs['minlat']
maxlat          = dset.attrs['maxlat']
dlon            = dset.attrs['dlon']
dlat            = dset.attrs['dlat']
Nlon            = int(dset.attrs['Nlon'])
Nlat            = int(dset.attrs['Nlat'])

lat_centre  = (maxlat+minlat)/2.0
lon_centre  = (maxlon+minlon)/2.0
projection  = 'lambert'

if projection=='merc':
    m       = Basemap(projection='merc', llcrnrlat=minlat-5., urcrnrlat=maxlat+5., llcrnrlon=minlon-5.,
                urcrnrlon=maxlon+5., lat_ts=20, resolution=resolution)
    m.drawparallels(np.arange(-80.0,80.0,5.0), labels=[1,0,0,1])
    m.drawmeridians(np.arange(-170.0,170.0,5.0), labels=[1,0,0,1])
    m.drawstates(color='g', linewidth=2.)
elif projection=='global':
    m       = Basemap(projection='ortho',lon_0=lon_centre, lat_0=lat_centre, resolution=resolution)
    m.drawparallels(np.arange(-80.0,80.0,10.0), labels=[1,0,0,1])
    m.drawmeridians(np.arange(-170.0,170.0,10.0), labels=[1,0,0,1])

elif projection=='regional_ortho':
    m1      = Basemap(projection='ortho', lon_0=minlon, lat_0=minlat, resolution='l')
    m       = Basemap(projection='ortho', lon_0=minlon, lat_0=minlat, resolution=resolution,\
                llcrnrx=0., llcrnry=0., urcrnrx=m1.urcrnrx/mapfactor, urcrnry=m1.urcrnry/3.5)
    m.drawparallels(np.arange(-80.0,80.0,10.0), labels=[1,0,0,0],  linewidth=2,  fontsize=20)
    m.drawmeridians(np.arange(-170.0,170.0,10.0),  linewidth=2)
elif projection=='lambert':
    distEW, az, baz = obspy.geodetics.gps2dist_azimuth(minlat, minlon,
                        minlat, maxlon) # distance is in m
    distNS, az, baz = obspy.geodetics.gps2dist_azimuth(minlat, minlon,
                        maxlat+2., minlon) # distance is in m
    m               = Basemap(width=distEW, height=distNS, rsphere=(6378137.00,6356752.3142), resolution='l', projection='lcc',\
                        lat_1=minlat, lat_2=maxlat, lon_0=lon_centre, lat_0=lat_centre+1)
    m.drawparallels(np.arange(-80.0,80.0,10.0), linewidth=1, dashes=[2,2], labels=[1,1,0,0], fontsize=15)
    m.drawmeridians(np.arange(-170.0,170.0,10.0), linewidth=1, dashes=[2,2], labels=[0,0,1,0], fontsize=15)
m.drawcoastlines(linewidth=1.0)
m.drawcountries(linewidth=1.)
m.fillcontinents(lake_color='#99ffff',zorder=0.2)
m.drawmapboundary(fill_color="white")
dset._get_lon_lat_arr()
m.drawstates()
#

# # plot app V
# data        = np.zeros(dset.lonArr.shape)
# mask        = np.ones(dset.lonArr.shape, dtype=np.bool)
# data[nlat_grad:-nlat_grad, nlon_grad:-nlon_grad]\
#                         = appV
# mask[nlat_grad:-nlat_grad, nlon_grad:-nlon_grad]\
#                         = reason_n != 0.
# mdata       = ma.masked_array(data, mask=mask )
# x, y    = m(dset.lonArr, dset.latArr)
# # 
# vmin=3.5
# vmax=4.3
# import pycpt
# cmap    = pycpt.load.gmtColormap('./cv.cpt')
# im      = m.pcolormesh(x, y, mdata, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
# cb      = m.colorbar(im, "bottom", size="3%", pad='2%')
# cb.ax.tick_params(labelsize=10)
# 
# 
data        = np.zeros(dset.lonArr.shape)
mask        = np.ones(dset.lonArr.shape, dtype=np.bool)
data[nlat_lplc:-nlat_lplc, nlat_lplc:-nlat_lplc]\
                        = lplc_amp
mask[nlat_lplc:-nlat_lplc, nlat_lplc:-nlat_lplc]\
                        = reason_n_helm != 0.
mdata       = ma.masked_array(data, mask=mask )
x, y    = m(dset.lonArr, dset.latArr)

vmin=-7e-3
vmax=7e-3
import pycpt
cmap    = pycpt.load.gmtColormap('./cv.cpt')
im      = m.pcolormesh(x, y, mdata, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
cb      = m.colorbar(im, "bottom", size="3%", pad='2%')
cb.ax.tick_params(labelsize=10)

# data        = np.zeros(dset.lonArr.shape)
# mask        = np.ones(dset.lonArr.shape, dtype=np.bool)
# data[nlat_lplc:-nlat_lplc, nlat_lplc:-nlat_lplc]\
#                         = corV
# mask[nlat_lplc:-nlat_lplc, nlat_lplc:-nlat_lplc]\
#                         = reason_n_helm != 0.
# mdata       = ma.masked_array(data, mask=mask )
# x, y    = m(dset.lonArr, dset.latArr)
# m.drawstates()
# vmin=3.5
# vmax=4.3
# import pycpt
# cmap    = pycpt.load.gmtColormap('./cv.cpt')
# im      = m.pcolormesh(x, y, mdata, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
# cb      = m.colorbar(im, "bottom", size="3%", pad='2%')
# cb.ax.tick_params(labelsize=10)


# plot app V
# data        = np.zeros(dset.lonArr.shape)
# mask        = np.ones(dset.lonArr.shape, dtype=np.bool)
# data        = amp
# mask[nlat_lplc:-nlat_lplc, nlat_lplc:-nlat_lplc]\
#                         = reason_n_helm != 0.
# mdata       = ma.masked_array(data, mask=mask )
# x, y    = m(dset.lonArr, dset.latArr)
# # 
# vmin=2e-7
# vmax=4e-7
# import pycpt
# cmap    = pycpt.load.gmtColormap('./cv.cpt')
# im      = m.pcolormesh(x, y, mdata, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
# # im      = m.pcolormesh(x, y, mdata, cmap=cmap, shading='gouraud')
# cb      = m.colorbar(im, "bottom", size="3%", pad='2%')
# cb.ax.tick_params(labelsize=10)

# t1=timeit.default_timer()
# dset.vel_stack(helmholtz=False)
# dset1=eikonaltomo.EikonalTomoDataSet('/scratch/summit/life9360/eikonal_quake_eik.h5')
# dset1.vel_stack(runid=0, helmholtz=False)
# 
# dset2=eikonaltomo.EikonalTomoDataSet('/scratch/summit/life9360/eikonal_quake_helm.h5')
# dset2.vel_stack(runid=0, helmholtz=True)
# # t2=timeit.default_timer()
# # print t2-t1
# # dset.eikonal_stack()
# # dset._get_lon_lat_arr()
# dset.get_data4plot(period=28.)
# dset.np2ma()
# dset.plot_vel_iso(vmin=3.4, vmax=4.0)