import quakedbase
import numpy as np
import pyaftan
dset=quakedbase.quakeASDF('/scratch/summit/life9360/WUS_quake.h5')
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
dset=eikonaltomo.EikonalTomoDataSet('/scratch/summit/life9360/eikonal_quake.h5')
# dset2=eikonaltomo.EikonalTomoDataSet('../eikonal_tomo_quake_mp.h5')
dset.set_input_parameters(minlon=235., maxlon=255., minlat=31., maxlat=50., pers=np.array([60.]))
# dset2.set_input_parameters(minlon=235., maxlon=255., minlat=31., maxlat=50., pers=np.array([60.]))
# dset.set_input_parameters(minlon=235., maxlon=255., minlat=31., maxlat=50.)
# dset.xcorr_eikonal_mp(inasdffname='../COR_WUS.h5', workingdir='./eikonal_working', fieldtype='Tph', channel='ZZ', data_type='FieldDISPpmf2interp', nprocess=10)
field=dset.quake_eikonal(inasdffname='/scratch/summit/life9360/WUS_quake.h5', workingdir='./eikonal_working', fieldtype='Tph', channel='Z',
            data_type='FieldDISPpmf2interp', amplplc=True)
# dset2.quake_eikonal_mp(inasdffname='../WUS_quake_eikonal.h5', workingdir='./eikonal_working', fieldtype='Tph', channel='Z',
#         data_type='FieldDISPpmf2interp', amplplc=True)

# t1=timeit.default_timer()
# dset.eikonal_stack()
# # t2=timeit.default_timer()
# # print t2-t1
# # dset.eikonal_stack()
# # dset._get_lon_lat_arr()
# dset.get_data4plot(period=28.)
# dset.np2ma()
# dset.plot_vel_iso(vmin=3.4, vmax=4.0)