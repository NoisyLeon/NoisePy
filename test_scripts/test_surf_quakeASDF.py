import ASDFDBase
import numpy as np
import timeit
dset=ASDFDBase.quakeASDF('../ref_ALASKA.h5')
# dset.get_events(startdate='2010-9-21', enddate='2015-9-21', Mmin=5.5)

# dset=ASDFDBase.quakeASDF('../WUS_quake_inv.h5')
# dset=ASDFDBase.quakeASDF('../WUS_quake_mp_test.h5')
dset=ASDFDBase.quakeASDF('../WUS_quake_test.h5')
# dset.get_events(startdate='2007-9-28', enddate='2008-1-29', Mmin=6.6, magnitudetype='ms')
# dset.get_stations(channel='LHZ', station='R11A')
# dset.get_events(startdate='2007-9-28', enddate='2007-9-29', Mmin=6.6, magnitudetype='ms', maxlatitude=0)
# dset.get_stations(startdate='2007-9-26', enddate='2007-9-30', channel='LHZ', 
#         minlatitude=31.0, maxlatitude=49.0, minlongitude=-124.0, maxlongitude=-102.0)
# dset.get_stations(startdate='2007-9-26', enddate='2007-9-30', channel='LHZ', 
#         minlatitude=44.0, maxlatitude=47.5, minlongitude=242.5, maxlongitude=245)
# dset.get_surf_waveforms(longitude=(-124.0-102.0)/2., latitude=(31+49)/2., channel='LHZ')
t1=timeit.default_timer()
dset.get_surf_waveforms(channel='LHZ', verbose=True)
# dset.get_surf_waveforms_mp(outdir='./downloaded_waveforms', subsize=1000, deletemseed=True, nprocess=4)
t2=timeit.default_timer()
print t2-t1, 'sec'
# st=dset.array_processing()
# dset.quake_prephp(outdir='../PRE_PHP')

# try:
#     del dset.auxiliary_data.DISPbasic1
# except:
#     pass
# try:
#     del dset.auxiliary_data.DISPbasic2
# except:
#     pass
# try:
#     del dset.auxiliary_data.DISPpmf1
# except:
#     pass
# try:
#     del dset.auxiliary_data.DISPpmf2
# except:
#     pass
# 
# dset.quake_aftan_mp(prephdir='../PRE_PHP_R', outdir='/work3/leon/WUS_workingdir', nprocess=10)
# try:
#     del dset.auxiliary_data.DISPpmf2interp
# except:
#     pass
# dset.interp_disp()
# try:
#     del dset.auxiliary_data.FieldDISPpmf2interp
# except:
#     pass
# dset.quake_get_field()
# atr=dset.quake_aftan(prephdir='../PRE_PHP_R')



# import eikonaltomo
# # # 
# dset=eikonaltomo.EikonalTomoDataSet('../eikonal_tomo_quake.h5')
# # dset2=eikonaltomo.EikonalTomoDataSet('../eikonal_tomo_quake_mp.h5')
# dset.set_input_parameters(minlon=235., maxlon=255., minlat=31., maxlat=50., pers=np.array([60.]))
# # dset2.set_input_parameters(minlon=235., maxlon=255., minlat=31., maxlat=50., pers=np.array([60.]))
# # dset.set_input_parameters(minlon=235., maxlon=255., minlat=31., maxlat=50.)
# # dset.xcorr_eikonal_mp(inasdffname='../COR_WUS.h5', workingdir='./eikonal_working', fieldtype='Tph', channel='ZZ', data_type='FieldDISPpmf2interp', nprocess=10)
# field=dset.quake_eikonal(inasdffname='../WUS_quake_eikonal.h5', workingdir='./eikonal_working', fieldtype='Tph', channel='Z',
#             data_type='FieldDISPpmf2interp', amplplc=True)
# dset2.quake_eikonal_mp(inasdffname='../WUS_quake_eikonal.h5', workingdir='./eikonal_working', fieldtype='Tph', channel='Z',
#         data_type='FieldDISPpmf2interp', amplplc=True)
#
# # t1=timeit.default_timer()
# dset.eikonal_stack()
# # t2=timeit.default_timer()
# # print t2-t1
# # dset.eikonal_stack()
# # dset._get_lon_lat_arr('Eikonal_run_0')
# dset.get_data4plot(period=28.)
# dset.np2ma()
# dset.plot_vel_iso(vmin=3.4, vmax=4.0)