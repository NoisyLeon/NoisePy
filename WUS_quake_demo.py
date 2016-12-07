import ASDFDBase
import numpy as np

dset=ASDFDBase.quakeASDF('/work3/leon/WUS_quake.h5')
# dset.get_events(startdate='2005-1-01', enddate='2011-12-31', Mmin=5.5, magnitudetype='MS')
# dset.get_stations(startdate='2005-1-01', enddate='2011-12-31', channel='LHZ', network='TA,US,IU,CI,AZ,BK,NN,UU' ,
#         minlatitude=25.0, maxlatitude=50.0, minlongitude=-130.0, maxlongitude=-100.0)
dset.get_surf_waveforms_mp(outdir='/work3/leon/downloaded_waveforms', subsize=1000, deletemseed=True, nprocess=6, snumb=1930)
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
# # dset._get_lon_lat_arr()
# dset.get_data4plot(period=28.)
# dset.np2ma()
# dset.plot_vel_iso(vmin=3.4, vmax=4.0)