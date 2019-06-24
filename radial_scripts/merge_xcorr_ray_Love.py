import eikonaltomo
import numpy as np
#-----------------------
#initialization
#-----------------------
dset    = eikonaltomo.hybridTomoDataSet('/work1/leon/ALASKA_work/hdf5_files/eikonal_hybrid_Love_20190318.h5')

# pers    = np.append( np.arange(18.)*2.+6., np.arange(4.)*5.+45.)
# dset.set_input_parameters(minlon=188, maxlon=238, minlat=52, maxlat=72, pers=pers)
# # # # 
# # # # #-----------------------------------------------
# # # # # read xcorr and earthquake eikonal resultls
# # # # #-----------------------------------------------
# # # # dset.read_xcorr(inh5fname = '/work1/leon/ALASKA_work/hdf5_files/eikonal_xcorr_Alaska_TA_AK_20190318_250km_Love.h5')
# # # # dset.read_quake(inh5fname = '/work1/leon/ALASKA_work/hdf5_files/eikonal_quake_Love_20190314.h5')
# # dset.hybrid_eikonal_stack()
# #-----------------------------------------------
# # read ray tomography results, uncertainties will also be estimated from eikonal maps
# # #-----------------------------------------------
# dset.merge_raytomo(inrayfname='/work1/leon/ALASKA_work/hdf5_files/ray_tomo_Alaska_Love_20190318.h5', inrunid=0, \
#                    percentage=0.9, num_thresh=50., gstd_thresh=200., Traymin=8., Traymax=50.)
# 
# #-----------------------------------------------------
# # interpolate to a given grid spacing
# #-----------------------------------------------------
# dset.interp_surface(dlon=1., dlat=0.5)

# import raytomo
# cmap    = raytomo.discrete_cmap(10, 'jet_r')
# dset.plot(0, 'vel_sem', 20., vmin=20., vmax=70., cmap=cmap, clabel='Uncertainties (m/sec)', semfactor=2.0, merged=True)