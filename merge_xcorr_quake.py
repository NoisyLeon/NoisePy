import eikonaltomo
import numpy as np
#-----------------------
#initialization
#-----------------------
dset    = eikonaltomo.hybridTomoDataSet('/work1/leon/ALASKA_work/hdf5_files/eikonal_hybrid_20190318.h5')

# pers    = np.append( np.arange(18.)*2.+6., np.arange(4.)*5.+45.)
# pers    = np.append( pers, np.arange(6.)*5.+65.)
# dset.set_input_parameters(minlon=188, maxlon=238, minlat=52, maxlat=72, pers=pers)
# # 
# # #-----------------------------------------------
# # # read xcorr and earthquake eikonal resultls
# # #-----------------------------------------------
# dset.read_xcorr(inh5fname = '/work1/leon/ALASKA_work/hdf5_files/eikonal_xcorr_tomo_Alaska_all_20190318_250km_snr_10.h5')
# dset.read_quake(inh5fname = '/work1/leon/ALASKA_work/hdf5_files/eikonal_quake_20190220.h5')
# dset.hybrid_eikonal_stack()
#-----------------------------------------------
# read ray tomography results, uncertainties will also be estimated from eikonal maps
#-----------------------------------------------
# dset.merge_raytomo(inrayfname='/work1/leon/ALASKA_work/hdf5_files/ray_tomo_Alaska_20190318.h5', inrunid=1, \
#                    percentage=0.9, num_thresh=50., gstd_thresh=150.)
# # # 
# # # #-----------------------------------------------------
# # # # interpolate to a given grid spacing
# # # #-----------------------------------------------------
# dset.interp_surface(dlon=1., dlat=0.5)

# import raytomo
# cmap    = raytomo.discrete_cmap(10, 'jet_r')
# dset.plot(0, 'vel_sem', 20., vmin=20., vmax=70., cmap=cmap, clabel='Uncertainties (m/sec)', semfactor=2.0, merged=True)

# dset.compare_eiktomo('/work1/leon/ALASKA_work/hdf5_files/eikonal_hybrid_Love_20190318.h5', 0, 0, 10., clabel='CLove - CRayleigh (km/sec)')