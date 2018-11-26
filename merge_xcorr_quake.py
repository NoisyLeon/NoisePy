import eikonaltomo
import numpy as np

dset    = eikonaltomo.hybridTomoDataSet('/work1/leon/ALASKA_work/hdf5_files/eikonal_hybrid_20181101.h5')

# pers    = np.append( np.arange(18.)*2.+6., np.arange(4.)*5.+45.)
# pers    = np.append( pers, np.arange(6.)*5.+65.)
# dset.set_input_parameters(minlon=188, maxlon=238, minlat=52, maxlat=72, pers=pers)
# 
# dset.read_xcorr(inh5fname = '/work1/leon/ALASKA_work/hdf5_files/eikonal_xcorr_tomo_Alaska_TA_AK_20180814_250km.h5')
# dset.read_quake(inh5fname = '/work1/leon/ALASKA_work/hdf5_files/eikonal_quake_20181030.h5')

# dset.hybrid_eikonal_stack()

# dset.merge_raytomo(inrayfname='/work1/leon/ALASKA_work/hdf5_files/ray_tomo_Alaska_20180920.h5', inrunid=2, \
#                    percentage=0.9, num_thresh=100., gstd_thresh=150.)

# dset.interp_surface(dlon=1., dlat=0.5)