import noisedbase
import numpy as np
import timeit
import matplotlib.pyplot as plt
import eikonaltomo
import raytomo
#-----------------------
#initialization
#-----------------------
# dset    = eikonaltomo.EikonalTomoDataSet('/work1/leon/ALASKA_work/hdf5_files/eikonal_xcorr_Alaska_TA_AK_20190317_250km_Love.h5')
# dset    = eikonaltomo.EikonalTomoDataSet('/work1/leon/ALASKA_work/hdf5_files/eikonal_xcorr_Alaska_all_20190318_250km_Love.h5')
dset    = eikonaltomo.EikonalTomoDataSet('/work1/leon/ALASKA_work/hdf5_files/eikonal_xcorr_Alaska_TA_AK_20190318_250km_Love_ani.h5')
# cmap = raytomo.discrete_cmap(6, 'jet')
# dset.set_input_parameters(minlon=188, maxlon=238, minlat=52, maxlat=72)
# dset.set_input_parameters(minlon=188, maxlon=238, minlat=52, maxlat=72, pers=np.array([20.]))
# # # # 
# # # # #------------------------------
# # # # # perform eikonal tomography
# # # # #------------------------------
# # dset.xcorr_eikonal_mp_lowmem(inasdffname='/work1/leon/ALASKA_work/ASDF_data/xcorr_Alaska_RTZ_lov_20190314_TA_AK.h5', \
# #                 workingdir='/work1/leon/ALASKA_work/eikonal_working_TA_AK_20190318_Love', \
# #                    fieldtype='Tph', channel='TT', data_type='FieldDISPpmf2interp', nprocess=12, subsize=1000, mindp=10., cdist=250.)
# # # 
# # # # dset.xcorr_eikonal(inasdffname='/scratch/summit/life9360/ALASKA_work/ASDF_data/xcorr_Alaska.h5', \
# # # #                       workingdir='/scratch/summit/life9360/ALASKA_work/eikonal_working_debug', \
# # # #                    fieldtype='Tph', channel='ZZ', data_type='FieldDISPpmf2interp', mindp=10., deletetxt=False)
# # 
# # #------------------------------
# # # perform eikonal stacking
# # #------------------------------
# # # # # # # t1=timeit.default_timer()
# # dset.eikonal_stack(anisotropic=True)
# dset.eikonal_stack(anisotropic=True, spacing_ani=.6, N_bin=20)
# t2=timeit.default_timer()
# print t2-t1
# dset.eikonal_stack(runid=0)


#------------------------------
# compare with ray tomo
#------------------------------
# dset.compare_raytomo('/work1/leon/ALASKA_work/hdf5_files/ray_tomo_Alaska_Love_20190318.h5', 1, 0, 0, 10.)

# f2d = dset.plot_travel_time(inasdffname='/work1/leon/ALASKA_work/ASDF_data/xcorr_Alaska_20190218.h5', netcode='TA', stacode='H22K', period=40.)
