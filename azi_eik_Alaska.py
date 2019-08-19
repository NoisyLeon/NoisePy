import noisedbase
import numpy as np
import timeit
import matplotlib.pyplot as plt
import eikonaltomo
import raytomo
#-----------------------
#initialization
#-----------------------
# dset    = eikonaltomo.EikonalTomoDataSet('/work1/leon/ALASKA_work/hdf5_files/eikonal_xcorr_tomo_Alaska_all_20190318_250km_snr_10.h5')
# dset    = eikonaltomo.EikonalTomoDataSet('/work1/leon/ALASKA_work/hdf5_files/eikonal_xcorr_tomo_Alaska_TA_AK_20190317_250km_azi_2deg_40bins.h5')
# dset    = eikonaltomo.EikonalTomoDataSet('/work1/leon/ALASKA_work/hdf5_files/eikonal_xcorr_tomo_Alaska_TA_AK_20190317_250km_azi_40sec_2deg.h5')
# dset    = eikonaltomo.EikonalTomoDataSet('/work1/leon/ALASKA_work/hdf5_files/eikonal_xcorr_tomo_Alaska_TA_AK_20190317_250km_azi_2deg.h5')
# dset    = eikonaltomo.EikonalTomoDataSet('/work1/leon/ALASKA_work/hdf5_files/eikonal_xcorr_tomo_Alaska_all_20190317_250km_azi_40sec_2deg.h5')
dset    = eikonaltomo.EikonalTomoDataSet('/work1/leon/ALASKA_work/hdf5_files/xcorr_bks/eikonal_xcorr_tomo_Alaska_TA_AK_20190317_250km.h5')
# dset    = eikonaltomo.EikonalTomoDataSet('/work1/leon/ALASKA_work/hdf5_files/eikonal_xcorr_tomo_Alaska_TA_AK_20190317_250km_azi_10_50.h5')
# cmap = raytomo.discrete_cmap(6, 'jet')
# dset.set_input_parameters(minlon=188, maxlon=238, minlat=52, maxlat=72)
# dset.set_input_parameters(minlon=188, maxlon=238, minlat=52, maxlat=72, pers=np.array([20.]))
# dset.set_input_parameters(minlon=188, maxlon=238, minlat=52, maxlat=72, pers=np.array([10., 20., 30., 40., 50.]))
# # # # # # # 
# # # # # # # #------------------------------
# # # # # # # # perform eikonal tomography
# # # # # # # #------------------------------
# # # # # dset.xcorr_eikonal_mp_lowmem(inasdffname='/work1/leon/ALASKA_work/ASDF_data/xcorr_Alaska_RTZ_ray_20190314_all.h5', \
# # # # #                 workingdir='/work1/leon/ALASKA_work/eikonal_working_all_20190318', \
# # # # #                    fieldtype='Tph', channel='ZZ', data_type='FieldDISPpmf2interp', nprocess=30, subsize=1000, mindp=10., cdist=250.)
# # # # # # # 
# # # # # # # # dset.xcorr_eikonal(inasdffname='/scratch/summit/life9360/ALASKA_work/ASDF_data/xcorr_Alaska.h5', \
# # # # # # # #                       workingdir='/scratch/summit/life9360/ALASKA_work/eikonal_working_debug', \
# # # # # # # #                    fieldtype='Tph', channel='ZZ', data_type='FieldDISPpmf2interp', mindp=10., deletetxt=False)
# # # # # # 
# # # # # # #------------------------------
# # # # # # # perform eikonal stacking
# # # # # # #------------------------------
# # 
# dset.eikonal_stack_mp(anisotropic=True, spacing_ani=2., N_bin=20)
# # # # t2=timeit.default_timer()
# # # # # print t2-t1
# # # dset.eikonal_stack(runid=0)
# # # 
# dset.compute_azi_aniso()
#------------------------------
# compare with ray tomo
#------------------------------
# dset.compare_raytomo('/work1/leon/ALASKA_work/hdf5_files/ray_tomo_Alaska_20190318.h5', 1, 1, 0, 10.)

# f2d = dset.plot_travel_time(inasdffname='/work1/leon/ALASKA_work/ASDF_data/xcorr_Alaska_20190218.h5', netcode='TA', stacode='H22K', period=40.)
