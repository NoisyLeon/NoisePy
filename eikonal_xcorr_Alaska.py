import noisedbase
import numpy as np
import timeit
import matplotlib.pyplot as plt
import eikonaltomo

# dset=eikonaltomo.EikonalTomoDataSet('/scratch/summit/life9360/ALASKA_work/hdf5_files/eikonal_xcorr_tomo_Alaska_TA_AK_20180718_10sec.h5')
# dset    = eikonaltomo.EikonalTomoDataSet('/scratch/summit/life9360/ALASKA_work/hdf5_files/eikonal_xcorr_tomo_Alaska_TA_AK_20180813.h5')

dset    = eikonaltomo.EikonalTomoDataSet('/work1/leon/ALASKA_work/hdf5_files/eikonal_xcorr_tomo_Alaska_TA_AK_20180814_250km.h5')
# # dset=eikonaltomo.EikonalTomoDataSet('/scratch/summit/life9360/ALASKA_work/hdf5_files/eikonal_xcorr_tomo_Alaska_raydbase_mp.h5')
# dset=eikonaltomo.EikonalTomoDataSet('/scratch/summit/life9360/ALASKA_work/hdf5_files/eikonal_xcorr_tomo_Alaska_raydbase.h5')

# dset=eikonaltomo.EikonalTomoDataSet('/scratch/summit/life9360/ALASKA_work/hdf5_files/eikonal_xcorr_tomo_Alaska_all_T_10sec_only_20180712.h5')
# dset=eikonaltomo.EikonalTomoDataSet('/scratch/summit/life9360/ALASKA_work/hdf5_files/eikonal_xcorr_tomo_Alaska_TA_AK_20180411.h5')
# dset.compare_raytomo('/scratch/summit/life9360/ALASKA_work/hdf5_files/ray_tomo_Alaska_gr.h5', 1, 2, 1, 16.)
# # 
# dset.set_input_parameters(minlon=188, maxlon=238, minlat=52, maxlat=72)
# dset.set_input_parameters(minlon=188, maxlon=238, minlat=52, maxlat=72, pers=np.array([30., 40., 50.]))
# # # # dset.xcorr_eikonal_raydbase(inh5fname='/scratch/summit/life9360/ALASKA_work/hdf5_files/ray_tomo_Alaska_20180719_phase_test.h5', \
# # # #         workingdir='/scratch/summit/life9360/ALASKA_work/eikonal_working_raydbase_20180719')
# # # 
# # 
# # # dset.xcorr_eikonal_raydbase_mp(inh5fname='/scratch/summit/life9360/ALASKA_work/hdf5_files/ray_tomo_Alaska_20180719_phase_test_03.h5', \
# # #         workingdir='/scratch/summit/life9360/ALASKA_work/eikonal_working_raydbase_20180719_mp', nprocess=24)
# # # # dset.set_input_parameters(minlon=188, maxlon=238, minlat=52, maxlat=72)
# # dset.set_input_parameters(minlon=188, maxlon=238, minlat=52, maxlat=72, pers=np.array([30.]))
# 
# # dset.xcorr_eikonal_mp_lowmem(inasdffname='/scratch/summit/life9360/ALASKA_work/ASDF_data/xcorr_Alaska_20180809.h5', \
# #                 workingdir='/scratch/summit/life9360/ALASKA_work/eikonal_working_TA_AK_20180813', \
# #                    fieldtype='Tph', channel='ZZ', data_type='FieldDISPpmf2interp', nprocess=24, subsize=1000, mindp=10., cdist=250.)
# 
# # dset.xcorr_eikonal(inasdffname='/scratch/summit/life9360/ALASKA_work/ASDF_data/xcorr_Alaska.h5', \
# #                       workingdir='/scratch/summit/life9360/ALASKA_work/eikonal_working_debug', \
# #                    fieldtype='Tph', channel='ZZ', data_type='FieldDISPpmf2interp', mindp=10., deletetxt=False)
# 
# 
# 
# # # # # t1=timeit.default_timer()
# dset.eikonal_stack(anisotropic=True)
# # # t2=timeit.default_timer()
# # # print t2-t1
# dset.eikonal_stack(runid=0)
# # dset._get_lon_lat_arr('Eikonal_run_0')
# dset.get_data4plot(period=28.)
# dset.np2ma()
# dset.plot_vel_iso(vmin=3.4, vmax=4.0)

# dset=eikonaltomo.EikonalTomoDataSet('/scratch/summit/life9360/ALASKA_work/hdf5_files/eikonal_xcorr_tomo_Alaska_gr.h5')

# # dset.set_input_parameters(minlon=188, maxlon=238, minlat=52, maxlat=72)
# # # 
# dset.xcorr_eikonal_mp(inasdffname='/scratch/summit/life9360/ALASKA_work/ASDF_data/xcorr_Alaska.h5', \
#                       workingdir='/scratch/summit/life9360/ALASKA_work/eikonal_working_gr', \
#                    fieldtype='Tgr', channel='ZZ', data_type='FieldDISPpmf2interp', nprocess=10, subsize=100)
# dset.compare_raytomo('/work1/leon/ALASKA_work/hdf5_files/ray_tomo_Alaska_working.h5', 1, 2, 0, 12.)

