import noisedbase
import numpy as np
import timeit
import matplotlib.pyplot as plt

import eikonaltomo
dset=eikonaltomo.EikonalTomoDataSet('/scratch/summit/life9360/ALASKA_work/hdf5_files/eikonal_xcorr_tomo_Alaska_150km.h5')
# dset.compare_raytomo('/scratch/summit/life9360/ALASKA_work/hdf5_files/ray_tomo_Alaska_gr.h5', 1, 2, 1, 16.)
# dset=eikonaltomo.EikonalTomoDataSet('/scratch/summit/life9360/ALASKA_work/hdf5_files/eikonal_xcorr_tomo_Alaska_no_cur_discard.h5')
# dset=eikonaltomo.EikonalTomoDataSet('/scratch/summit/life9360/ALASKA_work/hdf5_files/eikonal_xcorr_tomo_Alaska_cur_no_near.h5')
# # dset.set_input_parameters(minlon=188, maxlon=238, minlat=52, maxlat=72, pers=np.array([16.]))
# dset.set_input_parameters(minlon=188, maxlon=238, minlat=52, maxlat=72, pers=np.array([8., 10., 16., 20., 30., 40., 50., 60.]))
# # # # 
# dset.xcorr_eikonal_mp(inasdffname='/scratch/summit/life9360/ALASKA_work/ASDF_data/xcorr_Alaska.h5', \
#                       workingdir='/scratch/summit/life9360/ALASKA_work/eikonal_working_150km', \
#                    fieldtype='Tph', channel='ZZ', data_type='FieldDISPpmf2interp', nprocess=24, subsize=100)



# dset=eikonaltomo.EikonalTomoDataSet('/work3/leon/eikonal_xcorr_tomo_Alaska.h5')
# dset.set_input_parameters(minlon=188, maxlon=238, minlat=52, maxlat=72, pers=np.array([16.]))
# dset.set_input_parameters(minlon=188, maxlon=238, minlat=52, maxlat=72)
# # # 
# dset.xcorr_eikonal_mp(inasdffname='/work3/leon/xcorr_Alaska.h5', \
#                       workingdir='/work3/leon/eikonal_working', \
#                    fieldtype='Tph', channel='ZZ', data_type='FieldDISPpmf2interp', nprocess=10, subsize=100)


# dset.xcorr_eikonal(inasdffname='/work3/leon/xcorr_Alaska.h5', workingdir='../eikonal_working_debug')
# dset.xcorr_eikonal(inasdffname='/scratch/summit/life9360/ALASKA_work/ASDF_data/xcorr_Alaska.h5', workingdir='/scratch/summit/life9360/ALASKA_work/eikonal_working_debug')
# # #
# # # t1=timeit.default_timer()
# dset.eikonal_stack()
# # t2=timeit.default_timer()
# # print t2-t1
# dset.eikonal_stack(runid=1)
# # dset._get_lon_lat_arr('Eikonal_run_0')
# dset.get_data4plot(period=28.)
# dset.np2ma()
# dset.plot_vel_iso(vmin=3.4, vmax=4.0)

# dset=eikonaltomo.EikonalTomoDataSet('/scratch/summit/life9360/ALASKA_work/hdf5_files/eikonal_xcorr_tomo_Alaska_gr.h5')
# dset.set_input_parameters(minlon=188, maxlon=238, minlat=52, maxlat=72, pers=np.array([10., 16., 20., 30., 40.]))
# # dset.set_input_parameters(minlon=188, maxlon=238, minlat=52, maxlat=72)
# # # 
# dset.xcorr_eikonal_mp(inasdffname='/scratch/summit/life9360/ALASKA_work/ASDF_data/xcorr_Alaska.h5', \
#                       workingdir='/scratch/summit/life9360/ALASKA_work/eikonal_working_gr', \
#                    fieldtype='Tgr', channel='ZZ', data_type='FieldDISPpmf2interp', nprocess=10, subsize=100)

