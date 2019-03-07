import noisedbase
import numpy as np
import timeit
import matplotlib.pyplot as plt
import eikonaltomo

dset    = eikonaltomo.EikonalTomoDataSet('/work1/leon/eikonal_tomo_WUS.h5')

# dset.set_input_parameters(minlon=235, maxlon=255, minlat=31, maxlat=50, pers=np.array([24.]), optimize_spacing=False)
# 
# dset.eikonal_stack(runid=0, anisotropic=True)


# # dset._get_lon_lat_arr('Eikonal_run_0')


# dset=eikonaltomo.EikonalTomoDataSet('/scratch/summit/life9360/ALASKA_work/hdf5_files/eikonal_xcorr_tomo_Alaska_gr.h5')
# dset.set_input_parameters(minlon=188, maxlon=238, minlat=52, maxlat=72, pers=np.array([10., 16., 20., 30., 40.]))
# # dset.set_input_parameters(minlon=188, maxlon=238, minlat=52, maxlat=72)
# # # 
# dset.xcorr_eikonal_mp(inasdffname='/scratch/summit/life9360/ALASKA_work/ASDF_data/xcorr_Alaska.h5', \
#                       workingdir='/scratch/summit/life9360/ALASKA_work/eikonal_working_gr', \
#                    fieldtype='Tgr', channel='ZZ', data_type='FieldDISPpmf2interp', nprocess=10, subsize=100)
# dset.compare_raytomo('/work1/leon/ALASKA_work/hdf5_files/ray_tomo_Alaska_working.h5', 1, 2, 0, 12.)

