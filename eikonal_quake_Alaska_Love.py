import quakedbase
import numpy as np
import timeit
import matplotlib.pyplot as plt
import pyaftan

import eikonaltomo
# # 
dset    = eikonaltomo.EikonalTomoDataSet('/work1/leon/ALASKA_work/hdf5_files/eikonal_quake_Love_20190314.h5')
# pers    = np.append( np.arange(11.)*2.+20., np.arange(10.)*5.+45.)
# dset.set_input_parameters(minlon=188, maxlon=238, minlat=52, maxlat=72, pers=pers)
# # dset.set_input_parameters(minlon=188, maxlon=238, minlat=52, maxlat=72, pers=np.array([50.]))
# # # # # 
# dset.quake_eikonal_mp_lowmem(inasdffname='/work1/leon/ALASKA_work/ASDF_data/surf_Alaska_Love.h5', \
#     workingdir='/work1/leon/Alaska_quake_eikonal_Love_working_mp_20190314', fieldtype='Tph', channel='T', \
#         data_type='FieldDISPpmf2interp', amplplc=False, cdist=250., nprocess=30, btime_qc='2006-01-01', deletetxt=True)
# # # 
# dset.eikonal_stack(runid=0)
# dset.helm_stack(runid=0, anisotropic=False, dv_thresh=0.2)
# 
# dset.compare_eiktomo(ineiktomofname='/work1/leon/ALASKA_work/hdf5_files/eikonal_xcorr_tomo_Alaska_TA_AK_20190218_250km.h5', \
#                      inrunid=0, runid=0, period=40., Nmeasure=50, helm=False)