import quakedbase
import numpy as np
import timeit
import matplotlib.pyplot as plt
import pyaftan

import eikonaltomo
# # 
dset    = eikonaltomo.EikonalTomoDataSet('/work1/leon/ALASKA_work/hdf5_files/eikonal_quake_20190220.h5')
# dset2    = eikonaltomo.EikonalTomoDataSet('/work1/leon/ALASKA_work/hdf5_files/eikonal_xcorr_tomo_Alaska_TA_AK_20190218_250km.h5')
# pers    = np.append( np.arange(11.)*2.+20., np.arange(10.)*5.+45.)
# dset.set_input_parameters(minlon=188, maxlon=238, minlat=52, maxlat=72, pers=pers)
# # # 
# # # dset.quake_eikonal_mp_lowmem(inasdffname='/work1/leon/ALASKA_work/ASDF_data/surf_Alaska.h5', \
# # #     workingdir='/work1/leon/Alaska_quake_eikonal_working_mp_20190220', fieldtype='Tph', channel='Z', \
# # #         data_type='FieldDISPpmf2interp', amplplc=True, cdist=250., nprocess=30, btime_qc='2006-01-01', deletetxt=True)
# # 
# dset.eikonal_stack(runid=0, anisotropic=True)
# dset.helm_stack(runid=0, anisotropic=False, dv_thresh=0.2)
# 
dset.compare_eiktomo(ineiktomofname='/work1/leon/ALASKA_work/hdf5_files/eikonal_xcorr_tomo_Alaska_all_20190318_250km_snr_10.h5', \
                     inrunid=0, runid=0, period=36., Nmeasure=50, helm=False)