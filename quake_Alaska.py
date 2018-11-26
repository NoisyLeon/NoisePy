import quakedbase
import numpy as np
import timeit
# import matplotlib.pyplot as plt
import pyaftan

# Initialize ASDF dataset
# dset    = quakedbase.quakeASDF('/work1/leon/ALASKA_work/ASDF_data/surf_Alaska_tmin_20_tmax_120.h5')
# # # dset.add_quakeml('/scratch/summit/life9360/ALASKA_work/quakeml/alaska_2017_aug.ml')
# # # print dset.events[0]
# # 
# # # Retrieving earthquake catalog
# # # ISC catalog
# # # dset.get_events(startdate='1991-01-01', enddate='2015-02-01', Mmin=5.5, magnitudetype='mb', gcmt=True)
# # # gcmt catalog
# # # dset.get_events(startdate='1991-01-01', enddate='2017-08-31', Mmin=5.5, magnitudetype='mb', gcmt=True)
# # # gcmt catalog, 20180831
# # # dset.get_events(startdate='2017-09-01', enddate='2018-04-30', Mmin=5.5, magnitudetype='mb', gcmt=True)
# # 
# # # Getting station information
# # # dset.get_stations(channel='LHZ', minlatitude=52., maxlatitude=72.5, minlongitude=-172., maxlongitude=-122.)
# # 
# # # Downloading data
# # # t1=timeit.default_timer()
# # 
# # # dset.read_surf_waveforms_DMT(datadir='/scratch/summit/life9360/ALASKA_work/surf_19950101_20170831', verbose=False)
# # # dset.get_surf_waveforms(startdate='2017-09-01', verbose=False)
# # 
# # # dset.quake_prephp(outdir='/work1/leon/ALASKA_work/quake_working_dir/pre_disp')
# inftan      = pyaftan.InputFtanParam()
# # # inftan.tmax = 100.
# # # inftan.tmin = 5.
# 
# inftan.tmax = 120.
# inftan.tmin = 20.
# dset.quake_aftan(prephdir='/work1/leon/ALASKA_work/quake_working_dir/pre_disp_R', inftan=inftan)
# pers    = np.append( np.arange(11.)*2.+20., np.arange(10.)*5.+45.)
# dset.interp_disp(verbose=True, pers=pers)
# dset.quake_get_field(pers=pers)

# 
import eikonaltomo
# 
# dset    = eikonaltomo.EikonalTomoDataSet('/work1/leon/ALASKA_work/hdf5_files/eikonal_quake_20181008_iso.h5')
# dset2   = eikonaltomo.EikonalTomoDataSet('/work1/leon/ALASKA_work/hdf5_files/eikonal_xcorr_tomo_Alaska_TA_AK_20180814_250km_ani.h5')
# 
# data1   = dset.plot_azimuthal_single_point(inlat=65., inlon=-145.+360., runid=1, period=50., fitdata=True, getdata=True)
# 
# data2   = dset2.plot_azimuthal_single_point(inlat=65., inlon=-145.+360., runid=0, period=50., fitdata=True, getdata=True)
# 
# import matplotlib.pyplot as plt
# plt.errorbar(data1[0]+180., data1[1], yerr=data1[2], fmt='o')
# plt.plot(data1[0]+180., data1[3]*np.ones(data1[0].size))
# plt.errorbar(data2[0]+180., data2[1], yerr=data2[2], fmt='o')
# plt.plot(data2[0]+180., data2[3]*np.ones(data2[0].size))
# plt.show()

dset    = eikonaltomo.EikonalTomoDataSet('/work1/leon/ALASKA_work/hdf5_files/eikonal_quake_debug_001.h5')
# # 
pers    = np.array([80.])
dset.set_input_parameters(minlon=188, maxlon=238, minlat=52, maxlat=72, pers=pers)

dset.quake_eikonal(inasdffname='/work1/leon/ALASKA_work/ASDF_data/surf_Alaska_for_mp.h5', \
    workingdir='/work1/leon/Alaska_quake_eikonal_working', fieldtype='Tph', channel='Z', \
        data_type='FieldDISPpmf2interp', amplplc=True, cdist=None, btime_qc='2006-01-01', deletetxt=False)

# dset.eikonal_stack(runid=0, anisotropic=False)
# dset.helm_stack(runid=4, anisotropic=False)