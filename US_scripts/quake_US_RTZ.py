import quakedbase
import numpy as np
import timeit
import matplotlib.pyplot as plt
import pyaftan

#------------------------------
# Initialize ASDF dataset
#------------------------------
dset    = quakedbase.quakeASDF('/work1/leon/US_work/ASDF_data/USCON_TA.h5')
# dset.copy_catalog_fromasdf(inasdffname = '/work1/leon/ALASKA_work/ASDF_data/surf_Alaska_Love.h5_cat_inv')

#------------------------------
# Retrieving earthquake catalog
#------------------------------
# ISC catalog
# dset.get_events(startdate='1991-01-01', enddate='2015-02-01', Mmin=5.5, magnitudetype='mb', gcmt=True)
# gcmt catalog
# dset.get_events(startdate='1991-01-01', enddate='2017-08-31', Mmin=5.5, magnitudetype='mb', gcmt=True)
# gcmt catalog, updated on 20180831
# dset.get_events(startdate='2017-09-01', enddate='2018-04-30', Mmin=5.5, magnitudetype='mb', gcmt=True)
# gcmt catalog, updated on 20190218
# dset.get_events(startdate='2018-05-01', enddate='2018-10-30', Mmin=5.5, magnitudetype='mb', gcmt=True)

#------------------------------
# Getting station information
#------------------------------
# # # dset.get_stations(channel='LHZ', minlatitude=52., maxlatitude=72.5, minlongitude=-172., maxlongitude=-122.)

#------------------------------
# Download/Read data
#------------------------------
# # # t1=timeit.default_timer()
# # 
# # # dset.read_surf_waveforms_DMT(datadir='/scratch/summit/life9360/ALASKA_work/surf_19950101_20170831', verbose=False)
# st = dset.get_love_waveforms(startdate='2004-04-01', verbose=False, channel='LHE,LHN,LHZ', minDelta=5.)

#------------------------------
# aftan analysis
#------------------------------
# dset.quake_prephp(outdir='/work1/leon/ALASKA_work/quake_working_dir_Love/pre_disp')
# inftan      = pyaftan.InputFtanParam()
# inftan.tmax = 100.
# inftan.tmin = 5.
# # # # 
# # # inftan.tmax = 120.
# # # inftan.tmin = 20.
# dset.quake_aftan(prephdir='/work1/leon/ALASKA_work/quake_working_dir_Love/pre_disp_L', inftan=inftan, channel='T')
# pers        = np.append( np.arange(11.)*2.+20., np.arange(10.)*5.+45.)
# dset.interp_disp(verbose=True, pers=pers, channel='T')
# dset.quake_get_field(pers=pers, channel='T')
