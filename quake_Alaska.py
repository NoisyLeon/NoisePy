import quakedbase
import numpy as np
import timeit
import matplotlib.pyplot as plt
import pyaftan

# Initialize ASDF dataset
dset=quakedbase.quakeASDF('/scratch/summit/life9360/ALASKA_work/ASDF_data/surf_Alaska.h5')
# dset.add_quakeml('/scratch/summit/life9360/ALASKA_work/quakeml/alaska_2017_aug.ml')
# print dset.events[0]
# Retrieving earthquake catalog
# ISC catalog
# dset.get_events(startdate='1991-01-01', enddate='2015-02-01', Mmin=5.5, magnitudetype='mb', gcmt=True)
# gcmt catalog
# dset.get_events(startdate='1991-01-01', enddate='2017-08-31', Mmin=5.5, magnitudetype='mb', gcmt=True)
# Getting station information
# dset.get_stations(channel='LHZ', minlatitude=52., maxlatitude=72.5, minlongitude=-172., maxlongitude=-122.)

# Downloading data
# t1=timeit.default_timer()

# dset.read_surf_waveforms_DMT(datadir='/scratch/summit/life9360/ALASKA_work/surf_19950101_20170831', verbose=False)

# dset.quake_prephp(outdir='/scratch/summit/life9360/ALASKA_work/quake_working_dir/pre_disp')
# inftan      = pyaftan.InputFtanParam()
# inftan.tmax = 100.
# inftan.tmin = 5.
# dset.quake_aftan(prephdir='/scratch/summit/life9360/ALASKA_work/quake_working_dir/pre_disp_R', inftan=inftan)
# dset.interp_disp(verbose=True)
dset.quake_get_field()