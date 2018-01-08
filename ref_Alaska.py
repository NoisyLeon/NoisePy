import quakedbase
import numpy as np
import timeit
import matplotlib.pyplot as plt

# Initialize ASDF dataset
# dset=quakedbase.quakeASDF('/scratch/summit/life9360/ALASKA_work/ASDF_data/ref_Alaska.h5')
dset=quakedbase.quakeASDF('ref_Alaska.h5')
# dset.cat = quakedbase.obspy.read_events('/scratch/summit/life9360/ALASKA_work/quakeml/alaska_2017_aug.ml')
dset.cat = quakedbase.obspy.read_events('test.ml')
# print dset.events[0]
# Retrieving earthquake catalog
# ISC catalog
# dset.get_events(startdate='1991-01-01', enddate='2015-02-01', Mmin=5.5, magnitudetype='mb', gcmt=True)
# gcmt catalog
# dset.get_events(startdate='1991-01-01', enddate='2017-08-31', Mmin=5.5, magnitudetype='mb', gcmt=True)
# Getting station information
# dset.get_stations(channel='BH*', minlatitude=52., maxlatitude=72.5, minlongitude=-172., maxlongitude=-122.)

# Downloading data
# t1=timeit.default_timer()
# # st=dset.get_body_waveforms()
# dset.get_body_waveforms_mp( outdir='/scratch/summit/life9360/ALASKA_work/downloaded_P', verbose=True, nprocess=24)
# dset.get_body_waveforms( verbose=True)
# t2=timeit.default_timer()
# print t2-t1, 'sec'
# 
# # Computing receiver function
# dset.compute_ref()
dset.compute_ref_mp(outdir='../test_ref', verbose=True, nprocess=4)
# try: del dset.auxiliary_data.RefRHS
# except: pass
# 
# # Harmonic analysis
# dset.harmonic_stripping(outdir='.')
# t2=timeit.default_timer()
# print t2-t1, 'sec'
# dset.plot_ref(network='AE', station='U15A', phase='P', datatype='RefRHS')
# plt.show()