import quakedbase
import numpy as np
import timeit
import matplotlib.pyplot as plt

# Initialize ASDF dataset
dset=quakedbase.quakeASDF('../ref_AK.FID.h5')
# # # # Retrieving earthquake catalog
# dset.get_events(startdate='2010-07-26', enddate='2016-12-31', Mmin=5.5, magnitudetype='mb', add2dbase=True)
# # # # # # # Getting station information
# dset.get_stations(channel='BH*', station='FID', network='AK')
# # # # # # Downloading data
# # # # t1=timeit.default_timer()
# st=dset.get_body_waveforms()
# st=dset.get_body_waveforms_mp( outdir='./downloaded_P', verbose=False, nprocess=6)
# t2=timeit.default_timer()
# print t2-t1, 'sec'
#
# # Computing receiver function
# dset.compute_ref()
# dset.compute_ref_mp(outdir='/work3/leon/ref_working', verbose=True, nprocess=4)
# try: del dset.auxiliary_data.RefRHS
# except: pass
# 
# # Harmonic analysis
# dset.read_quakeml('U15A_cat.ml')
# dset.harmonic_stripping(outdir='../test_ref_working')
# dset.harmonic_stripping()
# t2=timeit.default_timer()
# print t2-t1, 'sec'
# dset.plot_ref(network='AE', station='U15A', phase='P', datatype='RefRHS')
# plt.show()