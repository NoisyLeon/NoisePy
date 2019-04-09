import noisedbase
import numpy as np
import timeit

#-----------------------------------------
# New xcorr database
#-----------------------------------------

dset = noisedbase.noiseASDF('/work1/leon/US_work/ASDF_data/temp_2009_2011.h5')

dset.compute_xcorr(datadir = '/work2/leon/temp_working_2009_2011',
        startdate='20090101', enddate='20111231', nprocess=8, fastfft=True)


