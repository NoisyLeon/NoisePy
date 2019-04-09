import noisedbase
import numpy as np
import timeit

#-----------------------------------------
# New xcorr database
#-----------------------------------------

dset = noisedbase.noiseASDF('/work1/leon/US_work/ASDF_data/temp_2004_2008.h5')

dset.compute_xcorr(datadir = '/work2/leon/temp_working_2004_2008',
        startdate='20060801', enddate='20081231', nprocess=15, fastfft=True)


