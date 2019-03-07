import noisedbase
import numpy as np
import timeit

#-----------------------------------------
# New xcorr database
#-----------------------------------------

dset = noisedbase.noiseASDF('/work1/leon/ALASKA_work/ASDF_data/temp_2001_2005.h5')

dset.compute_xcorr(datadir = '/work2/leon/temp_working_2001_2005',
        startdate='20010101', enddate='20051231', nprocess=6, fastfft=True)


