import noisedbase
import numpy as np
import timeit

#-----------------------------------------
# New xcorr database
#-----------------------------------------

dset = noisedbase.noiseASDF('/work1/leon/ALASKA_work/ASDF_data/temp_2011_2015.h5')

dset.compute_xcorr(datadir = '/work2/leon/temp_working_2011_2015',
        startdate='20110101', enddate='20151231', nprocess=6, fastfft=True)

