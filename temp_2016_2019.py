import noisedbase
import numpy as np
import timeit

#-----------------------------------------
# New xcorr database
#-----------------------------------------

dset = noisedbase.noiseASDF('/work1/leon/ALASKA_work/ASDF_data/temp_2016_2019.h5')

dset.compute_xcorr(datadir = '/work2/leon/temp_working_2016_2019',
        startdate='20160101', enddate='20190131', nprocess=10, fastfft=True)

