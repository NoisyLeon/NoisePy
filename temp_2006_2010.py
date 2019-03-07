import noisedbase
import numpy as np
import timeit

#-----------------------------------------
# New xcorr database
#-----------------------------------------

dset = noisedbase.noiseASDF('/work1/leon/ALASKA_work/ASDF_data/temp_2006_2010.h5')

dset.compute_xcorr(datadir = '/work2/leon/temp_working_2006_2010',
        startdate='20060101', enddate='20101231', nprocess=6, fastfft=True)

