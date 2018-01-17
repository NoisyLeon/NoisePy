import noisedbase
import numpy as np
import timeit

dset = noisedbase.noiseASDF('/scratch/summit/life9360/ALASKA_work/ASDF_data/xcorr_Alaska.h5')

ch=dset.xcorr_stack(datadir='/scratch/summit/life9360/ALASKA_work/COR_work_dir', startyear=1991, startmonth=1, endyear=2018, endmonth=1)

