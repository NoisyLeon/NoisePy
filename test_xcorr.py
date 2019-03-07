import noisedbase
import numpy as np
import time

#-----------------------------------------
# New xcorr database
#-----------------------------------------

dset = noisedbase.noiseASDF('/work1/leon/ALASKA_work/ASDF_data/xcorr_Alaska_20190306_Love.h5')
# dset.copy_stations(inasdffname = '/work1/leon/ALASKA_work/ASDF_data/xcorr_Alaska_20190218.h5', channel='LH*')

# xcorr_lst = dset.compute_xcorr(datadir = '/work2/leon/temp_working_2001_2005', startdate='20010101', enddate='20051231')

dset.compute_xcorr(datadir = '/work2/leon/temp_working_2001_2005', startdate='20010101', enddate='20051231', nprocess=10, fastfft=True, subsize=20)

# t1=time.time()
# for i in range(30):
#     xcorr_lst[i*3].convert_amph_to_xcorr(datadir='/work2/leon/temp_working_2001_2005', fastfft=True)
# # 
# t2=time.time()
# for i in range(30):
#     xcorr_lst[i*3].convert_amph_to_xcorr(datadir='/work2/leon/temp_working_2001_2005', fastfft=False)
# t3=time.time()
# # 
# print t3-t2, t2-t1

# xcorr_lst[0].convert_amph_to_xcorr(datadir='/work2/leon/temp_working_2001_2005', verbose = True)
# 
# xcorr_lst[0].convert_amph_to_xcorr(datadir='/work2/leon/temp_working_2001_2005', Nref=Nref, fftw_plan=fftw_plan, verbose = True)

# t1=time.time()
# for i in range(30):
#     xcorr_lst[0].convert_amph_to_xcorr(datadir='/work2/leon/temp_working_2001_2005', fastfft=True)
# t2=time.time()
# 
# # t3=time.time()
# for i in range(30):
#     xcorr_lst[0].convert_amph_to_xcorr(datadir='/work2/leon/temp_working_2001_2005', fastfft=False)
# t3=time.time()
# 
# print t3-t2, t2-t1