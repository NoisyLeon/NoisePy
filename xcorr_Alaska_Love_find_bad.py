import noisedbase
import numpy as np
import timeit

#-----------------------------------------
# New xcorr database
#-----------------------------------------

dset = noisedbase.noiseASDF('/work1/leon/ALASKA_work/ASDF_data/xcorr_Alaska_RTZ_lov_20190314_TA_AK.h5')
# dset.copy_stations(inasdffname = '/work1/leon/ALASKA_work/ASDF_data/xcorr_Alaska_20190218.h5', channel='LH*')

# dset.compute_xcorr(datadir = '/work2/leon/temp_working_2001_2005',
#         startdate='20010101', enddate='20051231', nprocess=10, fastfft=True)

# dset.xcorr_stack(datadir='/work2/leon/COR_ALASKA_dir', startyear=2001, startmonth=1, endyear=2019, endmonth=2, \
#                  inchannels=['LHN', 'LHE', 'LHZ'], fnametype=1)

# dset.xcorr_rotation()
# # # 
# # # dset.xcorr_prephp(outdir='/work1/leon/ALASKA_work/xcorr_working_dir/pre_disp', mapfile='./MAPS_ALASKA/smpkolya_phv')
# # # # # 
# dset.xcorr_aftan(prephdir='/work1/leon/ALASKA_work/xcorr_working_dir/pre_disp_L', channel='TT')
# # # # # # 
# dset.interp_disp(channel='TT')
# # dset.xcorr_raytomoinput(outdir='/work1/leon/ALASKA_work/xcorr_working_dir/raytomo_input_20190131_selected_three_lambda', staxml='/home/leon/code/DataRequest/ALASKA.xml',\
# #         netcodelst=['AK', 'TA', 'PO', 'XR', 'AV', 'XN', 'XY', 'CN', 'US'])
# dset.xcorr_raytomoinput(outdir='/work1/leon/ALASKA_work/xcorr_working_dir_Love/raytomo_input_20190131_all_three_lambda', snr_thresh=10., channel='TT')
# exclude_stalst='ex_sta.lst', \
dset.xcorr_raytomoinput_debug(outdir='/work1/leon/ALASKA_work/xcorr_debug', exclude_stalst='ex_sta.lst', \
                              snr_thresh=10., channel='TT', pers=np.array([50.]))
# dset.xcorr_get_field( channel='TT', snr_thresh=10., netcodelst=['AK', 'TA'])
# dset.write_stationtxt('sta_alaska.lst')

#-----------------------------------------
# Appending xcorr database
#-----------------------------------------
# dset = noisedbase.noiseASDF('/work1/leon/ALASKA_work/ASDF_data/xcorr_Alaska_20190218.h5')
# dset.xcorr_append(inasdffname='/work1/leon/ALASKA_work/ASDF_data/xcorr_Alaska_20180809.h5',\
#                    datadir='/work1/leon/ALASKA_work/COR_work_dir', startyear=2018, startmonth=7, endyear=2019, endmonth=1)

