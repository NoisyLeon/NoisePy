import noisedbase
import numpy as np
import timeit

#-----------------------------------------
# New xcorr database
#-----------------------------------------
# dset = noisedbase.noiseASDF('/work1/leon/ALASKA_work/ASDF_data/xcorr_Alaska_20180221.h5')
# dset = noisedbase.noiseASDF('/work1/leon/ALASKA_work/ASDF_data/xcorr_Alaska_20190218.h5')
dset = noisedbase.noiseASDF('/work1/leon/ALASKA_work/ASDF_data/xcorr_Alaska_RTZ_ray_20190314_all_snr_10.h5')
# 
# 
# # dset.xcorr_stack(datadir='/scratch/summit/life9360/ALASKA_work/COR_work_dir', startyear=1991, startmonth=1, endyear=2018, endmonth=1)
# 
# dset.xcorr_prephp(outdir='/work1/leon/ALASKA_work/xcorr_working_dir/pre_disp', mapfile='./MAPS_ALASKA/smpkolya_phv')
# # # # # 
# dset.xcorr_aftan(prephdir='/work1/leon/ALASKA_work/xcorr_working_dir/pre_disp_R')
# # # # # # 
# dset.interp_disp()
# # # dset.xcorr_raytomoinput(outdir='/work1/leon/ALASKA_work/xcorr_working_dir/raytomo_input_20190131_selected_three_lambda', staxml='/home/leon/code/DataRequest/ALASKA.xml',\
# # #         netcodelst=['AK', 'TA', 'PO', 'XR', 'AV', 'XN', 'XY', 'CN', 'US'])
# dset.xcorr_raytomoinput(outdir='/work1/leon/ALASKA_work/xcorr_working_dir_Rayleigh/raytomo_input_20190131_all_three_lambda_snr_10', snr_thresh=10.)
# dset.xcorr_get_field(snr_thresh=10.)
# dset.write_stationtxt('sta_alaska.lst')

#-----------------------------------------
# Appending xcorr database
#-----------------------------------------
# dset = noisedbase.noiseASDF('/work1/leon/ALASKA_work/ASDF_data/xcorr_Alaska_20190312.h5')
# dset.xcorr_append(inasdffname='/work1/leon/ALASKA_work/ASDF_data/xcorr_Alaska_20180221.h5', inchannels=['LHZ'], verbose=False, \
#                    datadir='/work1/leon/ALASKA_work/COR_work_dir', startyear=2018, startmonth=1, endyear=2019, endmonth=1, fnametype=2)


# dset = noisedbase.noiseASDF('/work1/leon/ALASKA_work/ASDF_data/xcorr_Alaska_20190311_002.h5')
# dset.xcorr_append(inasdffname='/work1/leon/ALASKA_work/ASDF_data/xcorr_Alaska_20180221.h5_bk', inchannels=['LHZ'], verbose=False, \
#                    datadir='/work1/leon/ALASKA_work/COR_work_dir', startyear=2018, startmonth=1, endyear=2019, endmonth=1, fnametype=2)

# dset.plot_waveforms_monthly(datadir='/work2/leon/COR_ALASKA_dir', monthdir='2018.JAN', staxml='../DataRequest/glims_5000.xml')