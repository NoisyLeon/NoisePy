import noisedbase
import numpy as np
import timeit

dset = noisedbase.noiseASDF('/scratch/summit/life9360/ALASKA_work/ASDF_data/xcorr_Alaska_20180809.h5')
# dset = noisedbase.noiseASDF('../xcorr_Alaska.h5')

# dset.xcorr_stack(datadir='/scratch/summit/life9360/ALASKA_work/COR_work_dir', startyear=1991, startmonth=1, endyear=2018, endmonth=1)
# 
# dset.xcorr_prephp(outdir='/scratch/summit/life9360/ALASKA_work/xcorr_working_dir/pre_disp', mapfile='./MAPS_ALASKA/smpkolya_phv')
# # 
# dset.xcorr_aftan(prephdir='/scratch/summit/life9360/ALASKA_work/xcorr_working_dir/pre_disp_R')
# # 
# dset.interp_disp()
# dset.xcorr_raytomoinput(outdir='/scratch/summit/life9360/ALASKA_work/xcorr_working_dir/raytomo_input_20180630')
# dset.xcorr_get_field(outdir='../eikonal_working_dir', staxml='/projects/life9360/code/DataRequest/ALASKA_TA_AK.xml')
# dset.write_stationtxt('sta_alaska.lst')

# dset = noisedbase.noiseASDF('/scratch/summit/life9360/ALASKA_work/ASDF_data/xcorr_Alaska_20180809.h5')
# dset.xcorr_append(inasdffname='/scratch/summit/life9360/ALASKA_work/ASDF_data/xcorr_Alaska.h5',\
#                   datadir='/scratch/summit/life9360/ALASKA_work/COR_work_dir', startyear=2018, startmonth=1, endyear=2018, endmonth=7)


# dset1 = noisedbase.noiseASDF('/scratch/summit/life9360/ALASKA_work/ASDF_data/xcorr_Alaska.h5')
# dset2 = noisedbase.noiseASDF('/scratch/summit/life9360/ALASKA_work/ASDF_data/xcorr_Alaska_20180809.h5')

# minlon = 999
# minlat = 999
# maxlon = -999
# maxlat = -999
# stalst = dset.waveforms.list()
# 
# for staid in stalst:
#     if minlon > dset.waveforms[staid].coordinates['longitude']:
#         minlon  = dset.waveforms[staid].coordinates['longitude']
#     if maxlon < dset.waveforms[staid].coordinates['longitude']:
#         maxlon  = dset.waveforms[staid].coordinates['longitude']
#     if minlat > dset.waveforms[staid].coordinates['latitude']:
#         minlat  = dset.waveforms[staid].coordinates['latitude']
#     if maxlat < dset.waveforms[staid].coordinates['latitude']:
#         maxlat  = dset.waveforms[staid].coordinates['latitude']