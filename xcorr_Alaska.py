import noisedbase
import numpy as np
import timeit

dset = noisedbase.noiseASDF('/scratch/summit/life9360/ALASKA_work/ASDF_data/xcorr_Alaska.h5')
# dset = noisedbase.noiseASDF('../xcorr_Alaska.h5')

# dset.xcorr_stack(datadir='/scratch/summit/life9360/ALASKA_work/COR_work_dir', startyear=1991, startmonth=1, endyear=2018, endmonth=1)
# 
# dset.xcorr_prephp(outdir='/scratch/summit/life9360/ALASKA_work/xcorr_working_dir/pre_disp', mapfile='./MAPS_ALASKA/smpkolya_phv')
# 
# dset.xcorr_aftan(prephdir='/scratch/summit/life9360/ALASKA_work/xcorr_working_dir/pre_disp_R')
# 
# dset.interp_disp()
# dset.xcorr_raytomoinput(outdir='/scratch/summit/life9360/ALASKA_work/xcorr_working_dir/raytomo_input')
dset.xcorr_get_field(outdir='../eikonal_working_dir')
