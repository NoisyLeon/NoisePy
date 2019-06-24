import noisedbase
import numpy as np
import timeit
import GeoPolygon
import raytomo

#-----------------------
#initialization
#-----------------------
# dset=raytomo.RayTomoDataSet('/work1/leon/ALASKA_work/hdf5_files/final_tomo_files/ray_tomo_Alaska_20190228_gr.h5')
dset=raytomo.RayTomoDataSet('/work1/leon/ALASKA_work/hdf5_files/ray_tomo_Alaska_20190318_gr.h5')
# 
# dset.set_input_parameters(minlon=188, maxlon=238, minlat=52, maxlat=72,  data_pfx='raytomo_in_', smoothpfx='N_INIT_', qcpfx='QC_')
# #-----------------------------
# # run the inversion
# #-----------------------------
# dset.run_smooth(datadir='/work1/leon/ALASKA_work/xcorr_working_dir_Rayleigh/raytomo_input_20190131_all_three_lambda',\
#             outdir='../ray_tomo_working_dir_gr', lengthcell=1., datatype='gr')
# # dset.run_qc(outdir='../ray_tomo_working_dir', dlon=0.2, dlat=0.1, isotropic=True, anipara=0, alphaAni4=1000, alphaAni0=850, betaAni0=1, sigmaAni0=50,\
# #             madfactor=3., lengthcell=0.5, lengthcellAni=.5, datatype='gr')
# dset.run_qc(outdir='../ray_tomo_working_dir_gr', dlon=0.2, dlat=0.1, isotropic=False, anipara=0, alphaAni4=1000, alphaAni0=850, betaAni0=1, sigmaAni0=50,\
#             madfactor=3., lengthcell=0.5, lengthcellAni=.5, datatype='gr')
# dset.run_qc(outdir='../ray_tomo_working_dir_gr', dlon=0.2, dlat=0.1, isotropic=False, anipara=1, alphaAni4=1000, alphaAni0=850, betaAni0=1, sigmaAni0=50, \
#             madfactor=3., lengthcell=0.5, lengthcellAni=.5, datatype='gr')

#-----------------------------------------------------
# get the finalized mask for Monte Carlo inversion 
#-----------------------------------------------------
# dset.get_mask_inv(Tmin=8., Tmax=50., runid=1)
# # 
# # #-----------------------------------------------------
# # # interpolate to a given grid spacing
# # #-----------------------------------------------------
# dset.interp_surface(dlon=1., dlat=0.5, runid = 1)