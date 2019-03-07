import noisedbase
import numpy as np
import timeit
import GeoPolygon
import raytomo

#-----------------------
#initialization
#-----------------------

dset    = raytomo.RayTomoDataSet('/work1/leon/ALASKA_work/hdf5_files/ray_tomo_Alaska_azi.h5')
# dset=raytomo.RayTomoDataSet('/work1/leon/ALASKA_work/hdf5_files/ray_tomo_Alaska_single_40.h5')

# dset=raytomo.RayTomoDataSet('/work1/leon/ALASKA_work/hdf5_files/ray_tomo_Alaska_formisha.h5')
# dset=raytomo.RayTomoDataSet('/scratch/summit/life9360/ALASKA_work/hdf5_files/ray_tomo_Alaska_20180806_all_001.h5')
# dset=raytomo.RayTomoDataSet('/work3/leon/ray_tomo_Alaska_20180410.h5')
# dset=raytomo.RayTomoDataSet('../ray_tomo_Alaska_20180410.h5')
# dset=raytomo.RayTomoDataSet('/scratch/summit/life9360/ALASKA_work/hdf5_files/ray_tomo_Alaska_20180410_un_from_TA_AK.h5')
# 
# dset.set_input_parameters(minlon=188, maxlon=238, minlat=52, maxlat=72,  data_pfx='raytomo_in_', smoothpfx='N_INIT_', qcpfx='QC_')
# # # #-----------------------------
# # # # run the inversion
# # # #-----------------------------
# # # # dset.set_input_parameters(minlon=188, maxlon=238, minlat=52, maxlat=72,  data_pfx='raytomo_in_', smoothpfx='N_INIT_', qcpfx='QC_', pers=np.array([40.]))
# dset.run_smooth(datadir='/work1/leon/ALASKA_work/xcorr_working_dir/raytomo_input_20190131_all_three_lambda', outdir='../ray_tomo_working_dir',\
#             lengthcell=1.)
# dset.run_qc(outdir='../ray_tomo_working_dir', dlon=0.2, dlat=0.1, isotropic=True, anipara=0, alphaAni4=1000, alphaAni0=850, betaAni0=1, sigmaAni0=50,\
#             madfactor=3., lengthcell=0.5, lengthcellAni=.5)
# dset.run_qc(outdir='../ray_tomo_working_dir', dlon=0.2, dlat=0.1, isotropic=False, anipara=0, alphaAni4=1000, alphaAni0=850, betaAni0=1, sigmaAni0=50,\
#             madfactor=3., lengthcell=0.5, lengthcellAni=.5)
# dset.run_qc(outdir='../ray_tomo_working_dir', dlon=0.2, dlat=0.1, isotropic=False, anipara=1, alphaAni4=1000, alphaAni0=850, betaAni0=1, sigmaAni0=50, \
#             madfactor=3., lengthcell=0.5, lengthcellAni=.5)
# 
# 
# dset.run_qc(outdir='../ray_tomo_working_dir', dlon=0.2, dlat=0.1, isotropic=False, anipara=1, alphaAni4=1000, alphaAni0=1850, betaAni0=1, sigmaAni0=150, \
#             alphaAni2=3000, sigmaAni2=200, madfactor=2., lengthcell=0.5, lengthcellAni=3.)
#-----------------------
# get corrected reference maps
#-----------------------

# dset.generate_corrected_map(dataid='qc_run_0', glbdir='./MAPS', outdir='./REG_MAPS')
# dset.plot_global_map(period=50., inglbpfx='./MAPS/smpkolya_phv_R')

# dset.plot(1,1,'v', 10., clabel='C (km/s)')

#-----------------------
# uncertainties
#-----------------------
# dset.get_uncertainty(runid=2, ineikfname='/work1/leon/ALASKA_work/hdf5_files/eikonal_xcorr_tomo_Alaska_TA_AK_20180814_250km.h5',\
#                 percentage=0.9, num_thresh=100., gstd_thresh=150.)
# import raytomo
# cmap = raytomo.discrete_cmap(10, 'jet_r')
# dset.plot(1, 2,'vel_sem', 20., vmin=20., vmax=70., cmap=cmap, thresh=100., clabel='Uncertainties (km/sec)', semfactor=2.0)
#------------------------------------------------
# interpolate both velocities and uncertainties
#------------------------------------------------
# dset.interp_surface(dlon=0.5, dlat=0.5, runid = 2)