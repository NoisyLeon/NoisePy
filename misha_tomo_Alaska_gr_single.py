import noisedbase
import numpy as np
import timeit
import GeoPolygon
import raytomo


dset=raytomo.RayTomoDataSet('/work1/leon/ALASKA_work/hdf5_files/ray_tomo_Alaska_20180823_gr_single.h5')

# dset.set_input_parameters(minlon=188, maxlon=238, minlat=52, maxlat=72,  data_pfx='raytomo_in_', smoothpfx='N_INIT_', qcpfx='QC_',\
#                           pers = np.array([8., 10., 12., 14., 16., 18.]))
# dset.run_smooth(datadir='/work1/leon/ALASKA_work/xcorr_working_dir/raytomo_input_20180630', outdir='../ray_tomo_working_dir',\
#             lengthcell=1., datatype='gr')
# dset.run_qc(outdir='../ray_tomo_working_dir', dlon=0.2, dlat=0.1, isotropic=True, anipara=0, alphaAni4=1000, alphaAni0=850, betaAni0=1, sigmaAni0=50,\
#             madfactor=3., lengthcell=0.5, lengthcellAni=.5, datatype='gr')
# dset.run_qc(outdir='../ray_tomo_working_dir', dlon=0.2, dlat=0.1, isotropic=False, anipara=0, alphaAni4=1000, alphaAni0=850, betaAni0=1, sigmaAni0=50,\
#             madfactor=3., lengthcell=0.5, lengthcellAni=.5, datatype='gr')
# dset.run_qc(outdir='../ray_tomo_working_dir', dlon=0.2, dlat=0.1, isotropic=False, anipara=1, alphaAni4=1000, alphaAni0=850, betaAni0=1, sigmaAni0=50, \
#             madfactor=3., lengthcell=0.5, lengthcellAni=.5, datatype='gr')