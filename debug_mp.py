import eikonaltomo
import hybridtomo
import numpy as np
#-----------------------
#initialization
#-----------------------
# dset    = hybridtomo.hybridTomoDataSet('/work1/leon/ALASKA_work/hdf5_files/debugmp_numba.h5')
# dset.set_input_parameters(minlon=188, maxlon=238, minlat=52, maxlat=72, pers=np.array([10.]))
# dset.hybrid_eikonal_stack(anisotropic=True, spacing_ani=.6, N_bin=20, Tmin=20.)
# dset.compute_azi_aniso()
# 
# dset    = hybridtomo.hybridTomoDataSet('/work1/leon/ALASKA_work/hdf5_files/debugmp_numba2.h5')
# dset.set_input_parameters(minlon=188, maxlon=238, minlat=52, maxlat=72, pers=np.array([10.]))
# dset.hybrid_eikonal_stack(anisotropic=True, spacing_ani=.6, N_bin=20, Tmin=20.)
# dset.compute_azi_aniso()



# 
# dset    = hybridtomo.hybridTomoDataSet('/work1/leon/ALASKA_work/hdf5_files/debugmp_multi.h5')
# dset.set_input_parameters(minlon=188, maxlon=238, minlat=52, maxlat=72, pers=np.array([10.]))
# dset.hybrid_eikonal_stack_mp(anisotropic=True, spacing_ani=.6, N_bin=20, workingdir='./debug_multi', enhanced=False)
# dset.compute_azi_aniso()

# 
# d1    = hybridtomo.hybridTomoDataSet('/work1/leon/ALASKA_work/hdf5_files/debugmp_numba.h5')
# d2    = hybridtomo.hybridTomoDataSet('/work1/leon/ALASKA_work/hdf5_files/debugmp_multi.h5')
# d3    = hybridtomo.hybridTomoDataSet('/work1/leon/ALASKA_work/hdf5_files/debugmp_numba2.h5')