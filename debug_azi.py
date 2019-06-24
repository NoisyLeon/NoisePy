import eikonaltomo
import hybridtomo
import numpy as np
#-----------------------
#initialization
#-----------------------
# dset    = hybridtomo.hybridTomoDataSet('/work1/leon/ALASKA_work/hdf5_files/comp_2deg_0.05_interp.h5')
dset    = hybridtomo.hybridTomoDataSet('/work1/leon/ALASKA_work/hdf5_files/comp_2deg_0.05.h5')
# dset.set_input_parameters(minlon=188, maxlon=238, minlat=52, maxlat=72, pers=np.array([10., 20., 30., 40., 50., 60., 70., 80.]))
# dset.set_input_parameters(minlon=188, maxlon=238, minlat=52, maxlat=72, pers=np.array([80.]))
# dset.hybrid_eikonal_stack_mp(anisotropic=True, spacing_ani=2., N_bin=20, azi_amp_tresh=0.05, workingdir='./comp_2deg_0.05', enhanced=False)
# dset.compute_azi_aniso()

# 
# dset    = hybridtomo.hybridTomoDataSet('/work1/leon/ALASKA_work/hdf5_files/comp_2deg_50_0.05.h5')
# dset    = hybridtomo.hybridTomoDataSet('/work1/leon/ALASKA_work/hdf5_files/comp_2deg_0.03.h5')
# dset.set_input_parameters(minlon=188, maxlon=238, minlat=52, maxlat=72, pers=np.array([10., 20., 30., 40., 50., 60., 70., 80.]))
# # dset.hybrid_eikonal_stack_mp(anisotropic=True, spacing_ani=2., N_bin=20, azi_amp_tresh=0.05, workingdir='./comp_2deg_50_0.05', enhanced=True)
# dset.hybrid_eikonal_stack(anisotropic=True, spacing_ani=2., N_bin=20, Tmin=20.)
# dset.compute_azi_aniso_enhanced()

# 
# dset    = hybridtomo.hybridTomoDataSet('/work1/leon/ALASKA_work/hdf5_files/debug_0.6deg_80.h5')
# dset.set_input_parameters(minlon=188, maxlon=238, minlat=52, maxlat=72, pers=np.array([10., 20.]))
# dset.hybrid_eikonal_stack_mp(anisotropic=True, spacing_ani=.6, N_bin=20, workingdir='./hybrid_eik_stack_0.6deg_80', enhanced=True)
# dset.compute_azi_aniso()



# dset    = hybridtomo.hybridTomoDataSet('/work1/leon/ALASKA_work/hdf5_files/debug2_2deg_60.h5')
# dset.set_input_parameters(minlon=188, maxlon=238, minlat=52, maxlat=72, pers=np.array([10., 20., 60., 70., 80.]))
# dset.hybrid_eikonal_stack_mp(anisotropic=True, spacing_ani=2., N_bin=20, workingdir='./debug2_2deg_60', enhanced=True, run=False)
# dset.compute_azi_aniso()

# dset    = hybridtomo.hybridTomoDataSet('/work1/leon/ALASKA_work/hdf5_files/mp_hybrid_001.h5')
# dset.set_input_parameters(minlon=188, maxlon=238, minlat=52, maxlat=72, pers=np.array([10.]))
# dset.set_input_parameters(minlon=188, maxlon=238, minlat=52, maxlat=72, pers=np.array([10., 20., 30., 40., 50., 60., 70., 80.]))
# dset.hybrid_eikonal_stack(anisotropic=True, spacing_ani=2., N_bin=20, Tmin=20.)
# dset.compute_azi_aniso_enhanced()

# dset.interp_surface_azi_eik(dlon=1., dlat=0.5)