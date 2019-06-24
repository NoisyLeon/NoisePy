import eikonaltomo
import raytomo
import numpy as np
#-----------------------
#initialization
#-----------------------

# dset    = eikonaltomo.EikonalTomoDataSet('/work1/leon/ALASKA_work/hdf5_files/debug2_quake_60.h5')
# dset.set_input_parameters(minlon=188, maxlon=238, minlat=52, maxlat=72, pers=np.array([28., 30., 32.]))
# dset.eikonal_stack_mp(anisotropic=True, spacing_ani=2., N_bin=20, workingdir='./debug2_quake_2deg_60', enhanced=True)
# dset.compute_azi_aniso()

# dset    = eikonaltomo.EikonalTomoDataSet('/work1/leon/ALASKA_work/hdf5_files/debug2_quake_90.h5')

# dset    = eikonaltomo.EikonalTomoDataSet('/work1/leon/ALASKA_work/hdf5_files/comp_quake_2deg.h5')
# dset.set_input_parameters(minlon=188, maxlon=238, minlat=52, maxlat=72, pers=np.array([28., 30., 32.]))
# dset.eikonal_stack_mp(anisotropic=True, spacing_ani=2., N_bin=20, azi_amp_tresh=0.1, workingdir='./comp_quake_2deg', enhanced=False)
# dset.compute_azi_aniso()

# dset    = eikonaltomo.EikonalTomoDataSet('/work1/leon/ALASKA_work/hdf5_files/comp_quake_2deg_0.05.h5')
# dset.set_input_parameters(minlon=188, maxlon=238, minlat=52, maxlat=72, pers=np.array([28., 30., 32.]))
# dset.set_input_parameters(minlon=188, maxlon=238, minlat=52, maxlat=72, pers=np.array([30.]))
# dset.eikonal_stack_mp(anisotropic=True, spacing_ani=2., N_bin=20, azi_amp_tresh=0.05, workingdir='./comp_quake_2deg_0.05', enhanced=False)
# dset.compute_azi_aniso()


# dset    = eikonaltomo.EikonalTomoDataSet('/work1/leon/ALASKA_work/hdf5_files/azi_quake_2deg_0.05.h5')
# dset.set_input_parameters(minlon=188, maxlon=238, minlat=52, maxlat=72, pers=np.array([20., 22., 24., 26., 28., 30., 32., 34., 36., 38., 40.]))
# dset.eikonal_stack_mp(anisotropic=True, spacing_ani=2., N_bin=20, azi_amp_tresh=0.05, workingdir='./azi_quake_2deg_0.05')
# dset.compute_azi_aniso()