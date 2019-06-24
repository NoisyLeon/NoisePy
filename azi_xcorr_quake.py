import eikonaltomo
import hybridtomo
import numpy as np
#-----------------------
#initialization
#-----------------------
dset    = hybridtomo.hybridTomoDataSet('/work1/leon/ALASKA_work/hdf5_files/azi_2deg_0.05_20190617.h5')

# pers    = np.append( np.arange(17.)*2.+8., np.arange(4.)*5.+45.)
# pers    = np.append( pers, np.arange(5.)*5.+65.)
# dset.set_input_parameters(minlon=188, maxlon=238, minlat=52, maxlat=72, pers=pers)
# dset.hybrid_eikonal_stack_mp(anisotropic=True, spacing_ani=2., N_bin=20, workingdir='./azi_2deg_0.05', enhanced=False, azi_amp_tresh=0.05, Tmin=20.)
# dset.compute_azi_aniso()
# dset.interp_surface_azi_eik(dlon=1., dlat=.5)




# dset    = hybridtomo.hybridTomoDataSet('/work1/leon/ALASKA_work/hdf5_files/azi_2deg_0.03_20190617.h5')
# pers    = np.append( np.arange(17.)*2.+8., np.arange(4.)*5.+45.)
# pers    = np.append( pers, np.arange(5.)*5.+65.)
# dset.set_input_parameters(minlon=188, maxlon=238, minlat=52, maxlat=72, pers=pers)
# dset.hybrid_eikonal_stack_mp(anisotropic=True, spacing_ani=2., N_bin=20, workingdir='./azi_2deg_0.03', enhanced=False, azi_amp_tresh=0.03, Tmin=20.)
# dset.compute_azi_aniso()