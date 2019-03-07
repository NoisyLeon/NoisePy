import noisedbase
import numpy as np
import timeit
import matplotlib.pyplot as plt
import eikonaltomo
import raytomo
import os
#-----------------------
#initialization
#-----------------------
dset        = eikonaltomo.EikonalTomoDataSet('/work1/leon/ALASKA_work/hdf5_files/eikonal_xcorr_tomo_Alaska_TA_AK_20190218_250km_aniso.h5')
# 
outdir      = '/work1/leon/ALASKA_work/figs_eik_azi_noise_with_psi1'
pers        = dset.attrs['period_array']
minlon      = dset.attrs['minlon']
maxlon      = dset.attrs['maxlon']
minlat      = dset.attrs['minlat']
maxlat      = dset.attrs['maxlat']
dlon        = 1.
dlat        = 1.
lons        = np.arange((maxlon-minlon)/dlon+1)*dlon+minlon
lats        = np.arange((maxlat-minlat)/dlat+1)*dlat+minlat
for per in pers:
    outdirper   = outdir+'/'+str(int(per))+'sec'
    if not os.path.isdir(outdirper):
        os.makedirs(outdirper)
    for lon in lons:
        for lat in lats:
            outfname    = outdirper+'/azi_'+str(abs(lon-360.))+'W_'+str(lat)+'N.jpg'
            dset.plot_azimuthal_single_point_all(inlon=lon, inlat=lat, runid=0, period=per, outfname=outfname, showfig=False)