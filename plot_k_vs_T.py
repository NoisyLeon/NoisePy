import noisedbase
import numpy as np
import timeit
import matplotlib.pyplot as plt
import raytomo
import matplotlib.pyplot as plt           
# dset=eikonaltomo.EikonalTomoDataSet('/scratch/summit/life9360/ALASKA_work/hdf5_files/eikonal_xcorr_tomo_Alaska_TA_AK_20180718_10sec.h5')
# dset    = eikonaltomo.EikonalTomoDataSet('/scratch/summit/life9360/ALASKA_work/hdf5_files/eikonal_xcorr_tomo_Alaska_TA_AK_20180813.h5')

dset    = raytomo.RayTomoDataSet('/work1/leon/ALASKA_work/hdf5_files/ray_tomo_Alaska_working.h5')
periods = dset.attrs['period_array'][1:-2]     
k       = np.zeros(periods.size)
for i in range(periods.size):                    
    k[i] = dset.plot_sem_curve(ineikfname='/work1/leon/ALASKA_work/hdf5_files/eikonal_xcorr_tomo_Alaska_TA_AK_20180814_250km.h5',\
                               period=periods[i], runid=2, dx=10., xmax=70., plotfig=False)[0]



ax  = plt.subplot()
plt.plot(periods, k, 'o', ms=15, label='observed')
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)
plt.ylabel('Slope ('+r'$10^{-3} \sec^{-1}$'+')', fontsize=30)
plt.xlabel('Period (s)', fontsize=30)
# plt.legend(fontsize=30)
plt.show()