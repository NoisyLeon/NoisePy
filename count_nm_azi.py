import hybridtomo
import matplotlib.pyplot as plt
import numpy as np
dset    = hybridtomo.hybridTomoDataSet('/work1/leon/ALASKA_work/hdf5_files/azi_2deg_0.05_20190617_psi1.h5')

pers    = np.append( np.arange(17.)*2.+8., np.arange(4.)*5.+45.)
pers    = np.append( pers, np.arange(5.)*5.+65.)

T2  = pers[(pers>=50.)]
# T1  = pers[(pers>=24.)*(pers<=40.)]
T1  = pers[(pers<=40.)]

az = np.arange(20)*18-180.
N2arr = np.zeros(20)
for per in T2:
    dataid = 'Eikonal_stack_0/%d_sec/histArr' %per
    h = dset[dataid].value[:, 141+1, 65+1]
    h /= h.sum()
    N2arr   += h
    
N2arr = np.roll(N2arr, -1)



az = np.arange(20)*18-180.
N1arr = np.zeros(20)
for per in T1:
    dataid = 'Eikonal_stack_0/%d_sec/histArr' %per
    h = dset[dataid].value[:, 141+1, 65+1]
    h /= h.sum()
    N1arr   += h
    



ax  = plt.subplot()
plt.plot(az, N2arr/N2arr.sum()*100., '-', lw=5, label='T = 50 - 80 s')
plt.plot(az, N1arr/N1arr.sum()*100., '--', lw=5, label='T = 24 - 40 s')
ax.tick_params(axis='x', labelsize=40)
ax.tick_params(axis='y', labelsize=40)
plt.xlabel('Azimuth (deg)', fontsize=30)
plt.ylabel('Measurement Percentage (%)', fontsize=30)
plt.legend(fontsize=40)
plt.ylim([0, 20.])
plt.show()