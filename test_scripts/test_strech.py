import quakedbase
import numpy as np
import timeit
import matplotlib.pyplot as plt
import CURefPy
# Initialize ASDF dataset
# dset=quakedbase.quakeASDF('/scratch/summit/life9360/ALASKA_work/ASDF_data/ref_Alaska.h5')
dset=quakedbase.quakeASDF('ref_Alaska.h5')
# dset.cat = quakedbase.obspy.read_events('/scratch/summit/life9360/ALASKA_work/quakeml/alaska_2017_aug.ml')
dset.cat = quakedbase.obspy.read_events('test.ml')

index   = 'TA_C23K_P'
data    = dset.auxiliary_data.RefR[index].E00001.data[:]
delta   = dset.auxiliary_data.RefR[index].E00001.parameters['delta']
fs      = 1./dset.auxiliary_data.RefR[index].E00001.parameters['delta']
b       = dset.auxiliary_data.RefR[index].E00001.parameters['b']
e       = dset.auxiliary_data.RefR[index].E00001.parameters['e']
o       = 0.
slow    = dset.auxiliary_data.RefR[index].E00001.parameters['hslowness']/111.12

nb          = int(np.ceil((o-b)*fs))  # index for t = 0.
nt          = np.arange(0+nb, 0+nb+20*fs, 1) # nt= nb ~ nb+ 20*fs, index array for data 
time        = np.arange(int((e-b)/delta)+1, dtype=np.float64) * delta + b
if len(nt)==1:
    data    = data[np.array([np.int_(nt)])]
else:
    data    = data[np.int_(nt)]
tarr1       = (nt - nb)/fs  # time array for move-outed data

slow=0.06
tarr2, data2  = CURefPy._stretch_old (tarr1, data, slow)
tarr2_n, data2_n  = CURefPy._stretch (tarr1, data, slow, refslow=0.06)
# tarr2_v, data2_v  = CURefPy._stretch_vera (tarr1, data, slow)
# tarr2_u, data2_u  = CURefPy._stretch_new_uncorrected (tarr1, data, slow)

plt.plot(tarr2, data2, '-', label='old')
# plt.plot(tarr2_u, data2_u, '-.', label='uncorrected')
# plt.plot(tarr2_v, data2_v, '-.', label='vera')
plt.plot(tarr1, data, '.-', label='unstretched')
plt.plot(tarr2_n, data2_n, '--', label='corrected')
plt.legend(fontsize=20)
plt.show()