import quakedbase
import numpy as np
import timeit
import matplotlib.pyplot as plt
import CURefPy

slow = 0.06
tarr1 = np.arange(0, 20., 0.05)
data = np.arange(tarr1.size) * 0.1

tarr2, data2  = CURefPy._stretch_old (tarr1, data, slow)
tarr2_n, data2_n  = CURefPy._stretch (tarr1, data, slow)
tarr2_v, data2_v  = CURefPy._stretch_vera (tarr1, data, slow)


plt.plot(tarr2, data2, '-', label='weisen')
plt.plot(tarr2_v, data2_v, '-.', label='vera')
plt.plot(tarr2_n, data2_n, '--', label='corrected')
plt.show()