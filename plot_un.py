import numpy as np
import matplotlib.pyplot as plt

unray   = np.loadtxt('un_ray.txt')
unlov   = np.loadtxt('un_lov.txt')
ungrp   = np.loadtxt('un_noise.txt')

ax  = plt.subplot()
ungrp[:11, 1]   = unray[:11, 1]
plt.plot(unray[:, 0], unray[:, 1], '-', ms=15, label='Rayleigh, phase', lw=5)
plt.plot(unlov[:, 0], unlov[:, 1], '--', ms=15, label='Love, phase', lw=5)
plt.plot(ungrp[:, 0], ungrp[:, 1]*2., '-', ms=15, label='Rayleigh, group', lw=5)

ax.tick_params(axis='x', labelsize=40)
ax.tick_params(axis='y', labelsize=40)
plt.ylabel('Uncertainties (m/s)', fontsize=50)
plt.xlabel('Period (s)', fontsize=50)
plt.legend(fontsize=40)
plt.xticks(np.arange(10.)*10.)
plt.xlim([5., 90.])
plt.ylim([0., 125.])
plt.show()