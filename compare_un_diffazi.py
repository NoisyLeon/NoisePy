import numpy as np

import matplotlib.pyplot as plt
import matplotlib

inarr1 = np.load('unazi.npz')
inarr2 = np.load('diffazi.npz')

un      = inarr1['arr_0']*3.5
mask1   = inarr1['arr_1']

diff    = inarr2['arr_0']
mask2   = inarr2['arr_1']

mask    = mask1 + mask2
ind     = np.logical_not(mask)

un  = un[ind]
un[un>90.] = 90.
r   = un - abs(diff[ind])

print r[r>=0.].size/float(r.size)

ax      = plt.subplot()
        
dbin    = 0.1
# bins    = np.arange(min(r), max(r) + dbin, dbin)
def to_percent(y, position):
     # Ignore the passed in position. This has the effect of scaling the default
     # tick locations.
     s = '%.0f' %(100. * y)
     # The percent symbol needs escaping in latex
     if matplotlib.rcParams['text.usetex'] is True:
         return s + r'$\%$'
     else:
         return s + '%'
        
weights = np.ones_like(r)/float(r.size)
# plt.hist(r, bins=bins, weights = weights)
plt.hist(r, bins=20, weights = weights)
import matplotlib.mlab as mlab
from matplotlib.ticker import FuncFormatter


plt.ylabel('Percentage (%)', fontsize=60)
plt.xlabel('Angle difference (deg)', fontsize=60, rotation=0)
plt.title('mean = %g , std = %g' %(r.mean(), r.std()), fontsize=30)
ax.tick_params(axis='x', labelsize=40)
ax.tick_params(axis='y', labelsize=40)
formatter = FuncFormatter(to_percent)
# Set the formatter
plt.gca().yaxis.set_major_formatter(formatter)
# plt.xlim([-4, 6.])
plt.show()