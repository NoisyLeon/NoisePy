import numpy as np
import matplotlib.pyplot as plt
import matplotlib
az1     = np.random.rand(10000)*179.
az2     = np.random.rand(10000)*179.

def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = '%.0f' %(100. * y)
    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'
ax      = plt.subplot()
data    = abs(az1 - az2)
data[data>90]   = 180. - data[data>90]
dbin    = 10.
bins    = np.arange(min(data), max(data) + dbin, dbin)

weights = np.ones_like(data)/float(data.size)
plt.hist(data, bins=bins, weights = weights)
import matplotlib.mlab as mlab
from matplotlib.ticker import FuncFormatter

plt.ylabel('Percentage (%)', fontsize=60)
plt.xlabel('Angle difference (deg)', fontsize=60, rotation=0)
plt.title('mean = %g , std = %g ' %(data.mean(), data.std()), fontsize=30)
ax.tick_params(axis='x', labelsize=40)
ax.tick_params(axis='y', labelsize=40)
formatter = FuncFormatter(to_percent)
# Set the formatter
plt.gca().yaxis.set_major_formatter(formatter)
plt.xlim([0, 90.])
plt.show()