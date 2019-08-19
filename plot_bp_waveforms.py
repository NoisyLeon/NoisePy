import pyaftan
import obspy
import matplotlib.pyplot as plt
import numpy as np
infname='./COR/TA.A21K/COR_TA.A21K_LHZ_TA.N25K_LHZ.SAC'

tr  = obspy.read(infname)[0]
atr = pyaftan.aftantrace(tr.data, tr.stats)
atr.makesym()
atr.aftanf77(tmin=2., tmax=60., phvelname='TA.A21K.TA.N25K.pre')
atr.plotftan(plotflag=1)
# plt.suptitle('fortran77 aftan results')
plt.show()
# 
# ax  = plt.subplot()
# time = np.arange(atr.data.size)*atr.stats.delta
# f, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(5, sharex=True, sharey=True)
# ax0.plot(time, atr.data/atr.data.max(), 'k', lw=2)
# y = atr.gaussian_filter_snr(fcenter=1/10.)
# y /= y.max()
# ax1.plot(time, y, 'k', lw=1)
# # ax1.set_title('Sharing both axes')
# y = atr.gaussian_filter_snr(fcenter=1/20.)
# y /= y.max()
# ax2.plot(time, y, 'k', lw=1)
# y = atr.gaussian_filter_snr(fcenter=1/30.)
# y /= y.max()
# ax3.plot(time, y, 'k', lw=1)
# y = atr.gaussian_filter_snr(fcenter=1/50.)
# y /= y.max()
# ax4.plot(time, y, 'k', lw=1)
# 
# # Fine-tune figure; make subplots close to each other and hide x ticks for
# # all but bottom plot.
# f.subplots_adjust(hspace=0)
# plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
# 
# # y = atr.gaussian_filter_aftan(fcenter=1/40.)
# # plt.plot(time, y, 'k-', lw=3)
# ax4.tick_params(axis='x', labelsize=30)
# plt.xlim([0., 1000.])
# plt.show()
