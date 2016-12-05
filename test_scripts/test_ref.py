import CURefPy
import timeit
import obspy
# tr=obspy.read('decon.out')[0]
# rf1=CURefPy.RFTrace(tr.data, tr.stats)
rf2=CURefPy.RFTrace()
# rf2.get_data(Ztr='2005-10-15T10:06:15.720000Z.03.sac', RTtr='2005-10-15T10:06:15.720000Z.01.sac')
rf2.get_data(Ztr='XR.YRT.01.BHZ.sac', RTtr='XR.YRT.01.BHR.sac')
# rf2.get_data(Ztr='TA.R22A..BHZ.sac', RTtr='TA.R22A..BHR.sac')
# # rf1.IterDeconv()
rf2.IterDeconv()
rf2.move_out()
rf2.stretch_back()
# rf1.read('R22A.eqr')
# rf2.write('ref.sac',format='sac')
# t1=timeit.default_timer()
# for i in xrange(100):
#     # rf1.IterDeconv()
#     rf2.IterDeconv()
# t2=timeit.default_timer()
# print t2-t1

# Rtr=obspy.read('2005-10-15T10:06:15.720000Z.01.sac')[0]
# stime=Rtr.stats.starttime
# etime=Rtr.stats.endtime
# Rtr.trim(stime+20, etime-30)
# Rtr.write('R.sac',format='sac')
# 
# Ztr=obspy.read('2005-10-15T10:06:15.720000Z.03.sac')[0]
# stime=Ztr.stats.starttime
# etime=Ztr.stats.endtime
# Ztr.trim(stime+20, etime-30)
# Ztr.write('Z.sac',format='sac')