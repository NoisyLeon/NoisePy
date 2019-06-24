import GeoPoint as GP
import matplotlib.pyplot as plt
datadir='/projects/life9360'
dirPFX='1_'
ycl={int(1):None, int(2):int(3)}
# ycl={int(1):None, int(2):None}
testP=GP.GeoPoint(lon=120.5, lat=35.5)
testP.SetName()
testP.SetAllfname()

testP.LoadPhDisp(datadir=datadir, dirPFX=dirPFX)
testP.LoadVProfile(datadir=datadir, dirPFX=dirPFX)
fig = plt.figure()
testP.PlotDisp(ycl=ycl,datatype='PhV')
fig = plt.figure()
testP.PlotVProfile(depLimit=10)
plt.show()