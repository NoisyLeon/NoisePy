import field2d_earth
import numpy as np
import matplotlib.pyplot as plt

minlon=235.
maxlon=260.
minlat=31.
maxlat=50.



field=field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=0.2, minlat=minlat, maxlat=maxlat, dlat=0.2, period=10.)
field.read(fname='/scratch/summit/life9360/eikonal_working_single/60.0sec/E00979.t.lst')
tension = .0
field.interp_surface(workingdir='./eik_working_debug', outfname='Tph_10sec', tension=tension)
field.coarse_data()


lon = 250.; dlon = 0.05

tension = .0
field.interp_surface(workingdir='./eik_working_debug', outfname='amp_60sec', tension=tension)
xin, din, x, d  = field.get_fit(lon=lon, dlon=dlon)

ax  = plt.subplot()
plt.plot(x, d, '-', lw=2, label='tension = '+str(tension))

tension = 0.2
field.interp_surface(workingdir='./eik_working_debug', outfname='amp_60sec', tension=tension)
xin, din, x, d  = field.get_fit(lon=lon, dlon=dlon)
plt.plot(x, d, '-', lw=2, label='tension = '+str(tension))

tension = 0.4
field.interp_surface(workingdir='./eik_working_debug', outfname='amp_60sec', tension=tension)
xin, din, x, d  = field.get_fit(lon=lon, dlon=dlon)
plt.plot(x, d, '-', lw=2, label='tension = '+str(tension))

tension = 0.6
field.interp_surface(workingdir='./eik_working_debug', outfname='amp_60sec', tension=tension)
xin, din, x, d  = field.get_fit(lon=lon, dlon=dlon)
plt.plot(x, d, '-', lw=2, label='tension = '+str(tension))

tension = 0.8
field.interp_surface(workingdir='./eik_working_debug', outfname='amp_60sec', tension=tension)
xin, din, x, d  = field.get_fit(lon=lon, dlon=dlon)
plt.plot(x, d, '-', lw=2, label='tension = '+str(tension))

tension = 1.0
field.interp_surface(workingdir='./eik_working_debug', outfname='amp_60sec', tension=tension)
xin, din, x, d  = field.get_fit(lon=lon, dlon=dlon)
plt.plot(x, d, '-', lw=2, label='tension = '+str(tension))


plt.plot(xin, din, 'ko', ms=10, label='input')
# xin, din, x, d  = field.get_fit(lon=lon, dlon=0.2)
# plt.plot(xin, din, 'bo', ms=10, label='input, dlon=0.2')

plt.ylabel('travel time', fontsize=30)
plt.xlabel('latitude', fontsize=30)
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)
plt.title('longitude = '+str(lon), fontsize=40.)
plt.legend(loc=0, fontsize=20, numpoints=1)

plt.show()
# 
# 
# lat = 35; dlat = 0.05
# 
# tension = .0
# field.interp_surface(workingdir='./eik_working_debug', outfname='amp_60sec', tension=tension)
# xin, din, x, d  = field.get_fit(lat=lat, dlat=dlat)
# 
# ax  = plt.subplot()
# plt.plot(x, d, '-', lw=2, label='tension = '+str(tension))
# 
# tension = 0.2
# field.interp_surface(workingdir='./eik_working_debug', outfname='amp_60sec', tension=tension)
# xin, din, x, d  = field.get_fit(lat=lat, dlat=dlat)
# plt.plot(x, d, '-', lw=2, label='tension = '+str(tension))
# 
# tension = 0.4
# field.interp_surface(workingdir='./eik_working_debug', outfname='amp_60sec', tension=tension)
# xin, din, x, d  = field.get_fit(lat=lat, dlat=dlat)
# plt.plot(x, d, '-', lw=2, label='tension = '+str(tension))
# 
# tension = 0.6
# field.interp_surface(workingdir='./eik_working_debug', outfname='amp_60sec', tension=tension)
# xin, din, x, d  = field.get_fit(lat=lat, dlat=dlat)
# plt.plot(x, d, '-', lw=2, label='tension = '+str(tension))
# 
# tension = 0.8
# field.interp_surface(workingdir='./eik_working_debug', outfname='amp_60sec', tension=tension)
# xin, din, x, d  = field.get_fit(lat=lat, dlat=dlat)
# plt.plot(x, d, '-', lw=2, label='tension = '+str(tension))
# 
# tension = 1.0
# field.interp_surface(workingdir='./eik_working_debug', outfname='amp_60sec', tension=tension)
# xin, din, x, d  = field.get_fit(lat=lat, dlat=dlat)
# plt.plot(x, d, '-', lw=2, label='tension = '+str(tension))
# 
# 
# 
# plt.plot(xin, din, 'ko', ms=10, label='input')
# 
# plt.ylabel('travel time', fontsize=30)
# plt.xlabel('longitude', fontsize=30)
# ax.tick_params(axis='x', labelsize=20)
# ax.tick_params(axis='y', labelsize=20)
# plt.title('latitude = '+str(lat), fontsize=40.)
# plt.title('latitude = '+str(lat), fontsize=40.)
# plt.legend(loc=0, fontsize=20, numpoints=1)
# 
# plt.show()