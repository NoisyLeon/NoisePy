import field2d_earth
import numpy as np
import matplotlib.pyplot as plt

minlon=235.
maxlon=260.
minlat=31.
maxlat=50.



field   = field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=0.2, minlat=minlat, maxlat=maxlat, dlat=0.2, period=10.)
field.read(fname='/scratch/summit/life9360/eikonal_working_single/60.0sec/E00979.amp.lst')
field.ZarrIn    = field.ZarrIn*1e9
field.fieldtype = 'amp'
tension = .0
field.interp_surface(workingdir='./eik_working_debug', outfname='Tph_10sec', tension=tension)
field.coarse_data()
field.Laplacian('diff2')

field1          = field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=0.1, minlat=minlat, maxlat=maxlat, dlat=0.1, period=10.)
field1.read_array(field.lonArrIn, field.latArrIn, field.ZarrIn)
field1.fieldtype= 'amp'
field1.interp_surface(workingdir='./eik_working_debug', outfname='Tph_10sec', tension=tension)
field1.Laplacian('diff2')

field2          = field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=0.2, minlat=minlat, maxlat=maxlat, dlat=0.2, period=10.)
field2.read_array(field.lonArrIn, field.latArrIn, field.ZarrIn)
field2.fieldtype= 'amp'
field2.interp_surface(workingdir='./eik_working_debug', outfname='Tph_10sec', tension=tension)
field2.Laplacian('diff2')

field3          = field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=0.5, minlat=minlat, maxlat=maxlat, dlat=0.5, period=10.)
field3.read_array(field.lonArrIn, field.latArrIn, field.ZarrIn)
field3.fieldtype= 'amp'
field3.interp_surface(workingdir='./eik_working_debug', outfname='Tph_10sec', tension=tension)
field3.Laplacian('diff2')

field4          = field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=1., minlat=minlat, maxlat=maxlat, dlat=1., period=10.)
field4.read_array(field.lonArrIn, field.latArrIn, field.ZarrIn)
field4.fieldtype= 'amp'
field4.interp_surface(workingdir='./eik_working_debug', outfname='Tph_10sec', tension=tension)
field4.Laplacian('diff2')



# 
# lon = 250.; dlon = 0.05
# 
# # 
# ax  = plt.subplot()
# 
# 
# x, d  = field1.get_line_lplc(lon=lon, dlon=dlon)
# plt.plot(x, d, '-', lw=3, label='h = 0.1')
# 
# x, d  = field2.get_line_lplc(lon=lon, dlon=dlon)
# plt.plot(x, d, '-', lw=3, label='h = 0.2')
# 
# x, d  = field3.get_line_lplc(lon=lon, dlon=dlon)
# plt.plot(x, d, '-', lw=3, label='h = 0.5')
# 
# x, d  = field4.get_line_lplc(lon=lon, dlon=dlon)
# plt.plot(x, d, '-', lw=3, label='h = 1.')
# 
# x, d  = field.get_line_lplc(lon=lon, dlon=dlon)
# plt.plot(x, d, 'ko', ms=5, lw=3, label='real')
# # 
# plt.ylabel('Laplacian', fontsize=30)
# plt.xlabel('latitude', fontsize=30)
# ax.tick_params(axis='x', labelsize=20)
# ax.tick_params(axis='y', labelsize=20)
# plt.title('longitude = '+str(lon), fontsize=40.)
# plt.legend(loc=0, fontsize=20, numpoints=1)
# # 
# plt.show()


lat = 35.; dlat = 0.05

# 
ax  = plt.subplot()


x, d  = field1.get_line_lplc(lat=lat, dlat=dlat)
plt.plot(x, d, '-', lw=3, label='h = 0.1')

x, d  = field2.get_line_lplc(lat=lat, dlat=dlat)
plt.plot(x, d, '-', lw=3, label='h = 0.2')

x, d  = field3.get_line_lplc(lat=lat, dlat=dlat)
plt.plot(x, d, '-', lw=3, label='h = 0.5')

x, d  = field4.get_line_lplc(lat=lat, dlat=dlat)
plt.plot(x, d, '-', lw=3, label='h = 1.')

x, d  = field.get_line_lplc(lat=lat, dlat=dlat)
plt.plot(x, d, 'ko', ms=5, lw=3, label='real')
# 
plt.ylabel('Laplacian', fontsize=30)
plt.xlabel('longitude', fontsize=30)
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)
plt.title('latitude = '+str(lat), fontsize=40.)
plt.legend(loc=0, fontsize=20, numpoints=1)
# 
plt.show()