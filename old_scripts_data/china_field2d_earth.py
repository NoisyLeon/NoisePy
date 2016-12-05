import field2d_earth
import GeoPolygon
import matplotlib.pyplot as plt
import numpy as np

basins=GeoPolygon.GeoPolygonLst()
basins.ReadGeoPolygonLst('basin1')

minlat=16
maxlat=54
minlon=75
maxlon=135

# minlat=20.
# maxlat=52.
# minlon=80.
# maxlon=134.

datadir='/work3/leon/china_data/china_field/'
field=field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=0.2, minlat=minlat, maxlat=maxlat, dlat=0.2, period=10., fieldtype='Amp')
# field=field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=0.2, minlat=minlat, maxlat=maxlat, dlat=0.2, period=10., fieldtype='Tph')
# tfield=field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=0.2, minlat=minlat, maxlat=maxlat, dlat=0.2, period=10., fieldtype='Tph')

# field.read_dbase(datadir='./output')
# # field.read(fname='./stf_10_20sec/Tph_10.0.txt')
# field.read_ind(datadir+'/10sec/E00003_10.txt', 2, 6)
field.read_ind(datadir+'/10sec/E00001_10.txt', 4)
# field.read(fname='../Pyfmst/Tph_10sec_0.2.lst')
# field.add_noise(sigma=5.)

workingdir='/work3/leon/china_data/field_working'
# field.interp_surface(workingdir=workingdir, outfname='Tph_10sec')
field.interp_surface(workingdir=workingdir, outfname='Amp_10sec')
field.get_az_dist_Arr()
ag=-51; mindist=700.
field.stalons=field.lonArr[(field.azArr>ag)*(field.azArr<ag+1)*(field.distArr>10.*5+mindist)]
field.stalats=field.latArr[(field.azArr>ag)*(field.azArr<ag+1)*(field.distArr>10.*5+mindist)]
sZarr  = field.Zarr[(field.azArr>ag)*(field.azArr<ag+1)*(field.distArr>10.*5+mindist)]
sdist  = field.distArr[(field.azArr>ag)*(field.azArr<ag+1)*(field.distArr>10.*5+mindist)]
# field.check_curvature(workingdir=workingdir, threshold=0.05)
# field.gradient_qc(workingdir=workingdir, nearneighbor=True)

# field.cut_edge(1,1)
# field.np2ma()
# m=field.plot_field(contour=False, geopolygons=basins, stations=True, event=False, showfig=False, vmin=0)#, vmax=80000)
m=field.plot_field(contour=True, geopolygons=basins, stations=True, event=False, showfig=False, vmin=0)#, vmax=80000)
field.plot_event(infname='/work3/leon/china_data/EA_quake.h5', evnumb=1, inbasemap=m)

#####################################################
plt.figure()
field2=field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=0.2, minlat=minlat, maxlat=maxlat, dlat=0.2, period=10., fieldtype='Amp')
field2.read_ind(datadir+'/10sec/E00005_10.txt', 4)
workingdir='/work3/leon/china_data/field_working'
# field.interp_surface(workingdir=workingdir, outfname='Tph_10sec')
field2.interp_surface(workingdir=workingdir, outfname='Amp_10sec')
field2.get_az_dist_Arr()
field2.stalons=field2.lonArr[(field2.azArr>ag)*(field2.azArr<ag+1)*(field.distArr>10.*5+mindist)]
field2.stalats=field2.latArr[(field2.azArr>ag)*(field2.azArr<ag+1)*(field.distArr>10.*5+mindist)]
sZarr2  = field2.Zarr[(field.azArr>ag)*(field.azArr<ag+1)*(field.distArr>10.*5+mindist)]
sdist2  = field2.distArr[(field.azArr>ag)*(field.azArr<ag+1)*(field.distArr>10.*5+mindist)]
m=field2.plot_field(contour=True, geopolygons=basins, stations=True, event=False, showfig=False, vmin=0)#, vmax=80000)
field2.plot_event(infname='/work3/leon/china_data/EA_quake.h5', evnumb=5, inbasemap=m)


plt.figure()
ind_min=sdist.argmin()
refArr = np.ones(sdist.size)/np.sqrt(sdist)*np.sqrt(sdist.min())
sZarr=sZarr/sZarr[ind_min]
ind_min2=sdist2.argmin()
sZarr2=sZarr2/sZarr2[ind_min2]
plt.plot( sdist, sZarr, 'bo', markersize=5, label='Mw 5.44')
plt.plot( sdist2, sZarr2, 'go', markersize=5, label='Mw 6.29')
plt.plot( sdist, refArr, 'r--', markersize=5)
plt.legend()
plt.show()
