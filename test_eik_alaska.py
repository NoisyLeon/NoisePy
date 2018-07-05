import field2d_earth
import numpy as np

minlat  = 50.
maxlat  = 75.
minlon  = 180.
maxlon  = 240.

workingdir='./alaska_debug_eik_working'
field=field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=0.2, minlat=minlat, maxlat=maxlat, dlat=0.2, period=10.)
field.read(fname='./example_H23K/travel_time_H23K.txt')
field.evla  = 65.83
field.evlo  = -149.54 + 360.
field.interp_surface(workingdir=workingdir, outfname='Tph_10sec')
field.check_curvature(workingdir=workingdir, threshold=0.005)
field.eikonal_operator(workingdir=workingdir, lplcthresh=0.05)

workingdir='./alaska_debug_eik_working'
field2  = field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=0.2, minlat=minlat, maxlat=maxlat, dlat=0.1, period=10.)
field2.read_HD('./example_H23K/slow_azi_H23K.txt.HD')

# field=field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=0.2, minlat=minlat, maxlat=maxlat, dlat=0.2, period=10.)
# # field.read_dbase(datadir='./output')
# # # field.read(fname='./stf_10_20sec/Tph_10.0.txt')
# field.read(fname='./stf_10sec_all/Tph_10.0.txt')
# # field.read(fname='../Pyfmst/Tph_10sec_0.2.lst')
# # field.add_noise(sigma=5.)
# workingdir='./field_working'
# field.interp_surface(workingdir=workingdir, outfname='Tph_10sec')
# # 
# # # field1 = field.copy()
# # # field2 = field.copy()
# # # field3 = field.copy()
# # # field4 = field.copy()
# # # field5 = field.copy()
# # # field1.Laplacian('diff')
# # # field2.Laplacian('green')
# # # field3.Laplacian('convolve', order=2)
# # # field4.Laplacian('convolve', order=4)
# # # field5.Laplacian('convolve', order=6)
# # # 
# # # field1.plot_lplc(vmin=-0.06, vmax=0.06,showfig=False)
# # # field2.plot_lplc(vmin=-0.06, vmax=0.06,showfig=False)
# # # field3.plot_lplc(vmin=-0.06, vmax=0.06,showfig=False)
# # # field4.plot_lplc(vmin=-0.06, vmax=0.06,showfig=False)
# # # field5.plot_lplc(vmin=-0.06, vmax=0.06,showfig=True)
# # # field3.Laplacian('convolve', order=2)
# # 
# field.check_curvature(workingdir=workingdir, threshold=0.005)
# field.eikonal_operator(workingdir=workingdir, lplcthresh=0.005)

# field       = field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=0.2, minlat=minlat, maxlat=maxlat, dlat=0.2, period=10.)
# field.read(fname='./stf_10sec_all/Tph_10.0.txt')
# lat0        = (minlat + maxlat)/2.
# lon0        = (minlon + maxlon)/2.
# # field.synthetic_field(lat0=lat0, lon0=lon0, v=3.5)
# 
# workingdir  = './field_working_debug'
# # field.evla  = lat0
# # field.evlo  = lon0
# field.interp_surface(workingdir=workingdir, outfname='Tph_10sec')
# # field.check_curvature(workingdir=workingdir, threshold=0.005)
# # field.eikonal_operator(workingdir=workingdir, lplcthresh=0.005)
# 
# field.read_HD('./old_helm/EA_am_laplace.txt.HD')
# 
# # 
# field.Laplacian('diff2')
# field.lplc_diff = field.lplc - field.lplc_gmt
# field.mask      = np.zeros((field.Nlat, field.Nlon), dtype=np.bool)
# 
# 
# field.diff_debug(lon0=lon0, lat0=lat0)
# field.check_curvature(workingdir=workingdir, threshold=0.005)
# field.helmholtz_operator(workingdir=workingdir, lplcthresh=0.005)




# field.Laplacian('green')
# field.plot_lplc(vmin=-0.06, vmax=0.06,showfig=True)
# 
# fieldamp            = field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=0.2, minlat=minlat, maxlat=maxlat, dlat=0.2, period=10.)
# fieldamp.fieldtype  = 'amp'
# fieldamp.read(fname='./stf_10sec_all/Amp_10.0.txt')
# workingdir          = './field_working'
# fieldamp.interp_surface(workingdir=workingdir, outfname='Amp_10sec')
# fieldamp.check_curvature_amp(workingdir=workingdir, threshold=0.5)
# fieldamp.helmholtz_operator(workingdir=workingdir, lplcthresh=0.5)
# 
# field.get_lplc_amp(fieldamp)


