import field2d_earth
import GeoPolygon
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

datadir='/home/lili/code/china_data/china_field/'
field=field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=0.2, minlat=minlat, maxlat=maxlat, dlat=0.2, period=10.)


# field.read_dbase(datadir='./output')
# # field.read(fname='./stf_10_20sec/Tph_10.0.txt')
# field.read_ind(datadir+'/10sec/E00005_10.txt', 2, 6)
field.read_ind(datadir+'/10sec/E00002_10.txt', 4)
# field.read(fname='../Pyfmst/Tph_10sec_0.2.lst')
# field.add_noise(sigma=5.)
workingdir='/home/lili/code/china_data/field_working'
field.interp_surface(workingdir=workingdir, outfname='Tph_10sec')
# field.check_curvature(workingdir=workingdir, threshold=0.05)
# field.gradient_qc(workingdir=workingdir, nearneighbor=True)

# field.cut_edge(1,1)
# field.np2ma()
field.plot_field(contour=True, geopolygons=basins, stations=False, event=True)
# 
# # field.plot_diffa()
# field.plot_appV(geopolygons=basins, vmin=2.9, vmax=3.4)
# # field.plot_lplc(vmin=-0.01, vmax=0.01)
# field.plot_lplc(vmin=-10, vmax=10)
# field.write_dbase(outdir='./fmst_dbase_0.2')
# field.get_distArr(evlo=129.0,evla=41.306)
# field.write_dbase(outdir='./output_ses3d_all6')



# field.read_dbase(datadir='./output_ses3d')
# field.np2ma()
# fieldFMM=field.copy()
# fieldFMM.read_dbase(datadir='./output_FMM')
# fieldFMM.np2ma()
# # fieldFMM.plot_diffa()
# field.compare(fieldFMM)
# # field.histogram()
# field.mean_std()
# field.plot_compare()

# field.Laplacian('convolve')
# field.plot_lplc()
