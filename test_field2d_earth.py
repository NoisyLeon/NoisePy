import field2d_earth


minlat=23.
maxlat=52.
minlon=85.
maxlon=133.

# minlat=20.
# maxlat=52.
# minlon=80.
# maxlon=134.


field=field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=0.2, minlat=minlat, maxlat=maxlat, dlat=0.2, period=10.)


# field.read_dbase(datadir='./output')
# # field.read(fname='./stf_10_20sec/Tph_10.0.txt')
field.read(fname='./Tph_10.0.txt')
# field.read(fname='../Pyfmst/Tph_10sec_0.2.lst')
# field.add_noise(sigma=5.)
workingdir='./field_working'
field.interp_surface(workingdir=workingdir, outfname='Tph_10sec')

# field1 = field.copy()
# field2 = field.copy()
# field3 = field.copy()
# field4 = field.copy()
# field5 = field.copy()
# field1.Laplacian('diff')
# field2.Laplacian('green')
# field3.Laplacian('convolve', order=2)
# field4.Laplacian('convolve', order=4)
# field5.Laplacian('convolve', order=6)
# 
# field1.plot_lplc(vmin=-0.06, vmax=0.06,showfig=False)
# field2.plot_lplc(vmin=-0.06, vmax=0.06,showfig=False)
# field3.plot_lplc(vmin=-0.06, vmax=0.06,showfig=False)
# field4.plot_lplc(vmin=-0.06, vmax=0.06,showfig=False)
# field5.plot_lplc(vmin=-0.06, vmax=0.06,showfig=True)
# field3.Laplacian('convolve', order=2)

field.check_curvature(workingdir=workingdir, threshold=0.005)
field.eikonal_operator(workingdir=workingdir, lplcthresh=0.001)

# field2= field.copy()
# fieldArr2, reason_n2 = field2.gradient_qc_new(workingdir=workingdir, nearneighbor=True)
# fieldArr, reason_n = field.gradient_qc(workingdir=workingdir, nearneighbor=True)
# # field.reset_reason()
# field.Laplacian_Green()
# field.cut_edge(1,1)
# # field.reason_n[field.lplc>0.002]=9
# # field.reason_n[field.lplc<-0.002]=9
# field.reset_reason_2()
# # field.reason_n[field.appV<2.8]=9
# field.np2ma()
# field.plot_field(contour=True, geopolygons=basins)
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
