import ASDFDBase
import numpy as np
import GeoPolygon
basins=GeoPolygon.GeoPolygonLst()
basins.ReadGeoPolygonLst('basin1')
dset=ASDFDBase.quakeASDF('./EA_quake_chinaarray1.h5')
# cat=dset.get_events(startdate='2012-01-01', enddate='2013-12-01', add2dbase=False, Mmin=5.0, Mmax=6.5, minlatitude=25, maxlatitude=35,\
#                 minlongitude=100, maxlongitude=110, magnitudetype='mw', maxdepth=10., gcmt=True)
# cat2=dset.get_events(startdate='2011-7-01', enddate='2013-12-01', add2dbase=False, Mmin=5.0, Mmax=6.5, minlatitude=15, maxlatitude=30,\
#                 minlongitude=120, maxlongitude=125, magnitudetype='mw', maxdepth=10., gcmt=True)
# cat3=dset.get_events(startdate='2011-7-01', enddate='2013-12-01', Mmin=5.0, add2dbase=False, Mmax=6.5, minlatitude=32, maxlatitude=48,\
#                 minlongitude=116, maxlongitude=128, magnitudetype='mw', maxdepth=10., gcmt=True)
# dset.plot_events(valuetype='mag', geopolygons=basins, gcmt=True)