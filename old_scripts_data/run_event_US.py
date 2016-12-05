import ASDFDBase
import numpy as np
import obspy 
dset=ASDFDBase.quakeASDF('../US_quake_explosion.h5')
# cat=obspy.read_events('query.quakeml')
# dset.get_events(startdate='2005-01-01', enddate='2016-01-01', Mmin=3.5, Mmax=5.5, minlatitude=25, maxlatitude=55,\
#                 minlongitude=-125, maxlongitude=-60, magnitudetype='mw', maxdepth=10., gcmt=True)
# dset.get_events(startdate='2005-01-01', enddate='2016-01-01', Mmin=3.5, Mmax=5.5, minlatitude=25, maxlatitude=55, \
#                 minlongitude=-125, maxlongitude=-60, magnitudetype='mw', gcmt=False)
# dset.get_events(startdate='2005-01-01', enddate='2016-01-01', Mmin=4.0, Mmax=5.5, minlatitude=25, maxlatitude=55,\
#                 minlongitude=-125, maxlongitude=-65, gcmt=True)
# dset.get_events(startdate='1995-01-01', enddate='2016-01-01', Mmin=3.5, Mmax=5.5, minlatitude=30, maxlatitude=50,\
#                 minlongitude=-116, maxlongitude=-65, magnitudetype='mw', maxdepth=10., gcmt=True)
# dset.get_events(startdate='2011-7-01', enddate='2013-12-01', Mmin=5.0, Mmax=6.5, minlatitude=15, maxlatitude=30,\
#                 minlongitude=120, maxlongitude=125, magnitudetype='mw', maxdepth=10., gcmt=True)
# dset.get_events(startdate='2011-7-01', enddate='2013-12-01', Mmin=5.0, Mmax=6.5, minlatitude=32, maxlatitude=48,\
#                 minlongitude=116, maxlongitude=128, magnitudetype='mw', maxdepth=10., gcmt=True)
# dset.plot_events(valuetype='mag', gcmt=True)