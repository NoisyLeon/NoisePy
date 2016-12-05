import ASDFDBase
import numpy as np
import timeit

dset=ASDFDBase.quakeASDF('../WUS_quake_mp_test.h5')
# dset=ASDFDBase.quakeASDF('../WUS_quake_test.h5')
dset.get_events(startdate='2007-9-28', enddate='2007-9-29', Mmin=6.6, magnitudetype='ms')
# dset.get_stations(channel='LHZ', station='R11A')
# dset.get_events(startdate='2007-9-28', enddate='2007-9-29', Mmin=6.6, magnitudetype='ms', maxlatitude=0)
dset.get_stations(startdate='2007-9-26', enddate='2007-9-30', channel='LHZ', 
        minlatitude=31.0, maxlatitude=49.0, minlongitude=-124.0, maxlongitude=-102.0)
# dset.get_stations(startdate='2007-9-26', enddate='2007-9-30', channel='LHZ', 
#         minlatitude=44.0, maxlatitude=47.5, minlongitude=242.5, maxlongitude=245)
# dset.get_surf_waveforms(longitude=(-124.0-102.0)/2., latitude=(31+49)/2., channel='LHZ')
t1=timeit.default_timer()
# dset.get_surf_waveforms(channel='LHZ', verbose=True)
dset.get_surf_waveforms_mp(outdir='./downloaded_waveforms', subsize=1000, deletemseed=True, nprocess=12)
t2=timeit.default_timer()
print t2-t1, 'sec'
