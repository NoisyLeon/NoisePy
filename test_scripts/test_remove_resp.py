import obspy
import numpy as np
import ASDFDBase
# 
# sacfname='/home/lili/code/china_data/Sta_dat/GD/CHZ/CHZ.20120408214331.GD.LHZ'
# st=obspy.read(sacfname)
# st[0].stats.channel='BHZ'
# st.write('GD.CHZ.sac', format='sac')
# respf='/home/lili/code/china_data/RESP4WeisenCUB/dbRESPCNV20131007/GD/GD.CHZ/RESP.GD.CHZ.*.BHZ.001'
# pzfname=''
# seedresp = {'filename': respf,  # RESP filename
#                 # when using Trace/Stream.simulate() the "date" parameter can
#                 # also be omitted, and the starttime of the trace is then used.
#                 # Units to return response in ('DIS', 'VEL' or ACC)
#                 'units': 'VEL'
#                 }
# st.simulate(paz_remove=None, pre_filt=(0.005, 0.006, 30.0, 35.0), seedresp=seedresp)
# st.write('removed_GD.CHZ.obs.sac', format='sac')


sacfname='/home/lili/code/china_data/X1/53063/53063.20120408214331.X1.BHZ'
st=obspy.read(sacfname)
st.write('53063.X1.BHZ', format='sac')
pzfname='/home/lili/code/china_data/response_files/SAC_*_X1_53063_BHZ'
obspy.io.sac.sacpz.attach_paz(st[0], pzfname)
seedresp = {#'filename': pzfname,  # RESP filename
                # when using Trace/Stream.simulate() the "date" parameter can
                # also be omitted, and the starttime of the trace is then used.
                # Units to return response in ('DIS', 'VEL' or ACC)
                'units': 'VEL'
                }
st.simulate(paz_remove=st[0].stats.paz, pre_filt=(0.001, 0.005, 1, 100.0))
# st.differentiate()
st.write('removed_X1.53063.obs_none.sac', format='sac')
# # 
# # from obspy.clients.fdsn.client import Client
# # 
# dset=ASDFDBase.quakeASDF('./EA_quake_chinaarray1.h5')
# dset2=ASDFDBase.quakeASDF('./EA_quake_iris.h5')
# # # dset.add_quakeml(dset2.events)
# # client=Client('IRIS')
# starttime=obspy.core.utcdatetime.UTCDateTime('2012-01-01')
# # 
# # # st += client.get_waveforms(network='CB', station='GYA', location='00', channel='BHZ',
# # #                             starttime=starttime, endtime=endtime, attach_response=True)
# # # inv = client.get_stations(network='CB,HK,IC', starttime=starttime,
# # #             minlatitude=21, maxlatitude=30, minlongitude=97, maxlongitude=111)
# # # dset2.add_quakeml(dset.events)
# # # dset2.add_stationxml(inv)
# # # st=obspy.Stream()
# # # for event in dset.events:
# # #     starttime=event.origins[0].time
# # #     endtime=event.origins[0].time+5000
# # #     st += client.get_waveforms(network='CB,IC,RM,SY', station='*', location='*', channel='*Z',
# #                                 # starttime=starttime, endtime=endtime, attach_response=True)
# darr=np.array([])
# for staid1 in dset.waveforms.list():
#     lat1, elev1, lon1 =  dset.waveforms[staid1].coordinates.values()
#     for staid2 in dset2.waveforms.list():
#         lat2, elev2, lon2 =  dset2.waveforms[staid2].coordinates.values()
#         # print lat1, lon1, lat2, lon2
#         dist, az, baz = obspy.geodetics.gps2dist_azimuth(lat1, lon1, lat2, lon2)
#         if dist/1000. < 25. and staid1!=staid2:
#             d1, az, baz = obspy.geodetics.gps2dist_azimuth(lat1, lon1, 23.89, 122.2)
#             d2, az, baz = obspy.geodetics.gps2dist_azimuth(lat2, lon2, 23.89, 122.2)
#             print staid1, staid2, d1, d2
#         darr=np.append(darr, dist/1000.)
    
    