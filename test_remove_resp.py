import obspy
import ASDFDBase
# 
# sacfname='/work3/leon/china_data/CEA_data/GD/CHZ/CHZ.20120408214331.GD.LHZ'
# st=obspy.read(sacfname)
# st[0].stats.channel='BHZ'
# st.write('GD.CHZ.sac', format='sac')
# respf='/work3/leon/china_data/RESP4WeisenCUB/dbRESPCNV20131007/GD/GD.CHZ/RESP.GD.CHZ.00.BHZ.001'
# pzfname=''
# seedresp = {'filename': respf,  # RESP filename
#                 # when using Trace/Stream.simulate() the "date" parameter can
#                 # also be omitted, and the starttime of the trace is then used.
#                 # Units to return response in ('DIS', 'VEL' or ACC)
#                 'units': 'VEL'
#                 }
# st.simulate(paz_remove=None, pre_filt=(0.005, 0.006, 30.0, 35.0), seedresp=seedresp)
# st.write('removed_GD.CHZ.obs.sac', format='sac')


# sacfname='/work3/leon/china_data/X1/51022/51022.20130721234558.X1.BHZ'
# st=obspy.read(sacfname)
# st.write('51022.X1.BHZ', format='sac')
# pzfname='/work3/leon/china_data/response_files/SAC_47_X1_51022_BHZ'
# obspy.io.sac.sacpz.attach_paz(st[0], pzfname)
# seedresp = {#'filename': pzfname,  # RESP filename
#                 # when using Trace/Stream.simulate() the "date" parameter can
#                 # also be omitted, and the starttime of the trace is then used.
#                 # Units to return response in ('DIS', 'VEL' or ACC)
#                 'units': 'VEL'
#                 }
# st.simulate(paz_remove=st[0].stats.paz, pre_filt=(0.005, 0.006, 30.0, 35.0))
# st.differentiate()
# st.write('removed_X1.51022.obs_vel.sac', format='sac')

from obspy.clients.fdsn.client import Client
dset=ASDFDBase.quakeASDF('/work3/leon/china_data/EA_quake.h5')
client=Client('IRIS')
st=obspy.Stream()
for event in dset.events:
    starttime=event.origins[0].time
    endtime=event.origins[0].time+5000
    st += client.get_waveforms(network='CB,IC,RM,SY', station='*', location='*', channel='*Z',
                                starttime=starttime, endtime=endtime, attach_response=True)