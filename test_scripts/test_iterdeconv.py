import quakedbase
import numpy as np
import timeit
import matplotlib.pyplot as plt
import CURefPy
import obspy
import copy
# Initialize ASDF dataset
# dset=quakedbase.quakeASDF('/scratch/summit/life9360/ALASKA_work/ASDF_data/ref_Alaska.h5')
dset=quakedbase.quakeASDF('ref_Alaska.h5')
# dset.cat = quakedbase.obspy.read_events('/scratch/summit/life9360/ALASKA_work/quakeml/alaska_2017_aug.ml')
dset.cat= quakedbase.obspy.read_events('test.ml')
staid   = 'TA.C23K'
st      = dset.waveforms[staid]['body_ev_00001']
stla, elev, stlo    = dset.waveforms[staid].coordinates.values()
porigin         = dset.cat[0].preferred_origin()
evlo            = porigin.longitude
evla            = porigin.latitude
evdp            = porigin.depth
for tr in st:
    tr.stats.sac        = obspy.core.util.attribdict.AttribDict()
    tr.stats.sac['evlo']= evlo
    tr.stats.sac['evla']= evla
    tr.stats.sac['evdp']= evdp
    tr.stats.sac['stlo']= stlo
    tr.stats.sac['stla']= stla

inrefparam  = CURefPy.InputRefparam()
refTr       = CURefPy.RFTrace()
refTr.get_data(Ztr=st.select(component='Z')[0], RTtr=st.select(component=inrefparam.reftype)[0], tbeg=inrefparam.tbeg, tend=inrefparam.tend)
refTr.IterDeconv( tdel=inrefparam.tdel, f0 = inrefparam.f0, niter=inrefparam.niter, minderr=inrefparam.minderr, phase='P' )

refTr.move_out()

refTrold = copy.deepcopy(refTr)
# # # refTr.stretch_back()
# # # refTrold.stretch_back_old()
