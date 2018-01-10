# -*- coding: utf-8 -*-
"""
This is a sub-module of noisepy.
Classes and functions for receiver function analysis.

References:

For iterative deconvolution algorithmn:
Ligorría, Juan Pablo, and Charles J. Ammon. "Iterative deconvolution and receiver-function estimation."
    Bulletin of the seismological Society of America 89.5 (1999): 1395-1400.
    
For harmonic stripping and related quality control details:
Shen, Weisen, et al. "Joint inversion of surface wave dispersion and receiver functions: A Bayesian Monte-Carlo approach."
    Geophysical Journal International (2012): ggs050.
    
Please consider citing them if you use this code for your research.
    
:Copyright:
    Author: Lili Feng
    Graduate Research Assistant
    CIEI, Department of Physics, University of Colorado Boulder
    email: lili.feng@colorado.edu
"""

import numpy as np
import scipy.signal
import copy
import numexpr as npr
import obspy
import os
import matplotlib.pylab as plb
import matplotlib.pyplot as plt
from obspy.taup import TauPyModel
import warnings
import numba
from scipy import fftpack
from scipy.interpolate import interp1d
# from numba import jit
# from pyproj import Geod
# 
# geodist = Geod(ellps='WGS84')
taupmodel           = TauPyModel(model="iasp91")

#---------------------------------------------------------
# preparing model arrays for stretching
#---------------------------------------------------------
dz4stretch          = 0.5

def _model4stretch_ak135(modelfname='ak135_Q'):
    ak135Arr    = np.loadtxt(modelfname)
    hak135      = ak135Arr[:, 0]
    vsak135     = ak135Arr[:, 1]
    vpak135     = ak135Arr[:, 2]
    zak135      = np.cumsum(hak135)
    zak135      = np.append(0., zak135)
    vsak135     = np.append(vsak135[0], vsak135)
    vpak135     = np.append(vpak135[0], vpak135)
    fak135vs    = interp1d(zak135, vsak135, kind='nearest')
    fak135vp    = interp1d(zak135, vpak135, kind='nearest')
    zmax        = 240.
    zarr        = np.arange(int(zmax/dz4stretch), dtype=np.float64)*dz4stretch
    
    vsarr       = fak135vs(zarr)
    vparr       = fak135vp(zarr)
    return vsarr, vparr

def _model4stretch_iasp91(modelfname='IASP91.mod'):
    modelfname='IASP91.mod'
    iasp91Arr   = np.loadtxt(modelfname)
    ziasp91     = iasp91Arr[:, 0]
    vsiasp91    = iasp91Arr[:, 3]
    vpiasp91    = iasp91Arr[:, 2]
    zmax        = 240.
    zarr        = np.arange(int(zmax/dz4stretch), dtype=np.float64)*dz4stretch
    vsarr       = np.interp(zarr, ziasp91, vsiasp91)
    vparr       = np.interp(zarr, ziasp91, vpiasp91)
    nz          = zarr.size
    return vsarr, vparr, nz

vs4stretch, vp4stretch, nz4stretch  = _model4stretch_iasp91()


# @numba.jit( numba.float64[:](numba.float64, numba.int32, numba.float64) )
def _gaussFilter( dt, nft, f0 ):
    """
    Compute a gaussian filter in the freq domain which is unit area in time domain
    private function for IterDeconv
    ================================================================================
    :::input parameters:::
    dt      - sampling time interval
    nft     - number freq points
    f0      - width of filter
    
    Output:
    gauss   - Gaussian filter array (numpy)
    filter has the form: exp( - (0.5*w/f0)^2 ) the units of the filter are 1/s
    ================================================================================
    """
    df                  = 1.0/(nft*dt)
    nft21               = 0.5*nft + 1
    # get frequencies
    f                   = df*np.arange(nft21)
    w                   = 2*np.pi*f
    w                   = w/f0
    kernel              = w**2
    # compute the gaussian filter
    gauss               = np.zeros(nft)
    gauss[:int(nft21)]  = np.exp( -0.25*kernel )/dt
    gauss[int(nft21):]  = np.flipud(gauss[1:int(nft21)-1])
    return gauss

# @numba.jit( numba.float64[:](numba.float64[:], numba.int32, numba.float64,  numba.float64) )
def _phaseshift( x, nfft, DT, TSHIFT ):
    """Add a shift to the data into the freq domain, private function for IterDeconv
    """
    Xf      = fftpack.fft(x)
    # Xf      = np.fft.fft(x)
    # phase shift in radians
    shift_i = round(TSHIFT/DT) # removed +1 from here.
    p       = np.arange(nfft)+1
    p       = 2*np.pi*shift_i/(nfft)*p
    # apply shift
    Xf      = Xf*(np.cos(p) - 1j*np.sin(p))
    # back into time
    x       = np.real( fftpack.ifft(Xf) )/np.cos(2*np.pi*shift_i/nfft)
    # x       = np.real( np.fft.ifft(Xf) )/np.cos(2*np.pi*shift_i/nfft)
    return x

# @numba.jit( numba.float64[:](numba.float64[:], numba.float64[:],  numba.float64) )
def _FreFilter(inW, FilterW, dt ):
    """Filter input array in frequency domain, private function for IterDeconv
    """
    # # FinW    = np.fft.fft(inW)
    FinW    = fftpack.fft(inW)
    FinW    = FinW*FilterW*dt
    # # FilterdW= np.real(np.fft.ifft(FinW))
    FilterdW= np.real(fftpack.ifft(FinW))
    return FilterdW

# @numba.jit( numba.types.UniTuple(numba.float64[:], 2) (numba.float64[:], numba.float64[:], numba.float64) )
def _stretch(tarr, data, slow, refslow=0.06, modeltype=0):
    """Stretch data to vertically incident receiver function given slowness, private function for move_out
    """
    dt          = tarr[1] - tarr[0]
    dz          = dz4stretch
    if modeltype == 0:
        vparr       = vp4stretch
        vsarr       = vs4stretch
        nz          = nz4stretch
    else:
        zmax        = 240.
        zarr        = np.arange(int(zmax/dz), dtype=np.float64)*dz
        nz          = zarr.size
        # layer array
        harr        = np.ones(nz, dtype=np.float64)*dz
        # velocity arrays
        vpvs        = 1.7
        vp          = 6.4
        vparr       = np.ones(nz, dtype=np.float64)*vp
        vparr       = vparr + (zarr>60.)*np.ones(nz, dtype = np.float64) * 1.4
        vsarr       = vparr/vpvs
    # 1/vsarr**2 and 1/vparr**2
    sv2         = vsarr**(-2)
    pv2         = vparr**(-2)
    # dz/vs - dz/vp, time array for vertically incident wave
    s1          = np.ones(nz, dtype=np.float64)*refslow*refslow
    difft       = np.zeros(nz+1, dtype=np.float64)
    difft[1:]   = (np.sqrt(sv2-s1) - np.sqrt(pv2-s1)) * dz
    cumdifft    = np.cumsum(difft)
    # # dz*(tan(a1p) - tan(a1s))*sin(a0p)/vp0 +dz / cos(a1s)/vs - dz / cos(a1p)/vp
    # # s = sin(a1s)/vs = sin(a1p)/vp = sin(a0p) / vp0
    # time array for wave with given slowness
    s2          = np.ones(nz, dtype=np.float64)*slow*slow
    difft2      = np.zeros(nz+1, dtype=np.float64)
    difft2[1:]  = (np.sqrt(sv2-s2)-np.sqrt(pv2-s2))*dz # dz/
    cumdifft2   = np.cumsum(difft2)
    # interpolate data to correspond to cumdifft2 array
    nseis       = np.interp(cumdifft2, tarr, data)
    # get new time array
    tf          = cumdifft[-1]
    ntf         = int(tf/dt)
    tarr2       = np.arange(ntf, dtype=np.float64)*dt
    data2       = np.interp(tarr2, cumdifft, nseis)
    return tarr2, data2


def _stretch_new_uncorrected(tarr, data, slow):
    """Stretch data to vertically incident receiver function given slowness, private function for move_out
    """
    dt          = tarr[1] - tarr[0]
    # depth arrays
    dz          = 0.5
    zmax        = 240.
    zarr        = np.arange(int(zmax/dz), dtype=np.float64)*dz
    nz          = zarr.size
    # layer array
    harr        = np.ones(nz, dtype=np.float64)*dz
    # velocity arrays
    vpvs        = 1.7
    vp          = 6.4
    vparr       = np.ones(nz, dtype=np.float64)*vp
    vparr       = vparr + (zarr>60.)*np.ones(nz, dtype = np.float64) * 1.4
    vsarr       = vparr/vpvs
    # 1/vsarr**2 and 1/vparr**2
    sv2         = vsarr**(-2)
    pv2         = vparr**(-2)
    # dz/vs - dz/vp, time array for vertically incident wave
    difft       = np.zeros(nz+1, dtype=np.float64)
    difft[1:]   = (np.sqrt(sv2) - np.sqrt(pv2)) * dz
    cumdifft    = np.cumsum(difft)
    # # dz*(tan(a1p) - tan(a1s))*sin(a0p)/vp0 +dz / cos(a1s)/vs - dz / cos(a1p)/vp
    # # s = sin(a1s)/vs = sin(a1p)/vp = sin(a0p) / vp0
    # time array for wave with given slowness
    s2          = np.ones(nz, dtype=np.float64)*slow*slow
    # # difft2      = np.zeros(nz+1, dtype=np.float64)
    # # difft2[1:]  = (np.sqrt(sv2-s2)-np.sqrt(pv2-s2))*dz # dz/
    difft2      = (np.sqrt(sv2-s2)-np.sqrt(pv2-s2))*dz # originally, and data2 = np.interp(tarr2, cumdifft[:-1][indt<npts], nseis)
    cumdifft2   = np.cumsum(difft2)
    # interpolate data to correspond to cumdifft2 array
    nseis       = np.interp(cumdifft2, tarr, data)
    # get new time array
    tf          = cumdifft[-1]
    ntf         = int(tf/dt)
    tarr2       = np.arange(ntf, dtype=np.float64)*dt
    # originally it is
    # data2     = np.interp(tarr2, cumdifft[:-1][indt<npts], nseis)
    # because originally cumdifft2[0] != 0, this should be wrong !
    data2       = np.interp(tarr2, cumdifft[:-1], nseis)
    # data2       = np.interp(tarr2, cumdifft, nseis)
    return tarr2, data2


def _stretch_vera(tarr, data, slow):
    """Stretch data to vertically incident receiver function given slowness, private function for move_out
    """
    dt          = tarr[1] - tarr[0]
    # depth arrays
    dz          = 0.5
    zmax        = 240.
    zarr        = np.arange(int(zmax/dz), dtype=np.float64)*dz
    nz          = zarr.size
    # layer array
    harr        = np.ones(nz, dtype=np.float64)*dz
    # velocity arrays
    vpvs        = 1.7
    vp          = 6.4
    vparr       = np.ones(nz, dtype=np.float64)*vp
    vparr       = vparr + (zarr>60.)*np.ones(nz, dtype = np.float64) * 1.4
    vsarr       = vparr/vpvs
    # 1/vsarr**2 and 1/vparr**2
    sv2         = vsarr**(-2)
    pv2         = vparr**(-2)
    # dz/vs - dz/vp, time array for vertically incident wave
    # # # difft       = np.zeros(nz+1, dtype=np.float64)
    # # # difft[1:]   = (np.sqrt(sv2) - np.sqrt(pv2)) * dz
    difft       = (np.sqrt(sv2) - np.sqrt(pv2)) * dz
    cumdifft    = np.cumsum(difft)
    # # dz*(tan(a1p) - tan(a1s))*sin(a0p)/vp0 +dz / cos(a1s)/vs - dz / cos(a1p)/vp
    # # s = sin(a1s)/vs = sin(a1p)/vp = sin(a0p) / vp0
    # time array for wave with given slowness
    s2          = np.ones(nz, dtype=np.float64)*slow*slow
    # # # difft2      = np.zeros(nz+1, dtype=np.float64)
    # # # difft2[1:]  = (np.sqrt(sv2-s2)-np.sqrt(pv2-s2))*dz # dz/
    difft2      = (np.sqrt(sv2-s2)-np.sqrt(pv2-s2))*dz # originally, and data2 = np.interp(tarr2, cumdifft[:-1][indt<npts], nseis)
    cumdifft2   = np.cumsum(difft2)
    # interpolate data to correspond to cumdifft2 array
    nseis       = np.interp(cumdifft2, tarr, data)
    # get new time array
    tf          = cumdifft[-1]
    ntf         = int(tf/dt)
    tarr2       = np.arange(ntf, dtype=np.float64)*dt
    # originally it is
    # data2     = np.interp(tarr2, cumdifft[:-1][indt<npts], nseis)
    # because originally cumdifft2[0] != 0, this should be wrong !
    # # # data2       = np.interp(tarr2, cumdifft[:-1], nseis)
    data2       = np.interp(tarr2, cumdifft, nseis)
    print cumdifft2[1]/cumdifft[1]
    # data2       = np.zeros(tarr2.size, dtype=np.float64)
    # j = 0
    # for tempt in tarr2:
    #     indarr  = np.where(cumdifft <= tempt)[0]
    #     index   = indarr[-1]
    #     data2[j]= nseis[index] + (nseis[index+1]-nseis[index])* \
    #                     (tempt-cumdifft[index])/(cumdifft[index+1]-cumdifft[index])
    #     j       += 1
    return tarr2, data2

def _stretch_old(t1, nd1, slow):
    """Stretch data given slowness, private function for move_out
    """
    dzi         = 0.5
    dzmax       = 240.
    dZ          = np.arange(int(dzmax/dzi))*0.5
    Rv          = 1.7
    dt          = t1[1] - t1[0]
    ndz         = dZ.size
    zthk        = np.ones(ndz)*dzi
    cpv         = 6.4
    pvel        = np.ones(ndz)*cpv
    pvel        = pvel+(dZ>60)*np.ones(ndz)*1.4
    svel1       = pvel/Rv
    sv2         = svel1**(-2)
    pv2         = (svel1*Rv)**(-2)
    cc          = (np.sqrt(sv2)-np.sqrt(pv2))*dzi
    cc          = np.append(0., cc)
    vtt         = np.cumsum(cc)
    p2          = np.ones(ndz)
    p2          = p2*slow*slow
    cc2         = (np.sqrt(sv2-p2)-np.sqrt(pv2-p2))*dzi
    mtt         = np.cumsum(cc2)
    ntt         = np.round(mtt/dt)
    ntt[0]      = 0.
    if len(ntt)==1:
        kk      = np.array([np.int_(ntt)])
    else:
        kk      = np.int_(ntt)
    Ldatain     = nd1.size
    kkk         = kk[kk<Ldatain]
    nseis       = nd1[kkk]
    time        = vtt[len(nseis)-1]
    n1          = int(time/dt)
    t2          = np.arange(n1)*dt
    Lt2         = t2.size
    d2          = np.array([])
    for tempt in t2:
        tempd   = 0.
        smallTF = np.where(vtt <= tempt)[0]
        indexj  = smallTF[-1]
        tempd   = nseis[indexj] + (nseis[indexj+1]-nseis[indexj])*(tempt-vtt[indexj])/(vtt[indexj+1]-vtt[indexj])
        d2      = np.append(d2, tempd)
    return t2, d2


def _group ( inbaz, indat):
    """Group data according to back-azimuth, private function for harmonic stripping
    """
    binwidth    = 30
    nbin        = int((360+1)/binwidth)
    outbaz      = np.array([])
    outdat      = np.array([])
    outun       = np.array([])
    for i in range(nbin):
        bazmin  = i*binwidth
        bazmax  = (i+1)*binwidth
        tbaz    = i*binwidth + float(binwidth)/2
        tdat    = indat[(inbaz>=bazmin)*(inbaz<bazmax)]
        if (len(tdat) > 0):
            outbaz      = np.append(outbaz, tbaz)
            outdat      = np.append(outdat, tdat.mean())
            if (len(tdat)>1):
                outun   = np.append(outun, tdat.std()/(np.sqrt(len(tdat))) )
            if (len(tdat)==1):
                outun   = np.append(outun, 0.1)
    return outbaz, outdat, outun

def _difference ( aa, bb, NN):
    """Compute difference between two input array, private function for harmonic stripping
    """
    if NN > 0:
        L   = min(len(aa), len(bb), NN)
    else:
        L   = min(len(aa), len(bb))
    aa      = aa[:L]
    bb      = bb[:L]
    diff    = np.sum((aa-bb)*(aa-bb))
    diff    = diff / L
    return np.sqrt(diff)

def _invert_A0 ( inbaz, indat, inun ):   #only invert for A0 part
    """invert by assuming only A0, private function for harmonic stripping
    """
    Nbaz    = inbaz.size 
    U       = np.zeros((Nbaz, Nbaz), dtype=np.float64)
    np.fill_diagonal(U, 1./inun)
    G       = np.ones((Nbaz, 1), dtype=np.float64)
    G       = np.dot(U, G)
    d       = indat.T
    d       = np.dot(U, d)
    model   = np.linalg.lstsq(G, d)[0]
    A0      = model[0]
    predat  = np.dot(G, model)
    predat  = predat[:Nbaz]
    inun    = inun[:Nbaz]
    predat  = predat*inun
    return A0, predat

def _invert_A1 ( inbaz, indat, inun ):
    """invert by assuming only A0 and A1, private function for harmonic stripping
        indat   = A0 + A1*sin(theta + phi1)
                = A0 + A1*cos(phi1)*sin(theta) + A1*sin(phi1)*cos(theta)
    """
    Nbaz    = inbaz.size 
    U       = np.zeros((Nbaz, Nbaz), dtype=np.float64)
    np.fill_diagonal(U, 1./inun)
    # construct forward operator matrix
    tG      = np.ones((Nbaz, 1), dtype=np.float64)
    tbaz    = np.pi*inbaz/180
    tGsin   = np.sin(tbaz)
    tGcos   = np.cos(tbaz)
    G       = np.append(tG, tGsin)
    G       = np.append(G, tGcos)
    G       = G.reshape((3, Nbaz))
    G       = G.T
    G       = np.dot(U, G)
    # data
    d       = indat.T
    d       = np.dot(U, d)
    # least square inversion
    model   = np.linalg.lstsq(G,d)[0]
    A0      = model[0]
    A1      = np.sqrt(model[1]**2 + model[2]**2)
    phi1    = np.arctan2(model[2], model[1])
    predat  = np.dot(G, model)
    predat  = predat*inun
    return A0, A1, phi1, predat

def _invert_A2 ( inbaz, indat, inun ):
    """invert by assuming only A0 and A2, private function for harmonic stripping
        indat   = A0 + A2*sin(2*theta + phi2)
                = A0 + A1*cos(phi1)*sin(theta) + A1*sin(phi1)*cos(theta)
    """
    Nbaz    = inbaz.size 
    U       = np.zeros((Nbaz, Nbaz), dtype=np.float64)
    np.fill_diagonal(U, 1./inun)
    # construct forward operator matrix
    tG      = np.ones((Nbaz, 1), dtype=np.float64)
    tbaz    = np.pi*inbaz/180.
    tGsin   = np.sin(tbaz*2.)
    tGcos   = np.cos(tbaz*2.)
    G       = np.append(tG, tGsin)
    G       = np.append(G, tGcos)
    G       = G.reshape((3, Nbaz))
    G       = G.T
    G       = np.dot(U, G)
    # data
    d       = indat.T
    d       = np.dot(U,d)
    # least square inversion
    model   = np.linalg.lstsq(G,d)[0]
    A0      = model[0]
    A2      = np.sqrt(model[1]**2 + model[2]**2)
    phi2    = np.arctan2(model[2],model[1])
    predat  = np.dot(G, model)
    predat  = predat*inun
    return A0, A2, phi2, predat

def _invert_A0_A1_A2( inbaz, indat, inun):
    """invert for A0, A1, A2, private function for harmonic stripping
    """
    Nbaz    = inbaz.size 
    U       = np.zeros((Nbaz, Nbaz), dtype=np.float64)
    np.fill_diagonal(U, 1./inun)
    # construct forward operator matrix
    tG      = np.ones((Nbaz, 1), dtype=np.float64)
    tbaz    = np.pi*inbaz/180
    tGsin   = np.sin(tbaz)
    tGcos   = np.cos(tbaz)
    tGsin2  = np.sin(tbaz*2)
    tGcos2  = np.cos(tbaz*2)
    G       = np.append(tG, tGsin)
    G       = np.append(G, tGcos)
    G       = np.append(G, tGsin2)
    G       = np.append(G, tGcos2)
    G       = G.reshape((5, Nbaz))
    G       = G.T
    G       = np.dot(U, G)
    # data
    d       = indat.T
    d       = np.dot(U,d)
    # least square inversion
    model   = np.linalg.lstsq(G, d)[0]
    A0      = model[0]
    A1      = np.sqrt(model[1]**2 + model[2]**2)
    phi1    = np.arctan2(model[2],model[1])                                           
    A2      = np.sqrt(model[3]**2 + model[4]**2)
    phi2    = np.arctan2(model[4], model[3])
    # compute forward
    predat  = np.dot(G, model)
    predat  = predat*inun
    return A0, A1, phi1, A2, phi2, predat

#------------------------------------------------
# Function for computing predictions
#------------------------------------------------

def A0pre ( inbaz, A0 ): return A0

def A1pre ( inbaz, A0, A1, SIG1): return A0 + A1*np.sin(inbaz+SIG1)

def A1pre1 ( inbaz, A1, SIG1): return A1*np.sin(inbaz+SIG1)

def A2pre ( inbaz, A0, A2, SIG2): return A0 + A2*np.sin(2*inbaz+SIG2)

def A2pre1 ( inbaz, A2, SIG2): return A2*np.sin(2*inbaz+SIG2)

def A3pre1 ( inbaz, A1, SIG1): return A1*np.sin(inbaz + SIG1)

def A3pre2 ( inbaz, A2, SIG2): return A2*np.sin(2*inbaz + SIG2)

def A3pre3 ( inbaz, A1, phi1, A2, phi2 ):
    return A1*np.sin(inbaz + phi1) + A2*np.sin(2*inbaz + phi2)

def A3pre ( inbaz, A0, A1, phi1, A2, phi2):
    return A0 + A1*np.sin(inbaz + phi1) + A2*np.sin(2*inbaz + phi2)

def _match1 ( data1, data2 ):
    """Compute matching of two input data
    """
    nn          = min(len(data1), len(data2))
    data1       = data1[:nn]
    data2       = data2[:nn]
    di          = data1-data2
    tempdata2   = np.abs(data2)
    meandi      = di.mean()
    X1          = np.sum((di-meandi)**2)
    return np.sqrt(X1/nn)
###########################################################################



class InputRefparam(object):
    """
    A subclass to store input parameters for receiver function analysis
    ===============================================================================================================
    Parameters:
    reftype     - type of receiver function('R' or 'T')
    tbeg, tend  - begin/end time for trim
    tdel        - phase delay
    f0          - Gaussian width factor
    niter       - number of maximum iteration
    minderr     - minimum misfit improvement, iteration will stop if improvement between two steps is smaller than minderr
    phase       - phase name, default is P, if set to '', also the possible phases will be included
    ===============================================================================================================
    """
    def __init__(self, reftype='R', tbeg=20.0, tend=-30.0, tdel=5., f0 = 2.5, niter=200, minderr=0.001, phase='P', refslow=0.06 ):
        self.reftype        = reftype
        self.tbeg           = tbeg
        self.tend           = tend
        self.tdel           = tdel
        self.f0             = f0
        self.niter          = niter
        self.minderr        = minderr
        self.phase          = phase
        self.refslow        = refslow
        return


class HStripStream(obspy.core.stream.Stream):
    """Harmonic stripping stream, derived from obspy.Stream
    """
    def get_trace(self, network, station, indata, baz, dt, starttime):
        """Get trace
        """
        tr=obspy.Trace(); tr.stats.network=network; tr.stats.station=station
        tr.stats.channel=str(int(baz)); tr.stats.delta=dt
        tr.data=indata
        tr.stats.starttime=starttime
        self.append(tr)
        return
    
            
    def PlotStreams(self, ampfactor=40, title='', ax=plt.subplot(), targetDT=0.02):
        """Plot harmonic stripping stream accoring to back-azimuth
        ===============================================================================================================
        ::: input parameters :::
        ampfactor   - amplication factor for visulization
        title       - title
        ax          - subplot object
        targetDT    - target dt for decimation
        ===============================================================================================================
        """
        ymax=361.
        ymin=-1.
        for trace in self.traces:
            downsamplefactor=int(targetDT/trace.stats.delta)
            if downsamplefactor!=1: trace.decimate(factor=downsamplefactor, no_filter=True)
            dt=trace.stats.delta
            time=dt*np.arange(trace.stats.npts)
            yvalue=trace.data*ampfactor
            backazi=float(trace.stats.channel)
            ax.plot(time, yvalue+backazi, '-k', lw=0.3)
            
            ax.fill_between(time, y2=backazi, y1=yvalue+backazi, where=yvalue>0, color='red', lw=0.01, interpolate=True)
            ax.fill_between(time, y2=backazi, y1=yvalue+backazi, where=yvalue<0, color='blue', lw=0.01, interpolate=True)
            
            
            # tfill=time[yvalue>0]
            # yfill=(yvalue+backazi)[yvalue>0]
            # ax.fill_between(tfill, backazi, yfill, color='blue', linestyle='--', lw=0.)
            # tfill=time[yvalue<0]
            # yfill=(yvalue+backazi)[yvalue<0]
            # ax.fill_between(tfill, backazi, yfill, color='red', linestyle='--', lw=0.)
            
            
        plt.axis([0., 10., ymin, ymax])
        plt.xlabel('Time(sec)')
        plt.title(title)
        return
    
    def SaveHSStream(self, outdir, prefix):
        """Save harmonic stripping stream to MiniSEED
        """
        outfname=outdir+'/'+prefix+'.mseed'
        self.write(outfname, format='mseed')
        return
    
    def LoadHSStream(self, datadir, prefix):
        """Load harmonic stripping stream from MiniSEED
        """
        infname=datadir+'/'+prefix+'.mseed'
        self.traces=obspy.read(infname)
        return
    
class HarmonicStrippingDataBase(object):
    """Harmonic stripping database, include 6 harmonic stripping streams
    """
    def __init__(self, obsST=HStripStream(), diffST=HStripStream(), repST=HStripStream(),\
        repST0=HStripStream(), repST1=HStripStream(), repST2=HStripStream()):
        self.obsST  = obsST
        self.diffST = diffST
        self.repST  = repST
        self.repST0 = repST0
        self.repST1 = repST1
        self.repST2 = repST2
        return
    
    def PlotHSStreams(self, outdir='', stacode='', ampfactor=40, targetDT=0.025, longitude='', latitude='', browseflag=False, saveflag=True,\
            obsflag=1, diffflag=0, repflag=1, rep0flag=1, rep1flag=1, rep2flag=1):
        """Plot harmonic stripping streams accoring to back-azimuth
        ===============================================================================================================
        ::: input parameters :::
        outdir              - output directory for saving figure
        stacode             - station code
        ampfactor           - amplication factor for visulization
        targetDT            - target dt for decimation
        longitude/latitude  - station location
        browseflag          - browse figure or not
        saveflag            - save figure or not
        obsflag             - plot observed receiver function or not
        diffflag            - plot difference of observed and predicted receiver function or not
        repflag             - plot predicted receiver function or not
        rep0flag            - plot A0 of receiver function or not
        rep1flag            - plot A1 of receiver function or not
        rep2flag            - plot A2 of receiver function or not
        ===============================================================================================================
        """
        totalpn = obsflag+diffflag+repflag+rep0flag+rep1flag+rep2flag
        cpn=1
        plt.close('all')
        fig=plb.figure(num=1, figsize=(12.,8.), facecolor='w', edgecolor='k')
        ylabelflag=False
        if obsflag==1:
            ax=plt.subplot(1, totalpn,cpn)
            cpn=cpn+1
            self.obsST.PlotStreams(ampfactor=ampfactor, targetDT=targetDT, title='Observed Refs', ax=ax)
            plt.ylabel('Backazimuth(deg)')
            ylabelflag=True
        if diffflag==1:
            ax=plt.subplot(1, totalpn,cpn)
            cpn=cpn+1
            self.diffST.PlotStreams(ampfactor=ampfactor, targetDT=targetDT, title='Residual Refs', ax=ax)
            if ylabelflag==False:
                plt.ylabel('Backazimuth(deg)')
        if repflag==1:
            ax=plt.subplot(1, totalpn,cpn)
            cpn=cpn+1
            self.repST.PlotStreams(ampfactor=ampfactor, targetDT=targetDT, title='Predicted Refs', ax=ax)
            if ylabelflag==False:
                plt.ylabel('Backazimuth(deg)')
        if rep0flag==1:
            ax=plt.subplot(1, totalpn,cpn)
            cpn=cpn+1
            self.repST0.PlotStreams(ampfactor=ampfactor, targetDT=targetDT, title='A0 Refs', ax=ax)
            if ylabelflag==False:
                plt.ylabel('Backazimuth(deg)')
        if rep1flag==1:
            ax=plt.subplot(1, totalpn,cpn)
            cpn=cpn+1
            self.repST1.PlotStreams(ampfactor=ampfactor, targetDT=targetDT, title='A1 Refs', ax=ax)
            if ylabelflag==False:
                plt.ylabel('Backazimuth(deg)')
        if rep2flag==1:
            ax=plt.subplot(1, totalpn,cpn)
            self.repST2.PlotStreams(ampfactor=ampfactor, targetDT=targetDT, title='A2 Refs', ax=ax)
            if ylabelflag==False:
                plt.ylabel('Backazimuth(deg)')
        fig.suptitle(stacode+' Longitude:'+str(longitude)+' Latitude:'+str(latitude), fontsize=15)
        if browseflag:
                plt.draw()
                plt.pause(1) # <-------
                raw_input("<Hit Enter To Close>")
                plt.close('all')
        if saveflag and outdir!='':
            fig.savefig(outdir+'/'+stacode+'_COM.ps', orientation='landscape', format='ps')
            
    def SaveHSDatabase(self, outdir, stacode=''):
        """Save harmonic stripping streams to MiniSEED
        """
        prefix=stacode+'_obs'
        self.obsST.SaveHSStream(outdir, prefix)
        prefix=stacode+'_diff'
        self.diffST.SaveHSStream(outdir, prefix)
        prefix=stacode+'_rep'
        self.repST.SaveHSStream(outdir, prefix)
        prefix=stacode+'_rep0'
        self.repST0.SaveHSStream(outdir, prefix)
        prefix=stacode+'_rep1'
        self.repST1.SaveHSStream(outdir, prefix)
        prefix=stacode+'_rep2'
        self.repST2.SaveHSStream(outdir, prefix)
        return
    
    def LoadHSDatabase(self, datadir, stacode=''):
        """Load harmonic stripping streams from MiniSEED
        """
        prefix=stacode+'_obs'
        self.obsST.LoadHSStream(datadir, prefix)
        prefix=stacode+'_diff'
        self.diffST.LoadHSStream(datadir, prefix)
        prefix=stacode+'_rep'
        self.repST.LoadHSStream(datadir, prefix)
        prefix=stacode+'_rep0'
        self.repST0.LoadHSStream(datadir, prefix)
        prefix=stacode+'_rep1'
        self.repST1.LoadHSStream(datadir, prefix)
        prefix=stacode+'_rep2'
        self.repST2.LoadHSStream(datadir, prefix)
        return

class PostDatabase(object):
    """
    A class to store post precessed receiver function
    ===============================================================================================================
    Parameters:
    MoveOutFlag - succeeded compute moveout or not
                    1   - valid move-outed receiver function
                    -1  - negative value at zero
                    -2  - too large amplitude, value1 = maximum amplitude 
                    -3  - too large or too small horizontal slowness, value1 = horizontal slowness
    ampC        - amplitude corrected receiver function
    ampTC       - moveout of receiver function (amplitude and time corrected)
    header      - receiver function header
    tdiff       - trace difference
    ===============================================================================================================
    """
    def __init__(self):
        self.MoveOutFlag    = None
        self.ampC           = np.array([]) # 0.06...out
        self.ampTC          = np.array([]) # stre...out
        self.header         = {}
        self.tdiff          = None
        
        
class PostRefLst(object):
    """
    A class to store as list of PostDatabase object
    """
    def __init__(self,PostDatas=None):
        self.PostDatas=[]
        if isinstance(PostDatas, PostDatabase):
            PostDatas = [PostDatas]
        if PostDatas:
            self.PostDatas.extend(PostDatas)
    
    def __add__(self, other):
        """
        Add two PostRefLst with self += other.
        """
        if isinstance(other, StaInfo):
            other = PostRefLst([other])
        if not isinstance(other, PostRefLst):
            raise TypeError
        PostDatas = self.PostDatas + other.PostDatas
        return self.__class__(PostDatas=PostDatas)

    def __len__(self):
        """
        Return the number of PostDatas in the PostRefLst object.
        """
        return len(self.PostDatas)

    def __getitem__(self, index):
        """
        __getitem__ method of PostRefLst objects.
        :return: PostDatabase objects
        """
        if isinstance(index, slice):
            return self.__class__(PostDatas=self.PostDatas.__getitem__(index))
        else:
            return self.PostDatas.__getitem__(index)

    def append(self, postdata):
        """
        Append a single PostDatabase object to the current PostRefLst object.
        """
        if isinstance(postdata, PostDatabase):
            self.PostDatas.append(postdata)
        else:
            msg = 'Append only supports a single PostDatabase object as an argument.'
            raise TypeError(msg)
        return self
    
    def __delitem__(self, index):
        """
        Passes on the __delitem__ method to the underlying list of traces.
        """
        return self.PostDatas.__delitem__(index)
    
    def remove_bad(self, outdir):
        """Remove bad measurements and group data
        ===============================================================================================================
        ::: input parameters :::
        outdir      - output directory
        ::: output :::
        outdir/wmean.txt, outdir/bin_%d_txt
        ===============================================================================================================
        """
        outlst      = PostRefLst()
        lens        = np.array([]) # array to store length for each moveout trace
        bazArr      = np.array([]) # array to store baz for each moveout trace
        for PostData in self.PostDatas:
            time    = PostData.ampTC[:,0]
            data    = PostData.ampTC[:,1]
            L       = time.size
            flag    = True
            if abs(data).max()>1:
                flag    = False
            if data[abs(time)<0.1].min()<0.02:
                flag    = False
            if flag:
                PostData.Len= L
                lens        = np.append(lens, L)
                outlst.append(PostData)
                bazArr      = np.append( bazArr, np.floor(PostData.header['baz']))
        #Group data array
        gbaz        = np.array([])
        gdata       = np.array([])
        gun         = np.array([])
        ## store the stacked RF#
        Lmin        = int(lens.min())
        dat_avg     = np.zeros(Lmin, dtype=np.float64)
        weight_avg  = np.zeros(Lmin, dtype=np.float64)
        time1       = outlst[0].ampTC[:,0]
        time1       = time1[:Lmin]
        NLst        = len(outlst)
        tdat        = np.zeros(NLst, dtype=np.float64)
        for i in range(Lmin):
            for j in range (NLst):
                tdat[j] = outlst[j].ampTC[i, 1]
            b1,d1,u1    = _group(bazArr, tdat)
            gbaz        = np.append(gbaz, b1)
            gdata       = np.append(gdata, d1)
            gun         = np.append(gun, u1)
            d1DIVu1     = d1/u1
            DIVu1       = 1./u1
            wmean       = np.sum(d1DIVu1)
            weight      = np.sum(DIVu1)
            if (weight > 0.):
                dat_avg[i]  = wmean/weight
            else:
                print "weight is zero!!! ", len(d1), u1, d1
                sys.exit()
            weight_avg[i]   = np.sum(u1)/len(u1)
        Ngbaz       = len(b1)
        gbaz        = gbaz.reshape((Lmin, Ngbaz))
        gdata       = gdata.reshape((Lmin, Ngbaz))
        gun         = gun.reshape((Lmin, Ngbaz))
        # Save average data
        outname     = outdir+"/wmean.txt"
        outwmeanArr = np.append(time1, dat_avg)
        outwmeanArr = np.append(outwmeanArr, weight_avg)
        outwmeanArr = outwmeanArr.reshape((3, Lmin))
        outwmeanArr = outwmeanArr.T
        np.savetxt(outname, outwmeanArr, fmt='%g')
        # Save baz bin data
        for i in range (Ngbaz): # back -azimuth
            outname     = outdir+"/bin_%d_txt" % (int(gbaz[0][i]))
            outbinArr   = np.append(time1[:Lmin], gdata[:, i])
            outbinArr   = np.append(outbinArr, gun[:, i])
            outbinArr   = outbinArr.reshape((3, Lmin ))
            outbinArr   = outbinArr.T
            np.savetxt(outname, outbinArr, fmt='%g')
        # compute and store trace difference
        for i in range(len(outlst)):
            time            = outlst[i].ampTC[:,0]
            data            = outlst[i].ampTC[:,1]
            Lmin            = min( len(time) , len(time1) )
            tdiff           = _difference ( data[:Lmin], dat_avg[:Lmin], 0)
            outlst[i].tdiff = tdiff
        return outlst
    
    def thresh_tdiff(self, tdiff=0.08):
        """Remove data given threshold trace difference value
        """
        outlst      = PostRefLst()
        for PostData in self.PostDatas:
            if PostData.tdiff<tdiff:
                outlst.append(PostData)
        return outlst
    
    
    def harmonic_stripping(self, stacode, outdir):
        """
        Harmonic stripping analysis for quality controlled data.
        ===============================================================================================================
        ::: input parameters :::
        stacode     - station code( e.g. TA.R11A )
        outdir      - output directory
        
        ::: output :::
        outdir/bin_%d_rf.dat, outdir/A0.dat, outdir/A1.dat, outdir/A2.dat, outdir/A0_A1_A2.dat
        outdir/average_vr.dat, outdir/variance_reduction.dat
        outdir/prestre_*, outdir/repstre_*, outdir/obsstre_*, outdir/repstre_*
        outdir/0repstre_*, outdir/1repstre_*, outdir/2repstre_*, outdir/diffstre_*
        ===============================================================================================================
        """
        NLst    = len(self.PostDatas)
        baz     = np.zeros(NLst, dtype=np.float64)
        lens    = np.zeros(NLst, dtype=np.float64)
        # # # atime   = []
        # # # adata   = []
        names   = []
        eventT  = []
        for i in range(NLst):
            PostData= self.PostDatas[i]
            time    = PostData.ampTC[:,0]
            # # # data    = PostData.ampTC[:,1]
            # # # adata.append(data)
            # # # atime.append(time)
            lens[i] = time.size
            baz[i]  = np.floor(PostData.header['baz'])
            name    = 'moveout_'+str(int(PostData.header['baz']))+'_'+stacode+'_'+str(PostData.header['otime'])
            names.append(name)
            eventT.append(PostData.header['otime'])
        # store all time and data arrays
        Lmin    = int(lens.min())
        atime   = np.zeros((NLst, Lmin), dtype=np.float64)
        adata   = np.zeros((NLst, Lmin), dtype=np.float64)
        for i in range(NLst):
            PostData        = self.PostDatas[i]
            time            = PostData.ampTC[:,0]
            data            = PostData.ampTC[:,1]
            adata[i, :]     = data[:Lmin]
            atime[i, :]     = time[:Lmin]
        # parameters in 3 different inversion
        # best fitting A0
        A0_0    = np.zeros(Lmin, dtype=np.float64)
        # best fitting A0 , A1 and phi1
        A0_1    = np.zeros(Lmin, dtype=np.float64)
        A1_1    = np.zeros(Lmin, dtype=np.float64)
        phi1_1  = np.zeros(Lmin, dtype=np.float64)
        # best fitting A0 , A2 and phi2
        A0_2    = np.zeros(Lmin, dtype=np.float64)
        A2_2    = np.zeros(Lmin, dtype=np.float64)
        phi2_2  = np.zeros(Lmin, dtype=np.float64)
        # best fitting A0, A1 and A2 
        A0      = np.zeros(Lmin, dtype=np.float64)
        A1      = np.zeros(Lmin, dtype=np.float64)
        A2      = np.zeros(Lmin, dtype=np.float64)
        phi1    = np.zeros(Lmin, dtype=np.float64)
        phi2    = np.zeros(Lmin, dtype=np.float64)
        
        mfArr0  = np.zeros(Lmin, dtype=np.float64)  # misfit between A0 and R[i]
        mfArr1  = np.zeros(Lmin, dtype=np.float64)  # misfit between A0+A1+A2 and R[i]
        mfArr2  = np.zeros(Lmin, dtype=np.float64)  # misfit between A0+A1+A2 and binned data
        mfArr3  = np.zeros(Lmin, dtype=np.float64)  # weighted misfit between A0+A1+A2 and binned data
        Aavg    = np.zeros(Lmin, dtype=np.float64)  # average amplitude
        Astd    = np.zeros(Lmin, dtype=np.float64)
        # grouped data
        gbaz    = np.array([], dtype=np.float64)
        gdata   = np.array([], dtype=np.float64)
        gun     = np.array([], dtype=np.float64)
        tdat    = np.zeros(NLst, dtype=np.float64)
        for i in range (Lmin):
            for j in range(NLst):
                tdat[j]         = self.PostDatas[j].ampTC[i, 1]
            # # datmean             = tdat.mean()
            # # datstd              = tdat.std()
            baz1,tdat1,udat1    = _group(baz, tdat)
            gbaz                = np.append(gbaz, baz1)
            gdata               = np.append(gdata, tdat1)
            gun                 = np.append(gun, udat1)
            #------------------------------------------------------
            # inversions
            #------------------------------------------------------
            # invert for best-fitting A0
            (tempA0, predat0)                   = _invert_A0(baz1, tdat1, udat1)
            A0_0[i]                             = tempA0
            # invert for best-fitting A0, A1 and phi1
            (tempA0, tempA1, tempphi1, predat1) = _invert_A1(baz1, tdat1, udat1)
            A0_1[i]                             = tempA0
            A1_1[i]                             = tempA1
            phi1_1[i]                           = tempphi1
            # invert for best-fitting A0, A2 and phi2
            (tempA0, tempA2, tempphi2, predat2) = _invert_A2(baz1, tdat1, udat1)
            A0_2[i]                             = tempA0
            A2_2[i]                             = tempA2
            phi2_2[i]                           = tempphi2
            # invert for best-fitting A0, A1 and A2
            (tempA0, tempA1, tempphi1, tempA2, tempphi2, predat) \
                                                = _invert_A0_A1_A2 (baz1,tdat1,udat1)
            A0[i]                               = tempA0
            A1[i]                               = tempA1
            phi1[i]                             = tempphi1
            A2[i]                               = tempA2
            phi2[i]                             = tempphi2
            # 
            Aavg                                = tdat.mean()
            Astd                                = tdat.std()
            # compute misfit for raw baz array
            misfit0         = np.sqrt(np.sum((A0[i] - adata[:, i])**2)/NLst)
            predatraw       = A3pre( baz*np.pi/180., A0=A0[i], A1=A1[i], phi1=phi1[i], A2=A2[i], phi2=phi2[i])
            misfit1         = np.sqrt(np.sum((predatraw - adata[:, i])**2)/NLst)
            if misfit0 < 0.005:
                misfit0     = 0.005
            if misfit1 < 0.005:
                misfit1     = 0.005
            mfArr0[i]       = misfit0
            mfArr1[i]       = misfit1
            # compute misfit for binned baz array
            Nbin            = baz1.size
            predatbin       = A3pre( baz1*np.pi/180., A0=A0[i], A1=A1[i], phi1=phi1[i], A2=A2[i], phi2=phi2[i])
            misfit2         = np.sqrt(np.sum((predatbin - tdat1)**2)/Nbin)
            wNbin           = np.sum(1./(udat1**2))
            misfit3         = np.sqrt(np.sum((predatbin - tdat1)**2 /(udat1**2))/wNbin)
            mfArr2[i]       = misfit2
            mfArr3[i]       = misfit3
            
            # # 
            # # 
            # # mf  = 0.
            # # mf1 = 0.
            # # for j in range (NLst):
            # #     mf  = mf + (A0[i] - adata[j][i])**2
            # #     vv  = A3pre(baz[j]*np.pi/180., tempv0, tempv1, tempv2, tempv3, tempv4)
            # #     mf1 = mf1 + (vv - adata[j][i])**2
            # # mf  = np.sqrt(mf/len(baz))
            # # mf1 = np.sqrt(mf1/len(baz))
            # # if (mf<0.005):
            # #     mf = 0.005
            # # if (mf1<0.005):
            # #     mf1 = 0.005
            # # MF0 = np.append(MF0, mf-0.)
            # # MF1 = np.append(MF1, mf1-0.)
            
            # # mf2 = 0.
            # # mf3 = 0.
            # # V1  = 0.
            # # for j in np.arange (len(baz1)):
            # #     vv  = A3pre(baz1[j]*np.pi/180.,tempv0,tempv1,tempv2,tempv3,tempv4)
            # #     mf2 = mf2 + (vv - tdat1[j])**2;
            # #     mf3 = mf3 + (vv - tdat1[j])**2/udat1[j]**2
            # #     V1  = V1 + 1./(udat1[j]**2)
            # # 
            # # mf2 = np.sqrt(mf2/len(baz1))
            # # mf3 = np.sqrt(mf3/V1)
            # # MF2 = np.append(MF2, mf2-0.)
            # # MF3 = np.append(MF3, mf3-0.)
        Nbin        = baz1.size
        gbaz        = gbaz.reshape((Lmin, Nbin))
        gdata       = gdata.reshape((Lmin, Nbin))
        gun         = gun.reshape((Lmin, Nbin))
        # #Output grouped data
        # for i in xrange (len(gbaz[0])): #baz
        #     tname       = outdir+"/bin_%g_rf.dat" % (gbaz[0][i])
        #     outbinArr   = np.append(atime[0][:Lmin], gdata[:,i])
        #     outbinArr   = np.append(outbinArr, gun[:,i])
        #     outbinArr   = outbinArr.reshape((3,Lmin ))
        #     outbinArr   = outbinArr.T
        #     np.savetxt(tname, outbinArr, fmt='%g')
        # 
        # time    = atime[0]
        # time    = time[:Lmin]
        # 
        # ttA     = zA0
        # timef0  = time[(ttA>-2)*(ttA<2)]
        # ttAf0   = ttA[(ttA>-2)*(ttA<2)]
        # Lf0     = timef0.size
        # outArrf0= np.append(timef0,ttAf0)
        # outArrf0= outArrf0.reshape((2,Lf0))
        # outArrf0= outArrf0.T
        # np.savetxt(outdir+"/A0.dat", outArrf0, fmt='%g')
        # 
        # ttA     = oA0
        # ttA1    = oA1
        # PHI1    = oSIG1
        # ttAf1   = ttA[(ttA>-2)*(ttA<2)]
        # ttA1f1  = ttA1[(ttA>-2)*(ttA<2)]
        # PHI1f1  = PHI1[(ttA>-2)*(ttA<2)]
        # timef1  = time[(ttA>-2)*(ttA<2)]
        # Lf1     = ttAf1.size
        # PHI1f1  = PHI1f1+(PHI1f1<0)*np.pi
        # outArrf1= np.append(timef1, ttAf1)
        # outArrf1= np.append(outArrf1, ttA1f1)
        # outArrf1= np.append(outArrf1, PHI1f1)
        # outArrf1= outArrf1.reshape((4,Lf1))
        # outArrf1= outArrf1.T
        # np.savetxt(outdir+"/A1.dat", outArrf1, fmt='%g')
        # 
        # ttA     = tA0[:Lmin]
        # ttA2    = tA2[:Lmin]
        # PHI2    = tSIG2[:Lmin]
        # ttAf2   = ttA[(ttA>-2)*(ttA<2)]
        # ttA2f2  = ttA2[(ttA>-2)*(ttA<2)]
        # PHI2f2  = PHI2[(ttA>-2)*(ttA<2)]
        # timef2  = time[(ttA>-2)*(ttA<2)]
        # Lf2     = ttAf2.size
        # PHI2f2  = PHI2f2+(PHI2f2<0)*np.pi
        # outArrf2= np.append(timef2, ttAf2)
        # outArrf2= np.append(outArrf2, ttA2f2)
        # outArrf2= np.append(outArrf2, PHI2f2)
        # outArrf2= outArrf2.reshape((4,Lf2))
        # outArrf2= outArrf2.T
        # np.savetxt(outdir+"/A2.dat", outArrf2, fmt='%g')
        # 
        # ttA     = A0
        # ttA1    = A1
        # ttA2    = A2
        # PHI1    = SIG1
        # PHI2    = SIG2
        # ttAf3   = ttA[(ttA>-200)*(ttA<200)]
        # ttA1f3  = ttA1[(ttA>-200)*(ttA<200)]
        # ttA2f3  = ttA2[(ttA>-200)*(ttA<200)];
        # PHI1f3  = PHI1[(ttA>-200)*(ttA<200)]*180/np.pi
        # PHI2f3  = PHI2[(ttA>-200)*(ttA<200)]*180/np.pi
        # timef3  = time[(ttA>-200)*(ttA<200)]
        # MF0f3   = MF0[(ttA>-200)*(ttA<200)]
        # MF1f3   = MF1[(ttA>-200)*(ttA<200)]
        # MF2f3   = MF2[(ttA>-200)*(ttA<200)]
        # MF3f3   = MF3[(ttA>-200)*(ttA<200)]
        # AAf3    = A_A[(ttA>-200)*(ttA<200)]
        # AAunf3  = A_A_un[(ttA>-200)*(ttA<200)]
        # Lf3     = ttAf3.size
        # outArrf3= np.append(timef3, ttAf3)
        # outArrf3= np.append(outArrf3, ttA1f3)
        # outArrf3= np.append(outArrf3, PHI1f3)
        # outArrf3= np.append(outArrf3, ttA2f3)
        # outArrf3= np.append(outArrf3, PHI2f3)
        # outArrf3= np.append(outArrf3, MF0f3)
        # outArrf3= np.append(outArrf3, MF1f3)
        # outArrf3= np.append(outArrf3, MF2f3)
        # outArrf3= np.append(outArrf3, MF3f3)
        # outArrf3= np.append(outArrf3, AAf3)
        # outArrf3= np.append(outArrf3, AAunf3)
        # outArrf3= outArrf3.reshape((12,Lf3))
        # outArrf3= outArrf3.T
        # np.savetxt(outdir+"/A0_A1_A2.dat", outArrf3, fmt='%g')
        # ##################################################################
        # Latime  = len(atime)
        # if len(baz)==1: fbaz=np.array([np.float_(baz)])
        # else: fbaz=np.float_(baz)
        # fbaz    = fbaz[:Latime]
        # lfadata = np.array([])
        # ##################################################################
        # rdata   = np.array([])
        # drdata  = np.array([]) # this is raw - 0 - 1 - 2
        # rdata0  = np.array([]) # only 0
        # lfrdata1= np.array([]) # 0+1
        # lfrdata2= np.array([]) # 0+2
        # vr0     = np.array([])
        # vr1     = np.array([])
        # vr2     = np.array([])
        # vr3     = np.array([])
        # for j in xrange(Latime): lfadata=np.append(lfadata, adata[j][:Lmin])
        # lfadata=lfadata.reshape((Latime, Lmin))
        # for i in xrange(Lmin):
        #     ttA     = A0[i]
        #     ttA1    = A1[i]
        #     ttA2    = A2[i]
        #     PHI1    = SIG1[i]
        #     PHI2    = SIG2[i]
        #     
        #     temp1   = ttA1*np.sin(fbaz/180.*np.pi + PHI1)
        #     temp2   = ttA2*np.sin(2*fbaz/180.*np.pi + PHI2)
        #     temp3   = ttA + temp1 + temp2
        #     rdata   = np.append(rdata, temp3)
        #     tempadata   = lfadata[:,i]-temp3
        #     drdata      = np.append(drdata, tempadata)
        #     lfrdata1    = np.append(lfrdata1, temp1)
        #     lfrdata2    = np.append(lfrdata2, temp2)
        # rdata   = rdata.reshape((Lmin, Latime))
        # drdata  = drdata.reshape((Lmin, Latime))
        # lfrdata1= lfrdata1.reshape((Lmin, Latime))
        # lfrdata2= lfrdata2.reshape((Lmin, Latime))
        # 
        # with open(outdir+"/variance_reduction.dat","w") as fVR:
        #     for i in xrange(len(baz)):
        #         tempbaz     = baz[i]
        #         tempbaz1    = float(baz[i])*np.pi/180.
        #         outname     = outdir+"/pre" + names[i]
        #         timeCut     = time[time<=10.]
        #         Ltimecut    = len(timeCut)
        #         obs         = adata[i][time<=10.]
        #         lfA0        = A0preArr(tempbaz1,zA0)[time<=10.]
        #         lfA1        = A1preArr(tempbaz1,oA0,oA1,oSIG1)[time<=10.]
        #         lfA1n       = A1pre1Arr(tempbaz1,oA1,oSIG1)[time<=10.]
        #         lfA2        = A2preArr(tempbaz1,tA0,tA2,tSIG2)[time<=10.]
        #         lfA2n       = A2pre1Arr(tempbaz1,tA2,tSIG2)[time<=10.]
        #         lfA3        = A3preArr(tempbaz1,A0,A1,SIG1,A2,SIG2)[time<=10.]
        #         lfA3n1      = A3pre1Arr(tempbaz1,A1,SIG1)[time<=10.]
        #         lfA3n2      = A3pre2Arr(tempbaz1,A2,SIG2)[time<=10.]
        #         
        #         outpreArr   = np.append(timeCut, obs)
        #         outpreArr   = np.append(outpreArr, lfA0)
        #         outpreArr   = np.append(outpreArr, lfA1)
        #         outpreArr   = np.append(outpreArr, lfA2)
        #         outpreArr   = np.append(outpreArr, lfA3)
        #         outpreArr   = np.append(outpreArr, lfA1n)
        #         outpreArr   = np.append(outpreArr, lfA2n)
        #         outpreArr   = np.append(outpreArr, lfA3n1)
        #         outpreArr   = np.append(outpreArr, lfA3n2)
        #         outpreArr   = outpreArr.reshape((10,Ltimecut))
        #         outpreArr   = outpreArr.T
        #         np.savetxt(outname, outpreArr, fmt='%g')
        #         
        #         vr0         = np.append(vr0, _match1(lfA0,adata[i][time<=10.]))
        #         vr1         = np.append(vr1, _match1(lfA1,adata[i][time<=10.]))
        #         vr2         = np.append(vr2, _match1(lfA2,adata[i][time<=10.]))
        #         vr3         = np.append(vr3, _match1(lfA3,adata[i][time<=10.]))
        #         tempstr = "%d %g %g %g %g %s\n" %(baz[i],vr0[i],vr1[i],vr2[i],vr3[i],names[i])
        #         fVR.write(tempstr)
        # with open(outdir+"/average_vr.dat","w") as favr:
        #     tempstr = "%g %g %g %g\n" %(vr0.mean(), vr1.mean(), vr2.mean(), vr3.mean())
        #     favr.write(tempstr)
        #     
        # dt      = time[1]-time[0]
        # lfadata = lfadata.T ## (Lmin, Latime)        
        # 
        # for i in xrange (len(names)):
        #     outname = outdir+"/diff" + names[i]
        #     outArr  = np.append(time, drdata[:,i])
        #     outArr  = outArr.reshape((2,Lmin))
        #     outArr  = outArr.T
        #     np.savetxt(outname, outArr, fmt='%g')
        # 
        # for i in np.arange (len(names)):
        #     outname = outdir+"/rep" + names[i]
        #     outArr  = np.append(time, rdata[:,i])
        #     outArr  = outArr.reshape((2,Lmin))
        #     outArr  = outArr.T
        #     np.savetxt(outname, outArr, fmt='%g')
        # 
        # for i in np.arange (len(names)):
        #     outname = outdir+"/0rep" + names[i]
        #     outArr  = np.append(time, A0)
        #     outArr  = outArr.reshape((2,Lmin))
        #     outArr  = outArr.T
        #     np.savetxt(outname, outArr, fmt='%g')
        # 
        # for i in np.arange (len(names)):
        #     outname = outdir+"/1rep" + names[i]
        #     outArr  = np.append(time, lfrdata1[:,i])
        #     outArr  = outArr.reshape((2,Lmin))
        #     outArr  = outArr.T
        #     np.savetxt(outname, outArr, fmt='%g')
        #     
        # for i in np.arange (len(names)):
        #     outname = outdir+"/2rep" + names[i]
        #     outArr  = np.append(time, lfrdata2[:,i])
        #     outArr  = outArr.reshape((2,Lmin))
        #     outArr  = outArr.T
        #     np.savetxt(outname, outArr, fmt='%g')
        #     
        # for i in np.arange (len(names)):
        #     outname = outdir+"/obs" + names[i]
        #     outArr  = np.append(time, lfadata[:,i])
        #     outArr  = outArr.reshape((2,Lmin))
        #     outArr  = outArr.T
        #     np.savetxt(outname, outArr, fmt='%g')
        return
    

class RFTrace(obspy.Trace):
    """
    Receiver function trace class, derived from obspy.Trace
    Addon parameters:
    Ztr, RTtr   - Input data, numerator(R/T) and denominator(Z)
    """
    def get_data(self, Ztr, RTtr, tbeg=20.0, tend=-30.0):
        """
        Read raw R/T/Z data for receiver function analysis
        Arrival time will be read/computed for given phase, then data will be trimed according to tbeg and tend.
        """
        if isinstance (Ztr, str):
            self.Ztr    = obspy.read(Ztr)[0]
        elif isinstance(Ztr, obspy.Trace):
            self.Ztr    = Ztr
        else:
            raise TypeError('Unexpecetd type for Ztr!')
        if isinstance (RTtr, str):
            self.RTtr   = obspy.read(RTtr)[0]
        elif isinstance(RTtr, obspy.Trace):
            self.RTtr   = RTtr
        else:
            raise TypeError('Unexpecetd type for RTtr!')
        stime           = self.Ztr.stats.starttime
        etime           = self.Ztr.stats.endtime
        self.Ztr.trim(starttime=stime+tbeg, endtime=etime+tend)
        self.RTtr.trim(starttime=stime+tbeg, endtime=etime+tend)
        return
    
    def IterDeconv(self, tdel=5., f0 = 2.5, niter=200, minderr=0.001, phase='P', addhs=True ):
        """
        Compute receiver function with iterative deconvolution algorithmn
        ========================================================================================================================
        ::: input parameters :::
        tdel       - phase delay
        f0         - Gaussian width factor
        niter      - number of maximum iteration
        minderr    - minimum misfit improvement, iteration will stop if improvement between two steps is smaller than minderr
        phase      - phase name, default is P

        ::: input data  :::
        Ztr        - read from self.Ztr
        RTtr       - read from self.RTtr
        
        ::: output data :::
        self.data  - data array(numpy)
        ::: SAC header :::
        b          - begin time
        e          - end time
        user0      - Gaussian Width factor
        user2      - Variance reduction, (1-rms)*100
        user4      - horizontal slowness
        ========================================================================================================================
        """
        Ztr         = self.Ztr
        RTtr        = self.RTtr
        dt          = Ztr.stats.delta
        npts        = Ztr.stats.npts
        RMS         = np.zeros(niter, dtype=np.float64)  # RMS errors
        nfft        = 2**(npts-1).bit_length() # number points in fourier transform
        P0          = np.zeros(nfft, dtype=np.float64) # predicted spikes
        # Resize and rename the numerator and denominator
        U0          = np.zeros(nfft, dtype=np.float64) #add zeros to the end
        W0          = np.zeros(nfft, dtype=np.float64)
        U0[:npts]   = RTtr.data 
        W0[:npts]   = Ztr.data 
        # get filter in Freq domain 
        gauss       = _gaussFilter( dt, nfft, f0 )
        # filter signals
        Wf0         = np.fft.fft(W0)
        FilteredU0  = _FreFilter(U0, gauss, dt )
        FilteredW0  = _FreFilter(W0, gauss, dt )
        R           = FilteredU0 #  residual numerator
        # Get power in numerator for error scaling
        powerU      = np.sum(FilteredU0**2)
        # Loop through iterations
        it          = 0
        sumsq_i     = 1
        d_error     = 100*powerU + minderr
        maxlag      = int(0.5*nfft)
        while( abs(d_error) > minderr  and  it < niter ):
            it          = it+1 # iteration advance
            #   Ligorria and Ammon method
            # # # RW          = np.real(np.fft.ifft(np.fft.fft(R)*np.conj(np.fft.fft(FilteredW0))))
            RW          = np.real(fftpack.ifft(fftpack.fft(R)*np.conj(fftpack.fft(FilteredW0))))
            sumW0       = np.sum(FilteredW0**2)
            RW          = RW/sumW0
            imax        = np.argmax(abs(RW[:maxlag]))
            amp         = RW[imax]/dt; # scale the max and get correct sign
            #   compute predicted deconvolution
            P0[imax]    = P0[imax] + amp  # get spike signal - predicted RF
            P           = _FreFilter(P0, gauss*Wf0, dt*dt ) # convolve with filter
            #   compute residual with filtered numerator
            R           = FilteredU0 - P
            sumsq       = np.sum(R**2)/powerU
            RMS[it-1]   = sumsq # scaled error
            d_error     = 100*(sumsq_i - sumsq)  # change in error 
            sumsq_i     = sumsq  # store rms for computing difference in next   
        # Compute final receiver function
        P                       = _FreFilter(P0, gauss, dt )
        # Phase shift
        P                       = _phaseshift(P, nfft, dt, tdel)
        # output first nt samples
        RFI                     = P[:npts]
        # output the rms values 
        RMS                     = RMS[:it]
        self.stats              = RTtr.stats
        self.data               = RFI
        self.stats.sac['b']     = -tdel
        self.stats.sac['e']     = -tdel+(npts-1)*dt
        self.stats.sac['user0'] = f0
        self.stats.sac['user2'] = (1.0-RMS[it-1])*100.0
        if addhs:
            self.addHSlowness(phase=phase)
        return
    
    def addHSlowness(self, phase='P'):
        """
        Add horizontal slowness to user4 SAC header, distance. az, baz will also be added
        Computed for a given phase using taup and iasp91 model
        """
        evla                    = self.stats.sac['evla']
        evlo                    = self.stats.sac['evlo']
        stla                    = self.stats.sac['stla']
        stlo                    = self.stats.sac['stlo']
        dist, az, baz           = obspy.geodetics.gps2dist_azimuth(evla, evlo, stla, stlo)
        dist                    = dist/1000.  # distance is in km
        self.stats.sac['dist']  = dist
        self.stats.sac['az']    = az
        self.stats.sac['baz']   = baz
        evdp                    = self.stats.sac['evdp']/1000.
        Delta                   = obspy.geodetics.kilometer2degrees(dist)
        arrivals                = taupmodel.get_travel_times(source_depth_in_km=evdp,\
                                                distance_in_degree=Delta, phase_list=[phase])
        arr                     = arrivals[0]
        rayparam                = arr.ray_param_sec_degree
        arr_time                = arr.time
        self.stats.sac['user4'] = rayparam
        self.stats.sac['user5'] = arr_time
        return
    
    def init_postdbase(self):
        """
        Initialize post-processing database
        """
        self.postdbase  = PostDatabase()
        return
    
    def move_out(self, refslow = 0.06, modeltype=0):
        """
        moveout for receiver function
        """
        self.init_postdbase()
        tslow       = self.stats.sac['user4']/111.12
        ratio       = self.stats.sac['user2']
        b           = self.stats.sac['b']
        e           = self.stats.sac['e']
        baz         = self.stats.sac['baz']
        dt          = self.stats.delta
        npts        = self.stats.npts
        fs          = 1./dt
        o           = 0.
        t           = np.arange(0, npts/fs, 1./fs)
        nb          = int(np.ceil((o-b)*fs))  # index for t = 0.
        nt          = np.arange(0+nb, 0+nb+20*fs, 1) # nt= nb ~ nb+ 20*fs, index array for data 
        if nt[-1]>npts:
            return False
        if len(nt)==1:
            data    = self.data[np.array([np.int_(nt)])]
        else:
            data    = self.data[np.int_(nt)]
        tarr1       = (nt - nb)/fs  # time array for move-outed data
        flag        = 0 # flag signifying whether postdatabase has been written or not
        #---------------------------------------------------------------------------------
        # Step 1: Discard data with too large or too small H slowness
        #---------------------------------------------------------------------------------
        if (tslow <= 0.04 or tslow > 0.1):
            self.postdbase.MoveOutFlag  = -3
            self.postdbase.value        = tslow
            flag                        = 1
        refvp       = 6.0
        #-----------------------------------------------------------------------------------------------
        # Step 2: Discard data with too large Amplitude in receiver function after amplitude correction
        #-----------------------------------------------------------------------------------------------
        # correct amplitude to reference horizontal slowness
        reffactor   = np.arcsin(refslow*refvp)/np.arcsin(tslow*refvp)
        data        = data*reffactor
        absdata     = np.abs(data)
        maxdata     = absdata.max()
        if ( maxdata > 1 and flag == 0):
            self.postdbase.MoveOutFlag  = -2
            self.postdbase.value1       = maxdata
            flag                        = 1
        #----------------------------------------
        # Step 3: Stretch Data
        #----------------------------------------
        tarr2, data2= _stretch(tarr1, data, tslow, refslow=refslow, modeltype=modeltype)
        #--------------------------------------------------------
        # Step 4: Discard data with negative value at zero time
        #--------------------------------------------------------
        if (data2[0] < 0 and flag == 0):
            self.postdbase.MoveOutFlag  = -1
            self.postdbase.value1       = data2[0]
            flag                        = 1     
        if (flag == 0):
            self.postdbase.MoveOutFlag  = 1
            self.postdbase.value1       = None
        #--------------------------------------------------------
        # Step 5: Store the original and stretched data
        #--------------------------------------------------------
        DATA1               = data/1.42
        L                   = DATA1.size
        self.postdbase.ampC = np.append(tarr1,DATA1)
        self.postdbase.ampC = self.postdbase.ampC.reshape((2, L))
        self.postdbase.ampC = self.postdbase.ampC.T
        DATA2               = data2/1.42
        L                   = DATA2.size
        self.postdbase.ampTC= np.append(tarr2, DATA2)
        self.postdbase.ampTC= self.postdbase.ampTC.reshape((2, L))
        self.postdbase.ampTC= self.postdbase.ampTC.T
        return True
    
    def save_data(self, outdir):
        """Save receiver function and post processed (moveout) data to output directory
        """
        outfname                = outdir+'/'+self.stats.sac['kuser0']+'.sac'
        warnings.filterwarnings('ignore', category=UserWarning, append=True)
        self.stats.sac['user6'] = self.postdbase.MoveOutFlag
        self.write(outfname, format = 'sac')
        try:
            np.savez( outdir+'/'+self.stats.sac['kuser0']+'.post', self.postdbase.ampC, self.postdbase.ampTC)
        except:
            return
        return 



    
