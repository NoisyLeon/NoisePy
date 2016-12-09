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
from numba import jit
# from pyproj import Geod
# 
# geodist = Geod(ellps='WGS84')
taupmodel = TauPyModel(model="iasp91")
stretchbackdatafname='./strechback.data'

def _gaussFilter( dt, nft, f0 ):
    """
    Compute a gaussian filter in the freq domain which is unit area in time domain
    private function for IterDeconv
    ================================================================================
    Input:
    dt  - sampling time interval
    nft - number freq points
    f0  - width of filter
    
    Output:
    gauss  - Gaussian filter array (numpy)
    filter has the form: exp( - (0.5*w/f0)^2 ) the units of the filter are 1/s
    ================================================================================
    """
    df      = 1.0/(nft*dt)
    nft21   = 0.5*nft + 1
    # get frequencies
    f       = df*np.arange(nft21)
    w       = 2*np.pi*f
    w       = w/f0
    kernel  = w**2
    # compute the gaussian filter
    gauss   = np.zeros(nft)
    gauss[:nft21]   = np.exp( -0.25*kernel )/dt
    gauss[nft21:]   = np.flipud(gauss[1:nft21-1])
    return gauss


def _phaseshift( x, nfft, DT, TSHIFT ):
    """Add a shift to the data into the freq domain, private function for IterDeconv
    """
    Xf      = np.fft.fft(x)
    # phase shift in radians
    shift_i = round(TSHIFT/DT) # removed +1 from here.
    p       = np.arange(nfft)+1
    p       = 2*np.pi*shift_i/(nfft)*p
    # apply shift
    Xf      = Xf*(np.cos(p) - 1j*np.sin(p))
    # back into time
    x       = np.real( np.fft.ifft(Xf) )/np.cos(2*np.pi*shift_i/nfft)
    return x


def _FreFilter(inW, FilterW, dt ):
    """Filter input array in frequency domain, private function for IterDeconv
    """
    FinW    = np.fft.fft(inW)
    FinW    = FinW*FilterW*dt
    FilterdW= np.real(np.fft.ifft(FinW))
    return FilterdW


def _stretch (t1, nd1, slow):
    """Stretch data given slowness, private function for move_out
    """
    slowi   = slow
    dzi     = 0.5
    dzmax   = 240.
    dZ      = np.arange(int(dzmax/dzi))*0.5
    Rv      = 1.7
    dt      = t1[1] - t1[0]
    ndz     = dZ.size
    zthk    = np.ones(ndz)*dzi
    cpv     = 6.4
    pvel    = np.ones(ndz)*cpv
    pvel    = pvel+(dZ>60)*np.ones(ndz)*1.4
    svel1   = pvel/Rv
    sv2     = svel1**(-2)
    pv2     = (svel1*Rv)**(-2)
    cc      = (np.sqrt(sv2)-np.sqrt(pv2))*dzi
    cc      = np.append(0., cc)
    vtt     = np.cumsum(cc)
    p2      = np.ones(ndz)
    p2      = p2*slowi*slowi
    cc2     = (np.sqrt(sv2-p2)-np.sqrt(pv2-p2))*dzi
    mtt     = np.cumsum(cc2)
    ntt     = np.round(mtt/dt)
    ntt[0]  = 0.
    if len(ntt)==1: kk = np.array([np.int_(ntt)])
    else: kk = np.int_(ntt)
    Ldatain = nd1.size
    kkk     = kk[kk<Ldatain]
    nseis   = nd1[kkk]
    time    = vtt[len(nseis)-1]
    n1      = int(time/dt)
    t2      = np.arange(n1)*dt
    Lt2     = t2.size
    d2      = np.array([])
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
    for i in xrange(nbin):
        mi      = i*binwidth
        ma      = (i+1)*binwidth
        tbaz    = i*binwidth + float(binwidth)/2
        ttdat   = indat[(inbaz>=mi)*(inbaz<ma)]
        if (len(ttdat) > 0):
            outbaz      = np.append(outbaz, tbaz)
            outdat      = np.append(outdat, ttdat.mean())
            if (len(ttdat)>1):
                outun   = np.append(outun, ttdat.std()/(np.sqrt(len(ttdat))) )
            if (len(ttdat)==1):
                outun   = np.append(outun, 0.1)
    return outbaz, outdat, outun

def _difference ( aa, bb, NN):
    """Compute difference between two input array, private function for harmonic stripping
    """
    if NN > 0: L = min(len(aa),len(bb),NN)
    else: L = min(len(aa),len(bb))
    aa  = aa[:L]
    bb  = bb[:L]
    core= np.sum((aa-bb)*(aa-bb))
    core= core / L
    return np.sqrt(core)

def _invert_A0 ( inbaz, indat, inun ):   #only invert for A0 part
    """Invert for A0, private function for harmonic stripping
    """
    m       = len(inbaz); # data space;
    n       = 1; # model space;
    U       = np.zeros((m,m))
    np.fill_diagonal(U, 1./inun)
    G1      = np.ones((m,1))
    G1      = np.dot(U,G1)
    d       = indat.T
    d       = np.dot(U,d)
    model   = np.linalg.lstsq(G1,d)[0]
    cvA0    = model[0]
    ccdat   = np.dot(G1,model)
    ccdat   = ccdat[:m]
    inun    = inun[:m]
    odat    = ccdat*inun
    return cvA0, odat

def _invert_A2 ( inbaz, indat, inun ):
    """Invert for A2, private function for harmonic stripping
    """
    m       = len(inbaz) # data space;
    n       = 3 # model space;
    U       = np.zeros((m,m))
    np.fill_diagonal(U, 1./inun)
    tG1     = np.ones((m,1))
    tbaz    = np.pi*inbaz/180
    tGsin   = np.sin(tbaz*2)
    tGcos   = np.cos(tbaz*2)
    G       = np.append(tG1, tGsin)
    G       = np.append(G, tGcos)
    G       = G.reshape((3,m))
    G1      = G.T
    G1      = np.dot(U,G1)
    d       = indat.T
    d       = np.dot(U,d)
    model   = np.linalg.lstsq(G1,d)[0]
    resid   = np.linalg.lstsq(G1,d)[1]
    A0      = model[0]
    A2      = np.sqrt(model[1]**2 + model[2]**2)
    fi2     = np.arctan2(model[2],model[1])
    ccdat   = np.dot(G1,model)
    odat    = ccdat*inun
    return A0, A2, fi2, odat;


def _invert_A1 ( inbaz, indat, inun ):
    """Invert for A1, private function for harmonic stripping
    """
    m       = len(inbaz) # data space;
    n       = 3 # model space;
    A0      = 0
    A2      = 0
    fi2     = 0
    U       = np.zeros((m,m))
    np.fill_diagonal(U, 1./inun)
    tG1     = np.ones((m,1))
    tbaz    = np.pi*inbaz/180
    tGsin   = np.sin(tbaz)
    tGcos   = np.cos(tbaz)
    G       = np.append(tG1, tGsin)
    G       = np.append(G, tGcos)
    G       = G.reshape((3,m))
    G1      = G.T
    G1      = np.dot(U,G1)
    d       = indat.T
    d       = np.dot(U,d)
    model   = np.linalg.lstsq(G1,d)[0]
    resid   = np.linalg.lstsq(G1,d)[1]
    A0      = model[0]
    A1      = np.sqrt(model[1]**2 + model[2]**2)
    fi1     = np.arctan2(model[2], model[1])
    ccdat   = np.dot(G1,model)
    odat    = ccdat*inun
    return A0, A1, fi1, odat

def _invert_1 ( inbaz, indat, inun):
    """Invert for A0, A1, A2, private function for harmonic stripping
    """
    m       = len(inbaz)# data space;
    n       = 5 # model space;
    #	print m,len(inun);
    U       = np.zeros((m,m))
    np.fill_diagonal(U, 1./inun)
    tG1     = np.ones((m,1))
    tbaz    = np.pi*inbaz/180
    tGsin   = np.sin(tbaz)
    tGcos   = np.cos(tbaz)
    tGsin2  = np.sin(tbaz*2)
    tGcos2  = np.cos(tbaz*2)
    G       = np.append(tG1, tGsin)
    G       = np.append(G, tGcos)
    G       = np.append(G, tGsin2)
    G       = np.append(G, tGcos2)
    G       = G.reshape((5,m))
    G1      = G.T
    G1      = np.dot(U,G1)
    d       = indat.T
    d       = np.dot(U,d)
    model   = np.linalg.lstsq(G1,d)[0]
    resid   = np.linalg.lstsq(G1,d)[1]
    A0      = model[0]
    A1      = np.sqrt(model[1]**2 + model[2]**2)
    fi1     = np.arctan2(model[2],model[1])                                           
    A2      = np.sqrt(model[3]**2 + model[4]**2)
    fi2     = np.arctan2(model[4],model[3])
    # compute forward:
    ccdat   = np.dot(G1,model)
    odat    = ccdat*inun
    return A0,A1,fi1,A2,fi2,odat

##############################################
# Function for computing predictions
##############################################

def A0preArr ( inbaz, A0 ): return A0

def A1preArr ( inbaz, A0, A1, SIG1): return A0 + A1*np.sin(inbaz+SIG1)

def A1pre1Arr ( inbaz, A1, SIG1): return A1*np.sin(inbaz+SIG1)

def A2preArr ( inbaz, A0, A2, SIG2): return A0 + A2*np.sin(2*inbaz+SIG2)

def A2pre1Arr ( inbaz, A2, SIG2): return A2*np.sin(2*inbaz+SIG2)

def A3preArr ( inbaz, A0, A1, SIG1, A2, SIG2): return A0 + A1*np.sin(inbaz + SIG1) + A2*np.sin(2*inbaz + SIG2)

def A3pre1Arr ( inbaz, A1, SIG1): return A1*np.sin(inbaz + SIG1)

def A3pre2Arr ( inbaz, A2, SIG2): return A2*np.sin(2*inbaz + SIG2)

def A3pre3Arr ( inbaz, A1, SIG1, A2, SIG2 ): return A1*np.sin(inbaz + SIG1) + A2*np.sin(2*inbaz + SIG2)

def A3pre ( inbaz, A0, A1, SIG1, A2, SIG2): return A0 + A1*np.sin(inbaz + SIG1) + A2*np.sin(2*inbaz + SIG2)

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
    def __init__(self, reftype='R', tbeg=20.0, tend=-30.0, tdel=5., f0 = 2.5, niter=200, minderr=0.001, phase='P' ):
        self.reftype        = reftype
        self.tbeg           = tbeg
        self.tend           = tend
        self.tdel           = tdel
        self.f0             = f0
        self.niter          = niter
        self.minderr        = minderr
        self.phase          = phase
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
        Input Parameters:
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
            tfill=time[yvalue>0]
            yfill=(yvalue+backazi)[yvalue>0]
            ax.fill_between(tfill, backazi, yfill, color='blue', linestyle='--', lw=0.)
            tfill=time[yvalue<0]
            yfill=(yvalue+backazi)[yvalue<0]
            ax.fill_between(tfill, backazi, yfill, color='red', linestyle='--', lw=0.)
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
    
    def PlotHSStreams(self, outdir='', stacode='', ampfactor=40, targetDT=0.02, longitude='', latitude='', browseflag=False, saveflag=True,\
            obsflag=1, diffflag=0, repflag=1, rep0flag=1, rep1flag=1, rep2flag=1):
        """Plot harmonic stripping streams accoring to back-azimuth
        ===============================================================================================================
        Input Parameters:
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
    ampC        - scaled receiver function
    ampTC       - moveout of receiver function
    strback     - stretched back receiver function
    header      - receiver function header
    tdiff       - trace difference
    ===============================================================================================================
    """
    def __init__(self):
        self.MoveOutFlag    = None
        self.ampC           = np.array([]) # 0.06...out
        self.ampTC          = np.array([]) # stre...out
        self.strback        = np.array([]) #stre...out.back
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
        Input Parameters:
        outdir      - output directory
        Output:
        outdir/wmean.txt, outdir/bin_%d_txt
        ===============================================================================================================
        """
        tempLst1    = PostRefLst()
        lens        = np.array([]) # array to store length for each stretched back trace
        baz1        = np.array([]) # array to store baz for each stretched back trace
        for PostData in self.PostDatas:
            time    = PostData.strback[:,0]
            data    = PostData.strback[:,1]
            L       = len(time)
            tflag   = True
            if abs(data).max()>1: tflag = False
            if data[abs(time)<0.1].min()<0.02: tflag = False;
            if tflag:
                PostData.Len = L
                lens= np.append(lens,L)
                tempLst1.append(PostData)
                baz1= np.append( baz1, np.floor(PostData.header['baz']))
        #Grouped data
        gbaz        = np.array([])
        gdata       = np.array([])
        gun         = np.array([])
        ## store the stacked RF#
        Lmin        = int(lens.min())
        nt1         = tempLst1[0].strback[:,0]
        nt1         = nt1[:Lmin]
        nv1         = np.array([])
        nu1         = np.array([])
        LengthLst   = len(tempLst1)
        for i in xrange( Lmin ):
            tdat=np.array([])
            for j in xrange (LengthLst): tdat=np.append(tdat, tempLst1[j].strback[i,1])
            b1,t1,u1= _group(baz1, tdat)
            gbaz    = np.append(gbaz,b1)
            gdata   = np.append(gdata,t1)
            gun     = np.append(gun,u1)
            t1DIVu1 = t1/u1
            DIVu1   = 1./u1
            wmean   = np.sum(t1DIVu1)
            weight  = np.sum(DIVu1)
            if (weight > 0.): nv1=np.append(nv1,wmean/weight)
            else:
                print "weight is zero!!! ", len(t1), u1, t1
                sys.exit()
            nu1     = np.append(nu1, np.sum(u1)/len(u1))
        lengthbaz   = len(b1)
        gbaz        = gbaz.reshape((Lmin, lengthbaz))
        gdata       = gdata.reshape((Lmin, lengthbaz))
        gun         = gun.reshape((Lmin, lengthbaz))

        # Save wmean.txt
        outname     = outdir+"/wmean.txt"
        Lnt1        = len(nt1)
        outwmeanArr = np.append(nt1, nv1)
        outwmeanArr = np.append(outwmeanArr, nu1)
        outwmeanArr = outwmeanArr.reshape((3,Lnt1))
        outwmeanArr = outwmeanArr.T
        np.savetxt(outname, outwmeanArr, fmt='%g')
        # Save bin_*_txt
        for i in xrange (lengthbaz): # back -azimuth
            outname     = outdir+"/bin_%d_txt" % (int(gbaz[0][i]))
            outbinArr   = np.append(nt1[:Lmin], gdata[:,i])
            outbinArr   = np.append(outbinArr, gun[:,i])
            outbinArr   = outbinArr.reshape((3,Lmin ))
            outbinArr   = outbinArr.T
            np.savetxt(outname, outbinArr, fmt='%g')
        
        tdiff = -1.
        for i in xrange(len(tempLst1)):
            time    = tempLst1[i].strback[:,0]
            data    = tempLst1[i].strback[:,1]
            tflag   = 1.
            Lmin    = min( len(time) , len(nt1) )
            AA      = data[:Lmin]
            BB      = nv1[:Lmin]
            tdiff   = _difference ( AA, BB, 0)
            if (tdiff > 0.1): tflag = 0.
            tempLst1[i].tdiff=tdiff
        return tempLst1
    
    
    
    def QControl_tdiff(self, tdiff=0.08):
        """Remove data given threshold tdiff
        """
        tempLst=PostRefLst()
        for PostData in self.PostDatas:
            if PostData.tdiff<tdiff : tempLst.append(PostData)
        return tempLst
    
    
    def HarmonicStripping(self, stacode, outdir):
        """
        Harmonic Stripping Analysis for quality controlled data.
        ===============================================================================================================
        Input Parameters:
        stacode     - station code( e.g. TA.R11A )
        outdir      - output directory
        
        Output:
        outdir/bin_%d_rf.dat, outdir/A0.dat, outdir/A1.dat, outdir/A2.dat, outdir/A0_A1_A2.dat
        outdir/average_vr.dat, outdir/variance_reduction.dat
        outdir/prestre_*, outdir/repstre_*, outdir/obsstre_*, outdir/repstre_*
        outdir/0repstre_*, outdir/1repstre_*, outdir/2repstre_*, outdir/diffstre_*
        ===============================================================================================================
        """
        baz     = np.array([])
        lens    = np.array([])
        atime   = []
        adata   = []
        names   = []
        eventT  = []
        for PostData in self.PostDatas:
            time = PostData.strback[:,0]
            data = PostData.strback[:,1]
            adata.append(data)
            atime.append(time)
            L    = len(time)
            lens = np.append(lens, L)
            baz  = np.append(baz, np.floor(PostData.header['baz']))
            name = 'stre_'+str(int(PostData.header['baz']))+'_'+stacode+'_'+str(PostData.header['otime'])+'.out.back'
            names.append(name)
            eventT.append(PostData.header['otime'])
        # parameters in 3 different inversion
        zA0     = np.array([])
        oA0     = np.array([])
        oA1     = np.array([])
        oSIG1   = np.array([])
        tA0     = np.array([])
        tA2     = np.array([])
        tSIG2   = np.array([])
        A0      = np.array([])
        A1      = np.array([])
        A2      = np.array([])
        SIG1    = np.array([])
        SIG2    = np.array([])
        A0123   = np.array([])
        MF0     = np.array([])  # misfit between A0 and R[i]
        MF1     = np.array([])  # misfit between A0+A1+A2 and R[i]
        MF2     = np.array([]) # misfit between A0+A1+A2 and binned data
        MF3     = np.array([]) # weighted misfit between A0+A1+A2 and binned data
        A_A     = np.array([])  # average amplitude
        A_A_un  = np.array([])
        # grouped data
        gbaz    = np.array([])
        gdata   = np.array([])
        gun     = np.array([])
        Lmin    = int(lens.min())
        for i in xrange (Lmin):
            tdat=np.array([])
            for PostData in self.PostDatas: tdat = np.append(tdat,PostData.strback[i,1])
            aa      = tdat.mean()
            naa     = tdat.std()
            baz1,tdat1,udat1=_group(baz, tdat)
            gbaz    = np.append(gbaz,baz1)
            gdata   = np.append(gdata,tdat1)
            gun     = np.append(gun,udat1)
            # now do inversions 
            (tempv0,odat1) = _invert_A0 (baz1,tdat1,udat1)
            zA0     = np.append(zA0, tempv0)
            
            (tempv0,tempv1, tempv2, odat1) = _invert_A1 (baz1,tdat1,udat1)
            oA0     = np.append(oA0, tempv0)
            oA1     = np.append(oA1, tempv1)
            oSIG1   = np.append(oSIG1, tempv2)
            
            (tempv0,tempv1, tempv2,odat1) = _invert_A2 (baz1,tdat1,udat1)
            tA0     = np.append(tA0,tempv0)
            tA2     = np.append(tA2,tempv1)
            tSIG2   = np.append(tSIG2, tempv2)
            
            (tempv0,tempv1, tempv2,tempv3,tempv4,odat1) = _invert_1 (baz1,tdat1,udat1)
            A0      = np.append(A0, tempv0)
            A1      = np.append(A1, tempv1)
            SIG1    = np.append(SIG1, tempv2)
            A2      = np.append(A2, tempv3)
            SIG2    = np.append(SIG2, tempv4)
            A_A     = np.append(A_A,aa)
            A_A_un  = np.append(A_A_un,naa)
            
            mf  = 0.
            mf1 = 0.
            for j in xrange (len(baz)):
                mf  = mf + (tempv0 - adata[j][i])**2
                vv  = A3pre(baz[j]*np.pi/180.,tempv0,tempv1,tempv2,tempv3,tempv4)
                mf1 = mf1 + (vv - adata[j][i])**2
            mf  = np.sqrt(mf/len(baz))
            mf1 = np.sqrt(mf1/len(baz))
            if (mf<0.005): mf = 0.005
            if (mf1<0.005): mf1 = 0.005
            MF0 = np.append(MF0, mf-0.)
            MF1 = np.append(MF1, mf1-0.)
            mf2 = 0.
            mf3 = 0.
            V1  = 0.
            for j in np.arange (len(baz1)):
                vv  = A3pre(baz1[j]*np.pi/180.,tempv0,tempv1,tempv2,tempv3,tempv4)
                mf2 = mf2 + (vv - tdat1[j])**2;
                mf3 = mf3 + (vv - tdat1[j])**2/udat1[j]**2
                V1  = V1 + 1./(udat1[j]**2)
            mf2 = np.sqrt(mf2/len(baz1))
            mf3 = np.sqrt(mf3/V1)
            MF2 = np.append(MF2, mf2-0.)
            MF3 = np.append(MF3, mf3-0.)
            
        lengthbaz   = len(baz1)
        gbaz        = gbaz.reshape((Lmin, lengthbaz))
        gdata       = gdata.reshape((Lmin, lengthbaz))
        gun         = gun.reshape((Lmin, lengthbaz))
        #Output grouped data
        for i in xrange (len(gbaz[0])): #baz
            tname       = outdir+"/bin_%g_rf.dat" % (gbaz[0][i])
            outbinArr   = np.append(atime[0][:Lmin], gdata[:,i])
            outbinArr   = np.append(outbinArr, gun[:,i])
            outbinArr   = outbinArr.reshape((3,Lmin ))
            outbinArr   = outbinArr.T
            np.savetxt(tname, outbinArr, fmt='%g')
        
        time    = atime[0]
        time    = time[:Lmin]
        
        ttA     = zA0
        timef0  = time[(ttA>-2)*(ttA<2)]
        ttAf0   = ttA[(ttA>-2)*(ttA<2)]
        Lf0     = timef0.size
        outArrf0= np.append(timef0,ttAf0)
        outArrf0= outArrf0.reshape((2,Lf0))
        outArrf0= outArrf0.T
        np.savetxt(outdir+"/A0.dat", outArrf0, fmt='%g')
        
        ttA     = oA0
        ttA1    = oA1
        PHI1    = oSIG1
        ttAf1   = ttA[(ttA>-2)*(ttA<2)]
        ttA1f1  = ttA1[(ttA>-2)*(ttA<2)]
        PHI1f1  = PHI1[(ttA>-2)*(ttA<2)]
        timef1  = time[(ttA>-2)*(ttA<2)]
        Lf1     = ttAf1.size
        PHI1f1  = PHI1f1+(PHI1f1<0)*np.pi
        outArrf1= np.append(timef1, ttAf1)
        outArrf1= np.append(outArrf1, ttA1f1)
        outArrf1= np.append(outArrf1, PHI1f1)
        outArrf1= outArrf1.reshape((4,Lf1))
        outArrf1= outArrf1.T
        np.savetxt(outdir+"/A1.dat", outArrf1, fmt='%g')
        
        ttA     = tA0[:Lmin]
        ttA2    = tA2[:Lmin]
        PHI2    = tSIG2[:Lmin]
        ttAf2   = ttA[(ttA>-2)*(ttA<2)]
        ttA2f2  = ttA2[(ttA>-2)*(ttA<2)]
        PHI2f2  = PHI2[(ttA>-2)*(ttA<2)]
        timef2  = time[(ttA>-2)*(ttA<2)]
        Lf2     = ttAf2.size
        PHI2f2  = PHI2f2+(PHI2f2<0)*np.pi
        outArrf2= np.append(timef2, ttAf2)
        outArrf2= np.append(outArrf2, ttA2f2)
        outArrf2= np.append(outArrf2, PHI2f2)
        outArrf2= outArrf2.reshape((4,Lf2))
        outArrf2= outArrf2.T
        np.savetxt(outdir+"/A2.dat", outArrf2, fmt='%g')
    
        ttA     = A0
        ttA1    = A1
        ttA2    = A2
        PHI1    = SIG1
        PHI2    = SIG2
        ttAf3   = ttA[(ttA>-200)*(ttA<200)]
        ttA1f3  = ttA1[(ttA>-200)*(ttA<200)]
        ttA2f3  = ttA2[(ttA>-200)*(ttA<200)];
        PHI1f3  = PHI1[(ttA>-200)*(ttA<200)]*180/np.pi
        PHI2f3  = PHI2[(ttA>-200)*(ttA<200)]*180/np.pi
        timef3  = time[(ttA>-200)*(ttA<200)]
        MF0f3   = MF0[(ttA>-200)*(ttA<200)]
        MF1f3   = MF1[(ttA>-200)*(ttA<200)]
        MF2f3   = MF2[(ttA>-200)*(ttA<200)]
        MF3f3   = MF3[(ttA>-200)*(ttA<200)]
        AAf3    = A_A[(ttA>-200)*(ttA<200)]
        AAunf3  = A_A_un[(ttA>-200)*(ttA<200)]
        Lf3     = ttAf3.size
        outArrf3= np.append(timef3, ttAf3)
        outArrf3= np.append(outArrf3, ttA1f3)
        outArrf3= np.append(outArrf3, PHI1f3)
        outArrf3= np.append(outArrf3, ttA2f3)
        outArrf3= np.append(outArrf3, PHI2f3)
        outArrf3= np.append(outArrf3, MF0f3)
        outArrf3= np.append(outArrf3, MF1f3)
        outArrf3= np.append(outArrf3, MF2f3)
        outArrf3= np.append(outArrf3, MF3f3)
        outArrf3= np.append(outArrf3, AAf3)
        outArrf3= np.append(outArrf3, AAunf3)
        outArrf3= outArrf3.reshape((12,Lf3))
        outArrf3= outArrf3.T
        np.savetxt(outdir+"/A0_A1_A2.dat", outArrf3, fmt='%g')
        ##################################################################
        Latime  = len(atime)
        if len(baz)==1: fbaz=np.array([np.float_(baz)])
        else: fbaz=np.float_(baz)
        fbaz    = fbaz[:Latime]
        lfadata = np.array([])
        ##################################################################
        rdata   = np.array([])
        drdata  = np.array([]) # this is raw - 0 - 1 - 2
        rdata0  = np.array([]) # only 0
        lfrdata1= np.array([]) # 0+1
        lfrdata2= np.array([]) # 0+2
        vr0     = np.array([])
        vr1     = np.array([])
        vr2     = np.array([])
        vr3     = np.array([])
        for j in xrange(Latime): lfadata=np.append(lfadata, adata[j][:Lmin])
        lfadata=lfadata.reshape((Latime, Lmin))
        for i in xrange(Lmin):
            ttA     = A0[i]
            ttA1    = A1[i]
            ttA2    = A2[i]
            PHI1    = SIG1[i]
            PHI2    = SIG2[i]
            
            temp1   = ttA1*np.sin(fbaz/180.*np.pi + PHI1)
            temp2   = ttA2*np.sin(2*fbaz/180.*np.pi + PHI2)
            temp3   = ttA + temp1 + temp2
            rdata   = np.append(rdata, temp3)
            tempadata   = lfadata[:,i]-temp3
            drdata      = np.append(drdata, tempadata)
            lfrdata1    = np.append(lfrdata1, temp1)
            lfrdata2    = np.append(lfrdata2, temp2)
        rdata   = rdata.reshape((Lmin, Latime))
        drdata  = drdata.reshape((Lmin, Latime))
        lfrdata1= lfrdata1.reshape((Lmin, Latime))
        lfrdata2= lfrdata2.reshape((Lmin, Latime))
        
        with open(outdir+"/variance_reduction.dat","w") as fVR:
            for i in xrange(len(baz)):
                tempbaz     = baz[i]
                tempbaz1    = float(baz[i])*np.pi/180.
                outname     = outdir+"/pre" + names[i]
                timeCut     = time[time<=10.]
                Ltimecut    = len(timeCut)
                obs         = adata[i][time<=10.]
                lfA0        = A0preArr(tempbaz1,zA0)[time<=10.]
                lfA1        = A1preArr(tempbaz1,oA0,oA1,oSIG1)[time<=10.]
                lfA1n       = A1pre1Arr(tempbaz1,oA1,oSIG1)[time<=10.]
                lfA2        = A2preArr(tempbaz1,tA0,tA2,tSIG2)[time<=10.]
                lfA2n       = A2pre1Arr(tempbaz1,tA2,tSIG2)[time<=10.]
                lfA3        = A3preArr(tempbaz1,A0,A1,SIG1,A2,SIG2)[time<=10.]
                lfA3n1      = A3pre1Arr(tempbaz1,A1,SIG1)[time<=10.]
                lfA3n2      = A3pre2Arr(tempbaz1,A2,SIG2)[time<=10.]
                
                outpreArr   = np.append(timeCut, obs)
                outpreArr   = np.append(outpreArr, lfA0)
                outpreArr   = np.append(outpreArr, lfA1)
                outpreArr   = np.append(outpreArr, lfA2)
                outpreArr   = np.append(outpreArr, lfA3)
                outpreArr   = np.append(outpreArr, lfA1n)
                outpreArr   = np.append(outpreArr, lfA2n)
                outpreArr   = np.append(outpreArr, lfA3n1)
                outpreArr   = np.append(outpreArr, lfA3n2)
                outpreArr   = outpreArr.reshape((10,Ltimecut))
                outpreArr   = outpreArr.T
                np.savetxt(outname, outpreArr, fmt='%g')
                
                vr0         = np.append(vr0, _match1(lfA0,adata[i][time<=10.]))
                vr1         = np.append(vr1, _match1(lfA1,adata[i][time<=10.]))
                vr2         = np.append(vr2, _match1(lfA2,adata[i][time<=10.]))
                vr3         = np.append(vr3, _match1(lfA3,adata[i][time<=10.]))
                tempstr = "%d %g %g %g %g %s\n" %(baz[i],vr0[i],vr1[i],vr2[i],vr3[i],names[i])
                fVR.write(tempstr)
        with open(outdir+"/average_vr.dat","w") as favr:
            tempstr = "%g %g %g %g\n" %(vr0.mean(), vr1.mean(), vr2.mean(), vr3.mean())
            favr.write(tempstr)
            
        dt      = time[1]-time[0]
        lfadata = lfadata.T ## (Lmin, Latime)        

        for i in xrange (len(names)):
            outname = outdir+"/diff" + names[i]
            outArr  = np.append(time, drdata[:,i])
            outArr  = outArr.reshape((2,Lmin))
            outArr  = outArr.T
            np.savetxt(outname, outArr, fmt='%g')
        
        for i in np.arange (len(names)):
            outname = outdir+"/rep" + names[i]
            outArr  = np.append(time, rdata[:,i])
            outArr  = outArr.reshape((2,Lmin))
            outArr  = outArr.T
            np.savetxt(outname, outArr, fmt='%g')
        
        for i in np.arange (len(names)):
            outname = outdir+"/0rep" + names[i]
            outArr  = np.append(time, A0)
            outArr  = outArr.reshape((2,Lmin))
            outArr  = outArr.T
            np.savetxt(outname, outArr, fmt='%g')
        
        for i in np.arange (len(names)):
            outname = outdir+"/1rep" + names[i]
            outArr  = np.append(time, lfrdata1[:,i])
            outArr  = outArr.reshape((2,Lmin))
            outArr  = outArr.T
            np.savetxt(outname, outArr, fmt='%g')
            
        for i in np.arange (len(names)):
            outname = outdir+"/2rep" + names[i]
            outArr  = np.append(time, lfrdata2[:,i])
            outArr  = outArr.reshape((2,Lmin))
            outArr  = outArr.T
            np.savetxt(outname, outArr, fmt='%g')
            
        for i in np.arange (len(names)):
            outname = outdir+"/obs" + names[i]
            outArr  = np.append(time, lfadata[:,i])
            outArr  = outArr.reshape((2,Lmin))
            outArr  = outArr.T
            np.savetxt(outname, outArr, fmt='%g')
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
        if isinstance (Ztr, str): self.Ztr=obspy.read(Ztr)[0]
        else: self.Ztr=Ztr
        if isinstance (RTtr, str): self.RTtr=obspy.read(RTtr)[0]
        else: self.RTtr=RTtr
        stime=self.Ztr.stats.starttime; etime=self.Ztr.stats.endtime
        self.Ztr.trim(starttime=stime+tbeg, endtime=etime+tend)
        self.RTtr.trim(starttime=stime+tbeg, endtime=etime+tend)
        return
    
    def IterDeconv(self, tdel=5., f0 = 2.5, niter=200, minderr=0.001, phase='P', addhs=True ):
        """
        Compute receiver function with iterative deconvolution algorithmn
        ========================================================================================================================
        Input Parameters:
        tdel       - phase delay
        f0         - Gaussian width factor
        niter      - number of maximum iteration
        minderr    - minimum misfit improvement, iteration will stop if improvement between two steps is smaller than minderr
        phase      - phase name, default is P

        Input:
        Ztr        - read from self.Ztr
        RTtr       - read from self.RTtr
        
        Output:
        self.data  - data array(numpy)
        SAC header:
        b          - begin time
        e          - end time
        user0      - Gaussian Width factor
        user2      - Variance reduction, (1-rms)*100
        user4      - horizontal slowness
        ========================================================================================================================
        """
        Ztr     = self.Ztr
        RTtr    = self.RTtr
        dt      = Ztr.stats.delta
        npts    = Ztr.stats.npts
        RMS     = np.zeros(niter)  # RMS errors
        nfft    = 2**(npts-1).bit_length() # number points in fourier transform
        P0      = np.zeros(nfft) # predicted spikes
        # Resize and rename the numerator and denominator
        U0          = np.zeros(nfft) #add zeros to the end
        W0          = np.zeros(nfft)
        U0[:npts]   = RTtr.data # clear UIN;
        W0[:npts]   = Ztr.data # clear WIN;
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
        maxlag      = 0.5*nfft
        while( abs(d_error) > minderr  and  it < niter ):
            it          = it+1 # iteration advance
            #   Ligorria and Ammon method
            RW          = np.real(np.fft.ifft(np.fft.fft(R)*np.conj(np.fft.fft(FilteredW0))))
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
        P   = _FreFilter(P0, gauss, dt )
        # Phase shift
        P   = _phaseshift(P, nfft, dt, tdel)
        # output first nt samples
        RFI = P[:npts]
        # output the rms values 
        RMS = RMS[:it]
        self.stats              = RTtr.stats
        self.data               = RFI
        self.stats.sac['b']     = -tdel
        self.stats.sac['e']     = -tdel+npts*dt
        self.stats.sac['user0'] = f0
        self.stats.sac['user2'] = (1.0-RMS[it-1])*100.0
        if addhs: self.addHSlowness(phase=phase)
        return
    
    def addHSlowness(self, phase='P'):
        """
        Add horizontal slowness to user4 SAC header, distance. az, baz will also be added
        Computed for a given phase using taup and iasp91 model
        """
        evla=self.stats.sac['evla']; evlo=self.stats.sac['evlo']
        stla=self.stats.sac['stla']; stlo=self.stats.sac['stlo']
        dist, az, baz           = obspy.geodetics.gps2dist_azimuth(evla, evlo, stla, stlo); dist=dist/1000.  # distance is in m
        self.stats.sac['dist']  = dist
        self.stats.sac['az']    = az
        self.stats.sac['baz']   = baz
        evdp                    = self.stats.sac['evdp']/1000.
        Delta                   = obspy.geodetics.kilometer2degrees(dist)
        arrivals = taupmodel.get_travel_times(source_depth_in_km=evdp, distance_in_degree=Delta, phase_list=[phase])
        arr=arrivals[0]; rayparam=arr.ray_param_sec_degree; arr_time=arr.time
        self.stats.sac['user4']=rayparam
        self.stats.sac['user5']=arr_time
        return
    
    def init_postdbase(self):
        """
        Initialize post-processing database
        """
        self.postdbase=PostDatabase()
        return
    
    def move_out(self):
        """
        moveout for receiver function
        """
        self.init_postdbase()
        tslow       = self.stats.sac['user4']/111.12
        ratio       = self.stats.sac['user2']
        b           = self.stats.sac['b']; e = self.stats.sac['e']
        baz         = self.stats.sac['baz']; dt = self.stats.delta; npts = self.stats.npts
        samprate    = 1./dt
        a           = 0.
        t           = np.arange(0, npts/samprate, 1./samprate)
        nb          = int(np.ceil((a-b)*samprate))
        t1          = np.arange((0+nb)/samprate, (0+nb)/samprate+35, 1/samprate) # t1= nb ~ nb+ 35s
        nt          = np.arange(0+nb, 0+nb+20*samprate, 1) # nt= nb ~ nb+ 20s
        if nt[-1]>npts: return False
        if len(nt)==1: data=self.data[np.array([np.int_(nt)])]
        else: data=self.data[np.int_(nt)]
        nt1         = (nt - nb)/samprate
        flag        = 0 # flag signifying whether postdatabase has been written or not
        # Step 1: Discard data with too large or too small H slowness
        if (tslow <= 0.04 or tslow > 0.1):
            self.postdbase.MoveOutFlag  = -3
            self.postdbase.value1       = tslow
            flag                        = 1
        refvp       = 6.0
        refslow     = 0.06 # Reference Horizontal Slowness
        # Step 2: Discard data with too large Amplitude in receiver function after amplitude correction
        reffactor   = np.arcsin(refslow*refvp)/np.arcsin(tslow*refvp)
        data        = data*reffactor
        absdata     = np.abs(data)
        maxdata     = absdata.max()
        if ( maxdata > 1 and flag == 0):
            self.postdbase.MoveOutFlag=-2
            self.postdbase.value1=maxdata
            flag = 1
        # Amplitude correction is done.
        # Step 3: Stretch Data
        nt2, data2  = _stretch (nt1, data, tslow)
        # Step 4: Discard data with negative value at zero time
        if (data2[0] < 0 and flag == 0):
            self.postdbase.MoveOutFlag  = -1
            self.postdbase.value1       = data2[0]
            flag = 1     
        if (flag == 0):
            self.postdbase.MoveOutFlag  = 1
            self.postdbase.value1       = None 
        DATA1               = data/1.42
        L                   = DATA1.size
        self.postdbase.ampC = np.append(nt1,DATA1)
        self.postdbase.ampC = self.postdbase.ampC.reshape((2, L))
        self.postdbase.ampC = self.postdbase.ampC.T
        DATA2               = data2/1.42
        L                   = DATA2.size
        self.postdbase.ampTC= np.append(nt2,DATA2)
        self.postdbase.ampTC= self.postdbase.ampTC.reshape((2, L))
        self.postdbase.ampTC= self.postdbase.ampTC.T
        win1 = 3.; win2 = 8.
        cutted_data2    = data2[(nt2>=win1)*(nt2<=win2)]
        cutted_nt2      = nt2[(nt2>=win1)*(nt2<=win2)]
        peak            = cutted_data2.max()
        time            = cutted_nt2[cutted_data2.argmax()]
        return True

    def stretch_back(self):
        """
        strech the receiver function back to slow = 0.06 using the strech ship from the result of strech_back.py
        """
        # streching factor
        t0          = self.postdbase.ampTC[:,0]
        a0          = self.postdbase.ampTC[:,1]
        n1          = len(t0)
        tt1         = t0.max()
        dt          = t0[1]-t0[0]
        templatedata= np.loadtxt(stretchbackdatafname)
        ta=templatedata[:,0]; tp=templatedata[:,1]
        n2          = len(ta)
        tt2         = ta.max()
        n3          = int(min(tt1,tt2)/dt)-1
        strebackT   = np.arange(n3)*dt
        strebackA   = np.array([])
        for tempt in strebackT:
            smallTF = np.where(tp>=tempt)[0]
            indexj  = smallTF[0]-1
            if indexj<0: indexj=0;
            newt    = ta[indexj];
            smallTF_t0  = np.where(t0<=newt)[0]
            indexk      = smallTF_t0[-1]
            newv        = a0[indexk] + (a0[indexk+1] - a0[indexk])*(newt - t0[indexk])/(t0[indexk+1]-t0[indexk])
            strebackA   = np.append(strebackA, newv)
        L=strebackA.size
        self.postdbase.strback=np.append(strebackT,strebackA)
        self.postdbase.strback=self.postdbase.strback.reshape((2, L))
        self.postdbase.strback=self.postdbase.strback.T
        return
    
    def save_data(self, outdir):
        """Save receiver function and post processed (moveout, stretchback) data to output directory
        """
        outfname=outdir+'/'+self.stats.sac['kuser0']+'.sac'
        warnings.filterwarnings('ignore', category=UserWarning, append=True)
        self.write(outfname, 'sac')
        try: np.savez( outdir+'/'+self.stats.sac['kuser0']+'.post', self.postdbase.ampC, self.postdbase.ampTC, self.postdbase.strback)
        except: return
        return 



    
