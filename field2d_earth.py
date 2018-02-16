# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
import numpy.ma as ma
import scipy.ndimage.filters 
from scipy.ndimage import convolve
import matplotlib
import multiprocessing
from functools import partial
import os
from subprocess import call
import obspy.geodetics
from mpl_toolkits.basemap import Basemap, shiftgrid, cm
from pyproj import Geod
import random
import copy
import colormaps
import pyasdf
import math
import numba

lon_diff_weight_2 = np.array([[1., 0., -1.]])/2.
lat_diff_weight_2 = lon_diff_weight_2.T
lon_diff_weight_4 = np.array([[-1., 8., 0., -8., 1.]])/12.
lat_diff_weight_4 = lon_diff_weight_4.T
lon_diff_weight_6 = np.array([[1./60., 	-3./20.,  3./4.,  0., -3./4., 3./20.,  -1./60.]])
lat_diff_weight_6 = lon_diff_weight_6.T

lon_diff2_weight_2 = np.array([[1., -2., 1.]])
lat_diff2_weight_2 = lon_diff2_weight_2.T
lon_diff2_weight_4 = np.array([[-1., 16., -30., 16., -1.]])/12.
lat_diff2_weight_4 = lon_diff2_weight_4.T
lon_diff2_weight_6 = np.array([[1./90., 	-3./20.,  3./2.,  -49./18., 3./2., -3./20.,  1./90.]])
lat_diff2_weight_6 = lon_diff2_weight_6.T

geodist = Geod(ellps='WGS84')

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""
    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

def _write_txt(fname, outlon, outlat, outZ):
    outArr  = np.append(outlon, outlat)
    outArr  = np.append(outArr, outZ)
    outArr  = outArr.reshape((3,outZ.size))
    outArr  = outArr.T
    np.savetxt(fname, outArr, fmt='%g')
    return


@numba.jit(numba.float64(numba.float64, numba.float64, numba.float64, numba.float64))
def vincenty_inverse(lat1, lon1, lat2, lon2):
    """
    Vincenty's formula (inverse method) to calculate the distance (in
    kilometers) between two points on the surface of a spheroid
    """
    # WGS 84
    a                       = 6378137  # meters
    f                       = 1 / 298.257223563
    b                       = 6356752.314245  # meters; b = (1 - f)a
    MAX_ITERATIONS          = 200
    CONVERGENCE_THRESHOLD   = 1e-12  # .000,000,000,001
    # short-circuit coincident points
    if lon1 == lon2 and lat1 == lat2:
        return 0.0

    U1      = math.atan((1 - f) * math.tan(math.radians(lat1)))
    U2      = math.atan((1 - f) * math.tan(math.radians(lat2)))
    L       = math.radians(lon2 - lon1)
    Lambda  = L

    sinU1   = math.sin(U1)
    cosU1   = math.cos(U1)
    sinU2   = math.sin(U2)
    cosU2   = math.cos(U2)

    for iteration in range(MAX_ITERATIONS):
        sinLambda       = math.sin(Lambda)
        cosLambda       = math.cos(Lambda)
        sinSigma        = math.sqrt((cosU2 * sinLambda) ** 2 +
                             (cosU1 * sinU2 - sinU1 * cosU2 * cosLambda) ** 2)
        if sinSigma == 0:
            return 0.0  # coincident points
        cosSigma        = sinU1 * sinU2 + cosU1 * cosU2 * cosLambda
        sigma           = math.atan2(sinSigma, cosSigma)
        sinAlpha        = cosU1 * cosU2 * sinLambda / sinSigma
        cosSqAlpha      = 1 - sinAlpha ** 2
        if cosSqAlpha == 0.:
            cos2SigmaM  = 0.
        C               = f / 16 * cosSqAlpha * (4 + f * (4 - 3 * cosSqAlpha))
        LambdaPrev      = Lambda
        Lambda          = L + (1 - C) * f * sinAlpha * (sigma + C * sinSigma *
                                               (cos2SigmaM + C * cosSigma *
                                                (-1 + 2 * cos2SigmaM ** 2)))
        if abs(Lambda - LambdaPrev) < CONVERGENCE_THRESHOLD:
            break  # successful convergence
    else:
        return 20000.  # failure to converge
    uSq         = cosSqAlpha * (a ** 2 - b ** 2) / (b ** 2)
    A           = 1 + uSq / 16384 * (4096 + uSq * (-768 + uSq * (320 - 175 * uSq)))
    B           = uSq / 1024 * (256 + uSq * (-128 + uSq * (74 - 47 * uSq)))
    deltaSigma  = B * sinSigma * (cos2SigmaM + B / 4 * (cosSigma *
                 (-1 + 2 * cos2SigmaM ** 2) - B / 6 * cos2SigmaM *
                 (-3 + 4 * sinSigma ** 2) * (-3 + 4 * cos2SigmaM ** 2)))
    s           = b * A * (sigma - deltaSigma)
    s           /= 1000.  # meters to kilometers
    return round(s, 6)


@numba.jit(numba.float64(numba.float64, numba.float64, numba.float64, numba.float64))
def distance(lat1, lon1, lat2, lon2, ):
    r       = 6371.009
    lat1    = math.radians(lat1)
    lon1    = math.radians(lon1)
    lat2    = math.radians(lat2)
    lon2    = math.radians(lon2)
    londelta = lon2 - lon1
    a = math.pow(math.cos(lat2) * math.sin(londelta), 2) + math.pow(math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(londelta), 2)
    b = math.sin(lat1) * math.sin(lat2) + math.cos(lat1) * math.cos(lat2) * math.cos(londelta)
    angle = math.atan2(math.sqrt(a), b)

    return angle * r

def _test_vincenty(N):
    lat1    = np.float64((np.random.random(N)-0.5)*90.)
    lat2    = np.float64((np.random.random(N)-0.5)*90.)
    lon1    = np.float64((np.random.random(N))*360.)
    lon2    = np.float64((np.random.random(N))*360.)
    
    az, baz, dist1   = geodist.inv(lon1, lat1, lon2, lat2)
    dist2   = np.zeros(N, dtype=np.float64)
    dist3   = np.zeros(N, dtype=np.float64)
    for i in range(N):
        dist3[i], az, baz                  = obspy.geodetics.gps2dist_azimuth(lat1[i], lon1[i], lat2[i], lon2[i])
        # dist2[i] = vincenty_inverse(lat1[i], lon1[i], lat2[i], lon2[i])
        dist2[i] = distance(lat1[i], lon1[i], lat2[i], lon2[i])
    return  dist1/1000., dist2, dist3/1000.

@numba.jit(numba.types.UniTuple(numba.float64[:, :], 2)(numba.float64[:, :], numba.float64[:, :], numba.float64[:], numba.float64[:], \
        numba.float64[:], numba.float64[:],   numba.float64) )
def _check_nearneighbor_station(fieldArr, reason_n, lons, lats, lonArrIn, latArrIn, cdist):
    Nlat, Nlon          = reason_n.shape
    Nin                 = lonArrIn.size
    for ilat in xrange(Nlat):
        for ilon in xrange(Nlon):
            if reason_n[ilat, ilon]==1:
                continue
            lon         = lons[ilon]
            lat         = lats[ilat]
            marker_EN   = np.zeros((2,2), dtype=np.int32)
            marker_nn   = 4
            tflag       = False
            for iv1 in xrange(Nin):
                lon2    = lonArrIn[iv1]
                lat2    = latArrIn[iv1]
                dist    = distance(lat, lon, lat2, lon2)
                if dist > cdist:
                    continue
                if lon2-lon<0:
                    marker_E    = 0
                else:
                    marker_E    = 1
                if lat2-lat<0:
                    marker_N    = 0
                else:
                    marker_N    = 1
                if marker_EN[marker_E , marker_N]==1:
                    continue
                # dist            = vincenty_inverse(lat, lon, lat2, lon2)
                # az, baz, dist   = geodist.inv(lon, lat, lon2, lat2) # loninArr/latinArr are initial points
                # dist            = dist/1000.
                if dist< cdist*2 and dist >= 1:
                    marker_nn   = marker_nn-1
                    if marker_nn==0:
                        tflag   = True
                        break
                    marker_EN[marker_E, marker_N]=1
            if not tflag:
                fieldArr[ilat, ilon]    = 0
                reason_n[ilat, ilon]    = 2
        
    return fieldArr, reason_n

@numba.jit(numba.float64[:, :](numba.float64[:, :, :], numba.float64[:, :, :], numba.float64[:, :, :], numba.float64[:, :, :], \
        numba.float64[:, :, :],   numba.float64) )
def _dist3D_check(dist3D, indexE, indexW, indexN, indexS, cdist):
    Nlat, Nlon, Ndata   = dist3D.shape
    reason_n            = np.zeros((Nlat, Nlon))
    for ilat in xrange(Nlat):
        for ilon in xrange(Nlon):
            Eflag       = False
            Wflag       = False
            Nflag       = False
            Sflag       = False
            for idata in xrange(Ndata):
                if indexE[ilat, ilon, idata] and dist3D[ilat, ilon, idata] < cdist:
                    Eflag   = True
                if indexW[ilat, ilon, idata] and dist3D[ilat, ilon, idata] < cdist:
                    Wflag   = True
                if indexN[ilat, ilon, idata] and dist3D[ilat, ilon, idata] < cdist:
                    Nflag   = True
                if indexS[ilat, ilon, idata] and dist3D[ilat, ilon, idata] < cdist:
                    Sflag   = True
            if Eflag*Wflag*Nflag*Sflag:
                reason_n[ilat, ilon]    = 1.
    return reason_n

class Field2d(object):
    """
    An object to analyze 2D spherical field data on Earth
    ===========================================================================
    ::: parameters :::
    dlon, dlat      - grid interval
    Nlon, Nlat      - grid number in longitude, latitude 
    lonArr, latArr  - arrays for grid location
    fieldtype       - field type (Tph, Tgr, Amp)
    
    ---------------------------------------------------------------------------
    Note: meshgrid's default indexing is 'xy', which means:
    lons, lats = np.meshgrid[lon, lat]
    in lons[i, j] or lats[i, j],  i->lat, j->lon
    ===========================================================================
    """
    def __init__(self, minlon, maxlon, dlon, minlat, maxlat, dlat, period, evlo=float('inf'), evla=float('inf'), fieldtype='Tph',\
                 evid='', nlat_grad=1, nlon_grad=1, nlat_lplc=2, nlon_lplc=2):
        self.Nlon               = int(round((maxlon-minlon)/dlon)+1)
        self.Nlat               = int(round((maxlat-minlat)/dlat)+1)
        self.dlon               = dlon
        self.dlat               = dlat
        self.lon                = np.arange(self.Nlon)*self.dlon+minlon
        self.lat                = np.arange(self.Nlat)*self.dlat+minlat
        self.lonArr, self.latArr= np.meshgrid(self.lon, self.lat)
        self.minlon             = minlon
        self.maxlon             = self.lon.max()
        self.minlat             = minlat
        self.maxlat             = self.lat.max()
        self._get_dlon_dlat_km()
        self.period             = period
        self.evid               = evid
        self.fieldtype          = fieldtype
        self.Zarr               = np.zeros((self.Nlat, self.Nlon), dtype=np.float64)
        self.evlo               = evlo
        self.evla               = evla
        #-----------------------------------------------------------
        # parameters indicate edge cutting for gradient/lplc arrays
        #-----------------------------------------------------------
        self.nlon_grad          = nlon_grad
        self.nlat_grad          = nlat_grad
        self.nlon_lplc          = nlon_lplc
        self.nlat_lplc          = nlat_lplc
        return
    
    def copy(self):
        return copy.deepcopy(self)
    
    def _get_dlon_dlat_km_slow(self):
        """Get longitude and latitude interval in km
        """
        self.dlon_km            = np.array([])
        self.dlat_km            = np.array([])
        for lat in self.lat:
            dist_lon, az, baz   = obspy.geodetics.gps2dist_azimuth(lat, 0., lat, self.dlon)
            dist_lat, az, baz   = obspy.geodetics.gps2dist_azimuth(lat, 0., lat+self.dlat, 0.)
            self.dlon_km        = np.append(self.dlon_km, dist_lon/1000.)
            self.dlat_km        = np.append(self.dlat_km, dist_lat/1000.)
        self.dlon_kmArr         = (np.tile(self.dlon_km, self.Nlon).reshape(self.Nlon, self.Nlat)).T
        self.dlat_kmArr         = (np.tile(self.dlat_km, self.Nlon).reshape(self.Nlon, self.Nlat)).T
        return
    
    def  _get_dlon_dlat_km(self):
        az, baz, dist_lon       = geodist.inv(np.zeros(self.lat.size), self.lat, np.ones(self.lat.size)*self.dlon, self.lat) 
        az, baz, dist_lat       = geodist.inv(np.zeros(self.lat.size), self.lat, np.zeros(self.lat.size), self.lat+self.dlat) 
        self.dlon_km            = dist_lon/1000.
        self.dlat_km            = dist_lat/1000.
        self.dlon_kmArr         = (np.tile(self.dlon_km, self.Nlon).reshape(self.Nlon, self.Nlat)).T
        self.dlat_kmArr         = (np.tile(self.dlat_km, self.Nlon).reshape(self.Nlon, self.Nlat)).T
        return
    
    def read(self, fname):
        """read field file
        """
        try:
            Inarray=np.loadtxt(fname)
            with open(fname) as f:
                inline = f.readline()
                if inline.split()[0] =='#':
                    evlostr = inline.split()[1]
                    evlastr = inline.split()[2]
                    if evlostr.split('=')[0] =='evlo':
                        self.evlo = float(evlostr.split('=')[1])
                    if evlastr.split('=')[0] =='evla':
                        self.evla = float(evlastr.split('=')[1])
        except:
            Inarray=np.load(fname)
        self.lonArrIn=Inarray[:,0]
        self.latArrIn=Inarray[:,1]
        self.ZarrIn=Inarray[:,2]
        return
    
    def read_ind(self, fname, zindex=2, dindex=None):
        """read field file
        """
        try:
            Inarray=np.loadtxt(fname)
            with open(fname) as f:
                inline = f.readline()
                if inline.split()[0] =='#':
                    evlostr = inline.split()[1]
                    evlastr = inline.split()[2]
                    if evlostr.split('=')[0] =='evlo':
                        self.evlo = float(evlostr.split('=')[1])
                    if evlastr.split('=')[0] =='evla':
                        self.evla = float(evlastr.split('=')[1])
        except:
            Inarray=np.load(fname)
        self.lonArrIn=Inarray[:,0]
        self.latArrIn=Inarray[:,1]
        self.ZarrIn=Inarray[:,zindex]*1e9
        if dindex!=None:
            darrIn=Inarray[:,dindex]
            self.ZarrIn=darrIn/Inarray[:,zindex]
        return
    
    def read_array(self, lonArr, latArr, ZarrIn):
        """read field file
        """
        self.lonArrIn   = lonArr
        self.latArrIn   = latArr
        self.ZarrIn     = ZarrIn
        return
    
    def add_noise(self, sigma=0.5):
        """Add Gaussian noise with standard deviation = sigma to the input data
        """
        for i in xrange(self.ZarrIn.size):
            self.ZarrIn[i]=self.ZarrIn[i] + random.gauss(0, sigma)
        return
    
    def load_field(self, inField):
        """Load field data from an input object
        """
        self.lonArrIn=inField.lonArr
        self.latArrIn=inField.latArr
        self.ZarrIn=inField.Zarr
        return
    
    def write(self, fname, fmt='npy'):
        """Save field file
        """
        OutArr=np.append(self.lonArr, self.latArr)
        OutArr=np.append(OutArr, self.Zarr)
        OutArr=OutArr.reshape(3, self.Nlon*self.Nlat)
        OutArr=OutArr.T
        if fmt=='npy':
            np.save(fname, OutArr)
        elif fmt=='txt':
            np.savetxt(fname, OutArr)
        else:
            raise TypeError('Wrong output format!')
        return
    
    def np2ma(self):
        """Convert all the data array to masked array according to reason_n array.
        """
        try:
            reason_n=self.reason_n
        except:
            raise AttrictError('No reason_n array!')
        self.Zarr=ma.masked_array(self.Zarr, mask=np.zeros(reason_n.shape) )
        self.Zarr.mask[reason_n!=0]=1
        try:
            self.diffaArr=ma.masked_array(self.diffaArr, mask=np.zeros(reason_n.shape) )
            self.diffaArr.mask[reason_n!=0]=1
        except:
            pass
        try:
            self.appV=ma.masked_array(self.appV, mask=np.zeros(reason_n.shape) )
            self.appV.mask[reason_n!=0]=1
        except:
            pass
        try:
            self.grad[0]=ma.masked_array(self.grad[0], mask=np.zeros(reason_n.shape) )
            self.grad[0].mask[reason_n!=0]=1
            self.grad[1]=ma.masked_array(self.grad[1], mask=np.zeros(reason_n.shape) )
            self.grad[1].mask[reason_n!=0]=1
        except:
            pass
        try:
            self.lplc=ma.masked_array(self.lplc, mask=np.zeros(reason_n.shape) )
            self.lplc.mask[reason_n!=0]=1
        except:
            print 'No Laplacian array!'
            pass
        return
    
    def ma2np(self):
        """Convert all the maksed data array to numpy array
        """
        self.Zarr=ma.getdata(self.Zarr)
        try:
            self.diffaArr=ma.getdata(self.diffaArr)
        except:
            pass
        try:
            self.appV=ma.getdata(self.appV)
        except:
            pass
        try:
            self.lplc=ma.getdata(self.lplc)
        except:
            pass
        return
    
    def cut_edge(self, nlon, nlat):
        """Cut edge
        =======================================================================================
        ::: input parameters :::
        nlon, nlon  - number of edge point in longitude/latitude to be cutted
        =======================================================================================
        """
        self.Nlon               = self.Nlon-2*nlon
        self.Nlat               = self.Nlat-2*nlat
        self.minlon             = self.minlon + nlon*self.dlon
        self.maxlon             = self.maxlon - nlon*self.dlon
        self.minlat             = self.minlat + nlat*self.dlat
        self.maxlat             = self.maxlat - nlat*self.dlat
        self.lon                = np.arange(self.Nlon)*self.dlon+self.minlon
        self.lat                = np.arange(self.Nlat)*self.dlat+self.minlat
        self.lonArr,self.latArr = np.meshgrid(self.lon, self.lat)
        self.Zarr               = self.Zarr[nlat:-nlat, nlon:-nlon]
        try:
            self.reason_n       = self.reason_n[nlat:-nlat, nlon:-nlon]
        except:
            pass
        self._get_dlon_dlat_km()
        return
    
    def interp_surface(self, workingdir, outfname, tension=0.0):
        """Interpolate input data to grid point with gmt surface command
        =======================================================================================
        ::: input parameters :::
        workingdir  - working directory
        outfname    - output file name for interpolation
        tension     - input tension for gmt surface(0.0-1.0)
        ---------------------------------------------------------------------------------------
        ::: output :::
        self.Zarr   - interpolated field data
        =======================================================================================
        """
        if not os.path.isdir(workingdir):
            os.makedirs(workingdir)
        OutArr      = np.append(self.lonArrIn, self.latArrIn)
        OutArr      = np.append(OutArr, self.ZarrIn)
        OutArr      = OutArr.reshape(3, self.lonArrIn.size)
        OutArr      = OutArr.T
        np.savetxt(workingdir+'/'+outfname, OutArr, fmt='%g')
        fnameHD     = workingdir+'/'+outfname+'.HD'
        tempGMT     = workingdir+'/'+outfname+'_GMT.sh'
        grdfile     = workingdir+'/'+outfname+'.grd'
        with open(tempGMT,'wb') as f:
            REG     = '-R'+str(self.minlon)+'/'+str(self.maxlon)+'/'+str(self.minlat)+'/'+str(self.maxlat)
            # f.writelines('gmtset MAP_FRAME_TYPE fancy \n')
            # f.writelines('surface %s -T%g -G%s -I%g %s \n' %( workingdir+'/'+outfname, tension, grdfile, self.dlon, REG ))
            # f.writelines('grd2xyz %s %s > %s \n' %( grdfile, REG, fnameHD ))
            f.writelines('gmt gmtset MAP_FRAME_TYPE fancy \n')
            f.writelines('gmt surface %s -T%g -G%s -I%g %s \n' %( workingdir+'/'+outfname, tension, grdfile, self.dlon, REG ))
            f.writelines('gmt grd2xyz %s %s > %s \n' %( grdfile, REG, fnameHD ))
        call(['bash', tempGMT])
        os.remove(grdfile)
        os.remove(tempGMT)
        inArr       = np.loadtxt(fnameHD)
        ZarrIn      = inArr[:, 2]
        self.Zarr   = (ZarrIn.reshape(self.Nlat, self.Nlon))[::-1, :]
        return
    
    def gradient(self, method='diff', edge_order=1, order=2):
        """Compute gradient of the field
        =============================================================================================================
        ::: input parameters :::
        edge_order  - edge_order : {1, 2}, optional, only has effect when method='default'
                        Gradient is calculated using Nth order accurate differences at the boundaries
        method      - method: 'default' : use numpy.gradient 'convolve': use convolution
        order       - order of finite difference scheme, only has effect when method='convolve'
        ::: note :::
        gradient arrays are of shape Nlat-1, Nlon-1
        =============================================================================================================
        """
        Zarr            = self.Zarr
        if method=='diff':
            # self.dlat_kmArr : dx here in numpy gradient since Zarr is Z[ilat, ilon]
            self.grad   = np.gradient( self.Zarr, self.dlat_kmArr, self.dlon_kmArr, edge_order=edge_order)
            self.grad[0]= self.grad[0][self.nlat_grad:-self.nlat_grad, self.nlon_grad:-self.nlon_grad]
            self.grad[1]= self.grad[1][self.nlat_grad:-self.nlat_grad, self.nlon_grad:-self.nlon_grad]
        elif method == 'convolve':
            dlat_km     = self.dlat_kmArr
            dlon_km     = self.dlon_kmArr
            if order==2:
                diff_lon= convolve(Zarr, lon_diff_weight_2)/dlon_km
                diff_lat= convolve(Zarr, lat_diff_weight_2)/dlat_km
            elif order==4:
                diff_lon= convolve(Zarr, lon_diff_weight_4)/dlon_km
                diff_lat= convolve(Zarr, lat_diff_weight_4)/dlat_km
            elif order==6:
                diff_lon= convolve(Zarr, lon_diff_weight_6)/dlon_km
                diff_lat= convolve(Zarr, lat_diff_weight_6)/dlat_km
            self.grad   = []
            self.grad.append(diff_lat[self.nlat_grad:-self.nlat_grad, self.nlon_grad:-self.nlon_grad])
            self.grad.append(diff_lon[self.nlat_grad:-self.nlat_grad, self.nlon_grad:-self.nlon_grad])
        self.proAngle   = np.arctan2(self.grad[0], self.grad[1])/np.pi*180.
        return
    
    def Laplacian(self, method='green', order=4, verbose=False):
        """Compute Laplacian of the field
        =============================================================================================================
        ::: input parameters :::
        method      - method: 'diff'    : use central finite difference scheme, similar to convolve with order =2
                              'convolve': use convolution
                              'green'   : use Green's theorem( 2D Gauss's theorem )
        order       - order of finite difference scheme, only has effect when method='convolve'
        =============================================================================================================
        """
        Zarr                = self.Zarr
        if method == 'diff':
            dlat_km         = self.dlat_kmArr[1:-1, 1:-1]
            dlon_km         = self.dlon_kmArr[1:-1, 1:-1]
            Zarr_latp       = Zarr[2:, 1:-1]
            Zarr_latn       = Zarr[:-2, 1:-1]
            Zarr_lonp       = Zarr[1:-1, 2:]
            Zarr_lonn       = Zarr[1:-1, :-2]
            Zarr            = Zarr[1:-1, 1:-1]
            lplc            = (Zarr_latp+Zarr_latn-2*Zarr) / (dlat_km**2) + (Zarr_lonp+Zarr_lonn-2*Zarr) / (dlon_km**2)
            dnlat           = self.nlat_lplc - 1
            dnlon           = self.nlon_lplc - 1
            if dnlat == 0 and dnlon == 0:
                self.lplc       = lplc
            elif dnlat == 0 and dnlon != 0:
                self.lplc       = lplc[:, dnlon:-dnlon]
            elif dnlat != 0 and dnlon == 0:
                self.lplc       = lplc[dnlat:-dnlat, :]
            else:
                self.lplc       = lplc[dnlat:-dnlat, dnlon:-dnlon]
        elif method == 'convolve':
            dlat_km         = self.dlat_kmArr
            dlon_km         = self.dlon_kmArr
            if order==2:
                diff2_lon   = convolve(Zarr, lon_diff2_weight_2)/dlon_km/dlon_km
                diff2_lat   = convolve(Zarr, lat_diff2_weight_2)/dlat_km/dlat_km
            elif order==4:
                diff2_lon   = convolve(Zarr, lon_diff2_weight_4)/dlon_km/dlon_km
                diff2_lat   = convolve(Zarr, lat_diff2_weight_4)/dlat_km/dlat_km
            elif order==6:
                diff2_lon   = convolve(Zarr, lon_diff2_weight_6)/dlon_km/dlon_km
                diff2_lat   = convolve(Zarr, lat_diff2_weight_6)/dlat_km/dlat_km
            self.lplc       = diff2_lon+diff2_lat
            self.lplc       = self.lplc[self.nlat_lplc:-self.nlat_lplc, self.nlon_lplc:-self.nlon_lplc]
        elif method=='green':
            #---------------
            # gradient arrays
            #---------------
            try:
                grad_y          = self.grad[0]
                grad_x          = self.grad[1]
            except:
                self.gradient('diff')
                grad_y          = self.grad[0]
                grad_x          = self.grad[1]
            grad_xp             = grad_x[1:-1, 2:]
            grad_xn             = grad_x[1:-1, :-2]
            grad_yp             = grad_y[2:, 1:-1]
            grad_yn             = grad_y[:-2, 1:-1]
            dlat_km             = self.dlat_kmArr[self.nlat_grad+1:-self.nlat_grad-1, self.nlon_grad+1:-self.nlon_grad-1]
            dlon_km             = self.dlon_kmArr[self.nlat_grad+1:-self.nlat_grad-1, self.nlon_grad+1:-self.nlon_grad-1]
            #------------------
            # Green's theorem
            #------------------
            loopsum             = (grad_xp - grad_xn)*dlat_km + (grad_yp - grad_yn)*dlon_km
            area                = dlat_km*dlon_km
            lplc                = loopsum/area
            #-----------------------------------------------
            # cut edges according to nlat_lplc, nlon_lplc
            #-----------------------------------------------
            dnlat               = self.nlat_lplc - self.nlat_grad - 1
            if dnlat < 0:
                self.nlat_lplc  = self.nlat_grad + 1
            dnlon               = self.nlon_lplc - self.nlon_grad - 1
            if dnlon < 0:
                self.nlon_lplc  = self.nlon_grad + 1
            if dnlat == 0 and dnlon == 0:
                self.lplc       = lplc
            elif dnlat == 0 and dnlon != 0:
                self.lplc       = lplc[:, dnlon:-dnlon]
            elif dnlat != 0 and dnlon == 0:
                self.lplc       = lplc[dnlat:-dnlat, :]
            else:
                self.lplc       = lplc[dnlat:-dnlat, dnlon:-dnlon]
        if verbose:
            print 'max lplc:',self.lplc.max(), 'min lplc:',self.lplc.min()
        return
    
    def get_appV(self):
        """Get the apparent velocity from gradient
        """
        slowness                = np.sqrt ( self.grad[0] ** 2 + self.grad[1] ** 2)
        slowness[slowness==0]   = 0.3
        self.appV               = 1./slowness
        return
      
    def check_curvature(self, workingdir, outpfx='', threshold=0.005):
        """
        Check and discard data points with large curvatures.
        Points at boundaries will be discarded.
        Two interpolation schemes with different tension (0, 0.2) will be applied to the quality controlled field data file. 
        =====================================================================================================================
        ::: input parameters :::
        workingdir  - working directory
        threshold   - threshold value for Laplacian
        ---------------------------------------------------------------------------------------------------------------------
        ::: output :::
        workingdir/outpfx+fieldtype_per_v1.lst         - output field file with data point passing curvature checking
        workingdir/outpfx+fieldtype_per_v1.lst.HD      - interpolated travel time file 
        workingdir/outpfx+fieldtype_per_v1.lst.HD_0.2  - interpolated travel time file with tension=0.2
        =====================================================================================================================
        """
        # Compute Laplacian
        self.Laplacian(method='green')
        tfield      = self.copy()
        tfield.cut_edge(nlon=self.nlon_lplc, nlat=self.nlat_lplc)
        #--------------------
        # quality control
        #--------------------
        LonLst      = tfield.lonArr.reshape(tfield.lonArr.size)
        LatLst      = tfield.latArr.reshape(tfield.latArr.size)
        TLst        = tfield.Zarr.reshape(tfield.Zarr.size)
        lplc        = self.lplc.reshape(self.lplc.size)
        index       = np.where((lplc>-threshold)*(lplc<threshold))[0]
        LonLst      = LonLst[index]
        LatLst      = LatLst[index]
        TLst        = TLst[index]
        # output to txt file
        outfname    = workingdir+'/'+outpfx+self.fieldtype+'_'+str(self.period)+'_v1.lst'
        TfnameHD    = outfname+'.HD'
        _write_txt(fname=outfname, outlon=LonLst, outlat=LatLst, outZ=TLst)
        # interpolate with gmt surface
        tempGMT     = workingdir+'/'+outpfx+self.fieldtype+'_'+str(self.period)+'_v1_GMT.sh'
        grdfile     = workingdir+'/'+outpfx+self.fieldtype+'_'+str(self.period)+'_v1.grd'
        with open(tempGMT,'wb') as f:
            REG     = '-R'+str(self.minlon)+'/'+str(self.maxlon)+'/'+str(self.minlat)+'/'+str(self.maxlat)
            # f.writelines('gmtset MAP_FRAME_TYPE fancy \n')
            # f.writelines('surface %s -T0.0 -G%s -I%g %s \n' %( outfname, grdfile, self.dlon, REG ))
            # f.writelines('grd2xyz %s %s > %s \n' %( grdfile, REG, TfnameHD ))
            # f.writelines('surface %s -T0.2 -G%s -I%g %s \n' %( outfname, grdfile+'.T0.2', self.dlon, REG ))
            # f.writelines('grd2xyz %s %s > %s \n' %( grdfile+'.T0.2', REG, TfnameHD+'_0.2' ))
            
            f.writelines('gmt gmtset MAP_FRAME_TYPE fancy \n')
            f.writelines('gmt surface %s -T0.0 -G%s -I%g %s \n' %( outfname, grdfile, self.dlon, REG ))
            f.writelines('gmt grd2xyz %s %s > %s \n' %( grdfile, REG, TfnameHD ))
            f.writelines('gmt surface %s -T0.2 -G%s -I%g %s \n' %( outfname, grdfile+'.T0.2', self.dlon, REG ))
            f.writelines('gmt grd2xyz %s %s > %s \n' %( grdfile+'.T0.2', REG, TfnameHD+'_0.2' ))
        call(['bash', tempGMT])
        os.remove(grdfile+'.T0.2')
        os.remove(grdfile)
        os.remove(tempGMT)
        return 
        
    def eikonal_operator(self, workingdir, inpfx='', nearneighbor=True, cdist=None, lplcthresh=0.002, lplcnearneighbor=True):
        """
        Generate slowness maps from travel time maps using eikonal equation
        Two interpolated travel time file with different tension will be used for quality control.
        =====================================================================================================================
        ::: input parameters :::
        workingdir      - working directory
        evlo, evla      - event location
        nearneighbor    - do near neighbor quality control or not
        cdist           - distance for quality control, default is 12*period
        ::: output format :::
        outdir/slow_azi_stacode.pflag.txt.HD.2.v2 - Slowness map
        ---------------------------------------------------------------------------------------------------------------------
        Note: edge has been cutting twice, one in check_curvature 
        =====================================================================================================================
        """
        if cdist is None:
            cdist   = 12.*self.period
        evlo        = self.evlo
        evla        = self.evla
        # Read data,
        # v1: data that pass check_curvature criterion
        # v1HD and v1HD02: interpolated v1 data with tension = 0. and 0.2
        fnamev1     = workingdir+'/'+inpfx+self.fieldtype+'_'+str(self.period)+'_v1.lst'
        fnamev1HD   = fnamev1+'.HD'
        fnamev1HD02 = fnamev1HD+'_0.2'
        InarrayV1   = np.loadtxt(fnamev1)
        loninV1     = InarrayV1[:,0]
        latinV1     = InarrayV1[:,1]
        fieldin     = InarrayV1[:,2]
        Inv1HD      = np.loadtxt(fnamev1HD)
        lonv1HD     = Inv1HD[:,0]
        latv1HD     = Inv1HD[:,1]
        fieldv1HD   = Inv1HD[:,2]
        Inv1HD02    = np.loadtxt(fnamev1HD02)
        lonv1HD02   = Inv1HD02[:,0]
        latv1HD02   = Inv1HD02[:,1]
        fieldv1HD02 = Inv1HD02[:,2]
        # Set field value to be zero if there is large difference between v1HD and v1HD02
        diffArr     = fieldv1HD-fieldv1HD02
        # fieldArr    = fieldv1HD*((diffArr<1.)*(diffArr>-1.))
        # old
        fieldArr    = fieldv1HD*((diffArr<2.)*(diffArr>-2.)) 
        fieldArr    = (fieldArr.reshape(self.Nlat, self.Nlon))[::-1, :]
        # reason_n 
        #   0: accepted point
        #   1: data point the has large difference between v1HD and v1HD02
        #   2: data point that does not have near neighbor points at all E/W/N/S directions
        #   3: slowness is too large/small
        #   4: near a zero field data point
        #   5: epicentral distance is too small
        reason_n    = np.ones(fieldArr.size, dtype=np.int32)
        # reason_n1   = np.int32(reason_n*(diffArr>1.))
        # reason_n2   = np.int32(reason_n*(diffArr<-1.))
        # old
        reason_n1   = np.int32(reason_n*(diffArr>2.))
        reason_n2   = np.int32(reason_n*(diffArr<-2.))
        
        reason_n    = reason_n1+reason_n2
        reason_n    = (reason_n.reshape(self.Nlat, self.Nlon))[::-1,:]
        #-------------------------------------------------------------------------------------------------------
        # check each data point if there are close-by four stations located at E/W/N/S directions respectively
        #-------------------------------------------------------------------------------------------------------
        if nearneighbor:
            for ilat in range(self.Nlat):
                for ilon in range(self.Nlon):
                    if reason_n[ilat, ilon]==1:
                        continue
                    lon         = self.lon[ilon]
                    lat         = self.lat[ilat]
                    dlon_km     = self.dlon_km[ilat]
                    dlat_km     = self.dlat_km[ilat]
                    difflon     = abs(self.lonArrIn-lon)/self.dlon*dlon_km
                    difflat     = abs(self.latArrIn-lat)/self.dlat*dlat_km
                    index       = np.where((difflon<cdist)*(difflat<cdist))[0]
                    marker_EN   = np.zeros((2,2), dtype=np.bool)
                    marker_nn   = 4
                    tflag       = False
                    for iv1 in index:
                        lon2    = self.lonArrIn[iv1]
                        lat2    = self.latArrIn[iv1]
                        if lon2-lon<0:
                            marker_E    = 0
                        else:
                            marker_E    = 1
                        if lat2-lat<0:
                            marker_N    = 0
                        else:
                            marker_N    = 1
                        if marker_EN[marker_E , marker_N]:
                            continue
                        az, baz, dist   = geodist.inv(lon, lat, lon2, lat2) # loninArr/latinArr are initial points
                        dist            = dist/1000.
                        if dist< cdist*2 and dist >= 1:
                            marker_nn   = marker_nn-1
                            if marker_nn==0:
                                tflag   = True
                                break
                            marker_EN[marker_E, marker_N]   = True
                    if not tflag:
                        fieldArr[ilat, ilon]    = 0
                        reason_n[ilat, ilon]    = 2
        # Start to Compute Gradient
        tfield                      = self.copy()
        tfield.Zarr                 = fieldArr
        tfield.gradient('diff')
        # if one field point has zero value, reason_n for four near neighbor points will all be set to 4
        tempZarr                    = tfield.Zarr[self.nlat_grad:-self.nlat_grad, self.nlon_grad:-self.nlon_grad]
        index0                      = np.where(tempZarr==0.)
        ilatArr                     = index0[0] + 1
        ilonArr                     = index0[1] + 1
        reason_n[ilatArr+1, ilonArr]= 4
        reason_n[ilatArr-1, ilonArr]= 4
        reason_n[ilatArr, ilonArr+1]= 4
        reason_n[ilatArr, ilonArr-1]= 4
        # reduce size of reason_n to be the same shape as gradient arrays
        reason_n                    = reason_n[self.nlat_grad:-self.nlat_grad, self.nlon_grad:-self.nlon_grad]
        # if slowness is too large/small, reason_n will be set to 3
        slowness                    = np.sqrt(tfield.grad[0]**2 + tfield.grad[1]**2)
        if self.fieldtype=='Tph' or self.fieldtype=='Tgr':
            reason_n[(slowness>0.5)*(reason_n==0)]  = 3
            reason_n[(slowness<0.2)*(reason_n==0)]  = 3
        #-------------------------------------
        # computing propagation deflection
        #-------------------------------------
        indexvalid                              = np.where(reason_n==0)
        diffaArr                                = np.zeros(reason_n.shape, dtype = np.float64)
        latinArr                                = self.lat[indexvalid[0] + 1]
        loninArr                                = self.lon[indexvalid[1] + 1]
        evloArr                                 = np.ones(loninArr.size, dtype=np.float64)*evlo
        evlaArr                                 = np.ones(latinArr.size, dtype=np.float64)*evla
        az, baz, distevent                      = geodist.inv(loninArr, latinArr, evloArr, evlaArr) # loninArr/latinArr are initial points
        distevent                               = distevent/1000.
        az                                      = az + 180.
        az                                      = 90.-az
        baz                                     = 90.-baz
        az[az>180.]                             = az[az>180.] - 360.
        az[az<-180.]                            = az[az<-180.] + 360.
        baz[baz>180.]                           = baz[baz>180.] - 360.
        baz[baz<-180.]                          = baz[baz<-180.] + 360.
        # az azimuth receiver -> source 
        diffaArr[indexvalid[0], indexvalid[1]]  = tfield.proAngle[indexvalid[0], indexvalid[1]] - az
        self.az                                 = np.zeros(self.proAngle.shape, dtype=np.float64)
        self.az[indexvalid[0], indexvalid[1]]   = az
        self.baz                                = np.zeros(self.proAngle.shape, dtype=np.float64)
        self.baz[indexvalid[0], indexvalid[1]]  = baz
        # if epicentral distance is too small, reason_n will be set to 5, and diffaArr will be 0.
        tempArr                                 = diffaArr[indexvalid[0], indexvalid[1]]
        tempArr[distevent<cdist+50.]            = 0.
        diffaArr[indexvalid[0], indexvalid[1]]  = tempArr
        diffaArr[diffaArr>180.]                 = diffaArr[diffaArr>180.]-360.
        diffaArr[diffaArr<-180.]                = diffaArr[diffaArr<-180.]+360.
        tempArr                                 = reason_n[indexvalid[0], indexvalid[1]]
        tempArr[distevent<cdist+50.]            = 5
        reason_n[indexvalid[0], indexvalid[1]]  = tempArr
        # # final check of curvature, discard grid points with large curvature
        self.Laplacian(method='green')
        dnlat                                   = self.nlat_lplc - self.nlat_grad
        dnlon                                   = self.nlon_lplc - self.nlon_grad
        tempind                                 = (self.lplc > lplcthresh) + (self.lplc < -lplcthresh)
        if dnlat == 0 and dnlon == 0:
            reason_n[tempind]                   = 6
        elif dnlat == 0 and dnlon != 0:
            (reason_n[:, dnlon:-dnlon])[tempind]= 6
        elif dnlat != 0 and dnlon == 0:
            (reason_n[dnlat:-dnlat, :])[tempind]= 6
        else:
            (reason_n[dnlat:-dnlat, dnlon:-dnlon])[tempind]\
                                                = 6
        # # near neighbor discard for large curvature
        if lplcnearneighbor:
            indexlplc                               = np.where(reason_n==6.)
            ilatArr                                 = indexlplc[0] 
            ilonArr                                 = indexlplc[1]
            reason_n_temp                           = np.zeros(self.lonArr.shape)
            reason_n_temp[self.nlat_grad:-self.nlat_grad, self.nlon_grad:-self.nlon_grad] \
                                                    = reason_n.copy()
            reason_n_temp[ilatArr+1, ilonArr]       = 6
            reason_n_temp[ilatArr-1, ilonArr]       = 6
            reason_n_temp[ilatArr, ilonArr+1]       = 6
            reason_n_temp[ilatArr, ilonArr-1]       = 6
            reason_n                                = reason_n_temp[self.nlat_grad:-self.nlat_grad, self.nlon_grad:-self.nlon_grad]
        # store final data
        self.diffaArr                           = diffaArr
        self.grad                               = tfield.grad
        self.get_appV()
        self.reason_n                           = reason_n
        self.mask                               = np.ones((self.Nlat, self.Nlon), dtype=np.bool)
        tempmask                                = reason_n != 0
        self.mask[self.nlat_grad:-self.nlat_grad, self.nlon_grad:-self.nlon_grad]\
                                                = tempmask
        return
    
    
    def eikonal_operator_new(self, workingdir, inpfx='', nearneighbor=True, cdist=None, lplcthresh=0.005):
        """
        Generate slowness maps from travel time maps using eikonal equation
        Two interpolated travel time file with different tension will be used for quality control.
        =====================================================================================================================
        ::: input parameters :::
        workingdir      - working directory
        evlo, evla      - event location
        nearneighbor    - do near neighbor quality control or not
        cdist           - distance for quality control, default is 12*period
        ::: output format :::
        outdir/slow_azi_stacode.pflag.txt.HD.2.v2 - Slowness map
        ---------------------------------------------------------------------------------------------------------------------
        Note: edge has been cutting twice, one in check_curvature 
        =====================================================================================================================
        """
        if cdist is None:
            cdist   = 12.*self.period
        evlo        = self.evlo
        evla        = self.evla
        # Read data,
        # v1: data that pass check_curvature criterion
        # v1HD and v1HD02: interpolated v1 data with tension = 0. and 0.2
        fnamev1     = workingdir+'/'+inpfx+self.fieldtype+'_'+str(self.period)+'_v1.lst'
        fnamev1HD   = fnamev1+'.HD'
        fnamev1HD02 = fnamev1HD+'_0.2'
        InarrayV1   = np.loadtxt(fnamev1)
        loninV1     = InarrayV1[:,0]
        latinV1     = InarrayV1[:,1]
        fieldin     = InarrayV1[:,2]
        Inv1HD      = np.loadtxt(fnamev1HD)
        lonv1HD     = Inv1HD[:,0]
        latv1HD     = Inv1HD[:,1]
        fieldv1HD   = Inv1HD[:,2]
        Inv1HD02    = np.loadtxt(fnamev1HD02)
        lonv1HD02   = Inv1HD02[:,0]
        latv1HD02   = Inv1HD02[:,1]
        fieldv1HD02 = Inv1HD02[:,2]
        # Set field value to be zero if there is large difference between v1HD and v1HD02
        diffArr     = fieldv1HD-fieldv1HD02
        # fieldArr    = fieldv1HD*((diffArr<1.)*(diffArr>-1.))
        # old
        fieldArr    = fieldv1HD*((diffArr<2.)*(diffArr>-2.)) 
        fieldArr    = (fieldArr.reshape(self.Nlat, self.Nlon))[::-1, :]
        # reason_n 
        #   0: accepted point
        #   1: data point the has large difference between v1HD and v1HD02
        #   2: data point that does not have near neighbor points at all E/W/N/S directions
        #   3: slowness is too large/small
        #   4: near a zero field data point
        #   5: epicentral distance is too small
        reason_n    = np.ones(fieldArr.size, dtype=np.int32)
        # reason_n1   = np.int32(reason_n*(diffArr>1.))
        # reason_n2   = np.int32(reason_n*(diffArr<-1.))
        # old
        reason_n1   = np.int32(reason_n*(diffArr>2.))
        reason_n2   = np.int32(reason_n*(diffArr<-2.))
        
        reason_n    = reason_n1+reason_n2
        reason_n    = (reason_n.reshape(self.Nlat, self.Nlon))[::-1,:]
        #-------------------------------------------------------------------------------------------------------
        # check each data point if there are close-by four stations located at E/W/N/S directions respectively
        #-------------------------------------------------------------------------------------------------------
        if nearneighbor:
            lon3D               = np.broadcast_to(self.lonArrIn, (self.Nlat, self.Nlon, self.lonArrIn.size))
            lat3D               = np.broadcast_to(self.latArrIn, (self.Nlat, self.Nlon, self.latArrIn.size))
            lon3Dgd             = np.swapaxes(np.broadcast_to(self.lon, (self.Nlat, self.lonArrIn.size, self.Nlon)), 1, 2)
            lat3Dgd             = np.swapaxes(np.broadcast_to(self.lat, (self.latArrIn.size, self.Nlon, self.Nlat)), 0, 2)
            size                = lon3D.size
            azALL, bazALL, distALL  \
                                = geodist.inv(lon3D.reshape(size), lat3D.reshape(size), lon3Dgd.reshape(size), lat3Dgd.reshape(size)) # loninArr/latinArr are initial points
            dist3D              = distALL.reshape((self.Nlat, self.Nlon, self.lonArrIn.size))
            
            difflon3D           = lon3D - lon3Dgd
            indexE              = (difflon3D>0.)*1
            indexW              = (difflon3D<0.)*1
            
            difflat3D           = lat3D - lat3Dgd
            indexN              = (difflat3D>0.)*1
            indexS              = (difflat3D<0.)*1
            treason_n           = _dist3D_check(dist3D, indexE, indexW, indexN, indexS, cdist)            
            reason_n[treason_n!=1.]   = 2
            fieldArr[treason_n!=1.]   = 0
            # 
            # 
            # 
            # 
            # 
            # for ilat in range(self.Nlat):
            #     for ilon in range(self.Nlon):
            #         if reason_n[ilat, ilon]==1:
            #             continue
            #         lon         = self.lon[ilon]
            #         lat         = self.lat[ilat]
            #         dlon_km     = self.dlon_km[ilat]
            #         dlat_km     = self.dlat_km[ilat]
            #         difflon     = abs(self.lonArrIn-lon)/self.dlon*dlon_km
            #         difflat     = abs(self.latArrIn-lat)/self.dlat*dlat_km
            #         index       = np.where((difflon<cdist)*(difflat<cdist))[0]
            #         marker_EN   = np.zeros((2,2), dtype=np.bool)
            #         marker_nn   = 4
            #         tflag       = False
            #         for iv1 in index:
            #             lon2    = self.lonArrIn[iv1]
            #             lat2    = self.latArrIn[iv1]
            #             if lon2-lon<0:
            #                 marker_E    = 0
            #             else:
            #                 marker_E    = 1
            #             if lat2-lat<0:
            #                 marker_N    = 0
            #             else:
            #                 marker_N    = 1
            #             if marker_EN[marker_E , marker_N]:
            #                 continue
            #             az, baz, dist   = geodist.inv(lon, lat, lon2, lat2) # loninArr/latinArr are initial points
            #             dist            = dist/1000.
            #             if dist< cdist*2 and dist >= 1:
            #                 marker_nn   = marker_nn-1
            #                 if marker_nn==0:
            #                     tflag   = True
            #                     break
            #                 marker_EN[marker_E, marker_N]   = True
            #         if not tflag:
            #             fieldArr[ilat, ilon]    = 0
            #             reason_n[ilat, ilon]    = 2
                        
            
        # Start to Compute Gradient
        tfield                      = self.copy()
        tfield.Zarr                 = fieldArr
        tfield.gradient('diff')
        # if one field point has zero value, reason_n for four near neighbor points will all be set to 4
        tempZarr                    = tfield.Zarr[self.nlat_grad:-self.nlat_grad, self.nlon_grad:-self.nlon_grad]
        index0                      = np.where(tempZarr==0.)
        ilatArr                     = index0[0] + 1
        ilonArr                     = index0[1] + 1
        reason_n[ilatArr+1, ilonArr]= 4
        reason_n[ilatArr-1, ilonArr]= 4
        reason_n[ilatArr, ilonArr+1]= 4
        reason_n[ilatArr, ilonArr-1]= 4
        # reduce size of reason_n to be the same shape as gradient arrays
        reason_n                    = reason_n[self.nlat_grad:-self.nlat_grad, self.nlon_grad:-self.nlon_grad]
        # if slowness is too large/small, reason_n will be set to 3
        slowness                    = np.sqrt(tfield.grad[0]**2 + tfield.grad[1]**2)
        if self.fieldtype=='Tph' or self.fieldtype=='Tgr':
            reason_n[(slowness>0.5)*(reason_n==0)]  = 3
            reason_n[(slowness<0.2)*(reason_n==0)]  = 3
        #-------------------------------------
        # computing propagation deflection
        #-------------------------------------
        indexvalid                              = np.where(reason_n==0)
        diffaArr                                = np.zeros(reason_n.shape, dtype = np.float64)
        latinArr                                = self.lat[indexvalid[0] + 1]
        loninArr                                = self.lon[indexvalid[1] + 1]
        evloArr                                 = np.ones(loninArr.size, dtype=np.float64)*evlo
        evlaArr                                 = np.ones(latinArr.size, dtype=np.float64)*evla
        az, baz, distevent                      = geodist.inv(loninArr, latinArr, evloArr, evlaArr) # loninArr/latinArr are initial points
        distevent                               = distevent/1000.
        az                                      = az + 180.
        az                                      = 90.-az
        baz                                     = 90.-baz
        az[az>180.]                             = az[az>180.] - 360.
        az[az<-180.]                            = az[az<-180.] + 360.
        baz[baz>180.]                           = baz[baz>180.] - 360.
        baz[baz<-180.]                          = baz[baz<-180.] + 360.
        # az azimuth receiver -> source 
        diffaArr[indexvalid[0], indexvalid[1]]  = tfield.proAngle[indexvalid[0], indexvalid[1]] - az
        self.az                                 = np.zeros(self.proAngle.shape, dtype=np.float64)
        self.az[indexvalid[0], indexvalid[1]]   = az
        self.baz                                = np.zeros(self.proAngle.shape, dtype=np.float64)
        self.baz[indexvalid[0], indexvalid[1]]  = baz
        # if epicentral distance is too small, reason_n will be set to 5, and diffaArr will be 0.
        tempArr                                 = diffaArr[indexvalid[0], indexvalid[1]]
        tempArr[distevent<cdist+50.]            = 0.
        diffaArr[indexvalid[0], indexvalid[1]]  = tempArr
        diffaArr[diffaArr>180.]                 = diffaArr[diffaArr>180.]-360.
        diffaArr[diffaArr<-180.]                = diffaArr[diffaArr<-180.]+360.
        tempArr                                 = reason_n[indexvalid[0], indexvalid[1]]
        tempArr[distevent<cdist+50.]            = 5
        reason_n[indexvalid[0], indexvalid[1]]  = tempArr
        # final check of curvature, discard grid points with large curvature
        self.Laplacian(method='green')
        dnlat                                   = self.nlat_lplc - self.nlat_grad
        dnlon                                   = self.nlon_lplc - self.nlon_grad
        tempind                                 = (self.lplc > lplcthresh) + (self.lplc < -lplcthresh)
        if dnlat == 0 and dnlon == 0:
            reason_n[tempind]                   = 6
        elif dnlat == 0 and dnlon != 0:
            (reason_n[:, dnlon:-dnlon])[tempind]= 6
        elif dnlat != 0 and dnlon == 0:
            (reason_n[dnlat:-dnlat, :])[tempind]= 6
        else:
            (reason_n[dnlat:-dnlat, dnlon:-dnlon])[tempind]\
                                                = 6
        # # near neighbor discard for large curvature
        # indexlplc                               = np.where(reason_n==6.)
        # ilatArr                                 = indexlplc[0] 
        # ilonArr                                 = indexlplc[1]
        # reason_n_temp                           = np.zeros(self.lonArr.shape)
        # reason_n_temp[self.nlat_grad:-self.nlat_grad, self.nlon_grad:-self.nlon_grad] \
        #                                         = reason_n.copy()
        # reason_n_temp[ilatArr+1, ilonArr]       = 6
        # reason_n_temp[ilatArr-1, ilonArr]       = 6
        # reason_n_temp[ilatArr, ilonArr+1]       = 6
        # reason_n_temp[ilatArr, ilonArr-1]       = 6
        # reason_n                                = reason_n_temp[self.nlat_grad:-self.nlat_grad, self.nlon_grad:-self.nlon_grad]
        # store final data
        self.diffaArr                           = diffaArr
        self.grad                               = tfield.grad
        self.get_appV()
        self.reason_n                           = reason_n
        self.mask                               = np.ones((self.Nlat, self.Nlon), dtype=np.bool)
        tempmask                                = reason_n != 0
        self.mask[self.nlat_grad:-self.nlat_grad, self.nlon_grad:-self.nlon_grad]\
                                                = tempmask
        return
    
    
    def get_lplc_amp(self):
        if self.fieldtype!='Amp': raise ValueError('Not amplitude field!')
        w=2*np.pi/self.period
        self.lplc_amp=np.zeros(self.Zarr.shape)
        self.lplc_amp[self.Zarr!=0]=self.lplc[self.Zarr!=0]/self.Zarr[self.Zarr!=0]/w**2
        return
    
    def write_binary(self, outfname, amplplc=False):
        if amplplc:
            np.savez( outfname, self.appV, self.reason_n, self.proAngle, self.az, self.baz, self.Zarr, self.lplc_amp, self.corV )
        else:
            np.savez( outfname, self.appV, self.reason_n, self.proAngle, self.az, self.baz, self.Zarr )
        return

    def _get_basemap(self, projection='lambert', geopolygons=None, resolution='i'):
        """Plot data with contour
        """
        # fig=plt.figure(num=None, figsize=(12, 12), dpi=80, facecolor='w', edgecolor='k')
        lat_centre  = (self.maxlat+self.minlat)/2.0
        lon_centre  = (self.maxlon+self.minlon)/2.0
        if projection=='merc':
            m       = Basemap(projection='merc', llcrnrlat=self.minlat-5., urcrnrlat=self.maxlat+5., llcrnrlon=self.minlon-5.,
                        urcrnrlon=self.maxlon+5., lat_ts=20, resolution=resolution)
            m.drawparallels(np.arange(-80.0,80.0,5.0), labels=[1,0,0,1])
            m.drawmeridians(np.arange(-170.0,170.0,5.0), labels=[1,0,0,1])
            m.drawstates(color='g', linewidth=2.)
        elif projection=='global':
            m       = Basemap(projection='ortho',lon_0=lon_centre, lat_0=lat_centre, resolution=resolution)
            m.drawparallels(np.arange(-80.0,80.0,10.0), labels=[1,0,0,1])
            m.drawmeridians(np.arange(-170.0,170.0,10.0), labels=[1,0,0,1])
        
        elif projection=='regional_ortho':
            m1      = Basemap(projection='ortho', lon_0=self.minlon, lat_0=self.minlat, resolution='l')
            m       = Basemap(projection='ortho', lon_0=self.minlon, lat_0=self.minlat, resolution=resolution,\
                        llcrnrx=0., llcrnry=0., urcrnrx=m1.urcrnrx/mapfactor, urcrnry=m1.urcrnry/3.5)
            m.drawparallels(np.arange(-80.0,80.0,10.0), labels=[1,0,0,0],  linewidth=2,  fontsize=20)
            m.drawmeridians(np.arange(-170.0,170.0,10.0),  linewidth=2)
        elif projection=='lambert':
            distEW, az, baz = obspy.geodetics.gps2dist_azimuth(self.minlat, self.minlon,
                                self.minlat, self.maxlon) # distance is in m
            distNS, az, baz = obspy.geodetics.gps2dist_azimuth(self.minlat, self.minlon,
                                self.maxlat+2., self.minlon) # distance is in m
            m               = Basemap(width=distEW, height=distNS, rsphere=(6378137.00,6356752.3142), resolution='l', projection='lcc',\
                                lat_1=self.minlat, lat_2=self.maxlat, lon_0=lon_centre, lat_0=lat_centre+1)
            m.drawparallels(np.arange(-80.0,80.0,10.0), linewidth=1, dashes=[2,2], labels=[1,1,0,0], fontsize=15)
            m.drawmeridians(np.arange(-170.0,170.0,10.0), linewidth=1, dashes=[2,2], labels=[0,0,1,0], fontsize=15)
        m.drawcoastlines(linewidth=1.0)
        m.drawcountries(linewidth=1.)
        m.fillcontinents(lake_color='#99ffff',zorder=0.2)
        m.drawmapboundary(fill_color="white")
        try:
            geopolygons.PlotPolygon(inbasemap=m)
        except:
            pass
        return m
    
    def plot(self, datatype, projection='lambert', cmap='cv', contour=False, geopolygons=None, showfig=True, vmin=None, vmax=None, stations=False, event=False):
        """Plot data with contour
        """
        m       = self._get_basemap(projection=projection, geopolygons=geopolygons)
        x, y    = m(self.lonArr, self.latArr)
        if event:
            try:
                evx, evy    = m(self.evlo, self.evla)
                m.plot(evx, evy, 'yo', markersize=10)
            except:
                pass
        if stations:
            try:
                stx, sty    = m(self.lonArrIn, self.latArrIn)
                m.plot(stx, sty, 'y^', markersize=6)
            except:
                pass
        try:
            stx, sty        = m(self.stalons, self.stalats)
            m.plot(stx, sty, 'b^', markersize=6)
        except:
            pass
        if datatype == 'v' or datatype == 'appv':
            data        = np.zeros(self.lonArr.shape)
            data[self.nlat_grad:-self.nlat_grad, self.nlon_grad:-self.nlon_grad]\
                        = self.appV
            mdata       = ma.masked_array(data, mask=self.mask )
        if cmap == 'ses3d':
            cmap        = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
                            0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
        elif cmap == 'cv':
            import pycpt
            cmap    = pycpt.load.gmtColormap('./cv.cpt')
        elif os.path.isfile(cmap):
            import pycpt
            cmap    = pycpt.load.gmtColormap(cmap)
        im      = m.pcolormesh(x, y, mdata, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
        cb      = m.colorbar(im, "bottom", size="3%", pad='2%')
        cb.ax.tick_params(labelsize=10)
        if self.fieldtype=='Tph' or self.fieldtype=='Tgr':
            cb.set_label('sec', fontsize=12, rotation=0)
        if self.fieldtype=='Amp':
            cb.set_label('nm', fontsize=12, rotation=0)
        # if contour:
        #     # levels=np.linspace(ma.getdata(self.Zarr).min(), ma.getdata(self.Zarr).max(), 20)
        #     levels=np.linspace(ma.getdata(self.Zarr).min(), ma.getdata(self.Zarr).max(), 60)
        #     m.contour(x, y, self.Zarr, colors='k', levels=levels, linewidths=0.5)
        if showfig:
            plt.show()
        return m
    
    def plot_field(self, projection='lambert', contour=True, geopolygons=None, showfig=True, vmin=None, vmax=None, stations=False, event=False):
        """Plot data with contour
        """
        m=self._get_basemap(projection=projection, geopolygons=geopolygons)
        x, y=m(self.lonArr, self.latArr)
        if event:
            try:
                evx, evy=m(self.evlo, self.evla)
                m.plot(evx, evy, 'yo', markersize=10)
            except: pass
        if stations:
            try:
                stx, sty=m(self.lonArrIn, self.latArrIn)
                m.plot(stx, sty, 'y^', markersize=6)
            except: pass
        try:
            stx, sty = m(self.stalons, self.stalats)
            m.plot(stx, sty, 'b^', markersize=6)
        except: pass
        im=m.pcolormesh(x, y, self.Zarr, cmap='gist_ncar_r', shading='gouraud', vmin=vmin, vmax=vmax)
        cb = m.colorbar(im, "bottom", size="3%", pad='2%')
        cb.ax.tick_params(labelsize=10)
        if self.fieldtype=='Tph' or self.fieldtype=='Tgr':
            cb.set_label('sec', fontsize=12, rotation=0)
        if self.fieldtype=='Amp':
            cb.set_label('nm', fontsize=12, rotation=0)
        if contour:
            # levels=np.linspace(ma.getdata(self.Zarr).min(), ma.getdata(self.Zarr).max(), 20)
            levels=np.linspace(ma.getdata(self.Zarr).min(), ma.getdata(self.Zarr).max(), 60)
            m.contour(x, y, self.Zarr, colors='k', levels=levels, linewidths=0.5)
        if showfig:
            plt.show()
        return m
    
    def plot_lplc(self, projection='lambert', contour=False, geopolygons=None, vmin=None, vmax=None, showfig=True):
        """Plot data with contour
        """
        plt.figure()
        m       = self._get_basemap(projection=projection, geopolygons=geopolygons)
        x, y    = m(self.lonArr, self.latArr)
        x       = x[self.nlat_lplc:-self.nlat_lplc, self.nlon_lplc:-self.nlon_lplc]
        y       = y[self.nlat_lplc:-self.nlat_lplc, self.nlon_lplc:-self.nlon_lplc]
        # cmap =discrete_cmap(int(vmax-vmin)/2+1, 'seismic')
        m.pcolormesh(x, y, self.lplc, cmap='seismic', shading='gouraud', vmin=vmin, vmax=vmax)
        cb      = m.colorbar()
        cb.ax.tick_params(labelsize=15) 
        levels  = np.linspace(self.lplc.min(), self.lplc.max(), 100)
        if contour:
            plt.contour(x, y, self.lplc, colors='k', levels=levels)
        if showfig:
            plt.show()
        return
    
    def plot_lplc_amp(self, projection='lambert', contour=False, geopolygons=None, vmin=None, vmax=None, showfig=True):
        """Plot data with contour
        """
        m=self._get_basemap(projection=projection, geopolygons=geopolygons)
        m.drawstates()
        if self.lonArr.shape[0]-2==self.lplc.shape[0] and self.lonArr.shape[1]-2==self.lplc.shape[1]:
            self.cut_edge(1,1)
        elif self.lonArr.shape[0]!=self.lplc.shape[0] or self.lonArr.shape[1]!=self.lplc.shape[1]:
            raise ValueError('Incompatible shape for lplc and lon/lat array!')

        lplc_amp=ma.masked_array(self.lplc_amp, mask=np.zeros(self.Zarr.shape) )
        lplc_amp.mask[self.reason_n!=0]=1
        x, y=m(self.lonArr, self.latArr)
        # cmap =discrete_cmap(int((vmax-vmin)*80)/2+1, 'seismic')
        im=m.pcolormesh(x, y, lplc_amp, cmap='seismic_r', shading='gouraud', vmin=vmin, vmax=vmax)
        cb = m.colorbar(im, "right", size="3%", pad='2%')
        cb.ax.tick_params(labelsize=15)
        # cb.set_label(r"$\frac{\mathrm{km}}{\mathrm{s}}$", fontsize=8, rotation=0)
        if showfig:
            plt.show()
        return
    
    def plot_diffa(self, projection='lambert', prop=True, geopolygons=None, cmap='seismic', vmin=-20, vmax=20, showfig=True):
        """Plot data with contour
        """
        m=self._get_basemap(projection=projection, geopolygons=geopolygons)
        if self.lonArr.shape[0]-2==self.diffaArr.shape[0] and self.lonArr.shape[1]-2==self.diffaArr.shape[1]:
            self.cut_edge(1,1)
        elif self.lonArr.shape[0]!=self.diffaArr.shape[0] or self.lonArr.shape[1]!=self.diffaArr.shape[1]:
            raise ValueError('Incompatible shape for deflection and lon/lat array!')
        x, y=m(self.lonArr, self.latArr)
        cmap=pycpt.load.gmtColormap('./GMT_panoply.cpt')
        cmap =discrete_cmap(int(vmax-vmin)/4, cmap)
        im=m.pcolormesh(x, y, self.diffaArr, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
        cb = m.colorbar(im, "bottom", size="3%", pad='2%')
        cb.ax.tick_params(labelsize=10)
        cb.set_label('degree', fontsize=12, rotation=0)
        if prop:
            self.plot_propagation(inbasemap=m)
        if showfig:
            plt.show()
        return
    
    def plot_propagation(self, projection='lambert', inbasemap=None, factor=3, showfig=False):
        """Plot propagation direction
        """
        if inbasemap==None:
            m=self._get_basemap(projection=projection)
        else:
            m=inbasemap
        if self.lonArr.shape[0]-2==self.grad[0].shape[0] and self.lonArr.shape[1]-2==self.grad[0].shape[1]:
            self.cut_edge(1,1)
        elif self.lonArr.shape[0]!=self.grad[0].shape[0] or self.lonArr.shape[1]!=self.grad[0].shape[1]:
            raise ValueError('Incompatible shape for gradient and lon/lat array!')
        normArr = np.sqrt ( ma.getdata(self.grad[0] )** 2 + ma.getdata(self.grad[1]) ** 2)
        x, y=m(self.lonArr, self.latArr)
        U=self.grad[1]/normArr
        V=self.grad[0]/normArr
        if factor!=None:
            x=x[0:self.Nlat:factor, 0:self.Nlon:factor]
            y=y[0:self.Nlat:factor, 0:self.Nlon:factor]
            U=U[0:self.Nlat:factor, 0:self.Nlon:factor]
            V=V[0:self.Nlat:factor, 0:self.Nlon:factor]
        Q = m.quiver(x, y, U, V, scale=50, width=0.001)
        if showfig:
            plt.show()
        return
    
    def plot_appV(self, projection='lambert', geopolygons=None, showfig=True, vmin=None, vmax=None):
        """Plot data with contour
        """
        plt.figure()
        m       = self._get_basemap(projection=projection, geopolygons=geopolygons)
        x, y    = m(self.lonArr, self.latArr)
        x       = x[self.nlat_grad:-self.nlat_grad, self.nlon_grad:-self.nlon_grad]
        y       = y[self.nlat_grad:-self.nlat_grad, self.nlon_grad:-self.nlon_grad]
        cmap    = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
                    0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
        im      = m.pcolormesh(x, y, self.appV, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
        cb      = m.colorbar(im, "right", size="3%", pad='2%')
        cb.ax.tick_params(labelsize=10)
        cb.set_label(r"$\frac{\mathrm{km}}{\mathrm{s}}$", fontsize=8, rotation=0)
        if showfig:
            plt.show()
        return
    
    
    def get_az_dist_Arr(self):
        """Get epicentral distance array
        """
        evloArr=np.ones(self.lonArr.shape)*self.evlo
        evlaArr=np.ones(self.lonArr.shape)*self.evla
        g = Geod(ellps='WGS84')
        az, baz, distevent = geodist.inv( evloArr, evlaArr, self.lonArr, self.latArr)
        distevent=distevent/1000.
        self.distArr=distevent
        self.azArr=az
        return
    
    def plot_event(self, infname, evnumb, inbasemap):
        from obspy.imaging.beachball import beach
        dset=pyasdf.ASDFDataSet(infname)
        event=dset.events[evnumb-1]
        event_id=event.resource_id.id.split('=')[-1]
        magnitude=event.magnitudes[0].mag; Mtype=event.magnitudes[0].magnitude_type
        otime=event.origins[0].time
        evlo=event.origins[0].longitude; evla=event.origins[0].latitude; evdp=event.origins[0].depth/1000.
        mtensor=event.focal_mechanisms[0].moment_tensor.tensor
        mt=[mtensor.m_rr, mtensor.m_tt, mtensor.m_pp, mtensor.m_rt, mtensor.m_rp, mtensor.m_tp]
        x, y=inbasemap(evlo, evla)
        b = beach(mt, xy=(x, y), width=200000, linewidth=1, facecolor='b')
        b.set_zorder(10)
        ax = plt.gca()
        ax.add_collection(b)
        plt.suptitle('Depth: '+str(evdp)+' km'+ ' Magnitude: ' +str(magnitude) )
            
                
                    
    

    

