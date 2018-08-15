

from pyproj import Geod
import numpy as np

def ctr_generator(lons, lats, outfname, d = 10.):
    if lons.size != lats.size:
        raise ValueError('Size of longitude and latitude list must be the same')
    g           = Geod(ellps='WGS84')
    N           = lons.size
    lonlats     = []
    for i in range(N):
        lon1    = lons[i]
        lat1    = lats[i]
        if i < N-1:
            lon2    = lons[i+1]
            lat2    = lats[i+1]
        else:
            lon2    = lons[0]
            lat2    = lats[0]
        az, baz, dist   = g.inv(lon1, lat1, lon2, lat2)
        dist            = dist/1000.
        if d < dist:
            d               = dist/float(int(dist/d))
            Nd              = int(dist/d)
            lonlats         += [(lon1, lat1)]
            lonlats         += g.npts(lon1, lat1, lon2, lat2, npts=Nd-1)
        else:
            lonlats         += [(lon1, lat1)]
    with open(outfname, 'w') as fid:
        npts        = len(lonlats)
        fid.writelines('0. 0. \n')
        fid.writelines('%g \n' %npts)
        for lonlat in lonlats:
            if lonlat[0] < 0.:
                outlon  = lonlat[0]+360.
            else:
                outlon  = lonlat[0]
            outlat      = lonlat[1]
            fid.writelines('%g  %g\n' %(outlon, outlat))
        fid.writelines('%g \n' %npts)
        for i in range(npts):
            if i < npts-1:
                fid.writelines('%g  %g\n' %(i+1, i+2))
            else:
                fid.writelines('%g  %g\n' %(i+1, 1))
    # return lonlats

if __name__ == "__main__":
    
    lons    = np.array([188., 188., 238., 238])
    lats    = np.array([52., 72., 72., 52.])
    ctr_generator(lons=lons, lats=lats, outfname='contour.ctr', d=10000.)

