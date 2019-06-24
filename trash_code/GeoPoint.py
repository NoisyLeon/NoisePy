
"""
This is a sub-module of noisepy.
Classes and functions for geographycal points analysis and plotting.

:Copyright:
    Author: Lili Feng
    Graduate Research Assistant
    CIEI, Department of Physics, University of Colorado Boulder
    email: lili.feng@colorado.edu
"""
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
from mpl_toolkits.basemap import Basemap
from matplotlib.mlab import griddata
import matplotlib.cm
from matplotlib.ticker import FuncFormatter
import numexpr as npr
import glob
# setup Lambert Conformal basemap.
# set resolution=None to skip processing of boundary datasets.
# m = Basemap(width=12000000,height=9000000,projection='lcc',
#             resolution=None,lat_1=45.,lat_2=55,lat_0=50,lon_0=-107.)
# m.etopo()

class GeoPoint(object):
    """
    A class for a geographycal point analysis
    ---------------------------------------------------------------------
    Parameters:
    name - lon_lat
    lon, lat
    depthP - depth profile (np.array)
    depthPfname - 
    DispGr - Group V dispersion Curve (np.array)
    GrDispfname - 
    DispPh - Phase V dispersion Curve (np.array)
    PhDispfname - 
    """
    def __init__(self, name='',lon=None, lat=None, depthP=np.array([]), depthPfname='', DispGr=np.array([]), GrDispfname='',\
        DispPh=np.array([]), PhDispfname=''):
        self.name=name
        self.lon=lon
        self.lat=lat
        self.depthP=depthP
        self.depthPfname=depthPfname
        self.DispGr=DispGr
        self.GrDispfname=GrDispfname
        self.DispPh=DispPh
        self.PhDispfname=PhDispfname
    
    def SetName(self,name=''):
        if name=='':
            self.name='%g_%g' %(self.lon, self.lat)
        else:
            self.name=name
        return
    
    def SetVProfileFname(self, prefix='MC.', suffix='.acc.average.mod.q'):
        self.depthPfname=prefix+self.name+suffix
        return
    
    def SetGrDispfname(self, prefix='MC.', suffix='.acc.average.g.disp'):
        self.GrDispfname=prefix+self.name+suffix
        return
    
    def SetPhDispfname(self, prefix='MC.', suffix='.acc.average.p.disp'):
        self.PhDispfname=prefix+self.name+suffix
        return
    
    def SetAllfname(self):
        self.SetVProfileFname()
        self.SetGrDispfname()
        self.SetPhDispfname()
        return
      
    def LoadVProfile(self, datadir='', dirPFX='', dirSFX='', depthPfname=''):
        if depthPfname!='':
            infname=depthPfname
        elif datadir=='':
            infname='./'+dirPFX+self.name+dirSFX+'/'+self.depthPfname
        else:
            infname=datadir+'/'+dirPFX+self.name+dirSFX+'/'+self.depthPfname
        if os.path.isfile(infname):
            self.depthP=np.loadtxt(infname)
        else:
            print 'Warning: No Depth Profile File for:'+str(self.lon)+' '+str(self.lat)
        return
    
    def GenerateSmooth(self, ):
        pass
    
    
    def LoadGrDisp(self, datadir='', dirPFX='', dirSFX='', GrDispfname=''):
        if GrDispfname!='':
            infname=GrDispfname
        elif datadir=='':
            infname='./'+dirPFX+self.name+dirSFX+'/'+self.GrDispfname
        else:
            infname=datadir+'/'+dirPFX+self.name+dirSFX+'/'+self.GrDispfname
        if os.path.isfile(infname):
            self.DispGr=np.loadtxt(infname)
        else:
            print 'Warning: No Group Vel Dispersion File for:'+str(self.lon)+' '+str(self.lat)
        return;
    
    def SaveGrDisp(self, outdir='', dirPFX='', dirSFX='', GrDispfname=''):
        if GrDispfname!='':
            outfname=GrDispfname
        elif outdir=='':
            outfname='./'+dirPFX+self.name+dirSFX+'/'+self.GrDispfname
            outdir='./'+dirPFX+self.name+dirSFX;
        else:
            outfname=outdir+'/'+dirPFX+self.name+dirSFX+'/'+self.GrDispfname
            outdir=outdir+'/'+dirPFX+self.name+dirSFX
        if not os.path.isdir(outdir):
            os.makedirs(outdir);
        np.savetxt(outfname, self.DispGr, fmt='%g')
        return;
    
    def LoadPhDisp(self, datadir='', dirPFX='', dirSFX='', PhDispfname=''):
        if PhDispfname!='':
            infname=PhDispfname
        elif datadir=='':
            infname='./'+dirPFX+self.name+dirSFX+'/'+self.PhDispfname
        else:
            infname=datadir+'/'+dirPFX+self.name+dirSFX+'/'+self.PhDispfname
        if os.path.isfile(infname):
            self.DispPh=np.loadtxt(infname)
        else:
            print 'Warning: No Phase Vel Dispersion File for:'+str(self.lon)+' '+str(self.lat)
        return

    def SavePhDisp(self, outdir='', dirPFX='', dirSFX='', PhDispfname=''):
        if PhDispfname!='':
            outfname=PhDispfname
        elif outdir=='':
            outfname='./'+dirPFX+self.name+dirSFX+'/'+self.PhDispfname;
            outdir='./'+dirPFX+self.name+dirSFX;
        else:
            outfname=outdir+'/'+dirPFX+self.name+dirSFX+'/'+self.PhDispfname;
            outdir=outdir+'/'+dirPFX+self.name+dirSFX;
        if not os.path.isdir(outdir):
            os.makedirs(outdir);

        np.savetxt(outfname, self.DispPh, fmt='%g')
        return;
    
    def PlotDisp(self, xcl=0, xlabel='Period(s)', ycl={int(1):None, int(2):int(3)}, ylabel='Velocity(km/s)', title='', datatype='PhV', ax=None):
        if ax==None:
            ax=plt.subplot()
        if datatype=='PhV':
            Inarray=self.DispPh
            if title=='':
                title='Phase Velocity Dispersion Curve'
        elif datatype=='GrV':
            if title=='':
                title='Group Velocity Dispersion Curve'
            Inarray=self.DispGr
        else:
            print 'Error: Unknow Data Type!'
            return
        if Inarray.size==0:
            print 'Warning: No Dispersion Data for:'+str(self.lon)+' '+str(self.lat)
            return
        X=Inarray[:,xcl]
        for yC in ycl.keys():
            Y=Inarray[:,yC]
            if ycl[yC]==None:
                line1=ax.plot(X, Y, '-',lw=3) #
            else:
                errC=ycl[yC]
                Yerr=Inarray[:,errC]
                plt.errorbar(X, Y, fmt='.', lw=2, yerr=Yerr)       
        ###
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)              
        return
    
    def PlotDispBoth(self, xcl=0, xlabel='Period(s)', ycl={int(1):None, int(2):int(3)}, ylabel='Velocity(km/s)', title='Dispersion Curve', ax=None):
        if ax==None:
            ax=plt.subplot()
        Inarray=self.DispPh
        X=Inarray[:,xcl]
        for yC in ycl.keys():
            Y=Inarray[:,yC]
            if ycl[yC]==None:
                line1, =ax.plot(X, Y, '-b',lw=3) #
            else:
                errC=ycl[yC]
                Yerr=Inarray[:,errC]
                plt.errorbar(X, Y, fmt='.g', lw=2, yerr=Yerr)
        Inarray=self.DispGr
        X=Inarray[:,xcl]
        for yC in ycl.keys():
            Y=Inarray[:,yC]
            if ycl[yC]==None:
                line2, =ax.plot(X, Y, '-k',lw=3) #
            else:
                errC=ycl[yC]
                Yerr=Inarray[:,errC]
                plt.errorbar(X, Y, fmt='.g', lw=2, yerr=Yerr)
        ax.legend([line1, line2], ['Phase V', 'Group V'], loc=0)
        [xmin, xmax, ymin, ymax]=plt.axis()
        plt.axis([xmin-1, xmax+0.5, ymin, ymax])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)              
        return
    
    def PlotVProfile(self, xcl={int(1):None, int(2):None}, xlabel='Velocity(km/s)', ycl=0, ylabel='Depth(km)', title='Depth Profile', depLimit=None, ax=None):
        if ax==None:
            ax=plt.subplot()
        Inarray=self.depthP
        if Inarray.size==0:
            print 'Warning: No Dispersion Data for:'+str(self.lon)+' '+str(self.lat)
            return
        Y=Inarray[:,ycl]
        if depLimit!=None:
            yindex=Y<depLimit
            Y=Y[yindex]
            Inarray=Inarray[:Y.size, :]
        for xC in xcl.keys():
            X=Inarray[:,xC]
            if xcl[xC]==None:
                ax.plot(X, Y, lw=3) #
            else:
                errC=xcl[xC]
                Yerr=Inarray[:,errC]
                plt.errorbar(X, Y, fmt='.g', lw=2, yerr=Yerr)
        plt.xlabel(xlabel)  
        plt.ylabel(ylabel)
        plt.title(title)
        plt.gca().invert_yaxis()
        return
    
    def IsInRegion(self, maxlon=360, minlon=0, maxlat=90, minlat=-90 ):
        if self.lon < maxlon and self.lon > minlon and self.lat < maxlat and self.lat > minlat:
            return True
        else:
            return False
    
class GeoMap(object):
    """
    A object contains a list of GeoPoint
    """
    def __init__(self,geopoints=None):
        self.geopoints=[]
        if isinstance(geopoints, GeoPoint):
            geopoints = [geopoints]
        if geopoints:
            self.geopoints.extend(geopoints)

    def __add__(self, other):
        """
        Add two GeoMaps with self += other.
        """
        if isinstance(other, GeoPoint):
            other = GeoMap([other])
        if not isinstance(other, GeoMap):
            raise TypeError
        geopoints = self.geopoints + other.geopoints
        return self.__class__(geopoints=geopoints)

    def __len__(self):
        """
        Return the number of GeoPoints in the GeoMap object.
        """
        return len(self.geopoints)

    def __getitem__(self, index):
        """
        __getitem__ method of GeoMap objects.
        :return: GeoPoint objects
        """
        if isinstance(index, slice):
            return self.__class__(geopoints=self.geopoints.__getitem__(index))
        else:
            return self.geopoints.__getitem__(index)

    def append(self, geopoint):
        """
        Append a single GeoPoint object to the current GeoMap object.
        """
        if isinstance(geopoint, GeoPoint):
            self.geopoints.append(geopoint)
        else:
            msg = 'Append only supports a single GeoPoint object as an argument.'
            raise TypeError(msg)
        return self
    
    def ReadGeoMapLst(self, mapfile ):
        """
        Read GeoPoint List from a txt file
        name longitude latidute
        """
        f = open(mapfile, 'r')
        Name=[]
        for lines in f.readlines():
            lines=lines.split()
            name=lines[0]
            lon=float(lines[1])
            lat=float(lines[2])
            if Name.__contains__(name):
                index=Name.index(name)
                if abs(self[index].lon-lon) >0.01 and abs(self[index].lat-lat) >0.01:
                    raise ValueError('Incompatible GeoPoint Location:' + netsta+' in GeoPoint List!')
                else:
                    print 'Warning: Repeated GeoPoint:' +name+' in GeoPoint List!'
                    continue
            Name.append(name)
            self.append(GeoPoint (name=name, lon=lon, lat=lat))
            f.close()
        return
    
    def Trim(self, maxlon=360, minlon=0, maxlat=90, minlat=-90):
        TrimedGeoMap=GeoMap()
        for geopoint in self.geopoints:
            if geopoint.IsInRegion(maxlon=maxlon, minlon=minlon, maxlat=maxlat, minlat=minlat):
                TrimedGeoMap.append(geopoint)
        return TrimedGeoMap
    
    def SetAllfname(self):
        for geopoint in self.geopoints:
            geopoint.SetAllfname()
        return
    
    def SetPhDispfname(self, prefix='MC.', suffix='.acc.average.p.disp'):
        for geopoint in self.geopoints:
            geopoint.SetPhDispfname(prefix=prefix, suffix=suffix)
        return
    
    def SetGrDispfname(self, prefix='MC.', suffix='.acc.average.p.disp'):
        for geopoint in self.geopoints:
            geopoint.SetGrDispfname(prefix=prefix, suffix=suffix)
        return
    
    def LoadVProfile(self, datadir='', dirPFX='', dirSFX='', depthPfname=''):
        for geopoint in self.geopoints:
            geopoint.LoadVProfile(datadir=datadir, dirPFX=dirPFX, depthPfname=depthPfname)
        return
    
    def LoadGrDisp(self, datadir='', dirPFX='', dirSFX='', GrDispfname=''):
        for geopoint in self.geopoints:
            geopoint.LoadGrDisp(datadir=datadir, dirPFX=dirPFX, GrDispfname=GrDispfname)
        return
    
    def LoadPhDisp(self, datadir='', dirPFX='', dirSFX='', PhDispfname=''):
        for geopoint in self.geopoints:
            geopoint.LoadPhDisp(datadir=datadir, dirPFX=dirPFX, PhDispfname=PhDispfname)
        return
    
    def GetGeoMapfromDir(self, datadir='', dirPFX='', dirSFX=''):
        pattern=datadir+'/*';
        LonLst=np.array([]);
        LatLst=np.array([]);
        for dirname in sorted(glob.glob(pattern)):
            dirname=dirname.split('/');
            dirname=dirname[len(dirname)-1];
            if dirPFX!='':
                dirname=dirname.split(dirPFX)[1];
            if dirSFX!='':
                dirname=dirname.split(dirSFX);
                if len(dirname) > 2:
                    raise ValueError('Directory Suffix Error!');
                dirname=dirname[0];
            lon=dirname.split('_')[0];
            lat=dirname.split('_')[1];
            LonLst=np.append(LonLst, float(lon));
            LatLst=np.append(LatLst, float(lat));
        indLst=np.lexsort((LonLst,LatLst));
        for indS in indLst:
            lon='%g' %(LonLst[indS])
            lat='%g' %(LatLst[indS])
            self.append(GeoPoint (name=lon+'_'+lat, lon=float(lon), lat=float(lat)));
        return
    
    def GetMapLimit(self):
        minlon=360.
        maxlon=0.
        minlat=90.
        maxlat=-90.
        for geopoint in self.geopoints:
            if geopoint.lon > maxlon:
                maxlon=geopoint.lon
            if geopoint.lon < minlon:
                minlon=geopoint.lon
            if geopoint.lat > maxlat:
                maxlat=geopoint.lat
            if geopoint.lat < minlat:
                minlat=geopoint.lat
        self.minlon=minlon
        self.maxlon=maxlon
        self.minlat=minlat
        self.maxlat=maxlat
        return
     
    def BrowseFigures(self, datadir='', dirPFX='', dirSFX='', datatype='All', depLimit=None, \
                      llcrnrlon=None, llcrnrlat=None,urcrnrlon=None,urcrnrlat=None, browseflag=True, saveflag=False):
        if llcrnrlon==None or llcrnrlat==None or urcrnrlon==None or urcrnrlat==None:
            llcrnrlon=self.minlon
            llcrnrlat=self.minlat
            urcrnrlon=self.maxlon
            urcrnrlat=self.maxlat
            # print llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat
        for geopoint in self.geopoints:
            print 'Plotting:'+geopoint.name
            if geopoint.depthP.size==0 and geopoint.DispGr.size==0 and geopoint.DispPh.size==0:
                continue
            plt.close('all')
            fig=plb.figure(num=None, figsize=(8, 12), dpi=80, facecolor='w', edgecolor='k')
            if datatype=='All':
                fig.add_subplot(3,1,1)
            else:
                fig.add_subplot(2,1,1)
            m = Basemap(llcrnrlon=llcrnrlon-1, llcrnrlat=llcrnrlat-1, urcrnrlon=urcrnrlon+1, urcrnrlat=urcrnrlat+1, \
                rsphere=(6378137.00,6356752.3142), resolution='l', projection='merc')
            lon = geopoint.lon
            lat = geopoint.lat
            x,y = m(lon, lat)
            m.plot(x, y, 'ro', markersize=5)
            m.drawcoastlines()
            m.etopo()
            # m.fillcontinents()
            # draw parallels
            m.drawparallels(np.arange(-90,90,10),labels=[1,1,0,1])
            # draw meridians
            m.drawmeridians(np.arange(-180,180,10),labels=[1,1,0,1])
            plt.title('Longitude:'+str(geopoint.lon)+' Latitude:'+str(geopoint.lat), fontsize=15)
            if datatype=='All':
                geopoint.PlotDispBoth(ax=plt.subplot(3,1,2))
                # geopoint.PlotDisp(datatype='GrV', ax=plt.subplot(3,1,2))
                geopoint.PlotVProfile(ax=plt.subplot(3,1,3), depLimit=depLimit)
            elif datatype=='DISP':
                # geopoint.PlotDisp(datatype='PhV',ax=plt.subplot(3,1,2))
                # geopoint.PlotDisp(datatype='GrV',ax=plt.subplot(3,1,2))
                geopoint.PlotDispBoth(ax=plt.subplot(3,1,2))
            elif datatype=='VPr':
                geopoint.PlotVProfile(depLimit=depLimit,ax=plt.subplot(2,1,2))
            else:
                raise ValueError('Unknown datatype')
            fig.suptitle('Longitude:'+str(geopoint.lon)+' Latitude:'+str(geopoint.lat), fontsize=15)
            if browseflag==True:
                plt.draw()
                plt.pause(1) # <-------
                raw_input("<Hit Enter To Close>")
                plt.close('all')
            if saveflag==True and datadir!='':
                fig.savefig(datadir+'/'+dirPFX+geopoint.name+dirSFX+'/'+datatype+'_'+geopoint.name+'.ps', format='ps')
        return;

    def SavePhDisp(self, outdir='', dirPFX='', dirSFX='', PhDispfname=''):
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        for geopoint in self.geopoints:
            geopoint.SavePhDisp(outdir=outdir, dirPFX=dirPFX, dirSFX=dirSFX, PhDispfname=PhDispfname);
        print 'End of Saving Phave V Points!';
        return;
    
    def SaveGrDisp(self, outdir='', dirPFX='', dirSFX='', GrDispfname=''):
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        for geopoint in self.geopoints:
            geopoint.SaveGrDisp(outdir=outdir, dirPFX=dirPFX, dirSFX=dirSFX, GrDispfname=GrDispfname);
        print 'End of Saving Group V Points!';
        return;
        
    
class PeriodMap(object):
    """
    A class to store Phase/Group Velocity map for a specific period.
    """
    def __init__(self, period=None, mapfname='', tomomapArr=np.array([])):
        self.mapfname=mapfname;
        self.tomomapArr=tomomapArr;
        self.period=period;
        return;
        
    def ReadMap(self):
        if not os.path.isfile(self.mapfname):
            print 'Velocity Map for period: ',self.period,' sec not exist!'
            print self.mapfname;
            return;
        self.tomomapArr=np.loadtxt(self.mapfname);
        return;
    

class MapDatabase(object):
    """
    Geographical Map Database class for Map Analysis.
    """
    def __init__(self, tomodatadir='', tomotype='misha', tomof_pre='', tomof_sfx='', geo_pre='', geo_sfx='', perarray=np.array([]), geomapsdir='', refdir=''):
        self.tomodatadir=tomodatadir;
        self.tomotype=tomotype;
        self.perarray=perarray;
        self.geomapsdir=geomapsdir;
        self.refdir=refdir;
        self.tomof_pre=tomof_pre;
        self.tomof_sfx=tomof_sfx;
        
    def ReadTomoResult(self, datatype='ph'):
        """
        Read Tomoraphic Maps for a period array.
        ---------------------------------------------------------------------
        Input format:
        self.tomodatadir/per_ph/self.tomof_pre+per+self.tomof_sfx
        e.g. tomodatadir/10_ph/QC_AZI_R_1200_200_1000_100_1_10.1
        Input data are saved to a list of permap object.
        ---------------------------------------------------------------------
        """
        self.permaps=[];
        if self.tomotype=='misha':
            for per in self.perarray:
                intomofname=self.tomodatadir+'/'+'%g' %( per )+'_'+datatype+'/'+self.tomof_pre+'%g' %( per )+self.tomof_sfx;
                temp_per_map=PeriodMap(period=per, mapfname=intomofname);
                temp_per_map.ReadMap()
                self.permaps.append(temp_per_map);
        elif self.tomotype=='EH':
            for per in perarray:
                intomofname=self.tomodatadir+'/'+'%g' %( per )+'sec'+'/'+self.tomof_pre+'%g' %( per )+self.tomof_sfx;
                temp_per_map=PeriodMap(period=per, mapfname=intomofname);
                temp_per_map.ReadMap()
                self.permaps.append(temp_per_map);
        return;
    
    def TomoMap2GeoPoints(self, lonlatCheck=True, datatype='ph'):
        """
        Convert Tomographic maps to GeoMap object ( GeoPoint List ), saved as "self.geomap"
        """
        self.geomap=GeoMap();
        SizeC=self.permaps[0].tomomapArr.size;
        lonLst=self.permaps[0].tomomapArr[:,0];
        latLst=self.permaps[0].tomomapArr[:,1];
        Vvalue=self.permaps[0].tomomapArr[:,2];
        per0=self.permaps[0].period;
        for i in np.arange(lonLst.size):
            tempGeo=GeoPoint(lon=lonLst[i], lat=latLst[i]);
            tempGeo.SetName();
            if datatype=='ph':
                tempGeo.DispPh=np.array([ per0, Vvalue[i]]);
                tempGeo.SetPhDispfname(prefix='',suffix='.phv');
            elif datatype=='gr':
                tempGeo.DispGr=np.array([ per0, Vvalue[i]]);
                tempGeo.SetGrDispfname(prefix='',suffix='.grv');
            self.geomap.append(tempGeo);
        Lper=1;
        for permap in self.permaps[1:]:
            period=permap.period;
            Vvalue=permap.tomomapArr[:,2];
            if SizeC!=permap.tomomapArr.size:
                raise ValueError('Different size in period maps!: ', permap.period);
            if lonlatCheck==True:
                clon=permap.tomomapArr[:,0];
                clat=permap.tomomapArr[:,1];
                sumlon=npr.evaluate('sum(abs(lonLst-clon))');
                sumlat=npr.evaluate('sum(abs(lonLst-clon))');
                if sumlon>0.1 or sumlat>0.1:
                    raise ValueError('Incompatible grid points in period maps!: ', permap.period);
            Lper=Lper+1;
            for i in np.arange(lonLst.size):
                if datatype=='ph':
                    self.geomap[i].DispPh=np.append( self.geomap[i].DispPh , np.array( [period, Vvalue[i]]));
                elif datatype=='gr':
                    self.geomap[i].DispGr=np.append( self.geomap[i].DispGr , np.array( [period, Vvalue[i]]));
        for i in np.arange(lonLst.size):
            if datatype=='ph':
                self.geomap[i].DispPh=self.geomap[i].DispPh.reshape((Lper, 2));
            elif datatype=='gr':
                self.geomap[i].DispGr=self.geomap[i].DispGr.reshape((Lper, 2));
        return;
    
    
        
def PlotTomoMap(fname, dlon=0.5, dlat=0.5, title='', datatype='ph', outfname='', browseflag=False, saveflag=True):
    """
    Plot Tomography Map
    longitude latidute ZValue
    """
    if title=='':
        title=fname;
    if outfname=='':
        outfname=fname;
    Inarray=np.loadtxt(fname)
    LonLst=Inarray[:,0]
    LatLst=Inarray[:,1]
    ZValue=Inarray[:,2]
    llcrnrlon=LonLst.min()
    llcrnrlat=LatLst.min()
    urcrnrlon=LonLst.max()
    urcrnrlat=LatLst.max()
    Nlon=int((urcrnrlon-llcrnrlon)/dlon)+1
    Nlat=int((urcrnrlat-llcrnrlat)/dlat)+1
    fig=plt.figure(num=None, figsize=(8, 12), dpi=80, facecolor='w', edgecolor='k')
    m = Basemap(llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat, \
        rsphere=(6378137.00,6356752.3142), resolution='l', projection='merc')
    
    lon = LonLst
    lat = LatLst
    x,y = m(lon, lat)
    xi = np.linspace(x.min(), x.max(), Nlon)
    yi = np.linspace(y.min(), y.max(), Nlat)
    xi, yi = np.meshgrid(xi, yi)
    
    #-- Interpolating at the points in xi, yi
    zi = griddata(x, y, ZValue, xi, yi)
    # m.pcolormesh(xi, yi, zi, cmap='seismic_r', shading='gouraud')
    cmap=matplotlib.cm.seismic_r
    cmap.set_bad('w',1.)
    m.imshow(zi, cmap=cmap)
    m.drawcoastlines()
    m.colorbar(location='bottom',size='2%')
    # m.fillcontinents()
    # draw parallels
    m.drawparallels(np.arange(-90,90,10),labels=[1,1,0,1])
    # draw meridians
    m.drawmeridians(np.arange(-180,180,10),labels=[1,1,1,0])
    plt.suptitle(title,y=0.9, fontsize=22);
    if browseflag==True:
        plt.draw()
        plt.pause(1) # <-------
        raw_input("<Hit Enter To Close>")
        plt.close('all')
    if saveflag==True:
        fig.savefig(outfname+'.ps', format='ps')
    return 
    
def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(100 * y)
    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] == True:
        return s + r'$\%$'
    else:
        return s + '%'
    
def PlotTomoResidualIsotropic(fname, bins=1000, xmin=-10, xmax=10, outfname='', browseflag=True, saveflag=True):

    if outfname=='':
        outfname=fname;
    Inarray=np.loadtxt(fname)
    res_tomo=Inarray[:,7];
    res_mod=Inarray[:,8];
    fig=plt.figure(num=None, figsize=(8, 12), dpi=80, facecolor='w', edgecolor='k')
    plt.subplot(2,1,1)
    n_tomo, bins_tomo, patches_tomo = plt.hist(res_tomo, bins=bins, normed=1, facecolor='blue', alpha=0.75)
    plt.axis([xmin, xmax, 0, n_tomo.max()+0.05])
    mean_tomo=res_tomo.mean();
    std_tomo=np.std(res_tomo)
    formatter = FuncFormatter(to_percent)
    plt.ylabel('Percentage', fontsize=20)
    plt.xlabel('Misfit( s )', fontsize=20)
    plt.title('Tomo residual (mean: %g std: %g)' % (mean_tomo, std_tomo))             
    plt.subplot(2,1,2)
    n_mod, bins_mod, patches_mod = plt.hist(res_mod, bins=bins , normed=1, facecolor='blue', alpha=0.75)
    plt.axis([xmin, xmax, 0, n_mod.max()+0.05])
    mean_mod=res_mod.mean();
    std_mod=np.std(res_mod)
    formatter = FuncFormatter(to_percent)
    plt.ylabel('Percentage', fontsize=20)
    plt.xlabel('Misfit( s )', fontsize=20)
    plt.title('RefMod residual (Mean: %g std: %g)' % (mean_mod,std_mod))   
    if browseflag==True:
        plt.draw()
        plt.pause(1) # <-------
        raw_input("<Hit Enter To Close>")
        plt.close('all')
    if saveflag==True:
        fig.savefig(outfname+'.ps', format='ps')
    return
    
    
def GenerateDepthArr(depth, dz):
    if depth.size != dz.size:
        raise ValueError('Size of depth and depth interval arrays NOT compatible!');
    outArr=np.array([]);
    for i in np.arange(depth.size):
        if i==0:
            temparr=np.arange(depth[i]/dz[i])*dz[i];
        else:
            temparr=np.arange(depth[i]/dz[i])*dz[i]+depth[i-1];
        outArr=np.append(outArr, temparr);
    return outArr;

    
    
    
  
            
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
       
                
                
                
            
        
    
        
    
    
    

    
        
        
    
        
    
    
    
    
        
        
        