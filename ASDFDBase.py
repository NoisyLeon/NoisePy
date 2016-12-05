# -*- coding: utf-8 -*-
"""
A python module for seismic data analysis based on ASDF database

:Methods:
    aftan analysis (use pyaftan or aftanf77)
    C3(Correlation of coda of Cross-Correlation) computation
    Automatic Receiver Function Analysis( Iterative Deconvolution and Harmonic Stripping )
    Preparing data for surface wave tomography (Barmin's method, Eikonal/Helmholtz tomography)
    Stacking/Rotation for Cross-Correlation Results from SEED2CORpp
    Bayesian Monte Carlo Inversion of Surface Wave and Receiver Function datasets (To be added soon)

:Dependencies:
    numpy >=1.9.1
    scipy >=0.18.0
    matplotlib >=1.4.3
    ObsPy >=1.0.1
    pyfftw 0.10.3 (optional)
    
:Copyright:
    Author: Lili Feng
    Graduate Research Assistant
    CIEI, Department of Physics, University of Colorado Boulder
    email: lili.feng@colorado.edu
"""
import pyasdf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import obspy
import warnings
import copy
import os, shutil
import numba
from functools import partial
import multiprocessing
import pyaftan
from subprocess import call
from obspy.clients.fdsn.client import Client
from mpl_toolkits.basemap import Basemap, shiftgrid, cm
import obspy.signal.array_analysis
from obspy.imaging.cm import obspy_sequential
from pyproj import Geod
from obspy.taup import TauPyModel
import CURefPy
import glob

# from obspy.signal.invsim import corn_freq_2_paz
sta_info_default={'rec_func': 0, 'xcorr': 1, 'isnet': 0}

xcorr_header_default={'netcode1': '', 'stacode1': '', 'netcode2': '', 'stacode2': '', 'chan1': '', 'chan2': '',
        'npts': 12345, 'b': 12345, 'e': 12345, 'delta': 12345, 'dist': 12345, 'az': 12345, 'baz': 12345, 'stackday': 0}

xcorr_sacheader_default = {'knetwk': '', 'kstnm': '', 'kcmpnm': '', 'stla': 12345, 'stlo': 12345, 
            'kuser0': '', 'kevnm': '', 'evla': 12345, 'evlo': 12345, 'evdp': 0., 'dist': 0., 'az': 12345, 'baz': 12345, 
                'delta': 12345, 'npts': 12345, 'user0': 0, 'b': 12345, 'e': 12345}

ref_header_default = {'otime': '', 'network': '', 'station': '', 'stla': 12345, 'stlo': 12345, 'evla': 12345, 'evlo': 12345, 'evdp': 0.,
                    'dist': 0., 'az': 12345, 'baz': 12345, 'delta': 12345, 'npts': 12345, 'b': 12345, 'e': 12345, 'arrival': 12345, 'phase': '',
                        'tbeg': 12345, 'tend': 12345, 'hslowness': 12345, 'ghw': 12345, 'VR':  12345, 'moveout': -1}

monthdict={1: 'JAN', 2: 'FEB', 3: 'MAR', 4: 'APR', 5: 'MAY', 6: 'JUN', 7: 'JUL', 8: 'AUG', 9: 'SEP', 10: 'OCT', 11: 'NOV', 12: 'DEC'}

geodist = Geod(ellps='WGS84')
taupmodel = TauPyModel(model="iasp91")

class noiseASDF(pyasdf.ASDFDataSet):
    """ An object to for ambient noise cross-correlation analysis based on ASDF database
    """
    
    def write_stationxml(self, staxml, source='CIEI'):
        inv=obspy.core.inventory.inventory.Inventory(networks=[], source=source)
        for staid in self.waveforms.list():
            inv+=self.waveforms[staid].StationXML
        inv.write(staxml, format='stationxml')
        return
    
    def write_stationtxt(self, stafile):
        """Write obspy inventory to txt station list(format used in SEED2COR)
        """
        try:
            auxiliary_info=self.auxiliary_data.StaInfo
            isStaInfo=True
        except:
            isStaInfo=False
        with open(stafile, 'w') as f:
            for staid in self.waveforms.list():
                stainv=self.waveforms[staid].StationXML
                netcode=stainv.networks[0].code
                stacode=stainv.networks[0].stations[0].code
                lon=stainv.networks[0].stations[0].longitude
                lat=stainv.networks[0].stations[0].latitude
                if isStaInfo:
                    staid_aux=netcode+'/'+stacode
                    ccflag=auxiliary_info[staid_aux].parameters['xcorr']
                    f.writelines('%s %3.4f %3.4f %d %s\n' %(stacode, lon, lat, ccflag, netcode) )
                else:
                    f.writelines('%s %3.4f %3.4f %s\n' %(stacode, lon, lat, netcode) )        
        return
    
    def read_stationtxt(self, stafile, source='CIEI', chans=['BHZ', 'BHE', 'BHN'], dnetcode='TA'):
        """Read txt station list 
        """
        sta_info=sta_info_default.copy()
        with open(stafile, 'r') as f:
            Sta=[]
            site=obspy.core.inventory.util.Site(name='01')
            creation_date=obspy.core.utcdatetime.UTCDateTime(0)
            inv=obspy.core.inventory.inventory.Inventory(networks=[], source=source)
            total_number_of_channels=len(chans)
            for lines in f.readlines():
                lines=lines.split()
                stacode=lines[0]
                lon=float(lines[1])
                lat=float(lines[2])
                netcode=dnetcode
                ccflag=None
                if len(lines)==5:
                    try:
                        ccflag=int(lines[3])
                        netcode=lines[4]
                    except ValueError:
                        ccflag=int(lines[4])
                        netcode=lines[3]
                if len(lines)==4:
                    try:
                        ccflag=int(lines[3])
                    except ValueError:
                        netcode=lines[3]
                netsta=netcode+'.'+stacode
                if Sta.__contains__(netsta):
                    index=Sta.index(netsta)
                    if abs(self[index].lon-lon) >0.01 and abs(self[index].lat-lat) >0.01:
                        raise ValueError('Incompatible Station Location:' + netsta+' in Station List!')
                    else:
                        print 'Warning: Repeated Station:' +netsta+' in Station List!'
                        continue
                channels=[]
                if lon>180.:
                    lon-=360.
                for chan in chans:
                    channel=obspy.core.inventory.channel.Channel(code=chan, location_code='01', latitude=lat, longitude=lon,
                            elevation=0.0, depth=0.0)
                    channels.append(channel)
                station=obspy.core.inventory.station.Station(code=stacode, latitude=lat, longitude=lon, elevation=0.0,
                        site=site, channels=channels, total_number_of_channels = total_number_of_channels, creation_date = creation_date)
                network=obspy.core.inventory.network.Network(code=netcode, stations=[station])
                networks=[network]
                inv+=obspy.core.inventory.inventory.Inventory(networks=networks, source=source)
                staid_aux=netcode+'/'+stacode
                if ccflag!=None:
                    sta_info['xcorr']=ccflag
                self.add_auxiliary_data(data=np.array([]), data_type='StaInfo', path=staid_aux, parameters=sta_info)
        print 'Writing obspy inventory to ASDF dataset'
        self.add_stationxml(inv)
        print 'End writing obspy inventory to ASDF dataset'
        return 
    
    def read_stationtxt_ind(self, stafile, source='CIEI', chans=['BHZ', 'BHE', 'BHN'], s_ind=1, lon_ind=2, lat_ind=3, n_ind=0):
        """Read txt station list, column index can be changed
        """
        sta_info=sta_info_default.copy()
        with open(stafile, 'r') as f:
            Sta=[]
            site=obspy.core.inventory.util.Site(name='01')
            creation_date=obspy.core.utcdatetime.UTCDateTime(0)
            inv=obspy.core.inventory.inventory.Inventory(networks=[], source=source)
            total_number_of_channels=len(chans)
            for lines in f.readlines():
                lines=lines.split()
                stacode=lines[s_ind]
                lon=float(lines[lon_ind])
                lat=float(lines[lat_ind])
                netcode=lines[n_ind]
                netsta=netcode+'.'+stacode
                if Sta.__contains__(netsta):
                    index=Sta.index(netsta)
                    if abs(self[index].lon-lon) >0.01 and abs(self[index].lat-lat) >0.01:
                        raise ValueError('Incompatible Station Location:' + netsta+' in Station List!')
                    else:
                        print 'Warning: Repeated Station:' +netsta+' in Station List!'
                        continue
                channels=[]
                if lon>180.:
                    lon-=360.
                for chan in chans:
                    channel=obspy.core.inventory.channel.Channel(code=chan, location_code='01', latitude=lat, longitude=lon,
                            elevation=0.0, depth=0.0)
                    channels.append(channel)
                station=obspy.core.inventory.station.Station(code=stacode, latitude=lat, longitude=lon, elevation=0.0,
                        site=site, channels=channels, total_number_of_channels = total_number_of_channels, creation_date = creation_date)
                network=obspy.core.inventory.network.Network(code=netcode, stations=[station])
                networks=[network]
                inv+=obspy.core.inventory.inventory.Inventory(networks=networks, source=source)
                staid_aux=netcode+'/'+stacode
                self.add_auxiliary_data(data=np.array([]), data_type='StaInfo', path=staid_aux, parameters=sta_info)
        print 'Writing obspy inventory to ASDF dataset'
        self.add_stationxml(inv)
        print 'End writing obspy inventory to ASDF dataset'
        return 
    
    def get_limits_lonlat(self):
        """Get the geographical limits of the stations
        """
        staLst=self.waveforms.list()
        minlat=90.
        maxlat=-90.
        minlon=360.
        maxlon=0.
        for staid in staLst:
            lat, elv, lon=self.waveforms[staid].coordinates.values()
            if lon<0: lon+=360.
            minlat=min(lat, minlat)
            maxlat=max(lat, maxlat)
            minlon=min(lon, minlon)
            maxlon=max(lon, maxlon)
        print 'latitude range: ', minlat, '-', maxlat, 'longitude range:', minlon, '-', maxlon
        self.minlat=minlat; self.maxlat=maxlat; self.minlon=minlon; self.maxlon=maxlon
        return
            
    def _get_basemap(self, projection='lambert', geopolygons=None, resolution='i'):
        """Get basemap for plotting results
        """
        # fig=plt.figure(num=None, figsize=(12, 12), dpi=80, facecolor='w', edgecolor='k')
        try:
            minlon=self.minlon; maxlon=self.maxlon; minlat=self.minlat; maxlat=self.maxlat
        except AttributeError:
            self.get_limits_lonlat()
            minlon=self.minlon; maxlon=self.maxlon; minlat=self.minlat; maxlat=self.maxlat
        lat_centre = (maxlat+minlat)/2.0
        lon_centre = (maxlon+minlon)/2.0
        if projection=='merc':
            m=Basemap(projection='merc', llcrnrlat=minlat-5., urcrnrlat=maxlat+5., llcrnrlon=minlon-5.,
                      urcrnrlon=maxlon+5., lat_ts=20, resolution=resolution)
            m.drawparallels(np.arange(-80.0,80.0,5.0), labels=[1,0,0,1])
            m.drawmeridians(np.arange(-170.0,170.0,5.0), labels=[1,0,0,1])
            m.drawstates(color='g', linewidth=2.)
        elif projection=='global':
            m=Basemap(projection='ortho',lon_0=lon_centre, lat_0=lat_centre, resolution=resolution)
        elif projection=='regional_ortho':
            m1 = Basemap(projection='ortho', lon_0=minlon, lat_0=minlat, resolution='l')
            m = Basemap(projection='ortho', lon_0=minlon, lat_0=minlat, resolution=resolution,\
                llcrnrx=0., llcrnry=0., urcrnrx=m1.urcrnrx/mapfactor, urcrnry=m1.urcrnry/3.5)
            m.drawparallels(np.arange(-80.0,80.0,10.0), labels=[1,0,0,0],  linewidth=2,  fontsize=20)
            m.drawmeridians(np.arange(-170.0,170.0,10.0),  linewidth=2)
        elif projection=='lambert':
            distEW, az, baz=obspy.geodetics.gps2dist_azimuth(minlat, minlon, minlat, maxlon) # distance is in m
            distNS, az, baz=obspy.geodetics.gps2dist_azimuth(minlat, minlon, maxlat+2., minlon) # distance is in m
            m = Basemap(width=distEW, height=distNS, rsphere=(6378137.00,6356752.3142), resolution='l', projection='lcc',\
                lat_1=minlat, lat_2=maxlat, lon_0=lon_centre, lat_0=lat_centre+1)
            m.drawparallels(np.arange(-80.0,80.0,10.0), linewidth=1, dashes=[2,2], labels=[1,1,0,0], fontsize=15)
            m.drawmeridians(np.arange(-170.0,170.0,10.0), linewidth=1, dashes=[2,2], labels=[0,0,1,0], fontsize=15)
        m.drawcoastlines(linewidth=1.0)
        m.drawcountries(linewidth=1.)
        # m.fillcontinents(lake_color='#99ffff',zorder=0.2)
        m.drawmapboundary(fill_color="white")
        try:
            geopolygons.PlotPolygon(inbasemap=m)
        except:
            pass
        return m
    
    def plot_stations(self, projection='lambert', geopolygons=None, showfig=True):
        self.minlon=85; self.maxlon=125; self.minlat=25; self.maxlat=45
        staLst=self.waveforms.list()
        stalons=np.array([]); stalats=np.array([])
        for staid in staLst:
            stla, evz, stlo=self.waveforms[staid].coordinates.values()
            stalons=np.append(stalons, stlo); stalats=np.append(stalats, stla)
        m=self._get_basemap(projection=projection, geopolygons=geopolygons)
        m.etopo()
        # m.shadedrelief()
        stax, stay=m(stalons, stalats)
        m.plot(stax, stay, '^', markersize=3)
        # plt.title(str(self.period)+' sec', fontsize=20)
        if showfig: plt.show()
    
    def wsac_xcorr(self, netcode1, stacode1, netcode2, stacode2, chan1, chan2, outdir='.', pfx='COR'):
        """Write cross-correlation data from ASDF to sac file
        ==============================================================================
        Input Parameters:
        netcode1, stacode1, chan1   - network/station/channel name for station 1
        netcode2, stacode2, chan2   - network/station/channel name for station 2
        outdir                      - output directory
        pfx                         - prefix
        Output:
        e.g. outdir/COR/TA.G12A/COR_TA.G12A_BHT_TA.R21A_BHT.SAC
        ==============================================================================
        """
        subdset=self.auxiliary_data.NoiseXcorr[netcode1][stacode1][netcode2][stacode2][chan1][chan2]
        sta1=self.waveforms[netcode1+'.'+stacode1].StationXML.networks[0].stations[0]
        sta2=self.waveforms[netcode2+'.'+stacode2].StationXML.networks[0].stations[0]
        xcorr_sacheader=xcorr_sacheader_default.copy()
        xcorr_sacheader['kuser0']=netcode1
        xcorr_sacheader['kevnm']=stacode1
        xcorr_sacheader['knetwk']=netcode2
        xcorr_sacheader['kstnm']=stacode2
        xcorr_sacheader['kcmpnm']=chan1+chan2
        xcorr_sacheader['evla']=sta1.latitude
        xcorr_sacheader['evlo']=sta1.longitude
        xcorr_sacheader['stla']=sta2.latitude
        xcorr_sacheader['stlo']=sta2.longitude
        xcorr_sacheader['dist']=subdset.parameters['dist']
        xcorr_sacheader['az']=subdset.parameters['az']
        xcorr_sacheader['baz']=subdset.parameters['baz']
        xcorr_sacheader['b']=subdset.parameters['b']
        xcorr_sacheader['e']=subdset.parameters['e']
        xcorr_sacheader['delta']=subdset.parameters['delta']
        xcorr_sacheader['npts']=subdset.parameters['npts']
        xcorr_sacheader['user0']=subdset.parameters['stackday']
        sacTr=obspy.io.sac.sactrace.SACTrace(data=subdset.data.value, **xcorr_sacheader)
        if not os.path.isdir(outdir+'/'+pfx+'/'+netcode1+'.'+stacode1):
            os.makedirs(outdir+'/'+pfx+'/'+netcode1+'.'+stacode1)
        sacfname=outdir+'/'+pfx+'/'+netcode1+'.'+stacode1+'/'+ \
                pfx+'_'+netcode1+'.'+stacode1+'_'+chan1+'_'+netcode2+'.'+stacode2+'_'+chan2+'.SAC'
        sacTr.write(sacfname)
        return
    
    def wsac_xcorr_all(self, netcode1, stacode1, netcode2, stacode2, outdir='.', pfx='COR'):
        """Write all components of cross-correlation data from ASDF to sac file
        ==============================================================================
        Input Parameters:
        netcode1, stacode1  - network/station name for station 1
        netcode2, stacode2  - network/station name for station 2
        outdir              - output directory
        pfx                 - prefix
        Output:
        e.g. outdir/COR/TA.G12A/COR_TA.G12A_BHT_TA.R21A_BHT.SAC
        ==============================================================================
        """
        subdset=self.auxiliary_data.NoiseXcorr[netcode1][stacode1][netcode2][stacode2]
        channels1=subdset.list()
        channels2=subdset[channels1[0]].list()
        for chan1 in channels1:
            for chan2 in channels2:
                self.wsac_xcorr(netcode1=netcode1, stacode1=stacode1, netcode2=netcode2,
                    stacode2=stacode2, chan1=chan1, chan2=chan2, outdir=outdir, pfx=pfx)
        return
    
    def get_xcorr_trace(self, netcode1, stacode1, netcode2, stacode2, chan1, chan2):
        """Get one single cross-correlation trace
        """
        subdset=self.auxiliary_data.NoiseXcorr[netcode1][stacode1][netcode2][stacode2][chan1][chan2]
        evla, evz, evlo=self.waveforms[netcode1+'.'+stacode1].coordinates.values()
        stla, stz, stlo=self.waveforms[netcode2+'.'+stacode2].coordinates.values()
        tr=obspy.core.Trace()
        tr.data=subdset.data.value
        tr.stats.sac={}
        tr.stats.sac.evla=evla
        tr.stats.sac.evlo=evlo
        tr.stats.sac.stla=stla
        tr.stats.sac.stlo=stlo
        tr.stats.sac.kuser0=netcode1
        tr.stats.sac.kevnm=stacode1
        tr.stats.network=netcode2
        tr.stats.station=stacode2
        tr.stats.sac.kcmpnm=chan1+chan2
        tr.stats.sac.dist=subdset.parameters['dist']
        tr.stats.sac.az=subdset.parameters['az']
        tr.stats.sac.baz=subdset.parameters['baz']
        tr.stats.sac.b=subdset.parameters['b']
        tr.stats.sac.e=subdset.parameters['e']
        tr.stats.sac.user0=subdset.parameters['stackday']
        tr.stats.delta=subdset.parameters['delta']
        return tr
        
    def read_xcorr(self, datadir, pfx='COR', fnametype=2, inchannels=None, verbose=True):
        """Read cross-correlation data in ASDF database
        ===========================================================================================================
        Input Parameters:
        datadir                 - data directory
        pfx                     - prefix
        inchannels              - input channels, if None, will read channel information from obspy inventory
        fnametype               - input sac file name type
                                    =1: datadir/COR/G12A/COR_G12A_BHZ_R21A_BHZ.SAC
                                    =2: datadir/COR/G12A/COR_G12A_R21A.SAC
        -----------------------------------------------------------------------------------------------------------
        Output:
        ASDF path           : self.auxiliary_data.NoiseXcorr[netcode1][stacode1][netcode2][stacode2][chan1][chan2]
        ===========================================================================================================
        """
        staLst=self.waveforms.list()
        # main loop for station pairs
        if inchannels!=None:
            try:
                if not isinstance(inchannels[0], obspy.core.inventory.channel.Channel):
                    channels=[]
                    for inchan in inchannels:
                        channels.append(obspy.core.inventory.channel.Channel(code=inchan, location_code='01',
                                        latitude=0, longitude=0, elevation=0, depth=0) )
                else:
                    channels=inchannels
            except:
                inchannels=None
        for staid1 in staLst:
            for staid2 in staLst:
                netcode1, stacode1=staid1.split('.')
                netcode2, stacode2=staid2.split('.')
                if staid1 >= staid2:
                    continue
                if fnametype==2 and not os.path.isfile(datadir+'/'+pfx+'/'+staid1+'/'+pfx+'_'+staid1+'_'+staid2+'.SAC'):
                    continue
                if inchannels==None:
                    channels1=self.waveforms[staid1].StationXML.networks[0].stations[0].channels
                    channels2=self.waveforms[staid2].StationXML.networks[0].stations[0].channels
                else:
                    channels1=channels
                    channels2=channels
                skipflag=False
                for chan1 in channels1:
                    if skipflag:
                        break
                    for chan2 in channels2:
                        if fnametype==1:
                            fname=datadir+'/'+pfx+'/'+staid1+'/'+pfx+'_'+staid1+'_'+chan1.code+'_'+staid2+'_'+chan2.code+'.SAC'
                        elif fnametype==2:
                            fname=datadir+'/'+pfx+'/'+staid1+'/'+pfx+'_'+staid1+'_'+staid2+'.SAC'
                        try:
                            tr=obspy.core.read(fname)[0]
                        except IOError:
                            skipflag=True
                            break
                        # write cross-correlation header information
                        xcorr_header=xcorr_header_default.copy()
                        xcorr_header['b']=tr.stats.sac.b
                        xcorr_header['e']=tr.stats.sac.e
                        xcorr_header['netcode1']=netcode1
                        xcorr_header['netcode2']=netcode2
                        xcorr_header['stacode1']=stacode1
                        xcorr_header['stacode2']=stacode2
                        xcorr_header['npts']=tr.stats.npts
                        xcorr_header['delta']=tr.stats.delta
                        xcorr_header['stackday']=tr.stats.sac.user0
                        try:
                            xcorr_header['dist']=tr.stats.sac.dist
                            xcorr_header['az']=tr.stats.sac.az
                            xcorr_header['baz']=tr.stats.sac.baz
                        except AttributeError:
                            lon1=self.waveforms[staid1].StationXML.networks[0].stations[0].longitude
                            lat1=self.waveforms[staid1].StationXML.networks[0].stations[0].latitude
                            lon2=self.waveforms[staid2].StationXML.networks[0].stations[0].longitude
                            lat2=self.waveforms[staid2].StationXML.networks[0].stations[0].latitude
                            dist, az, baz=obspy.geodetics.gps2dist_azimuth(lat1, lon1, lat2, lon2)
                            dist=dist/1000.
                            xcorr_header['dist']=dist
                            xcorr_header['az']=az
                            xcorr_header['baz']=baz
                        staid_aux=netcode1+'/'+stacode1+'/'+netcode2+'/'+stacode2
                        xcorr_header['chan1']=chan1.code
                        xcorr_header['chan2']=chan2.code
                        self.add_auxiliary_data(data=tr.data, data_type='NoiseXcorr', path=staid_aux+'/'+chan1.code+'/'+chan2.code, parameters=xcorr_header)
                if verbose and not skipflag:
                    print 'reading xcorr data: '+netcode1+'.'+stacode1+'_'+netcode2+'.'+stacode2
        return
        
    def xcorr_stack(self, datadir, startyear, startmonth, endyear, endmonth, pfx='COR', outdir=None, inchannels=None, fnametype=1):
        """Stack cross-correlation data from monthly-stacked sac files
        ===========================================================================================================
        Input Parameters:
        datadir                 - data directory
        startyear, startmonth   - start date for stacking
        endyear, endmonth       - end date for stacking
        pfx                     - prefix
        outdir                  - output directory (None is not to save sac files)
        inchannels              - input channels, if None, will read channel information from obspy inventory
        fnametype               - input sac file name type
                                    =1: datadir/COR/G12A/COR_G12A_BHZ_R21A_BHZ.SAC
                                    =2: datadir/COR/G12A/COR_G12A_R21A.SAC
        -----------------------------------------------------------------------------------------------------------
        Output:
        ASDF path           : self.auxiliary_data.NoiseXcorr[netcode1][stacode1][netcode2][stacode2][chan1][chan2]
        sac file(optional)  : outdir/COR/TA.G12A/COR_TA.G12A_BHT_TA.R21A_BHT.SAC
        ===========================================================================================================
        """
        # prepare year/month list for stacking
        utcdate=obspy.core.utcdatetime.UTCDateTime(startyear, startmonth, 1)
        ylst=np.array([], dtype=int)
        mlst=np.array([], dtype=int)
        while (utcdate.year<endyear or (utcdate.year<=endyear and utcdate.month<=endmonth) ):
            ylst=np.append(ylst, utcdate.year)
            mlst=np.append(mlst, utcdate.month)
            try:
                utcdate.month+=1
            except ValueError:
                utcdate.year+=1
                utcdate.month=1
        mnumb=mlst.size
        # determine channels if inchannels is specified
        if inchannels!=None:
            try:
                if not isinstance(inchannels[0], obspy.core.inventory.channel.Channel):
                    channels=[]
                    for inchan in inchannels:
                        channels.append(obspy.core.inventory.channel.Channel(code=inchan, location_code='01',
                                        latitude=0, longitude=0, elevation=0, depth=0) )
                else:
                    channels=inchannels
            except:
                inchannels=None
        if inchannels==None:
            fnametype==1
        else:
            if len(channels)!=1:
                fnametype==1
        staLst=self.waveforms.list()
        # main loop for station pairs
        for staid1 in staLst:
            for staid2 in staLst:
                netcode1, stacode1=staid1.split('.')
                netcode2, stacode2=staid2.split('.')
                if stacode1 >= stacode2:
                    continue
                stackedST=[]
                cST=[]
                initflag=True
                if inchannels==None:
                    channels1=self.waveforms[staid1].StationXML.networks[0].stations[0].channels
                    channels2=self.waveforms[staid2].StationXML.networks[0].stations[0].channels
                else:
                    channels1=channels
                    channels2=channels
                for im in xrange(mnumb):
                    skipflag=False
                    for chan1 in channels1:
                        if skipflag:
                            break
                        for chan2 in channels2:
                            month=monthdict[mlst[im]]
                            yrmonth=str(ylst[im])+'.'+month
                            if fnametype==1:
                                fname=datadir+'/'+yrmonth+'/'+pfx+'/'+stacode1+'/'+pfx+'_'+stacode1+'_'+chan1.code+'_'+stacode2+'_'+chan2.code+'.SAC'
                            elif fnametype==2:
                                fname=datadir+'/'+yrmonth+'/'+pfx+'/'+stacode1+'/'+pfx+'_'+stacode1+'_'+stacode2+'.SAC'
                            if not os.path.isfile(fname):
                                skipflag=True
                                break
                            try:
                                tr=obspy.core.read(fname)[0]
                            except TypeError:
                                warnings.warn('Unable to read SAC for: ' + stacode1 +'_'+stacode2 +' Month: '+yrmonth, UserWarning, stacklevel=1)
                                skipflag=True
                            if np.isnan(tr.data).any() or abs(tr.data.max())>1e20:
                                warnings.warn('NaN monthly SAC for: ' + stacode1 +'_'+stacode2 +' Month: '+yrmonth, UserWarning, stacklevel=1)
                                skipflag=True
                                break
                            cST.append(tr)
                    if len(cST)!=len(channels1)*len(channels2) or skipflag:
                        cST=[]
                        continue
                    if initflag:
                        stackedST=copy.deepcopy(cST)
                        initflag=False
                    else:
                        for itr in xrange(len(cST)):
                            mtr=cST[itr]
                            stackedST[itr].data+=mtr.data
                            stackedST[itr].stats.sac.user0+=mtr.stats.sac.user0
                    cST=[]
                if len(stackedST)==len(channels1)*len(channels2):
                    print 'Finished Stacking for:'+netcode1+'.'+stacode1+'_'+netcode2+'.'+stacode2
                    # create sac output directory 
                    if outdir!=None:
                        if not os.path.isdir(outdir+'/'+pfx+'/'+netcode1+'.'+stacode1):
                            os.makedirs(outdir+'/'+pfx+'/'+netcode1+'.'+stacode1)
                    # write cross-correlation header information
                    xcorr_header=xcorr_header_default.copy()
                    xcorr_header['b']=stackedST[0].stats.sac.b
                    xcorr_header['e']=stackedST[0].stats.sac.e
                    xcorr_header['netcode1']=netcode1
                    xcorr_header['netcode2']=netcode2
                    xcorr_header['stacode1']=stacode1
                    xcorr_header['stacode2']=stacode2
                    xcorr_header['npts']=stackedST[0].stats.npts
                    xcorr_header['delta']=stackedST[0].stats.delta
                    xcorr_header['stackday']=stackedST[0].stats.sac.user0
                    try:
                        xcorr_header['dist']=stackedST[0].stats.sac.dist
                        xcorr_header['az']=stackedST[0].stats.sac.az
                        xcorr_header['baz']=stackedST[0].stats.sac.baz
                    except AttributeError:
                        lon1=self.waveforms[staid1].StationXML.networks[0].stations[0].longitude
                        lat1=self.waveforms[staid1].StationXML.networks[0].stations[0].latitude
                        lon2=self.waveforms[staid2].StationXML.networks[0].stations[0].longitude
                        lat2=self.waveforms[staid2].StationXML.networks[0].stations[0].latitude
                        dist, az, baz=obspy.geodetics.gps2dist_azimuth(lat1, lon1, lat2, lon2)
                        dist=dist/1000.
                        xcorr_header['dist']=dist
                        xcorr_header['az']=az
                        xcorr_header['baz']=baz
                    staid_aux=netcode1+'/'+stacode1+'/'+netcode2+'/'+stacode2
                    i=0
                    for chan1 in channels1:
                        for chan2 in channels2:
                            stackedTr=stackedST[i]
                            if outdir!=None:
                                outfname=outdir+'/'+pfx+'/'+netcode1+'.'+stacode1+'/'+ \
                                    pfx+'_'+netcode1+'.'+stacode1+'_'+chan1.code+'_'+netcode2+'.'+stacode2+'_'+chan2.code+'.SAC'
                                stackedTr.write(outfname,format='SAC')
                            xcorr_header['chan1']=chan1.code
                            xcorr_header['chan2']=chan2.code
                            self.add_auxiliary_data(data=stackedTr.data, data_type='NoiseXcorr', path=staid_aux+'/'+chan1.code+'/'+chan2.code, parameters=xcorr_header)
                            i+=1
        return
    
    def xcorr_stack_mp(self, datadir, outdir, startyear, startmonth, endyear, endmonth,
                    pfx='COR', inchannels=None, fnametype=1, subsize=1000, deletesac=True, nprocess=4):
        """Stack cross-correlation data from monthly-stacked sac files with multiprocessing
        ===========================================================================================================
        Input Parameters:
        datadir                 - data directory
        outdir                  - output directory 
        startyear, startmonth   - start date for stacking
        endyear, endmonth       - end date for stacking
        pfx                     - prefix
        inchannels              - input channels, if None, will read channel information from obspy inventory
        fnametype               - input sac file name type
                                    =1: datadir/COR/G12A/COR_G12A_BHZ_R21A_BHZ.SAC
                                    =2: datadir/COR/G12A/COR_G12A_R21A.SAC
        subsize                 - subsize of processing list, use to prevent lock in multiprocessing process
        deletesac               - delete output sac files
        nprocess                - number of processes
        -----------------------------------------------------------------------------------------------------------
        Output:
        ASDF path           : self.auxiliary_data.NoiseXcorr[netcode1][stacode1][netcode2][stacode2][chan1][chan2]
        sac file(optional)  : outdir/COR/TA.G12A/COR_TA.G12A_BHT_TA.R21A_BHT.SAC
        ===========================================================================================================
        """
        utcdate=obspy.core.utcdatetime.UTCDateTime(startyear, startmonth, 1)
        ylst=np.array([], dtype=int)
        mlst=np.array([], dtype=int)
        print 'Preparing data for stacking'
        while (utcdate.year<endyear or (utcdate.year<=endyear and utcdate.month<=endmonth) ):
            ylst=np.append(ylst, utcdate.year)
            mlst=np.append(mlst, utcdate.month)
            try:
                utcdate.month+=1
            except ValueError:
                utcdate.year+=1
                utcdate.month=1
        mnumb=mlst.size
        staLst=self.waveforms.list()
        if inchannels!=None:
            try:
                if not isinstance(inchannels[0], obspy.core.inventory.channel.Channel):
                    channels=[]
                    for inchan in inchannels:
                        channels.append(obspy.core.inventory.channel.Channel(code=inchan, location_code='01',
                                        latitude=0, longitude=0, elevation=0, depth=0) )
                else:
                    channels=inchannels
            except:
                inchannels=None
        if inchannels==None:
            fnametype==1
        else:
            if len(channels)!=1:
                fnametype==1
        stapairInvLst=[]
        for staid1 in staLst:
            if not os.path.isdir(outdir+'/'+pfx+'/'+staid1):
                os.makedirs(outdir+'/'+pfx+'/'+staid1)
            for staid2 in staLst:
                netcode1, stacode1=staid1.split('.')
                netcode2, stacode2=staid2.split('.')
                if stacode1 >= stacode2:
                    continue
                inv = self.waveforms[staid1].StationXML + self.waveforms[staid2].StationXML
                if inchannels!=None:
                    inv.networks[0].stations[0].channels=channels
                    inv.networks[1].stations[0].channels=channels
                stapairInvLst.append(inv) 
        print 'Start multiprocessing stacking !'
        if len(stapairInvLst) > subsize:
            Nsub = int(len(stapairInvLst)/subsize)
            for isub in xrange(Nsub):
                print isub,'in',Nsub
                cstapairs=stapairInvLst[isub*subsize:(isub+1)*subsize]
                STACKING = partial(stack4mp, datadir=datadir, outdir=outdir, ylst=ylst, mlst=mlst, pfx=pfx, fnametype=fnametype)
                pool = multiprocessing.Pool(processes=nprocess)
                pool.map(STACKING, cstapairs) #make our results with a map call
                pool.close() #we are not adding any more processes
                pool.join() #tell it to wait until all threads are done before going on
            cstapairs=stapairInvLst[(isub+1)*subsize:]
            STACKING = partial(stack4mp, datadir=datadir, outdir=outdir, ylst=ylst, mlst=mlst, pfx=pfx, fnametype=fnametype)
            pool = multiprocessing.Pool(processes=nprocess)
            pool.map(STACKING, cstapairs) 
            pool.close() 
            pool.join() 
        else:
            STACKING = partial(stack4mp, datadir=datadir, outdir=outdir, ylst=ylst, mlst=mlst, pfx=pfx, fnametype=fnametype)
            pool = multiprocessing.Pool(processes=nprocess)
            pool.map(STACKING, stapairInvLst) 
            pool.close() 
            pool.join() 
        print 'End of multiprocessing stacking !'
        print 'Reading data into ASDF database'
        for inv in stapairInvLst:
            channels1=inv.networks[0].stations[0].channels
            netcode1=inv.networks[0].code
            stacode1=inv.networks[0].stations[0].code
            channels2=inv.networks[1].stations[0].channels
            netcode2=inv.networks[1].code
            stacode2=inv.networks[1].stations[0].code
            skipflag=False
            xcorr_header=xcorr_header_default.copy()
            xcorr_header['netcode1']=netcode1
            xcorr_header['netcode2']=netcode2
            xcorr_header['stacode1']=stacode1
            xcorr_header['stacode2']=stacode2
            staid_aux=netcode1+'/'+stacode1+'/'+netcode2+'/'+stacode2
            for chan1 in channels1:
                if skipflag:
                    break
                for chan2 in channels2:
                    sacfname=outdir+'/'+pfx+'/'+netcode1+'.'+stacode1+'/'+ \
                        pfx+'_'+netcode1+'.'+stacode1+'_'+chan1.code+'_'+netcode2+'.'+stacode2+'_'+chan2.code+'.SAC'
                    try:
                        tr=obspy.read(sacfname)[0]
                        # cross-correlation header 
                        xcorr_header['b']=tr.stats.sac.b
                        xcorr_header['e']=tr.stats.sac.e
                        xcorr_header['npts']=tr.stats.npts
                        xcorr_header['delta']=tr.stats.delta
                        xcorr_header['stackday']=tr.stats.sac.user0
                        try:
                            xcorr_header['dist']=tr.stats.sac.dist
                            xcorr_header['az']=tr.stats.sac.az
                            xcorr_header['baz']=tr.stats.sac.baz
                        except AttributeError:
                            lon1=inv.networks[0].stations[0].longitude
                            lat1=inv.networks[0].stations[0].latitude
                            lon2=inv.networks[1].stations[0].longitude
                            lat2=inv.networks[1].stations[0].latitude
                            dist, az, baz=obspy.geodetics.gps2dist_azimuth(lat1, lon1, lat2, lon2)
                            dist=dist/1000.
                            xcorr_header['dist']=dist
                            xcorr_header['az']=az
                            xcorr_header['baz']=baz
                        xcorr_header['chan1']=chan1.code
                        xcorr_header['chan2']=chan2.code
                        self.add_auxiliary_data(data=tr.data, data_type='NoiseXcorr', path=staid_aux+'/'+chan1.code+'/'+chan2.code, parameters=xcorr_header)
                    except IOError:
                        skipflag=True
                        break
        if deletesac:
            shutil.rmtree(outdir+'/'+pfx)
        print 'End read data into ASDF database'
        return
                    
    def xcorr_rotation(self, outdir=None, pfx='COR'):
        """Rotate cross-correlation data 
        ===========================================================================================================
        Input Parameters:
        outdir                  - output directory for sac files (None is not to write)
        pfx                     - prefix
        -----------------------------------------------------------------------------------------------------------
        Output:
        ASDF path           : self.auxiliary_data.NoiseXcorr[netcode1][stacode1][netcode2][stacode2][chan1][chan2]
        sac file(optional)  : outdir/COR/TA.G12A/COR_TA.G12A_BHT_TA.R21A_BHT.SAC
        ===========================================================================================================
        """
        staLst=self.waveforms.list()
        for staid1 in staLst:
            for staid2 in staLst:
                netcode1, stacode1=staid1.split('.')
                netcode2, stacode2=staid2.split('.')
                if stacode1 >= stacode2:
                    continue
                chan1E=None; chan1N=None; chan1Z=None; chan2E=None; chan2N=None; chan2Z=None
                try:
                    channels1=self.auxiliary_data.NoiseXcorr[netcode1][stacode1][netcode2][stacode2].list()
                    channels2=self.auxiliary_data.NoiseXcorr[netcode1][stacode1][netcode2][stacode2][channels1[0]].list()
                    cpfx1=channels1[0][:2]
                    cpfx2=channels2[0][:2]
                    for chan in channels1:
                        if chan[2]=='E': chan1E=chan
                        if chan[2]=='N': chan1N=chan
                        if chan[2]=='Z': chan1Z=chan
                    for chan in channels2:
                        if chan[2]=='E': chan2E=chan
                        if chan[2]=='N': chan2N=chan
                        if chan[2]=='Z': chan2Z=chan
                except AttributeError:
                    continue
                subdset=self.auxiliary_data.NoiseXcorr[netcode1][stacode1][netcode2][stacode2]
                if chan1E==None or chan1N==None or chan2E==None or chan2N==None:
                    continue
                if chan1Z==None or chan2Z==None:
                    print 'Do rotation(RT) for:'+netcode1+'.'+stacode1+' and '+netcode2+'.'+stacode2
                else:
                    print 'Do rotation(RTZ) for:'+netcode1+'.'+stacode1+' and '+netcode2+'.'+stacode2
                dsetEE=subdset[chan1E][chan2E]
                dsetEN=subdset[chan1E][chan2N]
                dsetNE=subdset[chan1N][chan2E]
                dsetNN=subdset[chan1N][chan2N]
                temp_header=dsetEE.parameters.copy()
                chan1R=cpfx1+'R'; chan1T=cpfx1+'T'; chan2R=cpfx2+'R'; chan2T=cpfx2+'T'
                theta=temp_header['az']
                psi=temp_header['baz']
                Ctheta=np.cos(np.pi*theta/180.)
                Stheta=np.sin(np.pi*theta/180.)
                Cpsi=np.cos(np.pi*psi/180.)
                Spsi=np.sin(np.pi*psi/180.)
                tempTT=-Ctheta*Cpsi*dsetEE.data.value+Ctheta*Spsi*dsetEN.data.value - \
                    Stheta*Spsi*dsetNN.data.value + Stheta*Cpsi*dsetNE.data.value
                
                tempRR=- Stheta*Spsi*dsetEE.data.value - Stheta*Cpsi*dsetEN.data.value \
                    - Ctheta*Cpsi*dsetNN.data.value - Ctheta*Spsi*dsetNE.data.value
                
                tempTR=-Ctheta*Spsi*dsetEE.data.value - Ctheta*Cpsi*dsetEN.data.value  \
                    + Stheta*Cpsi*dsetNN.data.value + Stheta*Spsi*dsetNE.data.value
                
                tempRT=-Stheta*Cpsi*dsetEE.data.value +Stheta*Spsi*dsetEN.data.value \
                    + Ctheta*Spsi*dsetNN.data.value - Ctheta*Cpsi*dsetNE.data.value
                staid_aux=netcode1+'/'+stacode1+'/'+netcode2+'/'+stacode2
                temp_header['chan1']=chan1T; temp_header['chan2']=chan2T
                self.add_auxiliary_data(data=tempTT, data_type='NoiseXcorr', path=staid_aux+'/'+chan1T+'/'+chan2T, parameters=temp_header)
                
                temp_header['chan1']=chan1R; temp_header['chan2']=chan2R
                self.add_auxiliary_data(data=tempRR, data_type='NoiseXcorr', path=staid_aux+'/'+chan1R+'/'+chan2R, parameters=temp_header)
                
                temp_header['chan1']=chan1T; temp_header['chan2']=chan2R
                self.add_auxiliary_data(data=tempTR, data_type='NoiseXcorr', path=staid_aux+'/'+chan1T+'/'+chan2R, parameters=temp_header)
                
                temp_header['chan1']=chan1R; temp_header['chan2']=chan2T
                self.add_auxiliary_data(data=tempRT, data_type='NoiseXcorr', path=staid_aux+'/'+chan1R+'/'+chan2T, parameters=temp_header)
                # write to sac files
                if outdir!=None:
                    self.wsac_xcorr(netcode1=netcode1, stacode1=stacode1, netcode2=netcode2,
                            stacode2=stacode2, chan1=chan1T, chan2=chan2T, outdir=outdir, pfx=pfx)
                    self.wsac_xcorr(netcode1=netcode1, stacode1=stacode1, netcode2=netcode2,
                            stacode2=stacode2, chan1=chan1R, chan2=chan2R, outdir=outdir, pfx=pfx)
                    self.wsac_xcorr(netcode1=netcode1, stacode1=stacode1, netcode2=netcode2,
                            stacode2=stacode2, chan1=chan1T, chan2=chan2R, outdir=outdir, pfx=pfx)
                    self.wsac_xcorr(netcode1=netcode1, stacode1=stacode1, netcode2=netcode2,
                            stacode2=stacode2, chan1=chan1R, chan2=chan2T, outdir=outdir, pfx=pfx)
                # RTZ rotation
                if chan1Z!=None and chan2Z!=None:
                    dsetEZ=subdset[chan1E][chan2Z]
                    dsetZE=subdset[chan1Z][chan2E]
                    dsetNZ=subdset[chan1N][chan2Z]
                    dsetZN=subdset[chan1Z][chan2N]
                    tempRZ = Ctheta*dsetNZ.data.value + Stheta*dsetEZ.data.value
                    tempZR = - Cpsi*dsetZN.data.value -Spsi*dsetZE.data.value
                    tempTZ = -Stheta*dsetNZ.data.value + Ctheta*dsetEZ.data.value
                    tempZT =  Spsi*dsetZN.data.value - Cpsi*dsetZE.data.value
                    temp_header['chan1']=chan1R; temp_header['chan2']=chan2Z
                    self.add_auxiliary_data(data=tempRZ, data_type='NoiseXcorr', path=staid_aux+'/'+chan1R+'/'+chan2Z, parameters=temp_header)
                    temp_header['chan1']=chan1Z; temp_header['chan2']=chan2R
                    self.add_auxiliary_data(data=tempZR, data_type='NoiseXcorr', path=staid_aux+'/'+chan1Z+'/'+chan2R, parameters=temp_header)
                    temp_header['chan1']=chan1T; temp_header['chan2']=chan2Z
                    self.add_auxiliary_data(data=tempTZ, data_type='NoiseXcorr', path=staid_aux+'/'+chan1T+'/'+chan2Z, parameters=temp_header)
                    temp_header['chan1']=chan1Z; temp_header['chan2']=chan2T
                    self.add_auxiliary_data(data=tempZT, data_type='NoiseXcorr', path=staid_aux+'/'+chan1Z+'/'+chan2T, parameters=temp_header)
                    # write to sac files
                    if outdir!=None:
                        self.wsac_xcorr(netcode1=netcode1, stacode1=stacode1, netcode2=netcode2,
                                stacode2=stacode2, chan1=chan1R, chan2=chan2Z, outdir=outdir, pfx=pfx)                        
                        self.wsac_xcorr(netcode1=netcode1, stacode1=stacode1, netcode2=netcode2,
                                stacode2=stacode2, chan1=chan1Z, chan2=chan2R, outdir=outdir, pfx=pfx)
                        self.wsac_xcorr(netcode1=netcode1, stacode1=stacode1, netcode2=netcode2,
                                stacode2=stacode2, chan1=chan1T, chan2=chan2Z, outdir=outdir, pfx=pfx)
                        self.wsac_xcorr(netcode1=netcode1, stacode1=stacode1, netcode2=netcode2,
                                stacode2=stacode2, chan1=chan1Z, chan2=chan2T, outdir=outdir, pfx=pfx)
        return
    
    def xcorr_prephp(self, outdir, mapfile='./MAPS/smpkolya_phv'):
        """
        Generate predicted phase velocity dispersion curves for cross-correlation pairs
        ====================================================================================
        Input Parameters:
        outdir  - output directory
        mapfile - phase velocity maps
        ------------------------------------------------------------------------------------
        Input format:
        prephaseEXE pathfname mapfile perlst staname
        
        Output format:
        outdirL(outdirR)/evid.staid.pre
        ====================================================================================
        """
        staLst=self.waveforms.list()
        for evid in staLst:
            evnetcode, evstacode=evid.split('.')
            evla, evz, evlo=self.waveforms[evid].coordinates.values()
            pathfname=evid+'_pathfile'
            prephaseEXE='./mhr_grvel_predict/lf_mhr_predict_earth'
            perlst='./mhr_grvel_predict/perlist_phase'
            if not os.path.isfile(prephaseEXE):
                print 'lf_mhr_predict_earth executable does not exist!'
                return
            if not os.path.isfile(perlst):
                print 'period list does not exist!'
                return
            with open(pathfname,'w') as f:
                ista=0
                for station_id in staLst:
                    stacode=station_id.split('.')[1]
                    if evid >= station_id:
                        continue
                    stla, stz, stlo=self.waveforms[station_id].coordinates.values()
                    if ( abs(stlo-evlo) < 0.1 and abs(stla-evla)<0.1 ):
                        continue
                    ista=ista+1
                    f.writelines('%5d%5d %15s %15s %10.5f %10.5f %10.5f %10.5f \n'
                            %(1, ista, evid, station_id, evla, evlo, stla, stlo ))
            call([prephaseEXE, pathfname, mapfile, perlst, evid])
            os.remove(pathfname)
            outdirL=outdir+'_L'
            outdirR=outdir+'_R'
            if not os.path.isdir(outdirL):
                os.makedirs(outdirL)
            if not os.path.isdir(outdirR):
                os.makedirs(outdirR)
            fout = open(evid+'_temp','wb')
            for l1 in open('PREDICTION_L'+'_'+evid):
                l2 = l1.rstrip().split()
                if (len(l2)>8):
                    fout.close()
                    outname = outdirL + "/%s.%s.pre" % (l2[3],l2[4])
                    fout = open(outname,"w")
                elif (len(l2)>7):
                    fout.close()
                    outname = outdirL + "/%s.%s.pre" % (l2[2],l2[3])
                    fout = open(outname,"w")                
                else:
                    fout.write("%g %g\n" % (float(l2[0]),float(l2[1])))
            for l1 in open('PREDICTION_R'+'_'+evid):
                l2 = l1.rstrip().split()
                if (len(l2)>8):
                    fout.close()
                    outname = outdirR + "/%s.%s.pre" % (l2[3],l2[4])
                    fout = open(outname,"w")
                elif (len(l2)>7):
                    fout.close()
                    outname = outdirR + "/%s.%s.pre" % (l2[2],l2[3])
                    fout = open(outname,"w")         
                else:
                    fout.write("%g %g\n" % (float(l2[0]),float(l2[1])))
            fout.close()
            os.remove(evid+'_temp')
            os.remove('PREDICTION_L'+'_'+evid)
            os.remove('PREDICTION_R'+'_'+evid)
        return
    
    def xcorr_aftan(self, channel='ZZ', tb=0., outdir=None, inftan=pyaftan.InputFtanParam(), basic1=True, basic2=True, \
            pmf1=True, pmf2=True, verbose=True, prephdir=None, f77=True, pfx='DISP'):
        """ aftan analysis of cross-correlation data 
        =======================================================================================
        Input Parameters:
        channel     - channel pair for aftan analysis(e.g. 'ZZ', 'TT', 'ZR', 'RZ'...)
        tb          - begin time (default = 0.0)
        outdir      - directory for output disp txt files (default = None, no txt output)
        inftan      - input aftan parameters
        basic1      - save basic aftan results or not
        basic2      - save basic aftan results(with jump correction) or not
        pmf1        - save pmf aftan results or not
        pmf2        - save pmf aftan results(with jump correction) or not
        prephdir    - directory for predicted phase velocity dispersion curve
        f77         - use aftanf77 or not
        pfx         - prefix for output txt DISP files
        ---------------------------------------------------------------------------------------
        Output:
        self.auxiliary_data.DISPbasic1, self.auxiliary_data.DISPbasic2,
        self.auxiliary_data.DISPpmf1, self.auxiliary_data.DISPpmf2
        =======================================================================================
        """
        print 'Start aftan analysis!'
        staLst=self.waveforms.list()
        for staid1 in staLst:
            for staid2 in staLst:
                netcode1, stacode1=staid1.split('.')
                netcode2, stacode2=staid2.split('.')
                if staid1 >= staid2: continue
                try:
                    channels1=self.auxiliary_data.NoiseXcorr[netcode1][stacode1][netcode2][stacode2].list()
                    channels2=self.auxiliary_data.NoiseXcorr[netcode1][stacode1][netcode2][stacode2][channels1[0]].list()
                    for chan in channels1:
                        if chan[2]==channel[0]: chan1=chan
                    for chan in channels2:
                        if chan[2]==channel[1]: chan2=chan
                except KeyError:
                    continue
                try:
                    tr=self.get_xcorr_trace(netcode1, stacode1, netcode2, stacode2, chan1, chan2)
                except NameError:
                    print netcode1+'.'+stacode1+'_'+netcode2+'.'+stacode2+'_'+channel+' not exists!'
                    continue
                aftanTr=pyaftan.aftantrace(tr.data, tr.stats)
                if abs(aftanTr.stats.sac.b+aftanTr.stats.sac.e)<aftanTr.stats.delta:
                    aftanTr.makesym()
                if prephdir !=None:
                    phvelname = prephdir + "/%s.%s.pre" %(netcode1+'.'+stacode1, netcode2+'.'+stacode2)
                else:
                    phvelname =''
                if f77:
                    aftanTr.aftanf77(pmf=inftan.pmf, piover4=inftan.piover4, vmin=inftan.vmin, vmax=inftan.vmax, tmin=inftan.tmin, tmax=inftan.tmax,
                        tresh=inftan.tresh, ffact=inftan.ffact, taperl=inftan.taperl, snr=inftan.snr, fmatch=inftan.fmatch, nfin=inftan.nfin,
                            npoints=inftan.npoints, perc=inftan.perc, phvelname=phvelname)
                else:
                    aftanTr.aftan(pmf=inftan.pmf, piover4=inftan.piover4, vmin=inftan.vmin, vmax=inftan.vmax, tmin=inftan.tmin, tmax=inftan.tmax,
                        tresh=inftan.tresh, ffact=inftan.ffact, taperl=inftan.taperl, snr=inftan.snr, fmatch=inftan.fmatch, nfin=inftan.nfin,
                            npoints=inftan.npoints, perc=inftan.perc, phvelname=phvelname)
                if verbose:
                    print 'aftan analysis for: ' + netcode1+'.'+stacode1+'_'+netcode2+'.'+stacode2+'_'+channel
                aftanTr.get_snr(ffact=inftan.ffact) # SNR analysis
                staid_aux=netcode1+'/'+stacode1+'/'+netcode2+'/'+stacode2+'/'+channel
                # save aftan results to ASDF dataset
                if basic1:
                    parameters={'Tc': 0, 'To': 1, 'Vgr': 2, 'Vph': 3, 'ampdb': 4, 'dis': 5, 'snrdb': 6, 'mhw': 7, 'amp': 8, 'Np': aftanTr.ftanparam.nfout1_1}
                    self.add_auxiliary_data(data=aftanTr.ftanparam.arr1_1, data_type='DISPbasic1', path=staid_aux, parameters=parameters)
                if basic2:
                    parameters={'Tc': 0, 'To': 1, 'Vgr': 2, 'Vph': 3, 'ampdb': 4, 'snrdb': 5, 'mhw': 6, 'amp': 7, 'Np': aftanTr.ftanparam.nfout2_1}
                    self.add_auxiliary_data(data=aftanTr.ftanparam.arr2_1, data_type='DISPbasic2', path=staid_aux, parameters=parameters)
                if inftan.pmf:
                    if pmf1:
                        parameters={'Tc': 0, 'To': 1, 'Vgr': 2, 'Vph': 3, 'ampdb': 4, 'dis': 5, 'snrdb': 6, 'mhw': 7, 'amp': 8, 'Np': aftanTr.ftanparam.nfout1_2}
                        self.add_auxiliary_data(data=aftanTr.ftanparam.arr1_2, data_type='DISPpmf1', path=staid_aux, parameters=parameters)
                    if pmf2:
                        parameters={'Tc': 0, 'To': 1, 'Vgr': 2, 'Vph': 3, 'ampdb': 4, 'snrdb': 5, 'mhw': 6, 'amp': 7, 'snr':8, 'Np': aftanTr.ftanparam.nfout2_2}
                        self.add_auxiliary_data(data=aftanTr.ftanparam.arr2_2, data_type='DISPpmf2', path=staid_aux, parameters=parameters)
                if outdir != None:
                    if not os.path.isdir(outdir+'/'+pfx+'/'+staid1):
                        os.makedirs(outdir+'/'+pfx+'/'+staid1)
                    foutPR=outdir+'/'+pfx+'/'+netcode1+'.'+stacode1+'/'+ \
                                    pfx+'_'+netcode1+'.'+stacode1+'_'+chan1+'_'+netcode2+'.'+stacode2+'_'+chan2+'.SAC'
                    aftanTr.ftanparam.writeDISP(foutPR)
        print 'End aftan analysis!'
        return
               
    def xcorr_aftan_mp(self, outdir, channel='ZZ', tb=0., inftan=pyaftan.InputFtanParam(), basic1=True, basic2=True,
            pmf1=True, pmf2=True, verbose=True, prephdir=None, f77=True, pfx='DISP', subsize=1000, deletedisp=True, nprocess=None):
        """ aftan analysis of cross-correlation data with multiprocessing
        =======================================================================================
        Input Parameters:
        channel     - channel pair for aftan analysis(e.g. 'ZZ', 'TT', 'ZR', 'RZ'...)
        tb          - begin time (default = 0.0)
        outdir      - directory for output disp binary files
        inftan      - input aftan parameters
        basic1      - save basic aftan results or not
        basic2      - save basic aftan results(with jump correction) or not
        pmf1        - save pmf aftan results or not
        pmf2        - save pmf aftan results(with jump correction) or not
        prephdir    - directory for predicted phase velocity dispersion curve
        f77         - use aftanf77 or not
        pfx         - prefix for output txt DISP files
        subsize     - subsize of processing list, use to prevent lock in multiprocessing process
        deletedisp  - delete output dispersion files or not
        nprocess    - number of processes
        ---------------------------------------------------------------------------------------
        Output:
        self.auxiliary_data.DISPbasic1, self.auxiliary_data.DISPbasic2,
        self.auxiliary_data.DISPpmf1, self.auxiliary_data.DISPpmf2
        =======================================================================================
        """
        print 'Preparing data for aftan analysis !'
        staLst=self.waveforms.list()
        inputStream=[]
        for staid1 in staLst:
            if not os.path.isdir(outdir+'/'+pfx+'/'+staid1):
                os.makedirs(outdir+'/'+pfx+'/'+staid1)
            for staid2 in staLst:
                netcode1, stacode1=staid1.split('.')
                netcode2, stacode2=staid2.split('.')
                if staid1 >= staid2: continue
                try:
                    channels1=self.auxiliary_data.NoiseXcorr[netcode1][stacode1][netcode2][stacode2].list()
                    channels2=self.auxiliary_data.NoiseXcorr[netcode1][stacode1][netcode2][stacode2][channels1[0]].list()
                    for chan in channels1:
                        if chan[2]==channel[0]: chan1=chan
                    for chan in channels2:
                        if chan[2]==channel[1]: chan2=chan
                except KeyError:
                    continue
                try:
                    tr=self.get_xcorr_trace(netcode1, stacode1, netcode2, stacode2, chan1, chan2)
                except NameError:
                    print netcode1+'.'+stacode1+'_'+netcode2+'.'+stacode2+'_'+channel+' not exists!'
                    continue
                if verbose:
                    print 'Preparing aftan data: '+ netcode1+'.'+stacode1+'_'+netcode2+'.'+stacode2+'_'+channel
                aftanTr=pyaftan.aftantrace(tr.data, tr.stats)
                inputStream.append(aftanTr)
        print 'Start multiprocessing aftan analysis !'
        if len(inputStream) > subsize:
            Nsub = int(len(inputStream)/subsize)
            for isub in xrange(Nsub):
                print 'Subset:', isub,'in',Nsub,'sets'
                cstream=inputStream[isub*subsize:(isub+1)*subsize]
                AFTAN = partial(aftan4mp, outdir=outdir, inftan=inftan, prephdir=prephdir, f77=f77, pfx=pfx)
                pool = multiprocessing.Pool(processes=nprocess)
                pool.map(AFTAN, cstream) #make our results with a map call
                pool.close() #we are not adding any more processes
                pool.join() #tell it to wait until all threads are done before going on
            cstream=inputStream[(isub+1)*subsize:]
            AFTAN = partial(aftan4mp, outdir=outdir, inftan=inftan, prephdir=prephdir, f77=f77, pfx=pfx)
            pool = multiprocessing.Pool(processes=nprocess)
            pool.map(AFTAN, cstream) #make our results with a map call
            pool.close() #we are not adding any more processes
            pool.join() #tell it to wait until all threads are done before going on
        else:
            AFTAN = partial(aftan4mp, outdir=outdir, inftan=inftan, prephdir=prephdir, f77=f77, pfx=pfx)
            pool = multiprocessing.Pool(processes=nprocess)
            pool.map(AFTAN, inputStream) #make our results with a map call
            pool.close() #we are not adding any more processes
            pool.join() #tell it to wait until all threads are done before going on
        print 'End of multiprocessing aftan analysis !'
        print 'Reading aftan results into ASDF Dataset !'
        for staid1 in staLst:
            for staid2 in staLst:
                netcode1, stacode1=staid1.split('.')
                netcode2, stacode2=staid2.split('.')
                if stacode1 >= stacode2: continue
                try:
                    channels1=self.auxiliary_data.NoiseXcorr[netcode1][stacode1][netcode2][stacode2].list()
                    channels2=self.auxiliary_data.NoiseXcorr[netcode1][stacode1][netcode2][stacode2][channels1[0]].list()
                    for chan in channels1:
                        if chan[2]==channel[0]: chan1=chan
                    for chan in channels2:
                        if chan[2]==channel[1]: chan2=chan
                except KeyError: continue
                finPR=pfx+'/'+netcode1+'.'+stacode1+'/'+ \
                        pfx+'_'+netcode1+'.'+stacode1+'_'+chan1+'_'+netcode2+'.'+stacode2+'_'+chan2+'.SAC'
                try:
                    f10=np.load(outdir+'/'+finPR+'_1_DISP.0.npz')
                    f11=np.load(outdir+'/'+finPR+'_1_DISP.1.npz')
                    f20=np.load(outdir+'/'+finPR+'_2_DISP.0.npz')
                    f21=np.load(outdir+'/'+finPR+'_2_DISP.1.npz')
                except IOError:
                    print 'NO aftan results: '+ netcode1+'.'+stacode1+'_'+netcode2+'.'+stacode2+'_'+channel
                    continue
                print 'Reading aftan results '+ netcode1+'.'+stacode1+'_'+netcode2+'.'+stacode2+'_'+channel
                if deletedisp:
                    os.remove(outdir+'/'+finPR+'_1_DISP.0.npz')
                    os.remove(outdir+'/'+finPR+'_1_DISP.1.npz')
                    os.remove(outdir+'/'+finPR+'_2_DISP.0.npz')
                    os.remove(outdir+'/'+finPR+'_2_DISP.1.npz')
                arr1_1=f10['arr_0']
                nfout1_1=f10['arr_1']
                arr2_1=f11['arr_0']
                nfout2_1=f11['arr_1']
                arr1_2=f20['arr_0']
                nfout1_2=f20['arr_1']
                arr2_2=f21['arr_0']
                nfout2_2=f21['arr_1']
                staid_aux=netcode1+'/'+stacode1+'/'+netcode2+'/'+stacode2+'/'+channel
                if basic1:
                    parameters={'Tc': 0, 'To': 1, 'Vgr': 2, 'Vph': 3, 'ampdb': 4, 'dis': 5, 'snrdb': 6, 'mhw': 7, 'amp': 8, 'Np': nfout1_1}
                    self.add_auxiliary_data(data=arr1_1, data_type='DISPbasic1', path=staid_aux, parameters=parameters)
                if basic2:
                    parameters={'Tc': 0, 'To': 1, 'Vgr': 2, 'Vph': 3, 'ampdb': 4, 'snrdb': 5, 'mhw': 6, 'amp': 7, 'Np': nfout2_1}
                    self.add_auxiliary_data(data=arr2_1, data_type='DISPbasic2', path=staid_aux, parameters=parameters)
                if inftan.pmf:
                    if pmf1:
                        parameters={'Tc': 0, 'To': 1, 'Vgr': 2, 'Vph': 3, 'ampdb': 4, 'dis': 5, 'snrdb': 6, 'mhw': 7, 'amp': 8, 'Np': nfout1_2}
                        self.add_auxiliary_data(data=arr1_2, data_type='DISPpmf1', path=staid_aux, parameters=parameters)
                    if pmf2:
                        parameters={'Tc': 0, 'To': 1, 'Vgr': 2, 'Vph': 3, 'ampdb': 4, 'snrdb': 5, 'mhw': 6, 'amp': 7, 'snr':8, 'Np': nfout2_2}
                        self.add_auxiliary_data(data=arr2_2, data_type='DISPpmf2', path=staid_aux, parameters=parameters)
        if deletedisp: shutil.rmtree(outdir+'/'+pfx)
        return
    
    def interp_disp(self, data_type='DISPpmf2', channel='ZZ', pers=np.array([]), verbose=True):
        """ Interpolate dispersion curve for a given period array.
        =======================================================================================================
        Input Parameters:
        data_type   - dispersion data type (default = DISPpmf2, pmf aftan results after jump detection)
        pers        - period array
        
        Output:
        self.auxiliary_data.DISPbasic1interp, self.auxiliary_data.DISPbasic2interp,
        self.auxiliary_data.DISPpmf1interp, self.auxiliary_data.DISPpmf2interp
        =======================================================================================================
        """
        if data_type=='DISPpmf2':
            ntype=6
        else:
            ntype=5
        if pers.size==0:
            pers=np.append( np.arange(18.)*2.+6., np.arange(4.)*5.+45.)
        staLst=self.waveforms.list()
        for staid1 in staLst:
            for staid2 in staLst:
                netcode1, stacode1=staid1.split('.')
                netcode2, stacode2=staid2.split('.')
                if staid1 >= staid2: continue
                try:
                    subdset=self.auxiliary_data[data_type][netcode1][stacode1][netcode2][stacode2][channel]
                except:
                    continue
                data=subdset.data.value
                index=subdset.parameters
                if verbose:
                    print 'Interpolating dispersion curve for '+ netcode1+'.'+stacode1+'_'+netcode2+'.'+stacode2+'_'+channel
                outindex={ 'To': 0, 'Vgr': 1, 'Vph': 2,  'amp': 3, 'snr': 4, 'inbound': 5, 'Np': pers.size }
                Np=int(index['Np'])
                if Np < 5:
                    warnings.warn('Not enough datapoints for: '+ netcode1+'.'+stacode1+'_'+netcode2+'.'+stacode2+'_'+channel, UserWarning, stacklevel=1)
                    continue
                obsT=data[index['To']][:Np]
                Vgr=np.interp(pers, obsT, data[index['Vgr']][:Np] )
                Vph=np.interp(pers, obsT, data[index['Vph']][:Np] )
                amp=np.interp(pers, obsT, data[index['amp']][:Np] )
                inbound=(pers > obsT[0])*(pers < obsT[-1])*1
                interpdata=np.append(pers, Vgr)
                interpdata=np.append(interpdata, Vph)
                interpdata=np.append(interpdata, amp)
                if data_type=='DISPpmf2':
                    snr=np.interp(pers, obsT, data[index['snr']][:Np] )
                    interpdata=np.append(interpdata, snr)
                interpdata=np.append(interpdata, inbound)
                interpdata=interpdata.reshape(ntype, pers.size)
                staid_aux=netcode1+'/'+stacode1+'/'+netcode2+'/'+stacode2+'/'+channel
                self.add_auxiliary_data(data=interpdata, data_type=data_type+'interp', path=staid_aux, parameters=outindex)
        return
    
    def xcorr_raytomoinput_deprecated(self, outdir, channel='ZZ', pers=np.array([]), outpfx='raytomo_in_', data_type='DISPpmf2interp', verbose=True):
        """
        Generate Input files for Barmine's straight ray surface wave tomography code.
        =======================================================================================================
        Input Parameters:
        outdir      - output directory
        channel     - channel for tomography
        pers        - period array
        outpfx      - prefix for output files, default is 'MISHA_in_'
        data_type   - dispersion data type (default = DISPpmf2, pmf aftan results after jump detection)
        -------------------------------------------------------------------------------------------------------
        Output format:
        outdir/outpfx+per_channel_ph.lst
        =======================================================================================================
        """
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        if pers.size==0:
            pers=np.append( np.arange(18.)*2.+6., np.arange(4.)*5.+45.)
        staLst=self.waveforms.list()
        for per in pers:
            print 'Generating Tomo Input for period:', per
            fname_ph=outdir+'/'+outpfx+'%g'%( per ) +'_'+channel+'_ph.lst' %( per )
            fname_gr=outdir+'/'+outpfx+'%g'%( per ) +'_'+channel+'_gr.lst' %( per )
            fph=open(fname_ph, 'w')
            fgr=open(fname_gr, 'w')
            i=-1
            for staid1 in staLst:
                for staid2 in staLst:
                    netcode1, stacode1=staid1.split('.')
                    netcode2, stacode2=staid2.split('.')
                    if staid1 >= staid2: continue
                    i=i+1
                    try:
                        subdset=self.auxiliary_data[data_type][netcode1][stacode1][netcode2][stacode2][channel]
                    except:
                        # warnings.warn('No interpolated dispersion curve: ' + netcode1+'.'+stacode1+'_'+netcode2+'.'+stacode2+'_'+channel,
                        #             UserWarning, stacklevel=1)
                        continue
                    lat1, elv1, lon1=self.waveforms[staid1].coordinates.values()
                    lat2, elv2, lon2=self.waveforms[staid2].coordinates.values()
                    dist, az, baz=obspy.geodetics.gps2dist_azimuth(lat1, lon1, lat2, lon2) # distance is in m
                    dist=dist/1000.
                    if dist < 2.*per*3.5: continue
                    if lon1<0: lon1+=360.
                    if lon2<0: lon2+=360.
                    data=subdset.data.value
                    index=subdset.parameters
                    ind_per=np.where(data[index['To']][:] == per)[0]
                    if ind_per.size==0:
                        raise AttributeError('No interpolated dispersion curve data for period='+str(per)+' sec!')
                    pvel=data[index['Vph']][ind_per]
                    gvel=data[index['Vgr']][ind_per]
                    snr=data[index['snr']][ind_per]
                    amp=data[index['amp']][ind_per]
                    inbound=data[index['inbound']][ind_per]
                    # quality control
                    if pvel < 0 or gvel < 0 or pvel>10 or gvel>10 or snr >1e10 or amp >1e10: continue
                    if inbound!=1.: continue
                    if snr < 15.: continue
                    fph.writelines("%d %g %g %g %g %g 1. %s %s 1 1 \n" %(i, lat1, lon1, lat2, lon2, pvel, staid1, staid2))
                    fgr.writelines("%d %g %g %g %g %g 1. %s %s 1 1 \n" %(i, lat1, lon1, lat2, lon2, gvel, staid1, staid2))
            fph.close()
            fgr.close()
        print 'End of Generating Misha Tomography Input File!'
        return
    
    def xcorr_raytomoinput(self, outdir, channel='ZZ', pers=np.array([]), outpfx='raytomo_in_', data_type='DISPpmf2interp', verbose=True):
        """
        Generate Input files for Barmine's straight ray surface wave tomography code.
        =======================================================================================================
        Input Parameters:
        outdir      - output directory
        channel     - channel for tomography
        pers        - period array
        outpfx      - prefix for output files, default is 'MISHA_in_'
        data_type   - dispersion data type (default = DISPpmf2, pmf aftan results after jump detection)
        -------------------------------------------------------------------------------------------------------
        Output format:
        outdir/outpfx+per_channel_ph.lst
        =======================================================================================================
        """
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        if pers.size==0:
            pers=np.append( np.arange(18.)*2.+6., np.arange(4.)*5.+45.)
        fph_lst=[]
        fgr_lst=[]
        for per in pers:
            fname_ph=outdir+'/'+outpfx+'%g'%( per ) +'_'+channel+'_ph.lst' %( per )
            fname_gr=outdir+'/'+outpfx+'%g'%( per ) +'_'+channel+'_gr.lst' %( per )
            fph=open(fname_ph, 'w')
            fgr=open(fname_gr, 'w')
            fph_lst.append(fph)
            fgr_lst.append(fgr)
        staLst=self.waveforms.list()
        i=-1
        for staid1 in staLst:
            for staid2 in staLst:
                netcode1, stacode1=staid1.split('.')
                netcode2, stacode2=staid2.split('.')
                if staid1 >= staid2: continue
                i=i+1
                try:
                    subdset=self.auxiliary_data[data_type][netcode1][stacode1][netcode2][stacode2][channel]
                except:
                    continue
                lat1, elv1, lon1=self.waveforms[staid1].coordinates.values()
                lat2, elv2, lon2=self.waveforms[staid2].coordinates.values()
                dist, az, baz=obspy.geodetics.gps2dist_azimuth(lat1, lon1, lat2, lon2) # distance is in m
                dist=dist/1000.
                if lon1<0: lon1+=360.
                if lon2<0: lon2+=360.
                data=subdset.data.value
                index=subdset.parameters
                for iper in xrange(pers.size):
                    per=pers[iper]
                    if dist < 2.*per*3.5: continue
                    ind_per=np.where(data[index['To']][:] == per)[0]
                    if ind_per.size==0:
                        raise AttributeError('No interpolated dispersion curve data for period='+str(per)+' sec!')
                    pvel=data[index['Vph']][ind_per]
                    gvel=data[index['Vgr']][ind_per]
                    snr=data[index['snr']][ind_per]
                    inbound=data[index['inbound']][ind_per]
                    # quality control
                    if pvel < 0 or gvel < 0 or pvel>10 or gvel>10 or snr >1e10: continue
                    if inbound!=1.: continue
                    if snr < 15.: continue
                    fph=fph_lst[iper]
                    fgr=fgr_lst[iper]
                    fph.writelines("%d %g %g %g %g %g 1. %s %s 1 1 \n" %(i, lat1, lon1, lat2, lon2, pvel, staid1, staid2))
                    fgr.writelines("%d %g %g %g %g %g 1. %s %s 1 1 \n" %(i, lat1, lon1, lat2, lon2, gvel, staid1, staid2))
        for iper in xrange(pers.size):
            fph=fph_lst[iper]
            fgr=fgr_lst[iper]
            fph.close()
            fgr.close()
        print 'End of Generating Misha Tomography Input File!'
        return
    
    def xcorr_get_field(self, outdir=None, channel='ZZ', pers=np.array([]), data_type='DISPpmf2interp', verbose=True):
        """ Get the field data for Eikonal tomography
        ============================================================================================================================
        Input Parameters:
        outdir      - directory for txt output (default is not to generate txt output)
        channel     - channel name
        pers        - period array
        datatype    - dispersion data type (default = DISPpmf2interp, interpolated pmf aftan results after jump detection)
        Output:
        self.auxiliary_data.FieldDISPpmf2interp
        ============================================================================================================================
        """
        if pers.size==0:
            pers=np.append( np.arange(18.)*2.+6., np.arange(4.)*5.+45.)
        outindex={ 'longitude': 0, 'latitude': 1, 'Vph': 2,  'Vgr':3, 'snr': 4, 'dist': 5 }
        staLst=self.waveforms.list()
        for staid1 in staLst:
            field_lst=[]
            Nfplst=[]
            for per in pers:
                field_lst.append(np.array([]))
                Nfplst.append(0)
            lat1, elv1, lon1=self.waveforms[staid1].coordinates.values()
            if verbose:
                print 'Getting field data for: '+staid1
            for staid2 in staLst:
                if staid1==staid2:
                    continue
                netcode1, stacode1=staid1.split('.')
                netcode2, stacode2=staid2.split('.')
                try:
                    subdset=self.auxiliary_data[data_type][netcode1][stacode1][netcode2][stacode2][channel]
                except:
                    try:
                        subdset=self.auxiliary_data[data_type][netcode2][stacode2][netcode1][stacode1][channel]
                    except:
                        continue
                lat2, elv2, lon2=self.waveforms[staid2].coordinates.values()
                dist, az, baz=obspy.geodetics.gps2dist_azimuth(lat1, lon1, lat2, lon2) # distance is in m
                dist=dist/1000.
                if lon1<0: lon1+=360.
                if lon2<0: lon2+=360.
                data=subdset.data.value
                index=subdset.parameters
                for iper in xrange(pers.size):
                    per=pers[iper]
                    if dist < 2.*per*3.5: continue
                    ind_per=np.where(data[index['To']][:] == per)[0]
                    if ind_per.size==0:
                        raise AttributeError('No interpolated dispersion curve data for period='+str(per)+' sec!')
                    pvel=data[index['Vph']][ind_per]
                    gvel=data[index['Vgr']][ind_per]
                    snr=data[index['snr']][ind_per]
                    inbound=data[index['inbound']][ind_per]
                    # quality control
                    if pvel < 0 or gvel < 0 or pvel>10 or gvel>10 or snr >1e10: continue
                    if inbound!=1.: continue
                    if snr < 15.: continue
                    field_lst[iper]=np.append(field_lst[iper], lon2)
                    field_lst[iper]=np.append(field_lst[iper], lat2)
                    field_lst[iper]=np.append(field_lst[iper], pvel)
                    field_lst[iper]=np.append(field_lst[iper], gvel)
                    field_lst[iper]=np.append(field_lst[iper], snr)
                    field_lst[iper]=np.append(field_lst[iper], dist)
                    Nfplst[iper]+=1
            # end of reading data from all receivers, taking staid1 as virtual source
            if outdir!=None:
                if not os.path.isdir(outdir):
                    os.makedirs(outdir)
            staid_aux=netcode1+'/'+stacode1+'/'+channel
            for iper in xrange(pers.size):
                per=pers[iper]
                del_per=per-int(per)
                if field_lst[iper].size==0:
                    continue
                field_lst[iper]=field_lst[iper].reshape(Nfplst[iper], 6)
                if del_per==0.:
                    staid_aux_per=staid_aux+'/'+str(int(per))+'sec'
                else:
                    dper=str(del_per)
                    staid_aux_per=staid_aux+'/'+str(int(per))+'sec'+dper.split('.')[1]
                self.add_auxiliary_data(data=field_lst[iper], data_type='Field'+data_type, path=staid_aux_per, parameters=outindex)
                if outdir!=None:
                    if not os.path.isdir(outdir+'/'+str(per)+'sec'):
                        os.makedirs(outdir+'/'+str(per)+'sec')
                    txtfname=outdir+'/'+str(per)+'sec'+'/'+staid1+'_'+str(per)+'.txt'
                    header = 'evlo='+str(lon1)+' evla='+str(lat1)
                    np.savetxt( txtfname, field_lst[iper], fmt='%g', header=header )
        return
            
def stack4mp(inv, datadir, outdir, ylst, mlst, pfx, fnametype):
    stackedST=[]
    cST=[]
    initflag=True
    channels1=inv.networks[0].stations[0].channels
    channels2=inv.networks[1].stations[0].channels
    netcode1=inv.networks[0].code
    stacode1=inv.networks[0].stations[0].code
    netcode2=inv.networks[1].code
    stacode2=inv.networks[1].stations[0].code
    mnumb=mlst.size
    for im in xrange(mnumb):
        skipflag=False
        for chan1 in channels1:
            if skipflag:
                break
            for chan2 in channels2:
                month=monthdict[mlst[im]]
                yrmonth=str(ylst[im])+'.'+month
                if fnametype==1:
                    fname=datadir+'/'+yrmonth+'/'+pfx+'/'+stacode1+'/'+pfx+'_'+stacode1+'_'+chan1.code+'_'+stacode2+'_'+chan2.code+'.SAC'
                elif fnametype==2:
                    fname=datadir+'/'+yrmonth+'/'+pfx+'/'+stacode1+'/'+pfx+'_'+stacode1+'_'+stacode2+'.SAC'
                if not os.path.isfile(fname):
                    skipflag=True
                    break
                try:
                    tr=obspy.core.read(fname)[0]
                except TypeError:
                    warnings.warn('Unable to read SAC for: ' + stacode1 +'_'+stacode2 +' Month: '+yrmonth, UserWarning, stacklevel=1)
                    skipflag=True
                if np.isnan(tr.data).any() or abs(tr.data.max())>1e20:
                    warnings.warn('NaN monthly SAC for: ' + stacode1 +'_'+stacode2 +' Month: '+yrmonth, UserWarning, stacklevel=1)
                    skipflag=True
                    break
                cST.append(tr)
        if len(cST)!=len(channels1)*len(channels2) or skipflag:
            cST=[]
            continue
        if initflag:
            stackedST=copy.deepcopy(cST)
            initflag=False
        else:
            for itr in xrange(len(cST)):
                mtr=cST[itr]
                stackedST[itr].data+=mtr.data
                stackedST[itr].stats.sac.user0+=mtr.stats.sac.user0
        cST=[]
    if len(stackedST)==len(channels1)*len(channels2):
        print 'Finished Stacking for:'+stacode1+'_'+stacode2
        i=0
        for chan1 in channels1:
            for chan2 in channels2:
                stackedTr=stackedST[i]
                outfname=outdir+'/'+pfx+'/'+netcode1+'.'+stacode1+'/'+ \
                    pfx+'_'+netcode1+'.'+stacode1+'_'+chan1.code+'_'+netcode2+'.'+stacode2+'_'+chan2.code+'.SAC'
                stackedTr.write(outfname, format='SAC')
                i+=1
    return

def aftan4mp(aTr, outdir, inftan, prephdir, f77, pfx):
    # print 'aftan analysis for: '+ aTr.stats.sac.kuser0+'.'+aTr.stats.sac.kevnm+'_'+chan1+'_'+aTr.stats.network+'.'+aTr.stats.station+'_'+chan2
    if prephdir !=None:
        phvelname = prephdir + "/%s.%s.pre" %(aTr.stats.sac.kuser0+'.'+aTr.stats.sac.kevnm, aTr.stats.network+'.'+aTr.stats.station)
    else:
        phvelname =''
    if abs(aTr.stats.sac.b+aTr.stats.sac.e)< aTr.stats.delta:
        aTr.makesym()
    if f77:
        aTr.aftanf77(pmf=inftan.pmf, piover4=inftan.piover4, vmin=inftan.vmin, vmax=inftan.vmax, tmin=inftan.tmin, tmax=inftan.tmax,
            tresh=inftan.tresh, ffact=inftan.ffact, taperl=inftan.taperl, snr=inftan.snr, fmatch=inftan.fmatch, nfin=inftan.nfin,
                npoints=inftan.npoints, perc=inftan.perc, phvelname=phvelname)
    else:
        aTr.aftan(pmf=inftan.pmf, piover4=inftan.piover4, vmin=inftan.vmin, vmax=inftan.vmax, tmin=inftan.tmin, tmax=inftan.tmax,
            tresh=inftan.tresh, ffact=inftan.ffact, taperl=inftan.taperl, snr=inftan.snr, fmatch=inftan.fmatch, nfin=inftan.nfin,
                npoints=inftan.npoints, perc=inftan.perc, phvelname=phvelname)
    aTr.get_snr(ffact=inftan.ffact) # SNR analysis
    chan1=aTr.stats.sac.kcmpnm[:3]
    chan2=aTr.stats.sac.kcmpnm[3:]
    foutPR=outdir+'/'+pfx+'/'+aTr.stats.sac.kuser0+'.'+aTr.stats.sac.kevnm+'/'+ \
                pfx+'_'+aTr.stats.sac.kuser0+'.'+aTr.stats.sac.kevnm+'_'+chan1+'_'+aTr.stats.network+'.'+aTr.stats.station+'_'+chan2+'.SAC'
    aTr.ftanparam.writeDISPbinary(foutPR)
    return
    
    
    
class requestInfo(object):
    def __init__(self, evnumb, network, station, location, channel, starttime, endtime, quality=None,
            minimumlength=None, longestonly=None, filename=None, attach_response=False, baz=0):
        self.evnumb         = evnumb
        self.network        = network
        self.station        = station
        self.location       = location
        self.channel        = channel
        self.starttime      = starttime
        self.endtime        = endtime
        self.quality        = quality
        self.minimumlength  = minimumlength
        self.longestonly    = longestonly
        self.filename       = filename
        self.attach_response= attach_response
        self.baz            = baz

class quakeASDF(pyasdf.ASDFDataSet):
    """ An object to for earthquake data analysis based on ASDF database
    """    
    def get_events(self, startdate, enddate, add2dbase=True, gcmt=False, Mmin=5.5, Mmax=None, minlatitude=None, maxlatitude=None, minlongitude=None, maxlongitude=None,
            latitude=None, longitude=None, minradius=None, maxradius=None, mindepth=None, maxdepth=None, magnitudetype=None):
        """Get earthquake catalog from IRIS server
        =======================================================================================================
        Input Parameters:
        startdate, enddata  - start/end date for searching
        Mmin, Mmax          - minimum/maximum magnitude for searching                
        minlatitude         - Limit to events with a latitude larger than the specified minimum.
        maxlatitude         - Limit to events with a latitude smaller than the specified maximum.
        minlongitude        - Limit to events with a longitude larger than the specified minimum.
        maxlongitude        - Limit to events with a longitude smaller than the specified maximum.
        latitude            - Specify the latitude to be used for a radius search.
        longitude           - Specify the longitude to the used for a radius search.
        minradius           - Limit to events within the specified minimum number of degrees from the
                                geographic point defined by the latitude and longitude parameters.
        maxradius           - Limit to events within the specified maximum number of degrees from the
                                geographic point defined by the latitude and longitude parameters.
        mindepth            - Limit to events with depth, in kilometers, larger than the specified minimum.
        maxdepth            - Limit to events with depth, in kilometers, smaller than the specified maximum.
        magnitudetype       - Specify a magnitude type to use for testing the minimum and maximum limits.
        =======================================================================================================
        """
        starttime=obspy.core.utcdatetime.UTCDateTime(startdate)
        endtime=obspy.core.utcdatetime.UTCDateTime(enddate)
        if not gcmt:
            client=Client('IRIS')
            try:
                catISC = client.get_events(starttime=starttime, endtime=endtime, minmagnitude=Mmin, maxmagnitude=Mmax, catalog='ISC',
                    minlatitude=minlatitude, maxlatitude=maxlatitude, minlongitude=minlongitude, maxlongitude=maxlongitude,
                    latitude=latitude, longitude=longitude, minradius=minradius, maxradius=maxradius, mindepth=mindepth,
                    maxdepth=maxdepth, magnitudetype=magnitudetype)
                endtimeISC=catISC[0].origins[0].time
            except:
                catISC=obspy.core.event.Catalog()
                endtimeISC=starttime
            if endtime.julday-endtimeISC.julday >1:
                try:
                    catPDE = client.get_events(starttime=endtimeISC, endtime=endtime, minmagnitude=Mmin, maxmagnitude=Mmax, catalog='NEIC PDE',
                        minlatitude=minlatitude, maxlatitude=maxlatitude, minlongitude=minlongitude, maxlongitude=maxlongitude,
                        latitude=latitude, longitude=longitude, minradius=minradius, maxradius=maxradius, mindepth=mindepth,
                        maxdepth=maxdepth, magnitudetype=magnitudetype)
                    catalog=catISC+catPDE
                except:
                    catalog=catISC
            else: catalog=catISC
            outcatalog=obspy.core.event.Catalog()
            # check magnitude
            for event in catalog:
                if event.magnitudes[0].mag < Mmin: continue
                outcatalog.append(event)
        else:
            gcmt_url_old='http://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog/jan76_dec13.ndk'
            gcmt_new='http://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog/NEW_MONTHLY'
            if starttime.year < 2005:
                cat_old=obspy.read_events(gcmt_url_old)
                outcatalog=cat_old.filter("magnitude >= %g" %Mmin, "time >= %s" %str(starttime), "time <= %s" %str(endtime) )
                if Mmax!=None: cat_old=cat_old.filter("magnitude <= %g" %Mmax)
                if maxlongitude!=None: cat_old=cat_old.filter("longitude <= %g" %maxlongitude)
                if minlongitude!=None: cat_old=cat_old.filter("longitude >= %g" %minlongitude)
                if maxlatitude!=None: cat_old=cat_old.filter("latitude <= %g" %maxlatitude)
                if minlatitude!=None: cat_old=cat_old.filter("latitude >= %g" %minlatitude)
                if maxdepth!=None: cat_old=cat_old.filter("depth <= %g" %(maxdepth*1000.))
                if mindepth!=None: cat_old=cat_old.filter("depth >= %g" %(mindepth*1000.))
                temp_stime=obspy.core.utcdatetime.UTCDateTime('2014-01-01')
                outcatalog=cat_old
            else:
                outcatalog=obspy.core.event.Catalog()
                temp_stime=copy.deepcopy(starttime); temp_stime.day=1
            while (temp_stime < endtime):
                year=temp_stime.year; month=temp_stime.month
                yearstr=str(int(year))[2:]; monstr=monthdict[month]; monstr=monstr.lower()
                if year==2005 and month==6: monstr='june'
                if year==2005 and month==7: monstr='july'
                if year==2005 and month==9: monstr='sept'
                gcmt_url_new=gcmt_new+'/'+str(int(year))+'/'+monstr+yearstr+'.ndk'
                print gcmt_url_new
                try: cat_new=obspy.read_events(gcmt_url_new)
                except: break
                cat_new=cat_new.filter("magnitude >= %g" %Mmin, "time >= %s" %str(starttime), "time <= %s" %str(endtime) )
                if Mmax!=None: cat_new=cat_new.filter("magnitude <= %g" %Mmax)
                if maxlongitude!=None: cat_new=cat_new.filter("longitude <= %g" %maxlongitude)
                if minlongitude!=None: cat_new=cat_new.filter("longitude >= %g" %minlongitude)
                if maxlatitude!=None: cat_new=cat_new.filter("latitude <= %g" %maxlatitude)
                if minlatitude!=None: cat_new=cat_new.filter("latitude >= %g" %minlatitude)
                if maxdepth!=None: cat_new=cat_new.filter("depth <= %g" %(maxdepth*1000.))
                if mindepth!=None: cat_new=cat_new.filter("depth >= %g" %(mindepth*1000.))
                outcatalog+=cat_new
                try: temp_stime.month+=1
                except: temp_stime.year+=1; temp_stime.month=1
        if add2dbase: self.add_quakeml(outcatalog)
        else: return outcatalog
        return
    
    def read_stationtxt(self, stafile, source='CIEI', chans=['BHZ', 'BHE', 'BHN']):
        """Read txt station list 
        """
        sta_info=sta_info_default.copy()
        with open(stafile, 'r') as f:
            Sta=[]
            site=obspy.core.inventory.util.Site(name='01')
            creation_date=obspy.core.utcdatetime.UTCDateTime(0)
            inv=obspy.core.inventory.inventory.Inventory(networks=[], source=source)
            total_number_of_channels=len(chans)
            for lines in f.readlines():
                lines=lines.split()
                netsta=lines[0]
                netcode=netsta[:2]
                stacode=netsta[2:]
                if stacode[-1]=='/':
                    stacode=stacode[:-1]
                    print netcode, stacode
                lon=float(lines[1])
                lat=float(lines[2])
                if lat>90.:
                    lon=float(lines[2])
                    lat=float(lines[1])
                netsta=netcode+'.'+stacode
                if Sta.__contains__(netsta):
                    index=Sta.index(netsta)
                    if abs(self[index].lon-lon) >0.01 and abs(self[index].lat-lat) >0.01:
                        raise ValueError('Incompatible Station Location:' + netsta+' in Station List!')
                    else:
                        print 'Warning: Repeated Station:' +netsta+' in Station List!'
                        continue
                channels=[]
                if lon>180.:
                    lon-=360.
                for chan in chans:
                    channel=obspy.core.inventory.channel.Channel(code=chan, location_code='01', latitude=lat, longitude=lon,
                            elevation=0.0, depth=0.0)
                    channels.append(channel)
                station=obspy.core.inventory.station.Station(code=stacode, latitude=lat, longitude=lon, elevation=0.0,
                        site=site, channels=channels, total_number_of_channels = total_number_of_channels, creation_date = creation_date)
                network=obspy.core.inventory.network.Network(code=netcode, stations=[station])
                networks=[network]
                inv+=obspy.core.inventory.inventory.Inventory(networks=networks, source=source)
        print 'Writing obspy inventory to ASDF dataset'
        self.add_stationxml(inv)
        print 'End writing obspy inventory to ASDF dataset'
        return
    
    def read_sac(self, datadir):
        """This function is a scratch for reading a specific datasets, DO NOT use this function!
        """
        L=len(self.events)
        evnumb=0
        import glob
        for event in self.events:
            event_id=event.resource_id.id.split('=')[-1]
            magnitude=event.magnitudes[0].mag; Mtype=event.magnitudes[0].magnitude_type
            event_descrip=event.event_descriptions[0].text+', '+event.event_descriptions[0].type
            evnumb+=1
            print '================================= Getting surface wave data ==================================='
            print 'Event ' + str(evnumb)+' : '+event_descrip+', '+Mtype+' = '+str(magnitude) 
            st=obspy.Stream()
            otime=event.origins[0].time
            evlo=event.origins[0].longitude; evla=event.origins[0].latitude
            tag='surf_ev_%05d' %evnumb
            # if lon0!=None and lat0!=None:
            #     dist, az, baz=obspy.geodetics.gps2dist_azimuth(evla, evlo, lat0, lon0) # distance is in m
            #     dist=dist/1000.
            #     starttime=otime+dist/vmax; endtime=otime+dist/vmin
            #     commontime=True
            # else:
            #     commontime=False
            odate=str(otime.year)+'%02d' %otime.month +'%02d' %otime.day
            for staid in self.waveforms.list():
                netcode, stacode=staid.split('.')
                print staid
                stla, elev, stlo=self.waveforms[staid].coordinates.values()
                # sta_datadir=datadir+'/'+netcode+'/'+stacode
                sta_datadir=datadir+'/'+netcode+'/'+stacode
                sacpfx=sta_datadir+'/'+stacode+'.'+odate
                
                pzpfx='/home/lili/code/china_data/response_files/SAC_*'+netcode+'_'+stacode
                respfx='/home/lili/code/china_data/RESP4WeisenCUB/dbRESPCNV20131007/'+netcode+'/'+staid+'/RESP.'+staid
                st=obspy.Stream()
                for chan in ['*Z', '*E', '*N']:
                    sacfname = sacpfx+chan
                    pzfpattern  = pzpfx+'_'+chan
                    respfpattern= respfx+'*BH'+chan[-1]+'*'
                    #################
                    try: respfname=glob.glob(respfpattern)[0]
                    except: break
                    seedresp = {'filename': respfname,  # RESP filename
                    # when using Trace/Stream.simulate() the "date" parameter can
                    # also be omitted, and the starttime of the trace is then used.
                    # Units to return response in ('DIS', 'VEL' or ACC)
                    'units': 'VEL'
                    }
                    try: tr=obspy.read(sacfname)[0]
                    except: break
                    tr.detrend()
                    tr.stats.channel='BH'+chan[-1]
                    tr.simulate(paz_remove=None, pre_filt=(0.001, 0.005, 1, 100.0), seedresp=seedresp)
                    ################
                    # try: pzfname = glob.glob(pzfpattern)[0]
                    # except: break
                    # try: tr=obspy.read(sacfname)[0]
                    # except: break
                    # obspy.io.sac.sacpz.attach_paz(tr, pzfname)
                    # tr.decimate(10)
                    # tr.detrend()
                    # tr.simulate(paz_remove=tr.stats.paz, pre_filt=(0.001, 0.005, 1, 100.0))
                    st.append(tr)
                self.add_waveforms(st, event_id=event_id, tag=tag)    
     
    
    def _get_basemap(self, projection='lambert', geopolygons=None, resolution='i'):
        """Plot data with contour
        """
        # fig=plt.figure(num=None, figsize=(12, 12), dpi=80, facecolor='w', edgecolor='k')
        lat_centre = (self.maxlat+self.minlat)/2.0
        lon_centre = (self.maxlon+self.minlon)/2.0
        if projection=='merc':
            m=Basemap(projection='merc', llcrnrlat=self.minlat-5., urcrnrlat=self.maxlat+5., llcrnrlon=self.minlon-5.,
                      urcrnrlon=self.maxlon+5., lat_ts=20, resolution=resolution)
            m.drawparallels(np.arange(-80.0,80.0,5.0), labels=[1,0,0,1])
            m.drawmeridians(np.arange(-170.0,170.0,5.0), labels=[1,0,0,1])
            m.drawstates(color='g', linewidth=2.)
        elif projection=='global':
            m=Basemap(projection='ortho',lon_0=lon_centre, lat_0=lat_centre, resolution=resolution)
            m.drawparallels(np.arange(-80.0,80.0,10.0), labels=[1,0,0,1])
            m.drawmeridians(np.arange(-170.0,170.0,10.0), labels=[1,0,0,1])
        
        elif projection=='regional_ortho':
            m1 = Basemap(projection='ortho', lon_0=self.minlon, lat_0=self.minlat, resolution='l')
            m = Basemap(projection='ortho', lon_0=self.minlon, lat_0=self.minlat, resolution=resolution,\
                llcrnrx=0., llcrnry=0., urcrnrx=m1.urcrnrx/mapfactor, urcrnry=m1.urcrnry/3.5)
            m.drawparallels(np.arange(-80.0,80.0,10.0), labels=[1,0,0,0],  linewidth=2,  fontsize=20)
            m.drawmeridians(np.arange(-170.0,170.0,10.0),  linewidth=2)
        elif projection=='lambert':
            distEW, az, baz=obspy.geodetics.gps2dist_azimuth(self.minlat, self.minlon,
                                self.minlat, self.maxlon) # distance is in m
            distNS, az, baz=obspy.geodetics.gps2dist_azimuth(self.minlat, self.minlon,
                                self.maxlat+2., self.minlon) # distance is in m
            m = Basemap(width=distEW, height=distNS, rsphere=(6378137.00,6356752.3142), resolution='l', projection='lcc',\
                lat_1=self.minlat, lat_2=self.maxlat, lon_0=lon_centre, lat_0=lat_centre+1)
            m.drawparallels(np.arange(-80.0,80.0,10.0), linewidth=1, dashes=[2,2], labels=[1,1,0,0], fontsize=15)
            m.drawmeridians(np.arange(-170.0,170.0,10.0), linewidth=1, dashes=[2,2], labels=[0,0,1,0], fontsize=15)
        m.drawcoastlines(linewidth=1.0)
        m.drawcountries(linewidth=1.)
        m.fillcontinents(lake_color='#99ffff',zorder=0.2)
        m.drawmapboundary(fill_color="white")
        m.drawstates()
        try:
            geopolygons.PlotPolygon(inbasemap=m)
        except:
            pass
        return m
    
    def plot_events(self, gcmt=False, projection='lambert', valuetype='depth', geopolygons=None, showfig=True, vmin=None, vmax=None):
        if gcmt: from obspy.imaging.beachball import beach; ax = plt.gca()
        evlons=np.array([])
        evlats=np.array([])
        values=np.array([])
        focmecs=[]
        for event in self.events:
            event_id=event.resource_id.id.split('=')[-1]
            magnitude=event.magnitudes[0].mag; Mtype=event.magnitudes[0].magnitude_type
            otime=event.origins[0].time
            evlo=event.origins[0].longitude; evla=event.origins[0].latitude; evdp=event.origins[0].depth/1000.
            if evlo > -80.: continue
            evlons=np.append(evlons, evlo); evlats = np.append(evlats, evla);
            if valuetype=='depth': values=np.append(values, evdp)
            elif valuetype=='mag': values=np.append(values, magnitude)
            if gcmt:
                mtensor=event.focal_mechanisms[0].moment_tensor.tensor
                mt=[mtensor.m_rr, mtensor.m_tt, mtensor.m_pp, mtensor.m_rt, mtensor.m_rp, mtensor.m_tp]
                # nodalP=event.focal_mechanisms[0].nodal_planes.values()[1]
                # mt=[nodalP.strike, nodalP.dip, nodalP.rake]
                focmecs.append(mt)
        self.minlat=evlats.min()-1.; self.maxlat=evlats.max()+1.
        self.minlon=evlons.min()-1.; self.maxlon=evlons.max()+1.
        # self.minlat=15; self.maxlat=50
        # self.minlon=95; self.maxlon=128
        m=self._get_basemap(projection=projection, geopolygons=geopolygons)
        import pycpt
        cmap=pycpt.load.gmtColormap('./GMT_panoply.cpt')
        # cmap =discrete_cmap(int((vmax-vmin)/0.1)+1, cmap)
        x, y=m(evlons, evlats)
        if vmax==None and vmin==None: vmax=values.max(); vmin=values.min()
        if gcmt:
            for i in xrange(len(focmecs)):
                value=values[i]
                rgbcolor=cmap( (value-vmin)/(vmax-vmin) )
                b = beach(focmecs[i], xy=(x[i], y[i]), width=100000, linewidth=1, facecolor=rgbcolor)
                b.set_zorder(10)
                ax.add_collection(b)
                # ax.annotate(str(i), (x[i]+50000, y[i]+50000))
            im=m.scatter(x, y, marker='o', s=1, c=values, cmap=cmap, vmin=vmin, vmax=vmax)
            cb = m.colorbar(im, "bottom", size="3%", pad='2%')
            cb.set_label(valuetype, fontsize=20)
        else:
            if values.size!=0:
                im=m.scatter(x, y, marker='o', s=300, c=values, cmap=cmap, vmin=vmin, vmax=vmax)
                cb = m.colorbar(im, "bottom", size="3%", pad='2%')
            else: m.plot(x,y,'o')
        if gcmt: stime=self.events[0].origins[0].time; etime=self.events[-1].origins[0].time
        else: etime=self.events[0].origins[0].time; stime=self.events[-1].origins[0].time
        plt.suptitle('Number of event: '+str(len(self.events))+' time range: '+str(stime)+' - '+str(etime), fontsize=20 )
        if showfig: plt.show()
        
        
    
    def get_stations(self, startdate=None, enddate=None,  network=None, station=None, location=None, channel=None,
            minlatitude=None, maxlatitude=None, minlongitude=None, maxlongitude=None, latitude=None, longitude=None, minradius=None, maxradius=None):
        """Get station inventory from IRIS server
        =======================================================================================================
        Input Parameters:
        startdate, enddata  - start/end date for searching
        network             - Select one or more network codes.
                                Can be SEED network codes or data center defined codes.
                                    Multiple codes are comma-separated (e.g. "IU,TA").
        station             - Select one or more SEED station codes.
                                Multiple codes are comma-separated (e.g. "ANMO,PFO").
        location            - Select one or more SEED location identifiers.
                                Multiple identifiers are comma-separated (e.g. "00,01").
                                As a special case “--“ (two dashes) will be translated to a string of two space
                                characters to match blank location IDs.
        channel             - Select one or more SEED channel codes.
                                Multiple codes are comma-separated (e.g. "BHZ,HHZ").             
        minlatitude         - Limit to events with a latitude larger than the specified minimum.
        maxlatitude         - Limit to events with a latitude smaller than the specified maximum.
        minlongitude        - Limit to events with a longitude larger than the specified minimum.
        maxlongitude        - Limit to events with a longitude smaller than the specified maximum.
        latitude            - Specify the latitude to be used for a radius search.
        longitude           - Specify the longitude to the used for a radius search.
        minradius           - Limit to events within the specified minimum number of degrees from the
                                geographic point defined by the latitude and longitude parameters.
        maxradius           - Limit to events within the specified maximum number of degrees from the
                                geographic point defined by the latitude and longitude parameters.
        =======================================================================================================
        """
        try: starttime=obspy.core.utcdatetime.UTCDateTime(startdate)
        except: starttime=None
        try: endtime=obspy.core.utcdatetime.UTCDateTime(enddate)
        except: endtime=None
        client=Client('IRIS')
        inv = client.get_stations(network=network, station=station, starttime=starttime, endtime=endtime, channel=channel, 
            minlatitude=minlatitude, maxlatitude=maxlatitude, minlongitude=minlongitude, maxlongitude=maxlongitude,
            latitude=latitude, longitude=longitude, minradius=minradius, maxradius=maxradius, level='channel')
        self.add_stationxml(inv)
        self.inv=inv
        return 
    
    def get_surf_waveforms(self, lon0=None, lat0=None, minDelta=-1, maxDelta=181, channel='LHZ', vmax=6.0, vmin=1.0, verbose=False):
        """Get surface wave data from IRIS server
        ====================================================================================================================
        Input Parameters:
        lon0, lat0      - center of array. If specified, all wave form will have the same starttime and endtime
        min/maxDelta    - minimum/maximum epicentral distance, in degree
        channel         - Channel code, e.g. 'BHZ'.
                            Last character (i.e. component) can be a wildcard (‘?’ or ‘*’) to fetch Z, N and E component.
        vmin, vmax      - minimum/maximum velocity for surface wave window
        =====================================================================================================================
        """
        client=Client('IRIS')
        evnumb=0
        L=len(self.events)
        for event in self.events:
            event_id=event.resource_id.id.split('=')[-1]
            magnitude=event.magnitudes[0].mag; Mtype=event.magnitudes[0].magnitude_type
            event_descrip=event.event_descriptions[0].text+', '+event.event_descriptions[0].type
            evnumb+=1
            print '================================= Getting surface wave data ==================================='
            print 'Event ' + str(evnumb)+' : '+event_descrip+', '+Mtype+' = '+str(magnitude) 
            st=obspy.Stream()
            otime=event.origins[0].time
            evlo=event.origins[0].longitude; evla=event.origins[0].latitude
            if lon0!=None and lat0!=None:
                dist, az, baz=obspy.geodetics.gps2dist_azimuth(evla, evlo, lat0, lon0) # distance is in m
                dist=dist/1000.
                starttime=otime+dist/vmax; endtime=otime+dist/vmin
                commontime=True
            else:
                commontime=False
            for staid in self.waveforms.list():
                netcode, stacode=staid.split('.')
                stla, elev, stlo=self.waveforms[staid].coordinates.values()
                if not commontime:
                    dist, az, baz=obspy.geodetics.gps2dist_azimuth(evla, evlo, stla, stlo) # distance is in m
                    dist=dist/1000.; Delta=obspy.geodetics.kilometer2degrees(dist)
                    if Delta<minDelta: continue
                    if Delta>maxDelta: continue
                    starttime=otime+dist/vmax; endtime=otime+dist/vmin
                # location=self.waveforms[staid].StationXML[0].stations[0].channels[0].location_code
                try:
                    # st += client.get_waveforms(network=netcode, station=stacode, location=location, channel=channel,
                    #         starttime=starttime, endtime=endtime, attach_response=True)
                    st += client.get_waveforms(network=netcode, station=stacode, location='00', channel=channel,
                            starttime=starttime, endtime=endtime, attach_response=True)
                except:
                    if verbose: print 'No data for:', staid
                    continue
                if verbose: print 'Getting data for:', staid
            print '===================================== Removing response ======================================='
            pre_filt = (0.001, 0.005, 1, 100.0)
            st.detrend()
            st.remove_response(pre_filt=pre_filt, taper_fraction=0.1)
            tag='surf_ev_%05d' %evnumb
            self.add_waveforms(st, event_id=event_id, tag=tag)
        return
    
    def get_surf_waveforms_mp(self, outdir, lon0=None, lat0=None, minDelta=-1, maxDelta=181, channel='LHZ', vmax=6.0, vmin=1.0, verbose=False,
            subsize=1000, deletemseed=False, nprocess=None, snumb=0, enumb=None):
        """Get surface wave data from IRIS server with multiprocessing
        ====================================================================================================================
        Input Parameters:
        lon0, lat0      - center of array. If specified, all wave form will have the same starttime and endtime
        min/maxDelta    - minimum/maximum epicentral distance, in degree
        channel         - Channel code, e.g. 'BHZ'.
                            Last character (i.e. component) can be a wildcard (‘?’ or ‘*’) to fetch Z, N and E component.
        vmin, vmax      - minimum/maximum velocity for surface wave window
        subsize         - subsize of processing list, use to prevent lock in multiprocessing process
        deletemseed     - delete output MiniSeed files
        nprocess        - number of processes
        snumb, enumb    - start/end number of processing block
        =====================================================================================================================
        """
        client=Client('IRIS')
        evnumb=0
        L=len(self.events)
        if not os.path.isdir(outdir): os.makedirs(outdir)
        reqwaveLst=[]
        swave=snumb*subsize
        iwave=0
        print '================================= Preparing for surface wave data download ==================================='
        for event in self.events:
            eventid=event.resource_id.id.split('=')[-1]
            magnitude=event.magnitudes[0].mag; Mtype=event.magnitudes[0].magnitude_type
            event_descrip=event.event_descriptions[0].text+', '+event.event_descriptions[0].type
            evnumb+=1
            otime=event.origins[0].time
            evlo=event.origins[0].longitude; evla=event.origins[0].latitude
            if lon0!=None and lat0!=None:
                dist, az, baz=obspy.geodetics.gps2dist_azimuth(evla, evlo, lat0, lon0) # distance is in m
                dist=dist/1000.
                starttime=otime+dist/vmax; endtime=otime+dist/vmin
                commontime=True
            else:
                commontime=False
            for staid in self.waveforms.list():
                netcode, stacode=staid.split('.')
                iwave+=1
                if iwave < swave: continue
                stla, elev, stlo=self.waveforms[staid].coordinates.values()
                if not commontime:
                    dist, az, baz=obspy.geodetics.gps2dist_azimuth(evla, evlo, stla, stlo) # distance is in m
                    dist=dist/1000.; Delta=obspy.geodetics.kilometer2degrees(dist)
                    if Delta<minDelta: continue
                    if Delta>maxDelta: continue
                    starttime=otime+dist/vmax; endtime=otime+dist/vmin
                location=self.waveforms[staid].StationXML[0].stations[0].channels[0].location_code
                reqwaveLst.append( requestInfo(evnumb=evnumb, network=netcode, station=stacode, location=location, channel=channel,
                            starttime=starttime, endtime=endtime, attach_response=True) )
        print '============================= Start multiprocessing download surface wave data ==============================='
        if len(reqwaveLst) > subsize:
            Nsub = int(len(reqwaveLst)/subsize)
            # if enumb==None: enumb=Nsub
            for isub in xrange(Nsub):
                # if isub < snumb: continue
                # if isub > enumb: continue
                print 'Subset:', isub+1,'in',Nsub,'sets'
                creqlst=reqwaveLst[isub*subsize:(isub+1)*subsize]
                GETDATA = partial(get_waveforms4mp, outdir=outdir, client=client, pre_filt = (0.001, 0.005, 1, 100.0), verbose=verbose, rotation=False)
                pool = multiprocessing.Pool(processes=nprocess)
                pool.map(GETDATA, creqlst) #make our results with a map call
                pool.close() #we are not adding any more processes
                pool.join() #tell it to wait until all threads are done before going on
            creqlst=reqwaveLst[(isub+1)*subsize:]
            GETDATA = partial(get_waveforms4mp, outdir=outdir, client=client, pre_filt = (0.001, 0.005, 1, 100.0), verbose=verbose, rotation=False)
            pool = multiprocessing.Pool(processes=nprocess)
            pool.map(GETDATA, creqlst) #make our results with a map call
            pool.close() #we are not adding any more processes
            pool.join() #tell it to wait until all threads are done before going on
        else:
            GETDATA = partial(get_waveforms4mp, outdir=outdir, client=client, pre_filt = (0.001, 0.005, 1, 100.0), verbose=verbose, rotation=False)
            pool = multiprocessing.Pool(processes=nprocess)
            pool.map(GETDATA, reqwaveLst) #make our results with a map call
            pool.close() #we are not adding any more processes
            pool.join() #tell it to wait until all threads are done before going on
        print '============================= End of multiprocessing download surface wave data =============================='
        print '==================================== Reading downloaded surface wave data ===================================='
        evnumb=0
        no_resp=0
        for event in self.events:
            event_id=event.resource_id.id.split('=')[-1]
            magnitude=event.magnitudes[0].mag; Mtype=event.magnitudes[0].magnitude_type
            event_descrip=event.event_descriptions[0].text+', '+event.event_descriptions[0].type
            evnumb+=1
            evid='E%05d' %evnumb
            tag='surf_ev_%05d' %evnumb
            print 'Event ' + str(evnumb)+' : '+event_descrip+', '+Mtype+' = '+str(magnitude) 
            for staid in self.waveforms.list():
                netcode, stacode=staid.split('.')
                infname=outdir+'/'+evid+'.'+staid+'.mseed'
                if os.path.isfile(infname):
                    self.add_waveforms(infname, event_id=event_id, tag=tag)
                    if deletemseed: os.remove(infname)
                elif os.path.isfile(outdir+'/'+evid+'.'+staid+'.no_resp.mseed'): no_resp+=1
        print '================================== End reading downloaded surface wave data =================================='
        print 'Number of file without resp:', no_resp
        return
    
    def get_body_waveforms(self, minDelta=30, maxDelta=150, channel='BHE,BHN,BHZ', phase='P',
                        startoffset=-30., endoffset=60.0, verbose=True, rotation=True):
        """Get body wave data from IRIS server
        ====================================================================================================================
        Input Parameters:
        min/maxDelta    - minimum/maximum epicentral distance, in degree
        channel         - Channel code, e.g. 'BHZ'.
                            Last character (i.e. component) can be a wildcard (‘?’ or ‘*’) to fetch Z, N and E component.
        phase           - body wave phase to be downloaded, arrival time will be computed using taup
        start/endoffset - start and end offset for downloaded data
        vmin, vmax      - minimum/maximum velocity for surface wave window
        rotation        - rotate the seismogram to RT or not
        =====================================================================================================================
        """
        client=Client('IRIS')
        evnumb=0
        L=len(self.events)
        print '================================== Getting body wave data ====================================='
        for event in self.events:
            event_id=event.resource_id.id.split('=')[-1]
            magnitude=event.magnitudes[0].mag; Mtype=event.magnitudes[0].magnitude_type
            event_descrip=event.event_descriptions[0].text+', '+event.event_descriptions[0].type
            evnumb+=1
            otime=event.origins[0].time 
            print 'Event ' + str(evnumb)+' : '+ str(otime)+' '+ event_descrip+', '+Mtype+' = '+str(magnitude) 
            evlo=event.origins[0].longitude; evla=event.origins[0].latitude; evdp=event.origins[0].depth/1000.
            tag='body_ev_%05d' %evnumb
            for staid in self.waveforms.list():
                netcode, stacode=staid.split('.')
                stla, elev, stlo=self.waveforms[staid].coordinates.values(); elev=elev/1000.
                az, baz, dist = geodist.inv(evlo, evla, stlo, stla); dist=dist/1000.
                if baz<0.: baz+=360.
                Delta=obspy.geodetics.kilometer2degrees(dist)
                if Delta<minDelta: continue
                if Delta>maxDelta: continue
                arrivals = taupmodel.get_travel_times(source_depth_in_km=evdp, distance_in_degree=Delta, phase_list=[phase])#, receiver_depth_in_km=0)
                try:
                    arr=arrivals[0]; arrival_time=arr.time; rayparam=arr.ray_param_sec_degree
                except IndexError: continue
                starttime=otime+arrival_time+startoffset; endtime=otime+arrival_time+endoffset
                location=self.waveforms[staid].StationXML[0].stations[0].channels[0].location_code
                try:
                    st= client.get_waveforms(network=netcode, station=stacode, location=location, channel=channel,
                            starttime=starttime, endtime=endtime, attach_response=True)
                except:
                    if verbose: print 'No data for:', staid
                    continue
                pre_filt = (0.04, 0.05, 20., 25.)
                st.detrend()
                st.remove_response(pre_filt=pre_filt, taper_fraction=0.1)
                if rotation: st.rotate('NE->RT', back_azimuth=baz)
                if verbose: print 'Getting data for:', staid
                self.add_waveforms(st, event_id=event_id, tag=tag, labels=phase)
            # print '===================================== Removing response ======================================='
        return
    
    def get_body_waveforms_mp(self, outdir, minDelta=30, maxDelta=150, channel='BHE,BHN,BHZ', phase='P', startoffset=-30., endoffset=60.0,
            verbose=False, subsize=1000, deletemseed=False, nprocess=6, snumb=0, enumb=None, rotation=True):
        """Get body wave data from IRIS server
        ====================================================================================================================
        Input Parameters:
        min/maxDelta    - minimum/maximum epicentral distance, in degree
        channel         - Channel code, e.g. 'BHZ'.
                            Last character (i.e. component) can be a wildcard (‘?’ or ‘*’) to fetch Z, N and E component.
        phase           - body wave phase to be downloaded, arrival time will be computed using taup
        start/endoffset - start and end offset for downloaded data
        vmin, vmax      - minimum/maximum velocity for surface wave window
        rotation        - rotate the seismogram to RT or not
        deletemseed     - delete output MiniSeed files
        nprocess        - number of processes
        snumb, enumb    - start/end number of processing block
        =====================================================================================================================
        """
        client=Client('IRIS')
        evnumb=0
        L=len(self.events)
        if not os.path.isdir(outdir): os.makedirs(outdir)
        reqwaveLst=[]
        print '================================== Preparing download body wave data ======================================'
        swave=snumb*subsize
        iwave=0
        for event in self.events:
            magnitude=event.magnitudes[0].mag; Mtype=event.magnitudes[0].magnitude_type
            event_descrip=event.event_descriptions[0].text+', '+event.event_descriptions[0].type
            evnumb+=1
            otime=event.origins[0].time 
            print 'Event ' + str(evnumb)+' : '+ str(otime)+' '+ event_descrip+', '+Mtype+' = '+str(magnitude) 
            evlo=event.origins[0].longitude; evla=event.origins[0].latitude; evdp=event.origins[0].depth/1000.
            for staid in self.waveforms.list():
                iwave+=1
                if iwave < swave: continue
                netcode, stacode=staid.split('.')
                stla, elev, stlo=self.waveforms[staid].coordinates.values(); elev=elev/1000.
                az, baz, dist = geodist.inv(evlo, evla, stlo, stla); dist=dist/1000.
                if baz<0.: baz+=360.
                Delta=obspy.geodetics.kilometer2degrees(dist)
                if Delta<minDelta: continue
                if Delta>maxDelta: continue
                arrivals = taupmodel.get_travel_times(source_depth_in_km=evdp, distance_in_degree=Delta, phase_list=[phase])#, receiver_depth_in_km=0)
                try:
                    arr=arrivals[0]; arrival_time=arr.time; rayparam=arr.ray_param_sec_degree
                except IndexError: continue
                starttime=otime+arrival_time+startoffset; endtime=otime+arrival_time+endoffset
                location=self.waveforms[staid].StationXML[0].stations[0].channels[0].location_code
                reqwaveLst.append( requestInfo(evnumb=evnumb, network=netcode, station=stacode, location=location, channel=channel,
                            starttime=starttime, endtime=endtime, attach_response=True, baz=baz) )
        print '============================= Start multiprocessing download body wave data ==============================='
        if len(reqwaveLst) > subsize:
            Nsub = int(len(reqwaveLst)/subsize)
            # if enumb==None: enumb=Nsub
            for isub in xrange(Nsub):
                # if isub < snumb: continue
                # if isub > enumb: continue
                print 'Subset:', isub+1,'in',Nsub,'sets'
                creqlst=reqwaveLst[isub*subsize:(isub+1)*subsize]
                GETDATA = partial(get_waveforms4mp, outdir=outdir, client=client, pre_filt = (0.04, 0.05, 20., 25.), verbose=verbose, rotation=rotation)
                pool = multiprocessing.Pool(processes=nprocess)
                pool.map(GETDATA, creqlst) #make our results with a map call
                pool.close() #we are not adding any more processes
                pool.join() #tell it to wait until all threads are done before going on
            creqlst=reqwaveLst[(isub+1)*subsize:]
            GETDATA = partial(get_waveforms4mp, outdir=outdir, client=client, pre_filt = (0.04, 0.05, 20., 25.), verbose=verbose, rotation=rotation)
            pool = multiprocessing.Pool(processes=nprocess)
            pool.map(GETDATA, creqlst) #make our results with a map call
            pool.close() #we are not adding any more processes
            pool.join() #tell it to wait until all threads are done before going on
        else:
            GETDATA = partial(get_waveforms4mp, outdir=outdir, client=client, pre_filt = (0.04, 0.05, 20., 25.), verbose=verbose, rotation=rotation)
            pool = multiprocessing.Pool(processes=nprocess)
            pool.map(GETDATA, reqwaveLst) #make our results with a map call
            pool.close() #we are not adding any more processes
            pool.join() #tell it to wait until all threads are done before going on
        print '============================= End of multiprocessing download body wave data =============================='
        print '==================================== Reading downloaded body wave data ===================================='
        evnumb=0
        no_resp=0
        for event in self.events:
            event_id=event.resource_id.id.split('=')[-1]
            magnitude=event.magnitudes[0].mag; Mtype=event.magnitudes[0].magnitude_type
            event_descrip=event.event_descriptions[0].text+', '+event.event_descriptions[0].type
            evnumb+=1
            evid='E%05d' %evnumb
            tag='body_ev_%05d' %evnumb
            print 'Event ' + str(evnumb)+' : '+event_descrip+', '+Mtype+' = '+str(magnitude) 
            for staid in self.waveforms.list():
                netcode, stacode=staid.split('.')
                infname=outdir+'/'+evid+'.'+staid+'.mseed'
                if os.path.isfile(infname):
                    self.add_waveforms(infname, event_id=event_id, tag=tag, labels=phase)
                    if deletemseed: os.remove(infname)
                elif os.path.isfile(outdir+'/'+evid+'.'+staid+'.no_resp.mseed'): no_resp+=1
        print '================================== End reading downloaded body wave data =================================='
        print 'Number of file without resp:', no_resp
        return
        
    def write2sac(self, network, station, evnumb, datatype='body'):
        """ Extract data from ASDF to SAC file
        ====================================================================================================================
        Input Parameters:
        network, station    - specify station
        evnumb              - event id
        datatype            - data type ('body' - body wave, 'sruf' - surface wave)
        =====================================================================================================================
        """
        event = self.events[evnumb-1]
        otime=event.origins[0].time
        tag=datatype+'_ev_%05d' %evnumb
        st=self.waveforms[network+'.'+station][tag]
        stla, elev, stlo=self.waveforms[network+'.'+station].coordinates.values()
        evlo=event.origins[0].longitude; evla=event.origins[0].latitude; evdp=event.origins[0].depth
        for tr in st:
            tr.stats.sac=obspy.core.util.attribdict.AttribDict()
            tr.stats.sac['evlo']=evlo; tr.stats.sac['evla']=evla; tr.stats.sac['evdp']=evdp
            tr.stats.sac['stlo']=stlo; tr.stats.sac['stla']=stla    
        st.write(str(otime)+'..sac', format='sac')
    
    def get_obspy_trace(self, network, station, evnumb, datatype='body'):
        """ Get obspy trace data from ASDF
        ====================================================================================================================
        Input Parameters:
        network, station    - specify station
        evnumb              - event id
        datatype            - data type ('body' - body wave, 'sruf' - surface wave)
        =====================================================================================================================
        """
        event = self.events[evnumb-1]
        tag=datatype+'_ev_%05d' %evnumb
        st=self.waveforms[network+'.'+station][tag]
        stla, elev, stlo=self.waveforms[network+'.'+station].coordinates.values()
        evlo=event.origins[0].longitude; evla=event.origins[0].latitude; evdp=event.origins[0].depth
        for tr in st:
            tr.stats.sac=obspy.core.util.attribdict.AttribDict()
            tr.stats.sac['evlo']=evlo; tr.stats.sac['evla']=evla; tr.stats.sac['evdp']=evdp
            tr.stats.sac['stlo']=stlo; tr.stats.sac['stla']=stla    
        return st
    
    def compute_ref(self, inrefparam=CURefPy.InputRefparam(), savescaled=True, savemoveout=True, verbose=True):
        """Compute receiver function and post processed data(moveout, stretchback)
        ====================================================================================================================
        Input Parameters:
        inrefparam      - input parameters, refer to InputRefparam in CURefPy for details
        savescaled      - save scaled post processed data
        savemoveout     - save moveout data
        =====================================================================================================================
        """
        print '================================== Receiver Function Analysis ======================================'
        for staid in self.waveforms.list():
            netcode, stacode=staid.split('.')
            print 'Station: '+staid
            stla, elev, stlo=self.waveforms[staid].coordinates.values()
            evnumb=0
            for event in self.events:
                evnumb+=1
                evid='E%05d' %evnumb
                tag='body_ev_%05d' %evnumb
                try: st=self.waveforms[staid][tag]
                except KeyError: continue
                phase=st[0].stats.asdf.labels[0]
                if inrefparam.phase != '' and inrefparam.phase != phase: continue
                evlo=event.origins[0].longitude; evla=event.origins[0].latitude; evdp=event.origins[0].depth
                otime=event.origins[0].time
                for tr in st:
                    tr.stats.sac=obspy.core.util.attribdict.AttribDict()
                    tr.stats.sac['evlo']=evlo; tr.stats.sac['evla']=evla; tr.stats.sac['evdp']=evdp
                    tr.stats.sac['stlo']=stlo; tr.stats.sac['stla']=stla
                if verbose:
                    magnitude=event.magnitudes[0].mag; Mtype=event.magnitudes[0].magnitude_type
                    event_descrip=event.event_descriptions[0].text+', '+event.event_descriptions[0].type
                    print 'Event ' + str(evnumb)+' : '+event_descrip+', '+Mtype+' = '+str(magnitude) 
                refTr=CURefPy.RFTrace()
                refTr.get_data(Ztr=st.select(component='Z')[0], RTtr=st.select(component=inrefparam.reftype)[0],
                        tbeg=inrefparam.tbeg, tend=inrefparam.tend)
                refTr.IterDeconv(tdel=inrefparam.tdel, f0 = inrefparam.f0, niter=inrefparam.niter,
                        minderr=inrefparam.minderr, phase=phase )
                ref_header=ref_header_default.copy()
                ref_header['otime']=str(otime); ref_header['network']=netcode; ref_header['station']=stacode
                ref_header['stla']=stla; ref_header['stlo']=stlo; ref_header['evla']=evla; ref_header['evlo']=evlo; ref_header['evdp']=evdp
                ref_header['dist']=refTr.stats.sac['dist']; ref_header['az']=refTr.stats.sac['az']; ref_header['baz']=refTr.stats.sac['baz']
                ref_header['delta']=refTr.stats.delta; ref_header['npts']=refTr.stats.npts; ref_header['b']=refTr.stats.sac['b']; ref_header['e']=refTr.stats.sac['e']
                ref_header['arrival']=refTr.stats.sac['user5']; ref_header['phase']=phase; ref_header['tbeg']=inrefparam.tbeg; ref_header['tend']=inrefparam.tend
                ref_header['hslowness']=refTr.stats.sac['user4']; ref_header['ghw']=inrefparam.f0; ref_header['VR']=refTr.stats.sac['user2']
                staid_aux=netcode+'_'+stacode+'_'+phase+'/'+evid
                self.add_auxiliary_data(data=refTr.data, data_type='Ref'+inrefparam.reftype, path=staid_aux, parameters=ref_header)
                if not refTr.move_out(): continue
                refTr.stretch_back()
                postdbase=refTr.postdbase
                ref_header['moveout']=postdbase.MoveOutFlag
                if savescaled: self.add_auxiliary_data(data=postdbase.ampC, data_type='Ref'+inrefparam.reftype+'scaled', path=staid_aux, parameters=ref_header)
                if savemoveout: self.add_auxiliary_data(data=postdbase.ampTC, data_type='Ref'+inrefparam.reftype+'moveout', path=staid_aux, parameters=ref_header)
                self.add_auxiliary_data(data=postdbase.strback, data_type='Ref'+inrefparam.reftype+'streback', path=staid_aux, parameters=ref_header)
        return
    
    def compute_ref_mp(self, outdir, inrefparam=CURefPy.InputRefparam(), savescaled=True, savemoveout=True, \
                verbose=False, subsize=1000, deleteref=True, deletepost=True, nprocess=None):
        """Compute receiver function and post processed data(moveout, stretchback) with multiprocessing
        ====================================================================================================================
        Input Parameters:
        inrefparam      - input parameters, refer to InputRefparam in CURefPy for details
        savescaled      - save scaled post processed data
        savemoveout     - save moveout data
        subsize         - subsize of processing list, use to prevent lock in multiprocessing process
        deleteref       - delete SAC receiver function data
        deletepost      - delete npz post processed data
        nprocess        - number of processes
        =====================================================================================================================
        """
        print '================================== Receiver Function Analysis ======================================'
        print 'Preparing data for multiprocessing'
        refLst=[]
        for staid in self.waveforms.list():
            netcode, stacode=staid.split('.')
            print 'Station: '+staid
            stla, elev, stlo=self.waveforms[staid].coordinates.values()
            evnumb=0
            outsta=outdir+'/'+staid
            if not os.path.isdir(outsta): os.makedirs(outsta)
            for event in self.events:
                evnumb+=1
                evid='E%05d' %evnumb
                tag='body_ev_%05d' %evnumb
                try: st=self.waveforms[staid][tag]
                except KeyError: continue
                phase=st[0].stats.asdf.labels[0]
                if inrefparam.phase != '' and inrefparam.phase != phase: continue
                evlo=event.origins[0].longitude; evla=event.origins[0].latitude; evdp=event.origins[0].depth
                otime=event.origins[0].time
                for tr in st:
                    tr.stats.sac=obspy.core.util.attribdict.AttribDict()
                    tr.stats.sac['evlo']=evlo; tr.stats.sac['evla']=evla; tr.stats.sac['evdp']=evdp
                    tr.stats.sac['stlo']=stlo; tr.stats.sac['stla']=stla; tr.stats.sac['kuser0']=evid; tr.stats.sac['kuser1']=phase
                if verbose:
                    magnitude=event.magnitudes[0].mag; Mtype=event.magnitudes[0].magnitude_type
                    event_descrip=event.event_descriptions[0].text+', '+event.event_descriptions[0].type
                    print 'Event ' + str(evnumb)+' : '+event_descrip+', '+Mtype+' = '+str(magnitude) 
                refTr=CURefPy.RFTrace()
                refTr.get_data(Ztr=st.select(component='Z')[0], RTtr=st.select(component=inrefparam.reftype)[0],
                        tbeg=inrefparam.tbeg, tend=inrefparam.tend)
                refLst.append( refTr )
        print 'Start multiprocessing receiver function analysis !'
        if len(refLst) > subsize:
            Nsub = int(len(refLst)/subsize)
            for isub in xrange(Nsub):
                print 'Subset:', isub,'in',Nsub,'sets'
                cstream=refLst[isub*subsize:(isub+1)*subsize]
                REF = partial(ref4mp, outdir=outsta, inrefparam=inrefparam)
                pool = multiprocessing.Pool(processes=nprocess)
                pool.map(AFTAN, cstream) #make our results with a map call
                pool.close() #we are not adding any more processes
                pool.join() #tell it to wait until all threads are done before going on
            cstream=refLst[(isub+1)*subsize:]
            REF = partial(ref4mp, outdir=outsta, inrefparam=inrefparam)
            pool = multiprocessing.Pool(processes=nprocess)
            pool.map(REF, cstream) #make our results with a map call
            pool.close() #we are not adding any more processes
            pool.join() #tell it to wait until all threads are done before going on
        else:
            REF = partial(ref4mp, outdir=outsta, inrefparam=inrefparam)
            pool = multiprocessing.Pool(processes=nprocess)
            pool.map(REF, refLst) #make our results with a map call
            pool.close() #we are not adding any more processes
            pool.join() #tell it to wait until all threads are done before going on
        print 'End of multiprocessing receiver function analysis !'
        print 'Start reading receiver function data !'
        for staid in self.waveforms.list():
            netcode, stacode=staid.split('.')
            print 'Station: '+staid
            stla, elev, stlo=self.waveforms[staid].coordinates.values()
            outsta=outdir+'/'+staid
            evnumb=0
            for event in self.events:
                evnumb+=1
                evid='E%05d' %evnumb
                sacfname=outsta+'/'+evid+'.sac'; postfname = outsta+'/'+evid+'.post.npz'
                if not os.path.isfile(sacfname): continue
                evlo=event.origins[0].longitude; evla=event.origins[0].latitude; evdp=event.origins[0].depth
                otime=event.origins[0].time
                refTr=obspy.read(sacfname)[0]
                ref_header=ref_header_default.copy()
                ref_header['otime']=str(otime); ref_header['network']=netcode; ref_header['station']=stacode
                ref_header['stla']=stla; ref_header['stlo']=stlo; ref_header['evla']=evla; ref_header['evlo']=evlo; ref_header['evdp']=evdp
                ref_header['dist']=refTr.stats.sac['dist']; ref_header['az']=refTr.stats.sac['az']; ref_header['baz']=refTr.stats.sac['baz']
                ref_header['delta']=refTr.stats.delta; ref_header['npts']=refTr.stats.npts; ref_header['b']=refTr.stats.sac['b']; ref_header['e']=refTr.stats.sac['e']
                ref_header['arrival']=refTr.stats.sac['user5']; ref_header['phase']=refTr.stats.sac['kuser1']; ref_header['tbeg']=inrefparam.tbeg; ref_header['tend']=inrefparam.tend
                ref_header['hslowness']=refTr.stats.sac['user4']; ref_header['ghw']=inrefparam.f0; ref_header['VR']=refTr.stats.sac['user2']
                staid_aux=netcode+'_'+stacode+'_'+phase+'/'+evid
                self.add_auxiliary_data(data=refTr.data, data_type='Ref'+inrefparam.reftype, path=staid_aux, parameters=ref_header)
                if deleteref: os.remove(sacfname)
                if not os.path.isfile(postfname): continue
                ref_header['moveout']=1
                postArr=np.load(postfname)
                ampC=postArr['arr_0']; ampTC=postArr['arr_1']; strback=postArr['arr_2']
                if deletepost: os.remove(postfname)
                if savescaled: self.add_auxiliary_data(data=ampC, data_type='Ref'+inrefparam.reftype+'scaled', path=staid_aux, parameters=ref_header)
                if savemoveout: self.add_auxiliary_data(data=ampTC, data_type='Ref'+inrefparam.reftype+'moveout', path=staid_aux, parameters=ref_header)
                self.add_auxiliary_data(data=strback, data_type='Ref'+inrefparam.reftype+'streback', path=staid_aux, parameters=ref_header)
            if deleteref*deletepost: shutil.rmtree(outsta)
        print 'End reading receiver function data !'       
        return
    
    def harmonic_stripping(self, outdir, data_type='RefRstreback', VR=80, tdiff=0.08, phase='P', reftype='R'):
        """Harmonic stripping analysis
        ====================================================================================================================
        Input Parameters:
        outdir          - output directory
        data_type       - datatype, default is 'RefRstreback', stretchback radial receiver function
        VR              - threshold variance reduction for quality control
        tdiff           - threshold trace difference for quality control
        phase           - phase, default = 'P'
        reftype         - receiver function type, default = 'R'
        =====================================================================================================================
        """
        print '================================== Harmonic Stripping Analysis ======================================'
        for staid in self.waveforms.list():
            netcode, stacode=staid.split('.')
            print 'Station: '+staid
            stla, elev, stlo=self.waveforms[staid].coordinates.values()
            evnumb=0
            postLst=CURefPy.PostRefLst()
            outsta=outdir+'/'+staid
            if not os.path.isdir(outsta): os.makedirs(outsta)
            for event in self.events:
                evnumb+=1
                evid='E%05d' %evnumb
                try: subdset=self.auxiliary_data[data_type][netcode+'_'+stacode+'_'+phase][evid]
                except KeyError: continue
                ref_header=subdset.parameters
                if ref_header['moveout'] <0 or ref_header['VR'] < VR: continue
                pdbase=CURefPy.PostDatabase()
                pdbase.strback=subdset.data.value; pdbase.header=subdset.parameters
                postLst.append(pdbase)
            qcLst = postLst.remove_bad(outsta)
            qcLst = qcLst.QControl_tdiff(tdiff=tdiff)
            qcLst.HarmonicStripping(outdir=outsta, stacode=staid)
            staid_aux=netcode+'_'+stacode+'_'+phase
            # wmean.txt
            wmeanArr=np.loadtxt(outsta+'/wmean.txt'); os.remove(outsta+'/wmean.txt')
            self.add_auxiliary_data(data=wmeanArr, data_type='Ref'+reftype+'HS',
                    path=staid_aux+'/wmean', parameters={})
            # bin_%d_txt
            for binfname in glob.glob(outsta+'/bin_*_txt'):
                binArr=np.loadtxt(binfname); os.remove(binfname)
                temp=binfname.split('/')[-1]
                self.add_auxiliary_data(data=binArr, data_type='Ref'+reftype+'HS',
                        path=staid_aux+'/bin/'+temp.split('_')[0]+'_'+temp.split('_')[1], parameters={})
            for binfname in glob.glob(outsta+'/bin_*_rf.dat'):
                binArr=np.loadtxt(binfname); os.remove(binfname)
                temp=binfname.split('/')[-1]
                self.add_auxiliary_data(data=binArr, data_type='Ref'+reftype+'HS',
                        path=staid_aux+'/bin_rf/'+temp.split('_')[0]+'_'+temp.split('_')[1], parameters={})
            # A0.dat
            A0Arr=np.loadtxt(outsta+'/A0.dat'); os.remove(outsta+'/A0.dat')
            self.add_auxiliary_data(data=A0Arr, data_type='Ref'+reftype+'HS',
                    path=staid_aux+'/A0', parameters={})
            # A1.dat
            A1Arr=np.loadtxt(outsta+'/A1.dat'); os.remove(outsta+'/A1.dat')
            self.add_auxiliary_data(data=A1Arr, data_type='Ref'+reftype+'HS',
                    path=staid_aux+'/A1', parameters={})
            # A2.dat
            A2Arr=np.loadtxt(outsta+'/A2.dat'); os.remove(outsta+'/A2.dat')
            self.add_auxiliary_data(data=A2Arr, data_type='Ref'+reftype+'HS',
                    path=staid_aux+'/A2', parameters={})
            # A0_A1_A2.dat
            A0A1A2Arr=np.loadtxt(outsta+'/A0_A1_A2.dat'); os.remove(outsta+'/A0_A1_A2.dat')
            self.add_auxiliary_data(data=A0A1A2Arr, data_type='Ref'+reftype+'HS',
                    path=staid_aux+'/A0_A1_A2', parameters={})
            evnumb=0
            for event in self.events:
                evnumb+=1
                evid='E%05d' %evnumb
                try: subdset=self.auxiliary_data[data_type][netcode+'_'+stacode+'_'+phase][evid]
                except KeyError: continue
                ref_header=subdset.parameters
                if ref_header['moveout'] <0 or ref_header['VR'] < VR: continue
                otime=ref_header['otime']; baz=ref_header['baz']
                fsfx=str(int(baz))+'_'+staid+'_'+otime+'.out.back'
                diff_fname=outsta+'/diffstre_'+fsfx
                obsfname=outsta+'/obsstre_'+fsfx
                repfname=outsta+'/repstre_'+fsfx
                rep0fname=outsta+'/0repstre_'+fsfx
                rep1fname=outsta+'/1repstre_'+fsfx
                rep2fname=outsta+'/2repstre_'+fsfx
                prefname=outsta+'/prestre_'+fsfx
                if not (os.path.isfile(diff_fname) and os.path.isfile(obsfname) and os.path.isfile(repfname) and \
                        os.path.isfile(rep0fname) and os.path.isfile(rep1fname) and os.path.isfile(rep2fname) and os.path.isfile(prefname) ):
                    continue
                diffArr=np.loadtxt(diff_fname); os.remove(diff_fname)
                obsArr=np.loadtxt(obsfname); os.remove(obsfname)
                repArr=np.loadtxt(repfname); os.remove(repfname)
                rep0Arr=np.loadtxt(rep0fname); os.remove(rep0fname)
                rep1Arr=np.loadtxt(rep1fname); os.remove(rep1fname)
                rep2Arr=np.loadtxt(rep2fname); os.remove(rep2fname)
                preArr=np.loadtxt(prefname); os.remove(prefname)
                self.add_auxiliary_data(data=obsArr, data_type='Ref'+reftype+'HS',
                    path=staid_aux+'/obs/'+evid, parameters=ref_header)
                self.add_auxiliary_data(data=diffArr, data_type='Ref'+reftype+'HS',
                    path=staid_aux+'/diff/'+evid, parameters=ref_header)
                self.add_auxiliary_data(data=repArr, data_type='Ref'+reftype+'HS',
                    path=staid_aux+'/rep/'+evid, parameters=ref_header)
                self.add_auxiliary_data(data=rep0Arr, data_type='Ref'+reftype+'HS',
                    path=staid_aux+'/rep0/'+evid, parameters=ref_header)
                self.add_auxiliary_data(data=rep1Arr, data_type='Ref'+reftype+'HS',
                    path=staid_aux+'/rep1/'+evid, parameters=ref_header)
                self.add_auxiliary_data(data=rep2Arr, data_type='Ref'+reftype+'HS',
                    path=staid_aux+'/rep2/'+evid, parameters=ref_header)
                self.add_auxiliary_data(data=preArr, data_type='Ref'+reftype+'HS',
                    path=staid_aux+'/pre/'+evid, parameters=ref_header)
        
        return
    
    def plot_ref(self, network, station, phase='P', datatype='RefRHS'):
        """plot receiver function
        ====================================================================================================================
        Input Parameters:
        network, station    - specify station
        phase               - phase, default = 'P'
        datatype            - datatype, default = 'RefRHS', harmonic striped radial receiver function
        =====================================================================================================================
        """
        obsHSstream=CURefPy.HStripStream()
        diffHSstream=CURefPy.HStripStream()
        repHSstream=CURefPy.HStripStream()
        rep0HSstream=CURefPy.HStripStream()
        rep1HSstream=CURefPy.HStripStream()
        rep2HSstream=CURefPy.HStripStream()
        subgroup=self.auxiliary_data[datatype][network+'_'+station+'_'+phase]
        stla, elev, stlo=self.waveforms[network+'.'+station].coordinates.values()
        for evid in subgroup.obs.list():
            ref_header=subgroup['obs'][evid].parameters
            dt=ref_header['delta']; baz=ref_header['baz']; eventT=ref_header['otime']
            obsArr=subgroup['obs'][evid].data.value
            starttime=obspy.core.utcdatetime.UTCDateTime(eventT)+ref_header['arrival']-ref_header['tbeg']+30.
            obsHSstream.get_trace(network=network, station=station, indata=obsArr[:, 1], baz=baz, dt=dt, starttime=starttime)
            
            diffArr=subgroup['diff'][evid].data.value
            diffHSstream.get_trace(network=network, station=station, indata=diffArr[:, 1], baz=baz, dt=dt, starttime=starttime)
            
            repArr=subgroup['rep'][evid].data.value
            repHSstream.get_trace(network=network, station=station, indata=repArr[:, 1], baz=baz, dt=dt, starttime=starttime)
            
            rep0Arr=subgroup['rep0'][evid].data.value
            rep0HSstream.get_trace(network=network, station=station, indata=rep0Arr[:, 1], baz=baz, dt=dt, starttime=starttime)
            
            rep1Arr=subgroup['rep1'][evid].data.value
            rep1HSstream.get_trace(network=network, station=station, indata=rep1Arr[:, 1], baz=baz, dt=dt, starttime=starttime)
            
            rep2Arr=subgroup['rep2'][evid].data.value
            rep2HSstream.get_trace(network=network, station=station, indata=rep2Arr[:, 1], baz=baz, dt=dt, starttime=starttime)
        self.HSDataBase=CURefPy.HarmonicStrippingDataBase(obsST=obsHSstream, diffST=diffHSstream, repST=repHSstream,\
            repST0=rep0HSstream, repST1=rep1HSstream, repST2=rep2HSstream)
        self.HSDataBase.PlotHSStreams(stacode=network+'.'+station, longitude=stlo, latitude=stla)
        return

    def array_processing(self, evnumb=1, win_len=20., win_frac=0.2, sll_x=-3.0, slm_x=3.0, sll_y=-3.0, slm_y=3.0, sl_s=0.03,
            frqlow=0.0125, frqhigh=0.02, semb_thres=-1e9, vel_thres=-1e9, prewhiten=0, verbose=True, coordsys='lonlat', timestamp='mlabday',
                method=0, minlat=None, maxlat=None, minlon=None, maxlon=None, lon0=None, lat0=None, radius=None, Tmin=None, Tmax=None, vmax=5.0, vmin=2.0):
        """Array processing ( beamforming/fk analysis )
        ==============================================================================================================================================
        Input Parameters:
        evnumb          - event number for analysis
        win_len         - Sliding window length in seconds
        win_frac        - Fraction of sliding window to use for step
        sll_x, slm_x    - slowness x min/max
        sll_y, slm_y    - slowness y min/max 
        sl_s            - slowness step
        semb_thres      - Threshold for semblance
        vel_thres       - Threshold for velocity
        frqlow, frqhigh - lower/higher frequency for fk/capon
        prewhiten       - Do prewhitening, values: 1 or 0
        coordsys        - valid values: ‘lonlat’ and ‘xy’, choose which stream attributes to use for coordinates
        timestamp       - valid values: ‘julsec’ and ‘mlabday’; ‘julsec’ returns the timestamp in seconds since 1970-01-01T00:00:00,
                            ‘mlabday’ returns the timestamp in days (decimals represent hours, minutes and seconds) since ‘0001-01-01T00:00:00’
                                as needed for matplotlib date plotting (see e.g. matplotlib’s num2date)
        method          - the method to use 0 == bf, 1 == capon
        minlat, maxlat  - latitude limit for stations
        minlon, maxlon  - longitude limit for stations
        lon0, lat0      - origin for radius selection
        radius          - radius for station selection
        Tmin, Tmax      - minimum/maximum time
        vmin, vmax      - minimum/maximum velocity for surface wave window, will not be used if Tmin or Tmax is specified
        ==============================================================================================================================================
        """
        # prepare Stream data
        tag='surf_ev_%05d' %evnumb
        st=obspy.Stream()
        lons=np.array([]); lats=np.array([])
        for staid in self.waveforms.list():
            stla, elev, stlo=self.waveforms[staid].coordinates.values()
            if minlat!=None:
                if stla<minlat: continue
            if maxlat!=None:
                if stla>maxlat: continue
            if minlon!=None:
                if stlo<minlon: continue
            if maxlon!=None:
                if stlo>maxlon: continue
            if lon0 !=None and lat0!=None and radius !=None:
                dist, az, baz=obspy.geodetics.gps2dist_azimuth(lat0, lon0, stla, stlo) # distance is in m
                if dist/1000>radius: continue
            try:
                tr = self.waveforms[staid][tag][0].copy()
                tr.stats.coordinates = obspy.core.util.attribdict.AttribDict({
                    'latitude': stla,
                    'elevation': elev,
                    'longitude': stlo})
                st.append(tr)
                lons=np.append(lons, stlo)
                lats=np.append(lats, stla)
            except KeyError:
                print 'no data:', staid
        if len(st)==0: print 'No data for array processing!'; return
        event = self.events[0]
        evlo=event.origins[0].longitude; evla=event.origins[0].latitude
        if lon0 !=None and lat0!=None:
            dist0, az, baz=obspy.geodetics.gps2dist_azimuth(lat0, lon0, evla, evlo) # distance is in m
        else:
            try: meanlat=(minlat+maxlat)/2; meanlon=(minlon+maxlon)/2; dist0, az, baz=obspy.geodetics.gps2dist_azimuth(lat0, lon0, meanlat, meanlon)
            except: dist0, az, baz=obspy.geodetics.gps2dist_azimuth(lat0, lon0, lats.mean(), lons.mean())
        dist0=dist0/1000.
        otime=event.origins[0].time
        if Tmin==None: stime=otime+dist0/vmax
        else: stime=otime+Tmin
        if Tmax==None: etime=otime+dist0/vmin
        else: etime=otime+Tmax
        # set input
        kwargs = dict(
            # slowness grid: X min, X max, Y min, Y max, Slow Step
            sll_x=sll_x, slm_x=slm_x, sll_y=sll_y, slm_y=slm_y, sl_s=sl_s,
            # sliding window properties
            win_len=win_len, win_frac=win_frac,
            # frequency properties
            frqlow=frqlow, frqhigh=frqhigh, prewhiten=0,
            # restrict output
            semb_thres=semb_thres, vel_thres=vel_thres, timestamp=timestamp,
            stime=stime, etime=etime, method=method,
            verbose=verbose
        )
        # array analysis
        out = obspy.signal.array_analysis.array_processing(st, **kwargs)
        # Plot
        labels = ['rel.power', 'abs.power', 'baz', 'slow']
        xlocator = mdates.AutoDateLocator()
        fig = plt.figure()
        for i, lab in enumerate(labels):
            ax = fig.add_subplot(4, 1, i + 1)
            ax.scatter(out[:, 0], out[:, i + 1], c=out[:, 1], alpha=0.6,
                       edgecolors='none', cmap=obspy_sequential)
            ax.set_ylabel(lab)
            ax.set_xlim(out[0, 0], out[-1, 0])
            ax.set_ylim(out[:, i + 1].min(), out[:, i + 1].max())
            ax.xaxis.set_major_locator(xlocator)
            ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(xlocator))
        
        fig.suptitle('Array analysis %s' % (
            stime.strftime('%Y-%m-%d'), ))
        fig.autofmt_xdate()
        fig.subplots_adjust(left=0.15, top=0.95, right=0.95, bottom=0.2, hspace=0)
        plt.show()
        return 
    
    def quake_prephp(self, outdir, mapfile='./MAPS/smpkolya_phv'):
        """
        Generate predicted phase velocity dispersion curves for event-station pairs
        ====================================================================================
        Input Parameters:
        outdir  - output directory
        mapfile - phase velocity maps
        ------------------------------------------------------------------------------------
        Input format:
        prephaseEXE pathfname mapfile perlst staname
        
        Output format:
        outdirL(outdirR)/evid.staid.pre
        ====================================================================================
        """
        staLst=self.waveforms.list()
        evnumb=0
        for event in self.events:
            evnumb+=1
            evlo=event.origins[0].longitude; evla=event.origins[0].latitude
            evid='E%05d' % evnumb
            pathfname=evid+'_pathfile'
            prephaseEXE='./mhr_grvel_predict/lf_mhr_predict_earth'
            perlst='./mhr_grvel_predict/perlist_phase'
            if not os.path.isfile(prephaseEXE):
                print 'lf_mhr_predict_earth executable does not exist!'
                return
            if not os.path.isfile(perlst):
                print 'period list does not exist!'
                return
            with open(pathfname,'w') as f:
                ista=0
                for station_id in staLst:
                    stacode=station_id.split('.')[1]
                    stla, stz, stlo=self.waveforms[station_id].coordinates.values()
                    if ( abs(stlo-evlo) < 0.1 and abs(stla-evla)<0.1 ): continue
                    ista=ista+1
                    f.writelines('%5d%5d %15s %15s %10.5f %10.5f %10.5f %10.5f \n'
                            %(1, ista, evid, station_id, evla, evlo, stla, stlo ))
            call([prephaseEXE, pathfname, mapfile, perlst, evid])
            os.remove(pathfname)
            outdirL=outdir+'_L'
            outdirR=outdir+'_R'
            if not os.path.isdir(outdirL): os.makedirs(outdirL)
            if not os.path.isdir(outdirR): os.makedirs(outdirR)
            fout = open(evid+'_temp','wb')
            for l1 in open('PREDICTION_L'+'_'+evid):
                l2 = l1.rstrip().split()
                if (len(l2)>8):
                    fout.close()
                    outname = outdirL + "/%s.%s.pre" % (l2[3],l2[4])
                    fout = open(outname,"w")
                elif (len(l2)>7):
                    fout.close()
                    outname = outdirL + "/%s.%s.pre" % (l2[2],l2[3])
                    fout = open(outname,"w")                
                else:
                    fout.write("%g %g\n" % (float(l2[0]),float(l2[1])))
            for l1 in open('PREDICTION_R'+'_'+evid):
                l2 = l1.rstrip().split()
                if (len(l2)>8):
                    fout.close()
                    outname = outdirR + "/%s.%s.pre" % (l2[3],l2[4])
                    fout = open(outname,"w")
                elif (len(l2)>7):
                    fout.close()
                    outname = outdirR + "/%s.%s.pre" % (l2[2],l2[3])
                    fout = open(outname,"w")         
                else:
                    fout.write("%g %g\n" % (float(l2[0]),float(l2[1])))
            fout.close()
            os.remove(evid+'_temp')
            os.remove('PREDICTION_L'+'_'+evid)
            os.remove('PREDICTION_R'+'_'+evid)
        return
    
    def quake_aftan(self, channel='Z', tb=0., outdir=None, inftan=pyaftan.InputFtanParam(), basic1=True, basic2=True, \
            pmf1=True, pmf2=True, verbose=True, prephdir=None, f77=True, pfx='DISP'):
        """ aftan analysis of earthquake data 
        =======================================================================================
        Input Parameters:
        channel     - channel pair for aftan analysis('Z', 'R', 'T')
        tb          - begin time (default = 0.0)
        outdir      - directory for output disp txt files (default = None, no txt output)
        inftan      - input aftan parameters
        basic1      - save basic aftan results or not
        basic2      - save basic aftan results(with jump correction) or not
        pmf1        - save pmf aftan results or not
        pmf2        - save pmf aftan results(with jump correction) or not
        prephdir    - directory for predicted phase velocity dispersion curve
        f77         - use aftanf77 or not
        pfx         - prefix for output txt DISP files
        ---------------------------------------------------------------------------------------
        Output:
        self.auxiliary_data.DISPbasic1, self.auxiliary_data.DISPbasic2,
        self.auxiliary_data.DISPpmf1, self.auxiliary_data.DISPpmf2
        =======================================================================================
        """
        print 'Start aftan analysis!'
        staLst=self.waveforms.list()
        evnumb=0
        for event in self.events:
            evnumb+=1
            evlo=event.origins[0].longitude; evla=event.origins[0].latitude
            otime=event.origins[0].time
            tag='surf_ev_%05d' %evnumb
            evid='E%05d' % evnumb
            for staid in staLst:
                netcode, stacode=staid.split('.')
                stla, stz, stlo=self.waveforms[staid].coordinates.values()
                az, baz, dist = geodist.inv(evlo, evla, stlo, stla); dist=dist/1000. 
                if baz<0: baz+=360.
                try:
                    if channel!='R' or channel!='T':
                        inST=self.waveforms[staid][tag].select(component=channel)
                    else:
                        st=self.waveforms[staid][tag]
                        st.rotate('NE->RT', backazimuth=baz) 
                        inST=st.select(component=channel)
                except KeyError: continue
                if len(inST)==0: continue
                else: tr=inST[0]
                stime=tr.stats.starttime; etime=tr.stats.endtime
                tr.stats.sac={}; tr.stats.sac['dist']= dist; tr.stats.sac['b']=stime-otime; tr.stats.sac['e']=etime-otime
                aftanTr=pyaftan.aftantrace(tr.data, tr.stats)
                if prephdir !=None: phvelname = prephdir + "/%s.%s.pre" %(evid, staid)
                else: phvelname =''
                if f77:
                    aftanTr.aftanf77(pmf=inftan.pmf, piover4=inftan.piover4, vmin=inftan.vmin, vmax=inftan.vmax, tmin=inftan.tmin, tmax=inftan.tmax,
                        tresh=inftan.tresh, ffact=inftan.ffact, taperl=inftan.taperl, snr=inftan.snr, fmatch=inftan.fmatch, nfin=inftan.nfin,
                            npoints=inftan.npoints, perc=inftan.perc, phvelname=phvelname)
                else:
                    aftanTr.aftan(pmf=inftan.pmf, piover4=inftan.piover4, vmin=inftan.vmin, vmax=inftan.vmax, tmin=inftan.tmin, tmax=inftan.tmax,
                        tresh=inftan.tresh, ffact=inftan.ffact, taperl=inftan.taperl, snr=inftan.snr, fmatch=inftan.fmatch, nfin=inftan.nfin,
                            npoints=inftan.npoints, perc=inftan.perc, phvelname=phvelname)
                if verbose: print 'aftan analysis for: ' + evid+' '+staid+'_'+channel
                aftanTr.get_snr(ffact=inftan.ffact) # SNR analysis
                staid_aux=evid+'/'+netcode+'_'+stacode+'_'+channel
                # save aftan results to ASDF dataset
                if basic1:
                    parameters={'Tc': 0, 'To': 1, 'Vgr': 2, 'Vph': 3, 'ampdb': 4, 'dis': 5, 'snrdb': 6, 'mhw': 7, 'amp': 8, 'Np': aftanTr.ftanparam.nfout1_1}
                    self.add_auxiliary_data(data=aftanTr.ftanparam.arr1_1, data_type='DISPbasic1', path=staid_aux, parameters=parameters)
                if basic2:
                    parameters={'Tc': 0, 'To': 1, 'Vgr': 2, 'Vph': 3, 'ampdb': 4, 'snrdb': 5, 'mhw': 6, 'amp': 7, 'Np': aftanTr.ftanparam.nfout2_1}
                    self.add_auxiliary_data(data=aftanTr.ftanparam.arr2_1, data_type='DISPbasic2', path=staid_aux, parameters=parameters)
                if inftan.pmf:
                    if pmf1:
                        parameters={'Tc': 0, 'To': 1, 'Vgr': 2, 'Vph': 3, 'ampdb': 4, 'dis': 5, 'snrdb': 6, 'mhw': 7, 'amp': 8, 'Np': aftanTr.ftanparam.nfout1_2}
                        self.add_auxiliary_data(data=aftanTr.ftanparam.arr1_2, data_type='DISPpmf1', path=staid_aux, parameters=parameters)
                    if pmf2:
                        parameters={'Tc': 0, 'To': 1, 'Vgr': 2, 'Vph': 3, 'ampdb': 4, 'snrdb': 5, 'mhw': 6, 'amp': 7, 'snr':8, 'Np': aftanTr.ftanparam.nfout2_2}
                        self.add_auxiliary_data(data=aftanTr.ftanparam.arr2_2, data_type='DISPpmf2', path=staid_aux, parameters=parameters)
                if outdir != None:
                    if not os.path.isdir(outdir+'/'+pfx+'/'+evid):
                        os.makedirs(outdir+'/'+pfx+'/'+evid)
                    foutPR=outdir+'/'+pfx+'/'+evid+'/'+ staid+'_'+channel+'.SAC'
                    aftanTr.ftanparam.writeDISP(foutPR)
        print 'End aftan analysis!'
        return
               
    def quake_aftan_mp(self, outdir, channel='Z', tb=0., inftan=pyaftan.InputFtanParam(), basic1=True, basic2=True,
            pmf1=True, pmf2=True, verbose=True, prephdir=None, f77=True, pfx='DISP', subsize=1000, deletedisp=True, nprocess=None):
        """ aftan analysis of earthquake data with multiprocessing
        =======================================================================================
        Input Parameters:
        channel     - channel pair for aftan analysis('Z', 'R', 'T')
        tb          - begin time (default = 0.0)
        outdir      - directory for output disp binary files
        inftan      - input aftan parameters
        basic1      - save basic aftan results or not
        basic2      - save basic aftan results(with jump correction) or not
        pmf1        - save pmf aftan results or not
        pmf2        - save pmf aftan results(with jump correction) or not
        prephdir    - directory for predicted phase velocity dispersion curve
        f77         - use aftanf77 or not
        pfx         - prefix for output txt DISP files
        subsize     - subsize of processing list, use to prevent lock in multiprocessing process
        deletedisp  - delete output dispersion files or not
        nprocess    - number of processes
        ---------------------------------------------------------------------------------------
        Output:
        self.auxiliary_data.DISPbasic1, self.auxiliary_data.DISPbasic2,
        self.auxiliary_data.DISPpmf1, self.auxiliary_data.DISPpmf2
        =======================================================================================
        """
        print 'Preparing data for aftan analysis !'
        staLst=self.waveforms.list()
        inputStream=[]
        evnumb=0
        for event in self.events:
            evnumb+=1
            evlo=event.origins[0].longitude; evla=event.origins[0].latitude
            otime=event.origins[0].time
            tag='surf_ev_%05d' %evnumb
            evid='E%05d' % evnumb
            if not os.path.isdir(outdir+'/'+pfx+'/'+evid): os.makedirs(outdir+'/'+pfx+'/'+evid)
            for staid in staLst:
                netcode, stacode=staid.split('.')
                stla, stz, stlo=self.waveforms[staid].coordinates.values()
                # event should be initial point, station is end point, then we use baz to to rotation!
                az, baz, dist = geodist.inv(evlo, evla, stlo, stla); dist=dist/1000. 
                if baz<0: baz+=360.
                try:
                    if channel!='R' or channel!='T':
                        inST=self.waveforms[staid][tag].select(component=channel)
                    else:
                        st=self.waveforms[staid][tag]
                        st.rotate('NE->RT', backazimuth=baz) # need check
                        inST=st.select(component=channel)
                except KeyError: continue
                if len(inST)==0: continue
                else: tr=inST[0]
                stime=tr.stats.starttime; etime=tr.stats.endtime
                tr.stats.sac={}; tr.stats.sac['dist']= dist; tr.stats.sac['b']=stime-otime; tr.stats.sac['e']=etime-otime
                tr.stats.sac['kuser0']=evid
                aftanTr=pyaftan.aftantrace(tr.data, tr.stats)
                if verbose: print 'Preparing aftan data: ' + evid+' '+staid+'_'+channel
                inputStream.append(aftanTr)
        print 'Start multiprocessing aftan analysis !'
        if len(inputStream) > subsize:
            Nsub = int(len(inputStream)/subsize)
            for isub in xrange(Nsub):
                print 'Subset:', isub,'in',Nsub,'sets'
                cstream=inputStream[isub*subsize:(isub+1)*subsize]
                AFTAN = partial(aftan4mp_quake, outdir=outdir, inftan=inftan, prephdir=prephdir, f77=f77, pfx=pfx)
                pool = multiprocessing.Pool(processes=nprocess)
                pool.map(AFTAN, cstream) #make our results with a map call
                pool.close() #we are not adding any more processes
                pool.join() #tell it to wait until all threads are done before going on
            cstream=inputStream[(isub+1)*subsize:]
            AFTAN = partial(aftan4mp_quake, outdir=outdir, inftan=inftan, prephdir=prephdir, f77=f77, pfx=pfx)
            pool = multiprocessing.Pool(processes=nprocess)
            pool.map(AFTAN, cstream) #make our results with a map call
            pool.close() #we are not adding any more processes
            pool.join() #tell it to wait until all threads are done before going on
        else:
            AFTAN = partial(aftan4mp_quake, outdir=outdir, inftan=inftan, prephdir=prephdir, f77=f77, pfx=pfx)
            pool = multiprocessing.Pool(processes=nprocess)
            pool.map(AFTAN, inputStream) #make our results with a map call
            pool.close() #we are not adding any more processes
            pool.join() #tell it to wait until all threads are done before going on
        print 'End of multiprocessing aftan analysis !'
        print 'Reading aftan results into ASDF Dataset !'
        for event in self.events:
            for staid in staLst:
                netcode, stacode=staid.split('.')
                evid='E%05d' % evnumb
                finPR=pfx+'/'+evid+'/'+staid+'_'+channel+'.SAC'
                try:
                    f10=np.load(outdir+'/'+finPR+'_1_DISP.0.npz')
                    f11=np.load(outdir+'/'+finPR+'_1_DISP.1.npz')
                    f20=np.load(outdir+'/'+finPR+'_2_DISP.0.npz')
                    f21=np.load(outdir+'/'+finPR+'_2_DISP.1.npz')
                except IOError:
                    print 'NO aftan results: '+ evid+' '+staid+'_'+channel
                    continue
                if verbose: print 'Reading aftan results '+ evid+' '+staid+'_'+channel
                if deletedisp:
                    os.remove(outdir+'/'+finPR+'_1_DISP.0.npz')
                    os.remove(outdir+'/'+finPR+'_1_DISP.1.npz')
                    os.remove(outdir+'/'+finPR+'_2_DISP.0.npz')
                    os.remove(outdir+'/'+finPR+'_2_DISP.1.npz')
                arr1_1=f10['arr_0']
                nfout1_1=f10['arr_1']
                arr2_1=f11['arr_0']
                nfout2_1=f11['arr_1']
                arr1_2=f20['arr_0']
                nfout1_2=f20['arr_1']
                arr2_2=f21['arr_0']
                nfout2_2=f21['arr_1']
                staid_aux=evid+'/'+netcode+'_'+stacode+'_'+channel
                if basic1:
                    parameters={'Tc': 0, 'To': 1, 'Vgr': 2, 'Vph': 3, 'ampdb': 4, 'dis': 5, 'snrdb': 6, 'mhw': 7, 'amp': 8, 'Np': nfout1_1}
                    self.add_auxiliary_data(data=arr1_1, data_type='DISPbasic1', path=staid_aux, parameters=parameters)
                if basic2:
                    parameters={'Tc': 0, 'To': 1, 'Vgr': 2, 'Vph': 3, 'ampdb': 4, 'snrdb': 5, 'mhw': 6, 'amp': 7, 'Np': nfout2_1}
                    self.add_auxiliary_data(data=arr2_1, data_type='DISPbasic2', path=staid_aux, parameters=parameters)
                if inftan.pmf:
                    if pmf1:
                        parameters={'Tc': 0, 'To': 1, 'Vgr': 2, 'Vph': 3, 'ampdb': 4, 'dis': 5, 'snrdb': 6, 'mhw': 7, 'amp': 8, 'Np': nfout1_2}
                        self.add_auxiliary_data(data=arr1_2, data_type='DISPpmf1', path=staid_aux, parameters=parameters)
                    if pmf2:
                        parameters={'Tc': 0, 'To': 1, 'Vgr': 2, 'Vph': 3, 'ampdb': 4, 'snrdb': 5, 'mhw': 6, 'amp': 7, 'snr':8, 'Np': nfout2_2}
                        self.add_auxiliary_data(data=arr2_2, data_type='DISPpmf2', path=staid_aux, parameters=parameters)
        if deletedisp: shutil.rmtree(outdir+'/'+pfx)
        return
    
    def interp_disp(self, data_type='DISPpmf2', channel='Z', pers=np.array([]), verbose=True):
        """ Interpolate dispersion curve for a given period array.
        =======================================================================================================
        Input Parameters:
        data_type   - dispersion data type (default = DISPpmf2, pmf aftan results after jump detection)
        pers        - period array
        
        Output:
        self.auxiliary_data.DISPbasic1interp, self.auxiliary_data.DISPbasic2interp,
        self.auxiliary_data.DISPpmf1interp, self.auxiliary_data.DISPpmf2interp
        =======================================================================================================
        """
        if data_type=='DISPpmf2': ntype=6
        else: ntype=5
        if pers.size==0:
            pers=np.append( np.arange(7.)*2.+28., np.arange(6.)*5.+45.)
        staLst=self.waveforms.list()
        evnumb=0
        for event in self.events:
            evnumb+=1
            evid='E%05d' % evnumb
            for staid in staLst:
                netcode, stacode=staid.split('.')
                try:
                    subdset=self.auxiliary_data[data_type][evid][netcode+'_'+stacode+'_'+channel]
                except KeyError: continue
                data=subdset.data.value
                index=subdset.parameters
                if verbose: print 'Interpolating dispersion curve for '+ evid+' '+staid+'_'+channel
                outindex={ 'To': 0, 'Vgr': 1, 'Vph': 2,  'amp': 3, 'snr': 4, 'inbound': 5, 'Np': pers.size }
                Np=int(index['Np'])
                if Np < 5:
                    warnings.warn('Not enough datapoints for: '+ evid+' '+staid+'_'+channel, UserWarning, stacklevel=1)
                    continue
                obsT=data[index['To']][:Np]
                Vgr=np.interp(pers, obsT, data[index['Vgr']][:Np] )
                Vph=np.interp(pers, obsT, data[index['Vph']][:Np] )
                amp=np.interp(pers, obsT, data[index['amp']][:Np] )
                inbound=(pers > obsT[0])*(pers < obsT[-1])*1
                interpdata=np.append(pers, Vgr)
                interpdata=np.append(interpdata, Vph)
                interpdata=np.append(interpdata, amp)
                if data_type=='DISPpmf2':
                    snr=np.interp(pers, obsT, data[index['snr']][:Np] )
                    interpdata=np.append(interpdata, snr)
                interpdata=np.append(interpdata, inbound)
                interpdata=interpdata.reshape(ntype, pers.size)
                staid_aux=evid+'/'+netcode+'_'+stacode+'_'+channel
                self.add_auxiliary_data(data=interpdata, data_type=data_type+'interp', path=staid_aux, parameters=outindex)
        return
    
    def quake_get_field(self, outdir=None, channel='Z', pers=np.array([]), data_type='DISPpmf2interp', verbose=True):
        """ Get the field data for Eikonal tomography
        ============================================================================================================================
        Input Parameters:
        outdir      - directory for txt output (default is not to generate txt output)
        channel     - channel name
        pers        - period array
        datatype    - dispersion data type (default = DISPpmf2interp, interpolated pmf aftan results after jump detection)
        Output:
        self.auxiliary_data.FieldDISPpmf2interp
        ============================================================================================================================
        """
        if pers.size==0:
            pers=np.append( np.arange(7.)*2.+28., np.arange(6.)*5.+45.)
        outindex={ 'longitude': 0, 'latitude': 1, 'Vph': 2,  'Vgr':3, 'amp': 4, 'snr': 5, 'dist': 6 }
        staLst=self.waveforms.list()
        evnumb=0
        for event in self.events:
            evnumb+=1
            evid='E%05d' % evnumb
            field_lst=[]
            Nfplst=[]
            for per in pers:
                field_lst.append(np.array([]))
                Nfplst.append(0)
            evlo=event.origins[0].longitude; evla=event.origins[0].latitude
            if verbose: print 'Getting field data for: '+evid
            for staid in staLst:
                netcode, stacode=staid.split('.')
                try:
                    subdset=self.auxiliary_data[data_type][evid][netcode+'_'+stacode+'_'+channel]
                except KeyError: continue
                stla, stel, stlo=self.waveforms[staid].coordinates.values()
                az, baz, dist = geodist.inv(stlo, stla, evlo, evla); dist=dist/1000.
                if stlo<0: stlo+=360.
                if evlo<0: evlo+=360.
                data=subdset.data.value
                index=subdset.parameters
                for iper in xrange(pers.size):
                    per=pers[iper]
                    if dist < 2.*per*3.5: continue
                    ind_per=np.where(data[index['To']][:] == per)[0]
                    if ind_per.size==0:
                        raise AttributeError('No interpolated dispersion curve data for period='+str(per)+' sec!')
                    pvel=data[index['Vph']][ind_per]
                    gvel=data[index['Vgr']][ind_per]
                    snr=data[index['snr']][ind_per]
                    amp=data[index['amp']][ind_per]
                    inbound=data[index['inbound']][ind_per]
                    # quality control
                    if pvel < 0 or gvel < 0 or pvel>10 or gvel>10 or snr >1e10: continue
                    if inbound!=1.: continue
                    if snr < 10.: continue # different from noise data
                    field_lst[iper]=np.append(field_lst[iper], stlo)
                    field_lst[iper]=np.append(field_lst[iper], stla)
                    field_lst[iper]=np.append(field_lst[iper], pvel)
                    field_lst[iper]=np.append(field_lst[iper], gvel)
                    field_lst[iper]=np.append(field_lst[iper], amp)
                    field_lst[iper]=np.append(field_lst[iper], snr)
                    field_lst[iper]=np.append(field_lst[iper], dist)
                    Nfplst[iper]+=1
            if outdir!=None:
                if not os.path.isdir(outdir): os.makedirs(outdir)
            staid_aux=evid+'_'+channel
            for iper in xrange(pers.size):
                per=pers[iper]
                del_per=per-int(per)
                if field_lst[iper].size==0: continue
                field_lst[iper]=field_lst[iper].reshape(Nfplst[iper], 7)
                if del_per==0.:
                    staid_aux_per=staid_aux+'/'+str(int(per))+'sec'
                else:
                    dper=str(del_per)
                    staid_aux_per=staid_aux+'/'+str(int(per))+'sec'+dper.split('.')[1]
                self.add_auxiliary_data(data=field_lst[iper], data_type='Field'+data_type, path=staid_aux_per, parameters=outindex)
                if outdir!=None:
                    if not os.path.isdir(outdir+'/'+str(per)+'sec'):
                        os.makedirs(outdir+'/'+str(per)+'sec')
                    txtfname=outdir+'/'+str(per)+'sec'+'/'+evid+'_'+str(per)+'.txt'
                    header = 'evlo='+str(evlo)+' evla='+str(evla)
                    np.savetxt( txtfname, field_lst[iper], fmt='%g', header=header )
        return
    
    def get_limits_lonlat(self):
        """Get the geographical limits of the stations
        """
        staLst=self.waveforms.list()
        minlat=90.
        maxlat=-90.
        minlon=360.
        maxlon=0.
        for staid in staLst:
            lat, elv, lon=self.waveforms[staid].coordinates.values()
            if lon<0: lon+=360.
            minlat=min(lat, minlat)
            maxlat=max(lat, maxlat)
            minlon=min(lon, minlon)
            maxlon=max(lon, maxlon)
        print 'latitude range: ', minlat, '-', maxlat, 'longitude range:', minlon, '-', maxlon
        self.minlat=minlat; self.maxlat=maxlat; self.minlon=minlon; self.maxlon=maxlon
        return
    
    
    
def aftan4mp_quake(aTr, outdir, inftan, prephdir, f77, pfx):
    # print aTr.stats.network+'.'+aTr.stats.station
    if prephdir !=None:
        phvelname = prephdir + "/%s.%s.pre" %(aTr.stats.sac.kuser0, aTr.stats.network+'.'+aTr.stats.station)
    else:
        phvelname =''
    if f77:
        aTr.aftanf77(pmf=inftan.pmf, piover4=inftan.piover4, vmin=inftan.vmin, vmax=inftan.vmax, tmin=inftan.tmin, tmax=inftan.tmax,
            tresh=inftan.tresh, ffact=inftan.ffact, taperl=inftan.taperl, snr=inftan.snr, fmatch=inftan.fmatch, nfin=inftan.nfin,
                npoints=inftan.npoints, perc=inftan.perc, phvelname=phvelname)
    else:
        aTr.aftan(pmf=inftan.pmf, piover4=inftan.piover4, vmin=inftan.vmin, vmax=inftan.vmax, tmin=inftan.tmin, tmax=inftan.tmax,
            tresh=inftan.tresh, ffact=inftan.ffact, taperl=inftan.taperl, snr=inftan.snr, fmatch=inftan.fmatch, nfin=inftan.nfin,
                npoints=inftan.npoints, perc=inftan.perc, phvelname=phvelname)
    aTr.get_snr(ffact=inftan.ffact) # SNR analysis
    foutPR=outdir+'/'+pfx+'/'+aTr.stats.sac.kuser0+'/'+aTr.stats.network+'.'+aTr.stats.station+'_'+aTr.stats.channel[-1]+'.SAC'
    aTr.ftanparam.writeDISPbinary(foutPR)
    return


def get_waveforms4mp(reqinfo, outdir, client, pre_filt, verbose=True, rotation=False):
    try:
        try:
            st = client.get_waveforms(network=reqinfo.network, station=reqinfo.station, location=reqinfo.location, channel=reqinfo.channel,
                    starttime=reqinfo.starttime, endtime=reqinfo.endtime, attach_response=reqinfo.attach_response)
            st.detrend()
        except:
            if verbose: print 'No data for:', reqinfo.network+'.'+reqinfo.station
            return
        if verbose: print 'Getting data for:', reqinfo.network+'.'+reqinfo.station
        # print '===================================== Removing response ======================================='
        evid='E%05d' %reqinfo.evnumb
        try:
            st.remove_response(pre_filt=pre_filt, taper_fraction=0.1)
        except :
            N=10; i=0; get_resp=False
            while (i < N) and (not get_resp):
                st = client.get_waveforms(network=reqinfo.network, station=reqinfo.station, location=reqinfo.location, channel=reqinfo.channel,
                        starttime=reqinfo.starttime, endtime=reqinfo.endtime, attach_response=reqinfo.attach_response)
                try:
                    st.remove_response(pre_filt=pre_filt, taper_fraction=0.1)
                    get_resp=True
                except : i+=1
            if not get_resp:
                st.write(outdir+'/'+evid+'.'+reqinfo.network+'.'+reqinfo.station+'.no_resp.mseed', format='mseed')
                return
        if rotation: st.rotate('NE->RT', back_azimuth=reqinfo.baz)
        st.write(outdir+'/'+evid+'.'+reqinfo.network+'.'+reqinfo.station+'.mseed', format='mseed')
    except:
        print 'Unknown error for:'+evid+'.'+reqinfo.network+'.'+reqinfo.station
    return

def ref4mp(refTr, outdir, inrefparam):
    refTr.IterDeconv(tdel=inrefparam.tdel, f0 = inrefparam.f0, niter=inrefparam.niter,
            minderr=inrefparam.minderr, phase=refTr.Ztr.stats.sac['kuser1'] )
    if not refTr.move_out(): return
    refTr.stretch_back()
    refTr.save_data(outdir)
    return

