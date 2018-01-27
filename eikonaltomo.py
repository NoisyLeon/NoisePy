# -*- coding: utf-8 -*-
"""
A python module to run surface wave eikonal/Helmholtz tomography
The code creates a datadbase based on hdf5 data format

:Dependencies:
    pyasdf and its dependencies
    GMT 5.x.x (for interpolation on Earth surface)
    numba
    numexpr
    
:Copyright:
    Author: Lili Feng
    Graduate Research Assistant
    CIEI, Department of Physics, University of Colorado Boulder
    email: lili.feng@colorado.edu
    
:References:
    Lin, Fan-Chi, Michael H. Ritzwoller, and Roel Snieder. "Eikonal tomography: surface wave tomography by phase front tracking across a regional broad-band seismic array."
        Geophysical Journal International 177.3 (2009): 1091-1110.
    Lin, Fan-Chi, and Michael H. Ritzwoller. "Helmholtz surface wave tomography for isotropic and azimuthally anisotropic structure."
        Geophysical Journal International 186.3 (2011): 1104-1120.
"""
import numpy as np
import numpy.ma as ma
import h5py, pyasdf
import os, shutil
from subprocess import call
from mpl_toolkits.basemap import Basemap, shiftgrid, cm
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
import colormaps
import obspy
import field2d_earth
import numexpr
import warnings
from functools import partial
import multiprocessing
from numba import jit, float32, int32

# compiled function to get weight for each event and each grid point
@jit(float32[:,:,:](float32[:,:,:], float32[:,:,:]))
def _get_azi_weight(aziArr, validArr):
    Nevent, Nlon, Nlat  = aziArr.shape
    weightArr           = np.zeros((Nevent, Nlon, Nlat))
    for ilon in xrange(Nlon):
        for ilat in xrange(Nlat):
            for i in xrange(Nevent):
                for j in xrange(Nevent):
                    delAzi                      = abs(aziArr[i, ilon, ilat] - aziArr[j, ilon, ilat])
                    if delAzi < 20. or delAzi > 340.:
                        weightArr[i, ilon, ilat]+= validArr[i, ilon, ilat]    
    return weightArr

class EikonalTomoDataSet(h5py.File):
    
    def set_input_parameters(self, minlon, maxlon, minlat, maxlat, pers=np.array([]), dlon=0.2, dlat=0.2):
        """
        Set input parameters for tomographic inversion.
        =================================================================================================================
        ::: input parameters :::
        minlon, maxlon  - minimum/maximum longitude
        minlat, maxlat  - minimum/maximum latitude
        pers            - period array, default = np.append( np.arange(18.)*2.+6., np.arange(4.)*5.+45.)
        dlon, dlat      - longitude/latitude interval
        =================================================================================================================
        """
        if pers.size==0:
            # pers=np.arange(13.)*2.+6.
            pers    = np.append( np.arange(18.)*2.+6., np.arange(4.)*5.+45.)
        self.attrs.create(name = 'period_array', data=pers, dtype='f')
        self.attrs.create(name = 'minlon', data=minlon, dtype='f')
        self.attrs.create(name = 'maxlon', data=maxlon, dtype='f')
        self.attrs.create(name = 'minlat', data=minlat, dtype='f')
        self.attrs.create(name = 'maxlat', data=maxlat, dtype='f')
        self.attrs.create(name = 'dlon', data=dlon)
        self.attrs.create(name = 'dlat', data=dlat)
        Nlon        = (maxlon-minlon)/dlon+1
        Nlat        = (maxlat-minlat)/dlat+1
        self.attrs.create(name = 'Nlon', data=Nlon)
        self.attrs.create(name = 'Nlat', data=Nlat)
        return
    
    def xcorr_eikonal(self, inasdffname, workingdir, fieldtype='Tph', channel='ZZ', data_type='FieldDISPpmf2interp', runid=0, deletetxt=True, verbose=True):
        """
        Compute gradient of travel time for cross-correlation data
        =================================================================================================================
        ::: input parameters :::
        inasdffname - input ASDF data file
        workingdir  - working directory
        fieldtype   - fieldtype (Tph or Tgr)
        channel     - channel for analysis
        data_type   - data type
                     (default='FieldDISPpmf2interp', aftan measurements with phase-matched filtering and jump correction)
        runid       - run id
        deletetxt   - delete output txt files in working directory
        =================================================================================================================
        """
        if fieldtype!='Tph' and fieldtype!='Tgr':
            raise ValueError('Wrong field type: '+fieldtype+' !')
        create_group=False
        while (not create_group):
            try:
                group=self.create_group( name = 'Eikonal_run_'+str(runid) )
                create_group=True
            except:
                runid+=1
                continue
        group.attrs.create(name = 'fieldtype', data=fieldtype[1:])
        inDbase=pyasdf.ASDFDataSet(inasdffname)
        pers = self.attrs['period_array']
        minlon=self.attrs['minlon']
        maxlon=self.attrs['maxlon']
        minlat=self.attrs['minlat']
        maxlat=self.attrs['maxlat']
        dlon=self.attrs['dlon']
        dlat=self.attrs['dlat']
        fdict={ 'Tph': 2, 'Tgr': 3}
        evLst=inDbase.waveforms.list()
        for per in pers:
            print 'Computing gradient for: '+str(per)+' sec'
            del_per=per-int(per)
            if del_per==0.:
                persfx=str(int(per))+'sec'
            else:
                dper=str(del_per)
                persfx=str(int(per))+'sec'+dper.split('.')[1]
            working_per=workingdir+'/'+str(per)+'sec'
            per_group=group.create_group( name='%g_sec'%( per ) )
            for evid in evLst:
                netcode1, stacode1=evid.split('.')
                try:
                    subdset = inDbase.auxiliary_data[data_type][netcode1][stacode1][channel][persfx]
                except KeyError:
                    print 'No travel time field for: '+evid
                    continue
                if verbose: print 'Event: '+evid
                lat1, elv1, lon1=inDbase.waveforms[evid].coordinates.values()
                if lon1<0.:
                    lon1+=360.
                dataArr = subdset.data.value
                field2d=field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=dlon,
                        minlat=minlat, maxlat=maxlat, dlat=dlat, period=per, evlo=lon1, evla=lat1, fieldtype=fieldtype)
                        # minlat=minlat, maxlat=maxlat, dlat=dlat, period=per, fieldtype=fieldtype)

                Zarr=dataArr[:, fdict[fieldtype]]
                distArr=dataArr[:, 5]
                field2d.read_array(lonArr=np.append(lon1, dataArr[:,0]), latArr=np.append(lat1, dataArr[:,1]), ZarrIn=np.append(0., distArr/Zarr) )
                outfname=evid+'_'+fieldtype+'_'+channel+'.lst'
                field2d.interp_surface(workingdir=working_per, outfname=outfname)
                field2d.check_curvature(workingdir=working_per, outpfx=evid+'_'+channel+'_')
                field2d.gradient_qc(workingdir=working_per, inpfx=evid+'_'+channel+'_', nearneighbor=True, cdist=None)
                # save data to hdf5 dataset
                event_group=per_group.create_group(name=evid)
                event_group.attrs.create(name = 'evlo', data=lon1)
                event_group.attrs.create(name = 'evla', data=lat1)
                appVdset     = event_group.create_dataset(name='appV', data=field2d.appV)
                reason_ndset = event_group.create_dataset(name='reason_n', data=field2d.reason_n)
                proAngledset = event_group.create_dataset(name='proAngle', data=field2d.proAngle)
                azdset       = event_group.create_dataset(name='az', data=field2d.az)
                bazdset      = event_group.create_dataset(name='baz', data=field2d.baz)
                Tdset        = event_group.create_dataset(name='travelT', data=field2d.Zarr)
        if deletetxt:
            shutil.rmtree(workingdir)
        return
    
    def xcorr_eikonal_mp(self, inasdffname, workingdir, fieldtype='Tph', channel='ZZ', data_type='FieldDISPpmf2interp', runid=0,
                deletetxt=True, verbose=True, subsize=1000, nprocess=None):
        """
        Compute gradient of travel time for cross-correlation data with multiprocessing
        =================================================================================================================
        ::: input parameters :::
        inasdffname - input ASDF data file
        workingdir  - working directory
        fieldtype   - fieldtype (Tph or Tgr)
        channel     - channel for analysis
        data_type   - data type
                     (default='FieldDISPpmf2interp', aftan measurements with phase-matched filtering and jump correction)
        runid       - run id
        deletetxt   - delete output txt files in working directory
        subsize     - subsize of processing list, use to prevent lock in multiprocessing process
        nprocess    - number of processes
        =================================================================================================================
        """
        if fieldtype!='Tph' and fieldtype!='Tgr':
            raise ValueError('Wrong field type: '+fieldtype+' !')
        create_group=False
        while (not create_group):
            try:
                group=self.create_group( name = 'Eikonal_run_'+str(runid) )
                create_group=True
            except:
                runid+=1
                continue
        group.attrs.create(name = 'fieldtype', data=fieldtype[1:])
        inDbase=pyasdf.ASDFDataSet(inasdffname)
        pers = self.attrs['period_array']
        minlon=self.attrs['minlon']
        maxlon=self.attrs['maxlon']
        minlat=self.attrs['minlat']
        maxlat=self.attrs['maxlat']
        dlon=self.attrs['dlon']
        dlat=self.attrs['dlat']
        fdict={ 'Tph': 2, 'Tgr': 3}
        evLst=inDbase.waveforms.list()
        fieldLst=[]
        # prepare data
        for per in pers:
            print 'Preparing data for gradient computation of '+str(per)+' sec'
            del_per=per-int(per)
            if del_per==0.:
                persfx=str(int(per))+'sec'
            else:
                dper=str(del_per)
                persfx=str(int(per))+'sec'+dper.split('.')[1]
            working_per=workingdir+'/'+str(per)+'sec'
            if not os.path.isdir(working_per): os.makedirs(working_per)
            for evid in evLst:
                netcode1, stacode1=evid.split('.')
                try:
                    subdset = inDbase.auxiliary_data[data_type][netcode1][stacode1][channel][persfx]
                except KeyError:
                    print 'No travel time field for: '+evid
                    continue
                lat1, elv1, lon1=inDbase.waveforms[evid].coordinates.values()
                if lon1<0.:
                    lon1+=360.
                dataArr = subdset.data.value
                field2d=field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=dlon, minlat=minlat, maxlat=maxlat, dlat=dlat,
                        period=per, evlo=lon1, evla=lat1, fieldtype=fieldtype, evid=evid)
                Zarr=dataArr[:, fdict[fieldtype]]
                distArr=dataArr[:, 5]
                field2d.read_array(lonArr=np.append(lon1, dataArr[:,0]), latArr=np.append(lat1, dataArr[:,1]), ZarrIn=np.append(0., distArr/Zarr) )
                fieldLst.append(field2d)
        # Computing gradient with multiprocessing
        if len(fieldLst) > subsize:
            Nsub = int(len(fieldLst)/subsize)
            for isub in xrange(Nsub):
                print 'Subset:', isub,'in',Nsub,'sets'
                cfieldLst=fieldLst[isub*subsize:(isub+1)*subsize]
                EIKONAL = partial(eikonal4mp, workingdir=workingdir, channel=channel)
                pool = multiprocessing.Pool(processes=nprocess)
                pool.map(EIKONAL, cfieldLst) #make our results with a map call
                pool.close() #we are not adding any more processes
                pool.join() #tell it to wait until all threads are done before going on
            cfieldLst=fieldLst[(isub+1)*subsize:]
            EIKONAL = partial(eikonal4mp, workingdir=workingdir, channel=channel)
            pool = multiprocessing.Pool(processes=nprocess)
            pool.map(EIKONAL, cfieldLst) #make our results with a map call
            pool.close() #we are not adding any more processes
            pool.join() #tell it to wait until all threads are done before going on
        else:
            EIKONAL = partial(eikonal4mp, workingdir=workingdir, channel=channel)
            pool = multiprocessing.Pool(processes=nprocess)
            pool.map(EIKONAL, fieldLst) #make our results with a map call
            pool.close() #we are not adding any more processes
            pool.join() #tell it to wait until all threads are done before going on
        # Read data into hdf5 dataset
        for per in pers:
            print 'Reading gradient data for: '+str(per)+' sec'
            working_per=workingdir+'/'+str(per)+'sec'
            per_group=group.create_group( name='%g_sec'%( per ) )
            for evid in evLst:
                infname=working_per+'/'+evid+'_field2d.npz'
                if not os.path.isfile(infname): print 'No data for:', evid; continue
                InArr=np.load(infname)
                appV=InArr['arr_0']; reason_n=InArr['arr_1']; proAngle=InArr['arr_2']
                az=InArr['arr_3']; baz=InArr['arr_4']; Zarr=InArr['arr_5']
                lat1, elv1, lon1=inDbase.waveforms[evid].coordinates.values()
                # save data to hdf5 dataset
                event_group=per_group.create_group(name=evid)
                event_group.attrs.create(name = 'evlo', data=lon1)
                event_group.attrs.create(name = 'evla', data=lat1)
                appVdset     = event_group.create_dataset(name='appV', data=appV)
                reason_ndset = event_group.create_dataset(name='reason_n', data=reason_n)
                proAngledset = event_group.create_dataset(name='proAngle', data=proAngle)
                azdset       = event_group.create_dataset(name='az', data=az)
                bazdset      = event_group.create_dataset(name='baz', data=baz)
                Tdset        = event_group.create_dataset(name='travelT', data=Zarr)
        if deletetxt: shutil.rmtree(workingdir)
        return
    
    def quake_eikonal(self, inasdffname, workingdir, fieldtype='Tph', channel='Z', data_type='FieldDISPpmf2interp',
            runid=0, merge=False, deletetxt=False, verbose=True, amplplc=False):
        """
        Compute gradient of travel time for earthquake data
        =================================================================================================================
        ::: input parameters :::
        inasdffname - input ASDF data file
        workingdir  - working directory
        fieldtype   - fieldtype (Tph or Tgr)
        channel     - channel for analysis
        data_type   - data type
                     (default='FieldDISPpmf2interp', aftan measurements with phase-matched filtering and jump correction)
        runid       - run id
        deletetxt   - delete output txt files in working directory
        amplplc     - compute amplitude Laplacian term or not
        =================================================================================================================
        """
        if fieldtype!='Tph' and fieldtype!='Tgr':
            raise ValueError('Wrong field type: '+fieldtype+' !')
        if merge:
            try:
                group=self.create_group( name = 'Eikonal_run_'+str(runid) )
                group.attrs.create(name = 'fieldtype', data=fieldtype[1:])
            except ValueError:
                print 'Merging Eikonal run id: ',runid
                pass
        else:
            create_group=False
            while (not create_group):
                try:
                    group=self.create_group( name = 'Eikonal_run_'+str(runid) )
                    create_group=True
                except:
                    runid+=1
                    continue
            group.attrs.create(name = 'fieldtype', data=fieldtype[1:])
        inDbase=pyasdf.ASDFDataSet(inasdffname)
        pers = self.attrs['period_array']
        minlon=self.attrs['minlon']
        maxlon=self.attrs['maxlon']
        minlat=self.attrs['minlat']
        maxlat=self.attrs['maxlat']
        dlon=self.attrs['dlon']
        dlat=self.attrs['dlat']
        fdict={ 'Tph': 2, 'Tgr': 3, 'Amp': 4}
        evLst=inDbase.events
        for per in pers:
            print 'Computing gradient for: '+str(per)+' sec'
            del_per=per-int(per)
            if del_per==0.:
                persfx=str(int(per))+'sec'
            else:
                dper=str(del_per)
                persfx=str(int(per))+'sec'+dper.split('.')[1]
            working_per=workingdir+'/'+str(per)+'sec'
            per_group=group.require_group( name='%g_sec'%( per ) )
            evnumb=0
            for event in evLst:
                evnumb+=1
                evid='E%05d' % evnumb
                try:
                    subdset = inDbase.auxiliary_data[data_type][evid+'_'+channel][persfx]
                except KeyError:
                    print 'No travel time field for: '+evid
                    continue
                magnitude=event.magnitudes[0].mag; Mtype=event.magnitudes[0].magnitude_type
                event_descrip=event.event_descriptions[0].text+', '+event.event_descriptions[0].type
                evlo=event.origins[0].longitude; evla=event.origins[0].latitude
                if verbose: print 'Event: '+event_descrip+', '+Mtype+' = '+str(magnitude) 
                if evlo<0.: evlo+=360.
                dataArr = subdset.data.value
                field2d=field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=dlon,
                        minlat=minlat, maxlat=maxlat, dlat=dlat, period=per, evlo=evlo, evla=evla, fieldtype=fieldtype)
                Zarr=dataArr[:, fdict[fieldtype]]
                distArr=dataArr[:, 6] # Note amplitude in added!!!
                field2d.read_array(lonArr=np.append(evlo, dataArr[:,0]), latArr=np.append(evla, dataArr[:,1]), ZarrIn=np.append(0., distArr/Zarr) )
                outfname=evid+'_'+fieldtype+'_'+channel+'.lst'
                field2d.interp_surface(workingdir=working_per, outfname=outfname)
                field2d.check_curvature(workingdir=working_per, outpfx=evid+'_'+channel+'_')
                field2d.gradient_qc(workingdir=working_per, inpfx=evid+'_'+channel+'_', nearneighbor=True, cdist=None)
                # save data to hdf5 dataset
                event_group=per_group.create_group(name=evid)
                event_group.attrs.create(name = 'evlo', data=evlo)
                event_group.attrs.create(name = 'evla', data=evla)
                appVdset     = event_group.create_dataset(name='appV', data=field2d.appV)
                reason_ndset = event_group.create_dataset(name='reason_n', data=field2d.reason_n)
                proAngledset = event_group.create_dataset(name='proAngle', data=field2d.proAngle)
                azdset       = event_group.create_dataset(name='az', data=field2d.az)
                bazdset      = event_group.create_dataset(name='baz', data=field2d.baz)
                Tdset        = event_group.create_dataset(name='travelT', data=field2d.Zarr)
                if amplplc:
                    field2dAmp=field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=dlon,
                        minlat=minlat, maxlat=maxlat, dlat=dlat, period=per, evlo=evlo, evla=evla, fieldtype='Amp')
                    field2dAmp.read_array(lonArr=dataArr[:,0], latArr=dataArr[:,1], ZarrIn=dataArr[:, fdict['Amp']] )
                    outfnameAmp=evid+'_Amp_'+channel+'.lst'
                    field2dAmp.interp_surface(workingdir=working_per, outfname=outfnameAmp)
                    field2dAmp.gradient()
                    field2dAmp.cut_edge(1,1)
                    field2dAmp.Laplacian()
                    field2dAmp.cut_edge(1,1)
                    field2dAmp.get_lplc_amp()
                    lplc_ampdset = event_group.create_dataset(name='lplc_amp', data=field2dAmp.lplc_amp)
                    field2dAmp.lplc_amp[field2dAmp.lplc_amp > 2e-2]=0
                    field2dAmp.lplc_amp[field2dAmp.lplc_amp < -2e-2]=0
                    slownessApp=-np.ones(field2d.appV.shape)
                    slownessApp[field2d.appV!=0]=1./field2d.appV[field2d.appV!=0]
                    temp=slownessApp**2-field2dAmp.lplc_amp
                    temp[temp<0]=0
                    slownessCor=np.sqrt(temp)
                    corV=np.zeros(slownessCor.shape)
                    corV[slownessCor!=0]=1./slownessCor[slownessCor!=0]
                    corV_ampdset = event_group.create_dataset(name='corV', data=corV)
                # field2d.appV=corV
                return field2d
        if deletetxt: shutil.rmtree(workingdir)
        return
    
    def quake_eikonal_mp(self, inasdffname, workingdir, fieldtype='Tph', channel='Z', data_type='FieldDISPpmf2interp', runid=0,
                merge=False, deletetxt=True, verbose=True, subsize=1000, nprocess=None, amplplc=False):
        """
        Compute gradient of travel time for cross-correlation data with multiprocessing
        =================================================================================================================
        ::: input parameters :::
        inasdffname - input ASDF data file
        workingdir  - working directory
        fieldtype   - fieldtype (Tph or Tgr)
        channel     - channel for analysis
        data_type   - data type
                     (default='FieldDISPpmf2interp', aftan measurements with phase-matched filtering and jump correction)
        runid       - run id
        deletetxt   - delete output txt files in working directory
        subsize     - subsize of processing list, use to prevent lock in multiprocessing process
        nprocess    - number of processes
        amplplc     - compute amplitude Laplacian term or not
        =================================================================================================================
        """
        if fieldtype!='Tph' and fieldtype!='Tgr':
            raise ValueError('Wrong field type: '+fieldtype+' !')
        if merge:
            try:
                group=self.create_group( name = 'Eikonal_run_'+str(runid) )
                group.attrs.create(name = 'fieldtype', data=fieldtype[1:])
            except ValueError:
                print 'Merging Eikonal run id: ',runid
                pass
        else:
            create_group=False
            while (not create_group):
                try:
                    group=self.create_group( name = 'Eikonal_run_'+str(runid) )
                    create_group=True
                except:
                    runid+=1
                    continue
            group.attrs.create(name = 'fieldtype', data=fieldtype[1:])
        inDbase=pyasdf.ASDFDataSet(inasdffname)
        pers = self.attrs['period_array']
        minlon=self.attrs['minlon']
        maxlon=self.attrs['maxlon']
        minlat=self.attrs['minlat']
        maxlat=self.attrs['maxlat']
        dlon=self.attrs['dlon']
        dlat=self.attrs['dlat']
        fdict={ 'Tph': 2, 'Tgr': 3, 'Amp': 4}
        evLst=inDbase.events
        fieldLst=[]
        # prepare data
        for per in pers:
            print 'Computing gradient for: '+str(per)+' sec'
            del_per=per-int(per)
            if del_per==0.:
                persfx=str(int(per))+'sec'
            else:
                dper=str(del_per)
                persfx=str(int(per))+'sec'+dper.split('.')[1]
            working_per=workingdir+'/'+str(per)+'sec'
            per_group=group.require_group( name='%g_sec'%( per ) )
            evnumb=0
            for event in evLst:
                evnumb+=1
                evid='E%05d' % evnumb
                try:
                    subdset = inDbase.auxiliary_data[data_type][evid+'_'+channel][persfx]
                except KeyError:
                    print 'No travel time field for: '+evid
                    continue
                magnitude=event.magnitudes[0].mag; Mtype=event.magnitudes[0].magnitude_type
                event_descrip=event.event_descriptions[0].text+', '+event.event_descriptions[0].type
                evlo=event.origins[0].longitude; evla=event.origins[0].latitude
                if verbose: print 'Event: '+event_descrip+', '+Mtype+' = '+str(magnitude) 
                if evlo<0.: evlo+=360.
                dataArr = subdset.data.value
                fieldpair=[]
                field2d=field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=dlon,
                        minlat=minlat, maxlat=maxlat, dlat=dlat, period=per, evlo=evlo, evla=evla, fieldtype=fieldtype, evid=evid)
                Zarr=dataArr[:, fdict[fieldtype]]
                distArr=dataArr[:, 6] # Note amplitude in added!!!
                field2d.read_array(lonArr=np.append(evlo, dataArr[:,0]), latArr=np.append(evla, dataArr[:,1]), ZarrIn=np.append(0., distArr/Zarr) )
                fieldpair.append(field2d)
                if amplplc:
                    field2dAmp=field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=dlon,
                        minlat=minlat, maxlat=maxlat, dlat=dlat, period=per, evlo=evlo, evla=evla, fieldtype='Amp', evid=evid)
                    field2dAmp.read_array(lonArr=dataArr[:,0], latArr=dataArr[:,1], ZarrIn=dataArr[:, fdict['Amp']] )
                    fieldpair.append(field2dAmp)
                fieldLst.append(fieldpair)
        # Computing gradient with multiprocessing
        if len(fieldLst) > subsize:
            Nsub = int(len(fieldLst)/subsize)
            for isub in xrange(Nsub):
                print 'Subset:', isub,'in',Nsub,'sets'
                cfieldLst=fieldLst[isub*subsize:(isub+1)*subsize]
                HELMHOTZ = partial(helmhotz4mp, workingdir=workingdir, channel=channel, amplplc=amplplc)
                pool = multiprocessing.Pool(processes=nprocess)
                pool.map(HELMHOTZ, cfieldLst) #make our results with a map call
                pool.close() #we are not adding any more processes
                pool.join() #tell it to wait until all threads are done before going on
            cfieldLst=fieldLst[(isub+1)*subsize:]
            HELMHOTZ = partial(helmhotz4mp, workingdir=workingdir, channel=channel, amplplc=amplplc)
            pool = multiprocessing.Pool(processes=nprocess)
            pool.map(HELMHOTZ, cfieldLst) #make our results with a map call
            pool.close() #we are not adding any more processes
            pool.join() #tell it to wait until all threads are done before going on
        else:
            HELMHOTZ = partial(helmhotz4mp, workingdir=workingdir, channel=channel, amplplc=amplplc)
            pool = multiprocessing.Pool(processes=nprocess)
            pool.map(HELMHOTZ, fieldLst) #make our results with a map call
            pool.close() #we are not adding any more processes
            pool.join() #tell it to wait until all threads are done before going on
        # Read data into hdf5 dataset
        for per in pers:
            print 'Reading gradient data for: '+str(per)+' sec'
            working_per=workingdir+'/'+str(per)+'sec'
            per_group=group.require_group( name='%g_sec'%( per ) )
            evnumb=0
            for event in evLst:
                evnumb+=1
                evid='E%05d' % evnumb
                infname=working_per+'/'+evid+'_field2d.npz'
                if not os.path.isfile(infname): print 'No data for:', evid; continue
                InArr=np.load(infname)
                appV=InArr['arr_0']; reason_n=InArr['arr_1']; proAngle=InArr['arr_2']
                az=InArr['arr_3']; baz=InArr['arr_4']; Zarr=InArr['arr_5']
                if amplplc:
                    lplc_amp=InArr['arr_6']; corV=InArr['arr_7']
                evlo=event.origins[0].longitude; evla=event.origins[0].latitude
                # save data to hdf5 dataset
                event_group=per_group.require_group(name=evid)
                event_group.attrs.create(name = 'evlo', data=evlo)
                event_group.attrs.create(name = 'evla', data=evla)
                appVdset     = event_group.create_dataset(name='appV', data=appV)
                reason_ndset = event_group.create_dataset(name='reason_n', data=reason_n)
                proAngledset = event_group.create_dataset(name='proAngle', data=proAngle)
                azdset       = event_group.create_dataset(name='az', data=az)
                bazdset      = event_group.create_dataset(name='baz', data=baz)
                Tdset        = event_group.create_dataset(name='travelT', data=Zarr)
                if amplplc:
                    lplc_ampdset = event_group.create_dataset(name='lplc_amp', data=lplc_amp)
                    corV_dset = event_group.create_dataset(name='corV', data=corV)
        if deletetxt: shutil.rmtree(workingdir)
        return
    
    
    
    def eikonal_stack(self, runid=0, minazi=-180, maxazi=180, N_bin=20, anisotropic=False, helmholtz=False, use_numba=True):
        """
        Stack gradient results to perform Eikonal Tomography
        =================================================================================================================
        ::: input parameters :::
        runid           - run id
        minazi/maxazi   - min/max azimuth for anisotropic parameters determination
        N_bin           - number of bins for anisotropic parameters determination
        anisotropic     - perform anisotropic parameters determination or not
        use_numba       - use numba for large array manipulation or not, faster and much less memory requirement
        -----------------------------------------------------------------------------------------------------------------
        Version History:
            Dec 6th, 2016   - add function to use numba, faster and much less memory consumption
        =================================================================================================================
        """
        pers = self.attrs['period_array']
        minlon=self.attrs['minlon']
        maxlon=self.attrs['maxlon']
        minlat=self.attrs['minlat']
        maxlat=self.attrs['maxlat']
        dlon=self.attrs['dlon']
        dlat=self.attrs['dlat']
        Nlon=self.attrs['Nlon']
        Nlat=self.attrs['Nlat']
        group=self['Eikonal_run_'+str(runid)]
        try:
            group_out=self.create_group( name = 'Eikonal_stack_'+str(runid) )
        except ValueError:
            warnings.warn('Eikonal_stack_'+str(runid)+' exists! Will be recomputed!', UserWarning, stacklevel=1)
            del self['Eikonal_stack_'+str(runid)]
            group_out=self.create_group( name = 'Eikonal_stack_'+str(runid) )
        group_out.attrs.create(name = 'anisotropic', data=anisotropic)
        group_out.attrs.create(name = 'N_bin', data=N_bin)
        group_out.attrs.create(name = 'minazi', data=minazi)
        group_out.attrs.create(name = 'maxazi', data=maxazi)
        group_out.attrs.create(name = 'fieldtype', data=group.attrs['fieldtype'])
        for per in pers:
            print 'Stacking Eikonal results for: '+str(per)+' sec'
            per_group=group['%g_sec'%( per )]
            Nevent=len(per_group.keys())
            Nmeasure=np.zeros((Nlat-4, Nlon-4))
            weightArr=np.zeros((Nevent, Nlat-4, Nlon-4))
            slownessArr=np.zeros((Nevent, Nlat-4, Nlon-4))
            aziArr=np.zeros((Nevent, Nlat-4, Nlon-4), dtype='float32')
            reason_nArr=np.zeros((Nevent, Nlat-4, Nlon-4))
            validArr=np.zeros((Nevent, Nlat-4, Nlon-4), dtype='float32')
            for iev in xrange(Nevent):
                evid=per_group.keys()[iev]
                event_group=per_group[evid]
                reason_n=event_group['reason_n'].value
                az=event_group['az'].value
                oneArr=np.ones((Nlat-4, Nlon-4))
                oneArr[reason_n!=0]=0
                Nmeasure+=oneArr
                if helmholtz: velocity=event_group['corV'].value
                else: velocity=event_group['appV'].value
                slowness=np.zeros((Nlat-4, Nlon-4))
                slowness[velocity!=0]=1./velocity[velocity!=0]
                slownessArr[iev, :, :]=slowness
                reason_nArr[iev, :, :]=reason_n
                aziArr[iev, :, :]=az
            if Nmeasure.max()<15:
                print 'No enough measurements for: '+str(per)+' sec'
                continue
            ###########################################
            # Get weight for each grid point per event
            ###########################################
            if use_numba:
                validArr[reason_nArr==0]=1
                weightArr = _get_azi_weight(aziArr, validArr)
                weightArr[reason_nArr!=0]=0
                weightArr[weightArr!=0]=1./weightArr[weightArr!=0]
                weightsumArr=np.sum(weightArr, axis=0)
            else:
                azi_event1=np.broadcast_to(aziArr, (Nevent, Nevent, Nlat-4, Nlon-4))
                azi_event2=np.swapaxes(azi_event1, 0, 1)
                validArr[reason_nArr==0]=1
                validArr4=np.broadcast_to(validArr, (Nevent, Nevent, Nlat-4, Nlon-4))
                # use numexpr for very large array manipulations
                del_aziArr=numexpr.evaluate('abs(azi_event1-azi_event2)')
                index_azi=numexpr.evaluate('(1*(del_aziArr<20)+1*(del_aziArr>340))*validArr4')
                weightArr=numexpr.evaluate('sum(index_azi, 0)')
                weightArr[reason_nArr!=0]=0
                weightArr[weightArr!=0]=1./weightArr[weightArr!=0]
                weightsumArr=np.sum(weightArr, axis=0)
            ###########################################
            # reduce large weight to some value.
            ###########################################
            avgArr=np.zeros((Nlat-4, Nlon-4))
            avgArr[Nmeasure!=0]=weightsumArr[Nmeasure!=0]/Nmeasure[Nmeasure!=0]
            stdArr=np.sum( (weightArr-avgArr)**2, axis=0)
            stdArr[Nmeasure!=0]=stdArr[Nmeasure!=0]/Nmeasure[Nmeasure!=0]
            stdArr=np.sqrt(stdArr)
            threshhold=np.broadcast_to(avgArr+3.*stdArr, weightArr.shape)
            weightArr[weightArr>threshhold]=threshhold[weightArr>threshhold]
            ###########################################
            # Compute mean/std of slowness
            ###########################################
            weightsumArr=np.sum(weightArr, axis=0)
            weightsumArr2=np.broadcast_to(weightsumArr, weightArr.shape)
            weightArr[weightsumArr2!=0]=weightArr[weightsumArr2!=0]/weightsumArr2[weightsumArr2!=0]
            slownessArr2=slownessArr*weightArr
            slowness_sumArr=np.sum(slownessArr2, axis=0)
            slowness_sumArr2=np.broadcast_to(slowness_sumArr, weightArr.shape)
            w2sumArr=np.sum(weightArr**2, axis=0)
            temp=weightArr*(slownessArr-slowness_sumArr2)**2
            temp=np.sum(temp, axis=0)
            slowness_stdArr=np.sqrt(temp/(1-w2sumArr))
            slowness_stdArr2=np.broadcast_to(slowness_stdArr, weightArr.shape)
            ###########################################
            # discard outliers of slowness
            ###########################################
            weightArrQC=weightArr.copy()
            index_outlier=(np.abs(slownessArr-slowness_sumArr2))>2*slowness_stdArr2 
            weightArrQC[index_outlier]=0
            weightsumArrQC=np.sum(weightArrQC, axis=0)
            NmArr=np.sign(weightArrQC)
            NmeasureQC=np.sum(NmArr, axis=0)
            weightsumArrQC2=np.broadcast_to(weightsumArrQC, weightArr.shape)
            weightArrQC[weightsumArrQC2!=0]=weightArrQC[weightsumArrQC2!=0]/weightsumArrQC2[weightsumArrQC2!=0]
            temp=weightArrQC*slownessArr
            slowness_sumArrQC=np.sum(temp, axis=0)
            w2sumArrQC=np.sum(weightArrQC**2, axis=0)
            temp=weightArrQC*(slownessArr-slowness_sumArrQC)**2
            temp=np.sum(temp, axis=0)
            slowness_stdArrQC=np.sqrt(temp/(1-w2sumArrQC))
            # save isotropic velocity to database
            per_group_out= group_out.create_group( name='%g_sec'%( per ) )
            sdset        = per_group_out.create_dataset(name='slowness', data=slowness_sumArrQC)
            s_stddset    = per_group_out.create_dataset(name='slowness_std', data=slowness_stdArrQC)
            Nmdset       = per_group_out.create_dataset(name='Nmeasure', data=Nmeasure)
            NmQCdset     = per_group_out.create_dataset(name='NmeasureQC', data=NmeasureQC)
            #####################################################
            # determine anisotropic parameters, need benchmark and further verification
            #####################################################
            if anisotropic:
                NmeasureAni=np.zeros((Nlat-4, Nlon-4))
                total_near_neighbor=Nmeasure[4:-4, 4:-4]+Nmeasure[:-8, :-8]+Nmeasure[8:, 8:]+Nmeasure[:-8, 4:-4]+\
                        Nmeasure[8:, 4:-4]+Nmeasure[4:-4, :-8]+Nmeasure[4:-4, 8:] + Nmeasure[8:, :-8]+Nmeasure[:-8, 8:]
                NmeasureAni[4:-4, 4:-4]=total_near_neighbor # for quality control
                # initialization of anisotropic parameters
                d_bin=(maxazi-minazi)/N_bin
                histArr=np.zeros((N_bin, Nlat-4, Nlon-4))
                histArr_cutted=histArr[:, 3:-3, 3:-3]
                slow_sum_ani=np.zeros((N_bin, Nlat-4, Nlon-4))
                slow_sum_ani_cutted=slow_sum_ani[:, 3:-3, 3:-3]
                slow_un=np.zeros((N_bin, Nlat-4, Nlon-4))
                slow_un_cutted=slow_un[:, 3:-3, 3:-3]
                azi_11=aziArr[:, :-6, :-6]; azi_12=aziArr[:, :-6, 3:-3]; azi_13=aziArr[:, :-6, 6:]
                azi_21=aziArr[:, 3:-3, :-6]; azi_22=aziArr[:, 3:-3, 3:-3]; azi_23=aziArr[:, 3:-3, 6:]
                azi_31=aziArr[:, 6:, :-6]; azi_32=aziArr[:, 6:, 3:-3]; azi_33=aziArr[:, 6:, 6:]
                slowsumQC_cutted=slowness_sumArrQC[3:-3, 3:-3]
                slownessArr_cutted=slownessArr[:, 3:-3, 3:-3]
                index_outlier_cutted=index_outlier[:, 3:-3, 3:-3]
                for ibin in xrange(N_bin):
                    sumNbin=(np.zeros((Nlat-4, Nlon-4)))[3:-3, 3:-3]
                    slowbin=(np.zeros((Nlat-4, Nlon-4)))[3:-3, 3:-3]
                    ibin11=np.floor((azi_11-minazi)/d_bin); temp1=1*(ibin==ibin11); temp1[index_outlier_cutted]=0
                    temp2=temp1*(slownessArr_cutted-slowsumQC_cutted); 
                    temp1=np.sum(temp1, 0); temp2=np.sum(temp2, 0); #temp2[temp1!=0]=temp2[temp1!=0]/temp1[temp1!=0]
                    sumNbin+=temp1; slowbin+=temp2; #print temp2.max(), temp2.min() 

                    ibin12=np.floor((azi_12-minazi)/d_bin); temp1=1*(ibin==ibin12); temp1[index_outlier_cutted]=0
                    temp2=temp1*(slownessArr_cutted-slowsumQC_cutted)
                    temp1=np.sum(temp1, 0); temp2=np.sum(temp2, 0); sumNbin+=temp1; slowbin+=temp2
                    
                    ibin13=np.floor((azi_13-minazi)/d_bin); temp1=1*(ibin==ibin13); temp1[index_outlier_cutted]=0
                    temp2=temp1*(slownessArr_cutted-slowsumQC_cutted)
                    temp1=np.sum(temp1, 0); temp2=np.sum(temp2, 0); sumNbin+=temp1; slowbin+=temp2
                    
                    ibin21=np.floor((azi_21-minazi)/d_bin); temp1=1*(ibin==ibin21); temp1[index_outlier_cutted]=0
                    temp2=temp1*(slownessArr_cutted-slowsumQC_cutted)
                    temp1=np.sum(temp1, 0); temp2=np.sum(temp2, 0); sumNbin+=temp1; slowbin+=temp2
                    
                    ibin22=np.floor((azi_22-minazi)/d_bin); temp1=1*(ibin==ibin22); temp1[index_outlier_cutted]=0
                    temp2=temp1*(slownessArr_cutted-slowsumQC_cutted)
                    temp1=np.sum(temp1, 0); temp2=np.sum(temp2, 0); sumNbin+=temp1; slowbin+=temp2
                    
                    ibin23=np.floor((azi_23-minazi)/d_bin); temp1=1*(ibin==ibin23); temp1[index_outlier_cutted]=0
                    temp2=temp1*(slownessArr_cutted-slowsumQC_cutted)
                    temp1=np.sum(temp1, 0); temp2=np.sum(temp2, 0); sumNbin+=temp1; slowbin+=temp2
                    
                    ibin31=np.floor((azi_31-minazi)/d_bin); temp1=1*(ibin==ibin31); temp1[index_outlier_cutted]=0
                    temp2=temp1*(slownessArr_cutted-slowsumQC_cutted)
                    temp1=np.sum(temp1, 0); temp2=np.sum(temp2, 0); sumNbin+=temp1; slowbin+=temp2
                    
                    ibin32=np.floor((azi_32-minazi)/d_bin); temp1=1*(ibin==ibin32); temp1[index_outlier_cutted]=0
                    temp2=temp1*(slownessArr_cutted-slowsumQC_cutted)
                    temp1=np.sum(temp1, 0); temp2=np.sum(temp2, 0); sumNbin+=temp1; slowbin+=temp2
                    
                    ibin33=np.floor((azi_33-minazi)/d_bin); temp1=1*(ibin==ibin33); temp1[index_outlier_cutted]=0
                    temp2=temp1*(slownessArr_cutted-slowsumQC_cutted)
                    temp1=np.sum(temp1, 0); temp2=np.sum(temp2, 0); sumNbin+=temp1; slowbin+=temp2
                   
                    histArr_cutted[ibin, :, :]=sumNbin
                    slow_sum_ani_cutted[ibin, :, :]=slowbin
                slow_sum_ani_cutted[histArr_cutted>10]=slow_sum_ani_cutted[histArr_cutted>10]/histArr_cutted[histArr_cutted>10]
                slow_sum_ani_cutted[histArr_cutted<=10]=0
                slow_iso_std=np.broadcast_to(slowness_stdArrQC[3:-3, 3:-3], histArr_cutted.shape)
                slow_un_cutted[histArr_cutted>10]=slow_iso_std[histArr_cutted>10]/np.sqrt(histArr_cutted[histArr_cutted>10])
                slow_un_cutted[histArr_cutted<=10]=0
                temp=np.broadcast_to(slowsumQC_cutted, slow_un_cutted.shape)
                temp=( temp + slow_sum_ani_cutted)**2
                slow_un_cutted=slow_un_cutted/temp
                slow_sum_ani[:, 3:-3, 3:-3]=slow_sum_ani_cutted
                slow_un[:, 3:-3, 3:-3]=slow_un_cutted
                slow_sum_ani[:, NmeasureAni<45]=0 # near neighbor quality control
                slow_un[:, NmeasureAni<45]=0
                histArr[:, 3:-3, 3:-3]=histArr_cutted
                # save data to database
                s_anidset    = per_group_out.create_dataset(name='slownessAni', data=slow_sum_ani)
                s_anistddset = per_group_out.create_dataset(name='slownessAni_std', data=slow_un)
                histdset     = per_group_out.create_dataset(name='histArr', data=histArr)
                NmAnidset    = per_group_out.create_dataset(name='NmeasureAni', data=NmeasureAni)
        return
           
    def _numpy2ma(self, inarray, reason_n=None):
        """Convert input numpy array to masked array
        """
        if reason_n==None:
            outarray=ma.masked_array(inarray, mask=np.zeros(self.reason_n.shape) )
            outarray.mask[self.reason_n!=0]=1
        else:
            outarray=ma.masked_array(inarray, mask=np.zeros(reason_n.shape) )
            outarray.mask[reason_n!=0]=1
        return outarray     
    
    def _get_lon_lat_arr(self, ncut=2):
        """Get longitude/latitude array
        """
        minlon=self.attrs['minlon']
        maxlon=self.attrs['maxlon']
        minlat=self.attrs['minlat']
        maxlat=self.attrs['maxlat']
        dlon=self.attrs['dlon']
        dlat=self.attrs['dlat']
        self.lons=np.arange((maxlon-minlon)/dlon+1-2*ncut)*dlon+minlon+ncut*dlon
        self.lats=np.arange((maxlat-minlat)/dlat+1-2*ncut)*dlat+minlat+ncut*dlat
        self.Nlon=self.lons.size; self.Nlat=self.lats.size
        self.lonArr, self.latArr = np.meshgrid(self.lons, self.lats)
        return
    
    def np2ma(self):
        """Convert numpy data array to masked data array
        """
        try:
            reason_n=self.reason_n
        except:
            raise AttrictError('No reason_n array!')
        self.vel_iso=self._numpy2ma(self.vel_iso)
        return
    
    def get_data4plot(self, period, runid=0, ncut=2, Nmin=15):
        """
        Get data for plotting
        =======================================================================================
        ::: input parameters :::
        period              - period
        runid               - run id
        ncut                - number of cutted edge points
        Nmin                - minimum required number of measurements
        ---------------------------------------------------------------------------------------
        generated data arrays:
        ----------------------------------- isotropic version ---------------------------------
        self.vel_iso        - isotropic velocity
        self.slowness_std   - slowness standard deviation
        self.Nmeasure       - number of measurements at each grid point
        self.reason_n       - array to represent valid/invalid data points
        ---------------------------------- anisotropic version --------------------------------
        include all the array above(but will be converted to masked array), and
        self.N_bin          - number of bins
        self.minazi/maxazi  - min/max azimuth
        self.slownessAni    - anisotropic slowness perturbation categorized for each bin
        self.slownessAni_std- anisotropic slowness perturbation std
        self.histArr        - number of measurements for each bins
        self.NmeasureAni    - number of measurements for near neighbor points
        =======================================================================================
        """
        self._get_lon_lat_arr(ncut=ncut)
        Nlon=self.attrs['Nlon']
        Nlat=self.attrs['Nlat']
        subgroup=self['Eikonal_stack_'+str(runid)+'/%g_sec'%( period )]
        self.period=period
        slowness=subgroup['slowness'].value
        self.vel_iso=np.zeros((Nlat-4, Nlon-4))
        self.vel_iso[slowness!=0]=1./slowness[slowness!=0]
        self.Nmeasure=subgroup['Nmeasure'].value
        self.slowness_std=subgroup['slowness_std'].value
        self.reason_n=np.zeros((Nlat-4, Nlon-4))
        self.reason_n[self.Nmeasure<Nmin]=1
        group=self['Eikonal_stack_'+str(runid)]
        self.anisotropic=group.attrs['anisotropic']
        self.fieldtype=group.attrs['fieldtype']
        if self.anisotropic:
            self.N_bin=group.attrs['N_bin']
            self.minazi=group.attrs['minazi']
            self.maxazi=group.attrs['maxazi']
            self.slownessAni=subgroup['slownessAni'].value
            self.slownessAni_std=subgroup['slownessAni_std'].value
            self.histArr=subgroup['histArr'].value
            self.NmeasureAni=subgroup['NmeasureAni'].value
        return
    
    def _get_basemap(self, projection='lambert', geopolygons=None, resolution='i'):
        """Get basemap for plotting results
        """
        # fig=plt.figure(num=None, figsize=(12, 12), dpi=80, facecolor='w', edgecolor='k')
        minlon=self.attrs['minlon']
        maxlon=self.attrs['maxlon']
        minlat=self.attrs['minlat']
        maxlat=self.attrs['maxlat']
        lat_centre = (maxlat+minlat)/2.0
        lon_centre = (maxlon+minlon)/2.0
        if projection=='merc':
            m=Basemap(projection='merc', llcrnrlat=minlat-5., urcrnrlat=maxlat+5., llcrnrlon=minlon-5.,
                      urcrnrlon=maxlon+5., lat_ts=20, resolution=resolution)
            # m.drawparallels(np.arange(minlat,maxlat,dlat), labels=[1,0,0,1])
            # m.drawmeridians(np.arange(minlon,maxlon,dlon), labels=[1,0,0,1])
            m.drawparallels(np.arange(-80.0,80.0,5.0), labels=[1,0,0,1])
            m.drawmeridians(np.arange(-170.0,170.0,5.0), labels=[1,0,0,1])
            m.drawstates(color='g', linewidth=2.)
        elif projection=='global':
            m=Basemap(projection='ortho',lon_0=lon_centre, lat_0=lat_centre, resolution=resolution)
            # m.drawparallels(np.arange(-80.0,80.0,10.0), labels=[1,0,0,1])
            # m.drawmeridians(np.arange(-170.0,170.0,10.0), labels=[1,0,0,1])
        elif projection=='regional_ortho':
            m1 = Basemap(projection='ortho', lon_0=minlon, lat_0=minlat, resolution='l')
            m = Basemap(projection='ortho', lon_0=minlon, lat_0=minlat, resolution=resolution,\
                llcrnrx=0., llcrnry=0., urcrnrx=m1.urcrnrx/mapfactor, urcrnry=m1.urcrnry/3.5)
            m.drawparallels(np.arange(-80.0,80.0,10.0), labels=[1,0,0,0],  linewidth=2,  fontsize=20)
            # m.drawparallels(np.arange(-90.0,90.0,30.0),labels=[1,0,0,0], dashes=[10, 5], linewidth=2,  fontsize=20)
            # m.drawmeridians(np.arange(10,180.0,30.0), dashes=[10, 5], linewidth=2)
            m.drawmeridians(np.arange(-170.0,170.0,10.0),  linewidth=2)
        elif projection=='lambert':
            distEW, az, baz=obspy.geodetics.gps2dist_azimuth(minlat, minlon, minlat, maxlon) # distance is in m
            distNS, az, baz=obspy.geodetics.gps2dist_azimuth(minlat, minlon, maxlat+2., minlon) # distance is in m
            m = Basemap(width=distEW, height=distNS, rsphere=(6378137.00,6356752.3142), resolution='l', projection='lcc',\
                lat_1=minlat, lat_2=maxlat, lon_0=lon_centre, lat_0=lat_centre+1)
            m.drawparallels(np.arange(-80.0,80.0,10.0), linewidth=1, dashes=[2,2], labels=[1,1,0,0], fontsize=15)
            m.drawmeridians(np.arange(-170.0,170.0,10.0), linewidth=1, dashes=[2,2], labels=[0,0,1,0], fontsize=15)
            # m.drawparallels(np.arange(-80.0,80.0,10.0), linewidth=0.5, dashes=[2,2], labels=[1,0,0,0], fontsize=5)
            # m.drawmeridians(np.arange(-170.0,170.0,10.0), linewidth=0.5, dashes=[2,2], labels=[0,0,0,1], fontsize=5)
        m.drawcoastlines(linewidth=1.0)
        m.drawcountries(linewidth=1.)
        # m.drawmapboundary(fill_color=[1.0,1.0,1.0])
        m.fillcontinents(lake_color='#99ffff',zorder=0.2)
        m.drawstates()
        m.drawmapboundary(fill_color="white")
        try:
            geopolygons.PlotPolygon(inbasemap=m)
        except:
            pass
        return m
    
            
    
    def plot_vel_iso(self, projection='lambert', fastaxis=False, geopolygons=None, showfig=True, vmin=2.9, vmax=3.5):
        """Plot isotropic velocity
        """
        m=self._get_basemap(projection=projection, geopolygons=geopolygons)
        x, y=m(self.lonArr, self.latArr)
        cmap = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
            0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
        im=m.pcolormesh(x, y, self.vel_iso, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
        cb = m.colorbar(im, "bottom", size="3%", pad='2%')
        cb.set_label('V'+self.fieldtype+'(km/s)', fontsize=12, rotation=0)
        plt.title(str(self.period)+' sec', fontsize=20)
        # if fastaxis:
        #     try:
        #         self.plot_fast_axis(inbasemap=m)
        #     except:
        #         pass
        if showfig:
            plt.show()
        

def eikonal4mp(infield, workingdir, channel):
    working_per=workingdir+'/'+str(infield.period)+'sec'
    outfname=infield.evid+'_'+infield.fieldtype+'_'+channel+'.lst'
    infield.interp_surface(workingdir=working_per, outfname=outfname)
    infield.check_curvature(workingdir=working_per, outpfx=infield.evid+'_'+channel+'_')
    infield.gradient_qc(workingdir=working_per, inpfx=infield.evid+'_'+channel+'_', nearneighbor=True, cdist=None)
    outfname_npz=working_per+'/'+infield.evid+'_field2d'
    infield.write_binary(outfname=outfname_npz)
    return

def helmhotz4mp(infieldpair, workingdir, channel, amplplc):
    tfield=infieldpair[0]
    working_per=workingdir+'/'+str(tfield.period)+'sec'
    outfname=tfield.evid+'_'+tfield.fieldtype+'_'+channel+'.lst'
    tfield.interp_surface(workingdir=working_per, outfname=outfname)
    tfield.check_curvature(workingdir=working_per, outpfx=tfield.evid+'_'+channel+'_')
    tfield.gradient_qc(workingdir=working_per, inpfx=tfield.evid+'_'+channel+'_', nearneighbor=True, cdist=None)
    outfname_npz=working_per+'/'+tfield.evid+'_field2d'
    if not amplplc: tfield.write_binary(outfname=outfname_npz)
    if amplplc:
        field2dAmp=infieldpair[1]
        outfnameAmp=field2dAmp.evid+'_Amp_'+channel+'.lst'
        field2dAmp.interp_surface(workingdir=working_per, outfname=outfnameAmp)
        field2dAmp.gradient()
        field2dAmp.cut_edge(1,1)
        field2dAmp.Laplacian()
        field2dAmp.cut_edge(1,1)
        field2dAmp.get_lplc_amp()
        slownessApp=-np.ones(tfield.appV.shape)
        slownessApp[tfield.appV!=0]=1./tfield.appV[tfield.appV!=0]
        temp=slownessApp**2-field2dAmp.lplc_amp
        temp[temp<0]=0
        slownessCor=np.sqrt(temp)
        corV=np.zeros(slownessCor.shape)
        corV[slownessCor!=0]=1./slownessCor[slownessCor!=0]
        tfield.corV=corV
        tfield.lplc_amp=field2dAmp.lplc_amp
        tfield.write_binary(outfname=outfname_npz, amplplc=amplplc)
    return 

