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
import matplotlib
import colormaps
import obspy
import field2d_earth
import numexpr
import warnings
from functools import partial
import multiprocessing
from numba import jit, float32, int32, boolean, float64
import numba
import time
import eikonaltomo

# compiled function to get weight for each event and each grid point
@jit(float32[:,:,:](float32[:,:,:], float32[:,:,:]))
def _get_azi_weight(aziALL, validALL):
    Nevent, Nlon, Nlat  = aziALL.shape
    weightALL           = np.zeros((Nevent, Nlon, Nlat), dtype=np.float32)
    for ilon in xrange(Nlon):
        for ilat in xrange(Nlat):
            for i in xrange(Nevent):
                for j in xrange(Nevent):
                    delAzi                      = abs(aziALL[i, ilon, ilat] - aziALL[j, ilon, ilat])
                    if delAzi < 20. or delAzi > 340.:
                        weightALL[i, ilon, ilat]+= validALL[i, ilon, ilat]    
    return weightALL

# compiled function to evaluate station distribution 
@jit(boolean(float64[:], float64[:], int32))
def _check_station_distribution_old(lons, lats, Nvalid_min):
    N       = lons.size
    Nvalid  = 0
    for i in range(N):
        lon1            = lons[i]
        lat1            = lats[i]
        NnearE          = 0
        NnearW          = 0
        NnearN          = 0
        NnearS          = 0
        for j in range(N):
            lon2        = lons[j]
            lat2        = lats[j]
            if i == j:
                continue
            if abs(lat1 - lat2) < 1.5:
                colat           = 90. - (lat1+lat2)/2.
                temp_R          = 6371. * np.sin(np.pi * colat/180.)
                dlon            = abs(lon1 - lon2)
                dist_lon        = temp_R * np.sin(dlon*np.pi/180.)
                if dist_lon < 150.:
                    if lon2 >= lon1:
                        NnearW  += 1
                    else:
                        NnearE  += 1
                    if lat2 >= lat1:
                        NnearN  += 1
                    else:
                        NnearS  += 1
        if NnearE > 0 and NnearW > 0 and NnearN > 0 and NnearS > 0:
            Nvalid  += 1
    if Nvalid >= Nvalid_min:
        return True
    else:
        return False
    
@jit(boolean(float64[:], float64[:], int32))
def _check_station_distribution(lons, lats, Nvalid_min):
    """check the station distribution
        Step 1. a station is counted as valid if there are at least four stations nearby
        Step 2. check if the number of valid stations is larger than Nvalid_min 
    """
    N       = lons.size
    Nvalid  = 0
    for i in range(N):
        lon1            = lons[i]
        lat1            = lats[i]
        Nnear           = 0
        for j in range(N):
            lon2        = lons[j]
            lat2        = lats[j]
            if i == j:
                continue
            if abs(lat1 - lat2) < 1.5:
                colat           = 90. - (lat1+lat2)/2.
                temp_R          = 6371. * np.sin(np.pi * colat/180.)
                dlon            = abs(lon1 - lon2)
                dist_lon        = temp_R * np.sin(dlon*np.pi/180.)
                if dist_lon < 150.:
                    Nnear       += 1
        if Nnear >= 4:
            Nvalid  += 1
    if Nvalid >= Nvalid_min:
        return True
    else:
        return False

def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    # # # s = str(100 * y)
    s = str(y)
    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'

@jit(boolean[:](boolean[:], float64[:], float64[:], float64[:], float64[:]))
def _get_mask_interp(mask_in, lons_in, lats_in, lons, lats):
    Nlat            = lats.size
    Nlon            = lons.size
    mask_out        = np.ones((Nlat, Nlon), dtype=np.bool)
    for i in range(Nlat):
        for j in range(Nlon):
            clat    = lats[i]
            clon    = lons[j]
            ind_lon = np.where(clon<=lons_in)[0][0]      
            ind_lat = np.where(clat<=lats_in)[0][0]
            if (clon - lons_in[ind_lon])< 0.001 and (clat - lats_in[ind_lat]) < 0.001:
                mask_out[i, j]      = mask_in[ind_lat, ind_lon]
                continue
            mask_out[i, j]          = mask_out[i, j]*mask_in[ind_lat, ind_lon]
            if ind_lat > 0:
                mask_out[i, j]      = mask_out[i, j]*mask_in[ind_lat-1, ind_lon]
                if ind_lon > 0:
                    mask_out[i, j]  = mask_out[i, j]*mask_in[ind_lat-1, ind_lon-1]
            if ind_lon > 0:
                mask_out[i, j]      = mask_out[i, j]*mask_in[ind_lat, ind_lon-1]
                if ind_lat > 0:
                    mask_out[i, j]  = mask_out[i, j]*mask_in[ind_lat-1, ind_lon-1]
    return mask_out

def plot_fault_lines(mapobj, infname, lw=2, color='red'):
    with open(infname, 'rb') as fio:
        is_new  = False
        lonlst  = []
        latlst  = []
        for line in fio.readlines():
            if line.split()[0] == '>':
                x, y  = mapobj(lonlst, latlst)
                mapobj.plot(x, y,  lw = lw, color=color)
                # # # m.plot(xslb, yslb,  lw = 3, color='white')
                lonlst  = []
                latlst  = []
                continue
            lonlst.append(float(line.split()[0]))
            latlst.append(float(line.split()[1]))
        x, y  = mapobj(lonlst, latlst)
        mapobj.plot(x, y,  lw = lw, color=color)

@jit(numba.types.Tuple((float64[:, :, :], float64[:, :, :], float64[:, :, :], float64[:, :]))\
     (int32, int32, float32, float32, int32, float64[:, :], float64[:, :, :], float64[:, :], float64[:, :, :], numba.boolean[:, :, :]))
def _anisotropic_stacking(gridx, gridy, maxazi, minazi, N_bin, Nmeasure, aziALL,\
        slowness_sumQC, slownessALL, index_outlier):
    Nevent, Nx, Ny  = aziALL.shape
    Nx_trim         = Nx - (gridx - 1)
    Ny_trim         = Ny - (gridy - 1)
    NmeasureAni     = np.zeros((Nx_trim, Ny_trim), dtype=np.float64) # for quality control
    for ishift_x in range(gridx):
        for ishift_y in range(gridy):
            for ix in range(Nx_trim):
                for iy in range(Ny_trim):
                    NmeasureAni[ix, iy]  += Nmeasure[ix + ishift_x, iy + ishift_y]
    # initialization of anisotropic parameters
    d_bin           = float((maxazi-minazi)/N_bin)
    # number of measurements in each bin
    histArr         = np.zeros((N_bin, Nx_trim, Ny_trim))
    # slowness in each bin
    dslow_sum_ani   = np.zeros((N_bin, Nx_trim, Ny_trim))
    # slowness uncertainties for each bin
    dslow_un        = np.zeros((N_bin, Nx_trim, Ny_trim))
    # velocity uncertainties for each bin
    vel_un          = np.zeros((N_bin, Nx_trim, Ny_trim))
    #----------------------------------------------------------------------------------
    # Loop over azimuth bins to get slowness, velocity and number of measurements
    #----------------------------------------------------------------------------------
    for ibin in range(N_bin):
        sumNbin                     = np.zeros((Nx_trim, Ny_trim))
        # slowness arrays
        dslowbin                    = np.zeros((Nx_trim, Ny_trim))
        dslow_un_ibin               = np.zeros((Nx_trim, Ny_trim))
        dslow_mean                  = np.zeros((Nx_trim, Ny_trim))
        # velocity arrays
        velbin                      = np.zeros((Nx_trim, Ny_trim))
        vel_un_ibin                 = np.zeros((Nx_trim, Ny_trim))
        vel_mean                    = np.zeros((Nx_trim, Ny_trim))
        for ix in range(Nx_trim):
            for iy in range(Ny_trim):
                for ishift_x in range(gridx):
                    for ishift_y in range(gridy):
                        for iev in range(Nevent):
                            azi         = aziALL[iev, ix + ishift_x, iy + ishift_y]
                            ibin_temp   = np.floor((azi - minazi)/d_bin)
                            if ibin_temp != ibin:
                                continue
                            is_outlier  = index_outlier[iev, ix + ishift_x, iy + ishift_y]
                            if is_outlier:
                                continue
                            temp_dslow  = slownessALL[iev, ix + ishift_x, iy + ishift_y] - slowness_sumQC[ix + ishift_x, iy + ishift_y]
                            if slownessALL[iev, ix + ishift_x, iy + ishift_y] != 0.:
                                temp_vel= 1./slownessALL[iev, ix + ishift_x, iy + ishift_y]
                            else:
                                temp_vel= 0.
                            sumNbin[ix, iy]     += 1
                            dslowbin[ix, iy]    += temp_dslow
                            velbin[ix, iy]      += temp_vel
                # end nested loop of grid shifting
                if sumNbin[ix, iy] >= 2:
                    vel_mean[ix, iy]            = velbin[ix, iy] / sumNbin[ix, iy]
                    dslow_mean[ix, iy]          = dslowbin[ix, iy] / sumNbin[ix, iy]
                else:
                    sumNbin[ix, iy]             = 0
        # compute uncertainties
        for ix in range(Nx_trim):
            for iy in range(Ny_trim):
                for ishift_x in range(gridx):
                    for ishift_y in range(gridy):
                        for iev in range(Nevent):
                            azi                     = aziALL[iev, ix + ishift_x, iy + ishift_y]
                            ibin_temp               = np.floor((azi - minazi)/d_bin)
                            if ibin_temp != ibin:
                                continue
                            is_outlier              = index_outlier[iev, ix + ishift_x, iy + ishift_y]
                            if is_outlier:
                                continue
                            if slownessALL[iev, ix + ishift_x, iy + ishift_y] != 0.:
                                temp_vel            = 1./slownessALL[iev, ix + ishift_x, iy + ishift_y]
                            else:
                                temp_vel            = 0.
                            temp_vel_mean           = vel_mean[ix, iy]
                            vel_un_ibin[ix, iy]     += (temp_vel - temp_vel_mean)**2
                            temp_dslow              = slownessALL[iev, ix + ishift_x, iy + ishift_y] - slowness_sumQC[ix + ishift_x, iy + ishift_y]
                            temp_dslow_mean         = dslow_mean[ix, iy]
                            dslow_un_ibin[ix, iy]   += (temp_dslow - temp_dslow_mean)**2
        for ix in range(Nx_trim):
            for iy in range(Ny_trim):
                if sumNbin[ix, iy] < 2:
                    continue
                vel_un_ibin[ix, iy]             = np.sqrt(vel_un_ibin[ix, iy]/(sumNbin[ix, iy] - 1)/sumNbin[ix, iy])
                vel_un[ibin, ix, iy]            = vel_un_ibin[ix, iy]
                dslow_un_ibin[ix, iy]           = np.sqrt(dslow_un_ibin[ix, iy]/(sumNbin[ix, iy] - 1)/sumNbin[ix, iy])
                dslow_un[ibin, ix, iy]          = dslow_un_ibin[ix, iy]
                histArr[ibin, ix, iy]           = sumNbin[ix, iy]
                dslow_sum_ani[ibin, ix, iy]     = dslow_mean[ix, iy]
    return dslow_sum_ani, dslow_un, vel_un, histArr, NmeasureAni
        



    
class hybridTomoDataSet(eikonaltomo.EikonalTomoDataSet):
    """
    Object for merging eikonal tomography results, ray tomography results
    """
    #==================================================
    # functions print the information of database
    #==================================================
    def print_attrs(self, print_to_screen=True):
        """
        Print the attrsbute information of the dataset.
        """
        outstr      =  '======================================== Surface wave hybrid tomography database ======================================\n'
        try:
            outstr      += '--- period (s):                             - '+str(self.attrs['period_array'])+'\n'
            try:
                # outstr  += '--- per_xcorr (s):                          - '+str(self.attrs['per_xcorr'])+'\n'
                outstr  += '    per_xcorr_min/per_xcorr_max (s):        - '+str(self.attrs['per_xcorr_min'])+'/'+str(self.attrs['per_xcorr_max'])+'\n'
            except:
                outstr  += '*** NO ambient noise eikonal data\n'
            try:
                # outstr  += '--- per_quake (s):                          - '+str(self.attrs['per_quake'])+'\n'
                outstr  += '    per_quake_min/per_quake_max (s):        - '+str(self.attrs['per_quake_min'])+'/'+str(self.attrs['per_quake_max'])+'\n'
            except:
                outstr  += '*** NO earthquake eikonal/Helmholtz data\n'
            outstr      += '--- period_array_ray (s):                   - '+str(self.attrs['period_array_ray'])+'\n'
            outstr      += '    longitude range                         - '+str(self.attrs['minlon'])+' ~ '+str(self.attrs['maxlon'])+'\n'
            outstr      += '    longitude spacing/npts                  - '+str(self.attrs['dlon'])+'/'+str(self.attrs['Nlon'])+'\n'
            outstr      += '    nlon_grad/nlon_lplc                     - '+str(self.attrs['nlon_grad'])+'/'+str(self.attrs['nlon_lplc'])+'\n'
            outstr      += '    latitude range                          - '+str(self.attrs['minlat'])+' ~ '+str(self.attrs['maxlat'])+'\n'
            outstr      += '    latitude spacing/npts                   - '+str(self.attrs['dlat'])+'/'+str(self.attrs['Nlat'])+'\n'
            outstr      += '    nlat_grad/nlat_lplc                     - '+str(self.attrs['nlat_grad'])+'/'+str(self.attrs['nlat_lplc'])+'\n'
            try:
                outstr  += '!!! interpolated dlon/dlat:                 - '+str(self.attrs['dlon_interp'])+'/'+str(self.attrs['dlat_interp'])+'\n'
            except:
                outstr  += '*** NO interpolated data\n'
            per_arr     = self.attrs['period_array']
        except:
            print 'Empty Database!'
            return None
        if print_to_screen:
            print outstr
        else:
            return outstr
        return
    
    def print_info(self, runid=0):
        """print the information of given eikonal/Helmholz run
        """
        outstr      = self.print_attrs(print_to_screen=False)
        if outstr is None:
            return
        try:
            xcorr_grp   = self['xcorr_run']
            perid       = '%d_sec' % self.attrs['per_xcorr_min']
            pergrp      = xcorr_grp[perid]
            Nevent      = len(pergrp.keys())
            outstr      += '============================================= ambient noise correlation ===============================================\n'
            outstr      += '--- number of virtual events                        - '+str(Nevent)+'\n'
            evid        = pergrp.keys()[0]
            evgrp       = pergrp[evid]
            outstr      += '--- attributes for each event                       - Nvalid_grd, Ntotal_grd \n'
            outstr      += '--- appV (apparent velocity)                        - '+str(evgrp['appV'].shape)+'\n'
            outstr      += '--- az (azimuth)                                    - '+str(evgrp['az'].shape)+'\n'
            outstr      += '--- reason_n (index array)                          - '+str(evgrp['reason_n'].shape)+'\n'
            outstr      += '        0: accepted point \n' + \
                           '        1: data point the has large difference between v1HD and v1HD02 \n' + \
                           '        2: data point that does not have near neighbor points at all E/W/N/S directions\n' + \
                           '        3: slowness is too large/small \n' + \
                           '        4: near a zero field data point \n' + \
                           '        5: epicentral distance is too small \n' + \
                           '        6: large curvature              \n'
        except:
            pass
        try:
            quake_grp   = self['quake_run']
            perid       = '%d_sec' % self.attrs['per_quake_min']
            pergrp      = quake_grp[perid]
            Nevent      = len(pergrp.keys())
            outstr      += '================================================== earthquake data ====================================================\n'
            outstr      += '--- number of events                                - period-dependent \n'
            evid        = pergrp.keys()[0]
            evgrp       = pergrp[evid]
            outstr      += '--- attributes for each event                       - Nvalid_grd, Ntotal_grd \n'
            outstr      += '--- appV (apparent velocity)                        - '+str(evgrp['appV'].shape)+'\n'
            outstr      += '--- az (azimuth)                                    - '+str(evgrp['az'].shape)+'\n'
            outstr      += '--- reason_n (index array)                          - '+str(evgrp['reason_n'].shape)+'\n'
            # outstr      += '        0: accepted point \n' + \
            #                '        1: data point the has large difference between v1HD and v1HD02 \n' + \
            #                '        2: data point that does not have near neighbor points at all E/W/N/S directions\n' + \
            #                '        3: slowness is too large/small \n' + \
            #                '        4: near a zero field data point \n' + \
            #                '        5: epicentral distance is too small \n' + \
            #                '        6: large curvature              \n'
        except:
            pass
        try:
            subgroup= self['Eikonal_stack_%d' %runid]
            outstr  += '============================================== eikonal stacked results id = %d'% runid +' =========================================\n'
        except KeyError:
            outstr  += '============================================= NO corresponding stacked results id = %d'% runid +'=================================\n'
            return
        if subgroup.attrs['anisotropic']:
            tempstr = 'anisotropic'
            outstr  += '--- isotropic/anisotropic                           - '+tempstr+'\n'
            outstr  += '--- N_bin (number of bins, for ani run)             - '+str(subgroup.attrs['N_bin'])+'\n'
            outstr  += '--- minazi/maxazi (min/max azi, for ani run)        - '+str(subgroup.attrs['minazi'])+'/'+str(subgroup.attrs['maxazi'])+'\n'
        else:
            tempstr = 'isotropic'
            outstr  += '--- isotropic/anisotropic                           - '+tempstr+'\n'
        pergrp      = subgroup[perid]
        outstr      += '--- Nmeasure (number of raw measurements)           - '+str(pergrp['Nmeasure'].shape)+'\n'
        outstr      += '--- NmeasureQC (number of qc measurements)          - '+str(pergrp['NmeasureQC'].shape)+'\n'
        outstr      += '--- slowness                                        - '+str(pergrp['slowness'].shape)+'\n'
        outstr      += '--- slowness_std                                    - '+str(pergrp['slowness_std'].shape)+'\n'
        outstr      += '--- mask                                            - '+str(pergrp['mask'].shape)+'\n'
        outstr      += '--- vel_iso (isotropic velocity)                    - '+str(pergrp['vel_iso'].shape)+'\n'
        outstr      += '--- vel_sem (uncertainties for velocity)            - '+str(pergrp['vel_sem'].shape)+'\n'
        
        try:
            subgroup= self['merged_tomo_%d' %runid]
            outstr  += '============================================== merged tomography results id = %d'% runid +' =======================================\n'
        except KeyError:
            outstr  += '============================================== NO corresponding merged results id = %d'% runid +'=================================\n'
            return
        outstr      += '--- T_ray_max (s)                                   - '+str(subgroup.attrs['T_ray_max'])+'\n'
        outstr      += '--- mask_ray (not attrs, determined over all pers)  - '+str(subgroup['mask_ray'].shape)+'\n'
        outstr      += '!!! mask_ray_interp (not attrs, from mask_ray, MC)  - '+str(subgroup['mask_ray_interp'].shape)+'\n'
        perid       = '%d_sec' % self.attrs['period_array'][-1]
        pergrp      = subgroup[perid]
        outstr      += '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ in the period subdirectory $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ \n'
        outstr      += '--- Nmeasure (number of (qc) measurements)          - '+str(pergrp['Nmeasure'].shape)+'\n'
        outstr      += '--- mask (mask_ray or mask_eik)                     - '+str(pergrp['mask'].shape)+'\n'
        outstr      += '--- mask_eik (mask of eikonal results)              - '+str(pergrp['mask_eik'].shape)+'\n'
        outstr      += '!!! mask_interp (from mask_eik, T > T_ray_max, MC)  - '+str(pergrp['mask_interp'].shape)+'\n'
        outstr      += '--- vel_iso (isotropic velocity)                    - '+str(pergrp['vel_iso'].shape)+'\n'
        outstr      += '!!! vel_iso_interp (used for MC inversion)          - '+str(pergrp['vel_iso_interp'].shape)+'\n'
        outstr      += '--- vel_sem (uncertainties for velocity)            - '+str(pergrp['vel_sem'].shape)+'\n'
        outstr      += '!!! vel_sem_interp (used for MC inversion)          - '+str(pergrp['vel_sem_interp'].shape)+'\n'
        print outstr
        return
    
    def _get_lon_lat_arr_interp(self, ncut=0):
        """Get longitude/latitude array
        """
        minlon      = self.attrs['minlon']
        maxlon      = self.attrs['maxlon']
        minlat      = self.attrs['minlat']
        maxlat      = self.attrs['maxlat']
        dlon        = self.attrs['dlon_interp']
        dlat        = self.attrs['dlat_interp']
        self.lons   = np.arange((maxlon-minlon)/dlon+1-2*ncut)*dlon+minlon+ncut*dlon
        self.lats   = np.arange((maxlat-minlat)/dlat+1-2*ncut)*dlat+minlat+ncut*dlat
        self.Nlon   = self.lons.size
        self.Nlat   = self.lats.size
        self.lonArr, self.latArr = np.meshgrid(self.lons, self.lats)
        return
    
    def read_xcorr(self, inh5fname, runid=0):
        """
        read noise correlation eikonal tomography results 
        =============================================================
        ::: input parameters :::
        inh5fname   - input xcorr eikonal tomography data file
        runid       - run id
        =============================================================
        """
        group               = self.create_group( name = 'xcorr_run' )
        # input data file
        in_dset             = EikonalTomoDataSet(inh5fname)
        in_group            = in_dset['Eikonal_run_'+str(runid)]
        #------------------------------------
        # period arrays and other attributes
        #------------------------------------
        try:
            pers            = self.attrs['period_array']
            minlon          = self.attrs['minlon']
            maxlon          = self.attrs['maxlon']
            minlat          = self.attrs['minlat']
            maxlat          = self.attrs['maxlat']
            dlon            = self.attrs['dlon']
            dlat            = self.attrs['dlat']
            nlat_grad       = self.attrs['nlat_grad']
            nlon_grad       = self.attrs['nlon_grad']
            nlat_lplc       = self.attrs['nlat_lplc']
            nlon_lplc       = self.attrs['nlon_lplc']
        except:
            pers            = in_dset.attrs['period_array']
            minlon          = in_dset.attrs['minlon']
            maxlon          = in_dset.attrs['maxlon']
            minlat          = in_dset.attrs['minlat']
            maxlat          = in_dset.attrs['maxlat']
            dlon            = in_dset.attrs['dlon']
            dlat            = in_dset.attrs['dlat']
            nlat_grad       = in_dset.attrs['nlat_grad']
            nlon_grad       = in_dset.attrs['nlon_grad']
            nlat_lplc       = in_dset.attrs['nlat_lplc']
            nlon_lplc       = in_dset.attrs['nlon_lplc']
            self.set_input_parameters(minlon=minlon, maxlon=maxlon, minlat=minlat, maxlat=maxlat, pers=pers,\
                dlon=dlon, dlat=dlat, nlat_grad=nlat_grad, nlon_grad=nlon_grad, nlat_lplc=nlat_lplc, nlon_lplc=nlon_lplc,\
                    optimize_spacing=False)
        # check attributes
        if minlon != in_dset.attrs['minlon'] or maxlon != in_dset.attrs['maxlon'] or \
                minlat != in_dset.attrs['minlat'] or maxlat != in_dset.attrs['maxlat'] or \
                dlon != in_dset.attrs['dlon'] or dlat != in_dset.attrs['dlat'] or\
                minlon != in_dset.attrs['minlon'] or minlon != in_dset.attrs['minlon'] or \
                dlon != in_dset.attrs['dlon'] or dlat != in_dset.attrs['dlat'] or \
                nlat_grad != in_dset.attrs['nlat_grad'] or nlon_grad != in_dset.attrs['nlon_grad'] or\
                nlat_lplc != in_dset.attrs['nlat_lplc'] or nlon_lplc != in_dset.attrs['nlon_lplc']:
            raise ValueError('Inconsistent attributes!')
        in_per              = in_dset.attrs['period_array']
        per_xcorr           = np.array([])
        # Loop over periods from input database to load xcorr eikonal data
        for per in in_per:
            try:
                in_per_group= in_group['%g_sec'%( per )]
            except:
                print 'No data for T = '+str(per)
                continue
            per_xcorr       = np.append(per_xcorr, per)
            per_group       = group.create_group( name='%g_sec'%( per ) )
            Nevent          = len(in_per_group.keys())
            print 'Reading xcorr eikonal results for: '+str(per)+' sec, '+str(Nevent)+ ' events'
            for iev in range(Nevent):
                # get data
                evid                        = in_per_group.keys()[iev]
                in_event_group              = in_per_group[evid]
                az                          = in_event_group['az'].value
                velocity                    = in_event_group['appV'].value
                reason_n                    = in_event_group['reason_n'].value
                Ntotal_grd                  = in_event_group.attrs['Ntotal_grd']
                Nvalid_grd                  = in_event_group.attrs['Nvalid_grd']
                # save data
                event_group                 = per_group.create_group(name=evid)
                event_group.attrs.create(name = 'Ntotal_grd', data=Ntotal_grd)
                event_group.attrs.create(name = 'Nvalid_grd', data=Nvalid_grd)
                azdset                      = event_group.create_dataset(name='az', data=az)
                appVdset                    = event_group.create_dataset(name='appV', data=velocity)
                reason_ndset                = event_group.create_dataset(name='reason_n', data=reason_n)
        # check period arrays
        for iper in range(pers.size):
            per             = pers[iper]
            if per < per_xcorr.min():
                continue
            if per > per_xcorr.max():
                break
            if not per in per_xcorr:
                raise KeyError('Inconsistent period arrays!')
        # save periods for xcorr
        self.attrs.create(name = 'per_xcorr', data=per_xcorr, dtype='f')
        self.attrs.create(name = 'per_xcorr_min', data=per_xcorr[0], dtype='f')
        self.attrs.create(name = 'per_xcorr_max', data=per_xcorr[-1], dtype='f')
        return
    
    def read_quake(self, inh5fname, runid=0):
        """
        read earthquake eikonal tomography results 
        =============================================================
        ::: input parameters :::
        inh5fname   - input quake eikonal tomography data file
        runid       - run id
        =============================================================
        """
        group               = self.create_group( name = 'quake_run' )
        # input data file
        in_dset             = EikonalTomoDataSet(inh5fname)
        in_group            = in_dset['Eikonal_run_'+str(runid)]
        try:
            pers            = self.attrs['period_array']
            minlon          = self.attrs['minlon']
            maxlon          = self.attrs['maxlon']
            minlat          = self.attrs['minlat']
            maxlat          = self.attrs['maxlat']
            dlon            = self.attrs['dlon']
            dlat            = self.attrs['dlat']
            nlat_grad       = self.attrs['nlat_grad']
            nlon_grad       = self.attrs['nlon_grad']
            nlat_lplc       = self.attrs['nlat_lplc']
            nlon_lplc       = self.attrs['nlon_lplc']
        except:
            pers            = in_dset.attrs['period_array']
            minlon          = in_dset.attrs['minlon']
            maxlon          = in_dset.attrs['maxlon']
            minlat          = in_dset.attrs['minlat']
            maxlat          = in_dset.attrs['maxlat']
            dlon            = in_dset.attrs['dlon']
            dlat            = in_dset.attrs['dlat']
            nlat_grad       = in_dset.attrs['nlat_grad']
            nlon_grad       = in_dset.attrs['nlon_grad']
            nlat_lplc       = in_dset.attrs['nlat_lplc']
            nlon_lplc       = in_dset.attrs['nlon_lplc']
            self.set_input_parameters(minlon=minlon, maxlon=maxlon, minlat=minlat, maxlat=maxlat, pers=pers,\
                dlon=dlon, dlat=dlat, nlat_grad=nlat_grad, nlon_grad=nlon_grad, nlat_lplc=nlat_lplc, nlon_lplc=nlon_lplc,\
                    optimize_spacing=False)
        # check attributes
        if minlon != in_dset.attrs['minlon'] or maxlon != in_dset.attrs['maxlon'] or \
                minlat != in_dset.attrs['minlat'] or maxlat != in_dset.attrs['maxlat'] or \
                dlon != in_dset.attrs['dlon'] or dlat != in_dset.attrs['dlat'] or\
                minlon != in_dset.attrs['minlon'] or minlon != in_dset.attrs['minlon'] or \
                dlon != in_dset.attrs['dlon'] or dlat != in_dset.attrs['dlat'] or \
                nlat_grad != in_dset.attrs['nlat_grad'] or nlon_grad != in_dset.attrs['nlon_grad'] or\
                nlat_lplc != in_dset.attrs['nlat_lplc'] or nlon_lplc != in_dset.attrs['nlon_lplc']:
            raise ValueError('Inconsistent attributes!')
        in_per              = in_dset.attrs['period_array']
        # Loop over periods from input database to load earthquake eikonal data
        per_quake           = np.array([])
        for per in in_per:
            try:
                in_per_group    = in_group['%g_sec'%( per )]
            except:
                print 'No data for T = '+str(per)
                continue
            per_quake       = np.append(per_quake, per)
            per_group       = group.create_group( name='%g_sec'%( per ) )
            Nevent          = len(in_per_group.keys())
            print 'Reading quake eikonal results for: '+str(per)+' sec, '+str(Nevent)+ ' events'
            for iev in range(Nevent):
                # get data
                evid                        = in_per_group.keys()[iev]
                in_event_group              = in_per_group[evid]
                az                          = in_event_group['az'].value
                velocity                    = in_event_group['appV'].value
                reason_n                    = in_event_group['reason_n'].value
                Ntotal_grd                  = in_event_group.attrs['Ntotal_grd']
                Nvalid_grd                  = in_event_group.attrs['Nvalid_grd']
                # save data
                event_group                 = per_group.create_group(name=evid)
                event_group.attrs.create(name = 'Ntotal_grd', data=Ntotal_grd)
                event_group.attrs.create(name = 'Nvalid_grd', data=Nvalid_grd)
                azdset                      = event_group.create_dataset(name='az', data=az)
                appVdset                    = event_group.create_dataset(name='appV', data=velocity)
                reason_ndset                = event_group.create_dataset(name='reason_n', data=reason_n)
        # check periods    
        new_pers            = pers.copy()
        try:
            per_xcorr       = self.attrs['per_xcorr']
        except:
            per_xcorr       = np.array([])
        for iper in range(pers.size):
            per             = pers[iper]
            if per < per_quake.min():
                continue
            if per > per_quake.max():
                break
            if (not per in per_quake) and (not per in per_xcorr):
                raise KeyError('Inconsistent period arrays!')
        self.attrs.create(name = 'per_quake', data=per_quake, dtype='f')
        self.attrs.create(name = 'per_quake_min', data=per_quake[0], dtype='f')
        self.attrs.create(name = 'per_quake_max', data=per_quake[-1], dtype='f')
        # append periods
        for iper in range(per_quake.size):
            per             = per_quake[iper]
            if per > new_pers[-1]:
                new_pers    = np.append(new_pers, per)
        self.attrs.create(name = 'period_array', data=new_pers, dtype='f')
        return
    
    def hybrid_eikonal_stack_old(self, Tmin=30., Tmax=60., minazi=-180, maxazi=180, N_bin=20, threshmeasure=80, anisotropic=False, \
                spacing_ani=0.6, use_numba=True, coverage=0.1):
        """
        Hybridly stack gradient results to perform Eikonal Tomography
        =================================================================================================================
        ::: input parameters :::
        Tmin/Tmax       - minimum/maximum period for merging xcorr and earthquake eikonal results
        minazi/maxazi   - min/max azimuth for anisotropic parameters determination
        N_bin           - number of bins for anisotropic parameters determination
        anisotropic     - perform anisotropic parameters determination or not
        use_numba       - use numba for large array manipulation or not, faster and much less memory requirement
        -----------------------------------------------------------------------------------------------------------------
        version history:
            Oct 17th, 2018  - first version
        =================================================================================================================
        """
        # read attribute information
        pers            = self.attrs['period_array']
        minlon          = self.attrs['minlon']
        maxlon          = self.attrs['maxlon']
        minlat          = self.attrs['minlat']
        maxlat          = self.attrs['maxlat']
        dlon            = self.attrs['dlon']
        dlat            = self.attrs['dlat']
        Nlon            = int(self.attrs['Nlon'])
        Nlat            = int(self.attrs['Nlat'])
        nlat_grad       = self.attrs['nlat_grad']
        nlon_grad       = self.attrs['nlon_grad']
        nlat_lplc       = self.attrs['nlat_lplc']
        nlon_lplc       = self.attrs['nlon_lplc']
        group_xcorr     = self['xcorr_run']
        group_quake     = self['quake_run']
        group_out       = self.create_group( name = 'Eikonal_stack_0')
        # attributes for output group
        group_out.attrs.create(name = 'anisotropic', data = anisotropic)
        group_out.attrs.create(name = 'N_bin', data = N_bin)
        group_out.attrs.create(name = 'minazi', data = minazi)
        group_out.attrs.create(name = 'maxazi', data = maxazi)
        for per in pers:
            stack_xcorr         = True
            stack_quake         = True
            if per < Tmin:
                stack_quake     = False
            if per > Tmax:
                stack_xcorr     = False
            try:
                per_group_xcorr = group_xcorr['%g_sec'%( per )]
                Nevent_xcorr    = len(per_group_xcorr.keys())
            except KeyError:
                stack_xcorr     = False
            try:
                per_group_quake = group_quake['%g_sec'%( per )]
                Nevent_quake    = len(per_group_quake.keys())
            except KeyError:
                stack_quake     = False
            if (not stack_xcorr) and (not stack_quake):
                print '=== Skip stacking eikonal results for: '+str(per)+' sec'
            print '=== Stacking eikonal results for: '+str(per)+' sec'
            # initialize data arrays
            Nevent              = 0
            ev_str              = ''
            if stack_xcorr:
                Nevent          += Nevent_xcorr
                ev_str          += ' Number of noise events = '+str(Nevent_xcorr)
            if stack_quake:
                Nevent          += Nevent_quake
                ev_str          += ' Number of quake events = '+str(Nevent_quake)
            ev_str              += ' Number of total events = '+str(Nevent)
            print ev_str
            Nmeasure            = np.zeros((Nlat-2*nlat_grad, Nlon-2*nlon_grad), dtype=np.int32)
            weightALL           = np.zeros((Nevent, Nlat-2*nlat_grad, Nlon-2*nlon_grad))
            slownessALL         = np.zeros((Nevent, Nlat-2*nlat_grad, Nlon-2*nlon_grad))
            aziALL              = np.zeros((Nevent, Nlat-2*nlat_grad, Nlon-2*nlon_grad), dtype='float32')
            reason_nALL         = np.zeros((Nevent, Nlat-2*nlat_grad, Nlon-2*nlon_grad))
            validALL            = np.zeros((Nevent, Nlat-2*nlat_grad, Nlon-2*nlon_grad), dtype='float32')
            #-----------------------------------------------------
            # Loop over events to get eikonal maps for each event
            #-----------------------------------------------------
            print '--- Reading data'
            for iev in range(Nevent):
                # get data
                if stack_xcorr and (not stack_quake):
                    evid                    = per_group_xcorr.keys()[iev]
                    event_group             = per_group_xcorr[evid]
                    az                      = event_group['az'].value
                    velocity                = event_group['appV'].value
                    reason_n                = event_group['reason_n'].value
                elif (not stack_xcorr) and stack_quake:
                    evid                    = per_group_quake.keys()[iev]
                    event_group             = per_group_quake[evid]
                    az                      = event_group['az'].value
                    velocity                = event_group['appV'].value
                    reason_n                = event_group['reason_n'].value
                else:
                    if iev < Nevent_xcorr:
                        evid                = per_group_xcorr.keys()[iev]
                        event_group         = per_group_xcorr[evid]
                        az                  = event_group['az'].value
                        velocity            = event_group['appV'].value
                        reason_n            = event_group['reason_n'].value
                    else:
                        evid                = per_group_quake.keys()[iev - Nevent_xcorr]
                        event_group         = per_group_quake[evid]
                        az                  = event_group['az'].value
                        velocity            = event_group['appV'].value
                        reason_n            = event_group['reason_n'].value
                oneArr                      = np.ones((Nlat-2*nlat_grad, Nlon-2*nlon_grad), dtype=np.int32)
                oneArr[reason_n!=0]         = 0
                slowness                    = np.zeros((Nlat-2*nlat_grad, Nlon-2*nlon_grad), dtype=np.float32)
                slowness[velocity!=0]       = 1./velocity[velocity!=0]                
                slownessALL[iev, :, :]      = slowness
                reason_nALL[iev, :, :]      = reason_n
                aziALL[iev, :, :]           = az
                Nmeasure                    += oneArr
                # quality control of coverage
                try:
                    Ntotal_grd              = event_group.attrs['Ntotal_grd']
                    Nvalid_grd              = event_group.attrs['Nvalid_grd']
                    if float(Nvalid_grd)/float(Ntotal_grd)< coverage:
                        reason_nALL[iev, :, :]  = np.ones((Nlat-2*nlat_grad, Nlon-2*nlon_grad))
                except:
                    pass
            print '--- Stacking data'
            if Nmeasure.max()< threshmeasure:
                print ('No enough measurements for: '+str(per)+' sec')
                continue
            # discard grid points where number of raw measurements is low, added Sep 26th, 2018
            index_discard                   = Nmeasure < 50
            reason_nALL[:, index_discard]   = 10
            #-----------------------------------------------
            # Get weight for each grid point per event
            #-----------------------------------------------
            if use_numba:
                validALL[reason_nALL==0]    = 1
                weightALL                   = _get_azi_weight(aziALL, validALL)
                weightALL[reason_nALL!=0]   = 0
                weightALL[weightALL!=0]     = 1./weightALL[weightALL!=0]
                weightsum                   = np.sum(weightALL, axis=0)
            else:
                azi_event1                  = np.broadcast_to(aziALL, (Nevent, Nevent, Nlat-2*nlat_grad, Nlon-2*nlon_grad))
                azi_event2                  = np.swapaxes(azi_event1, 0, 1)
                validALL[reason_nALL==0]    = 1
                validALL4                   = np.broadcast_to(validALL, (Nevent, Nevent, Nlat-2*nlat_grad, Nlon-2*nlon_grad))
                # use numexpr for very large array manipulations
                del_aziALL                  = numexpr.evaluate('abs(azi_event1-azi_event2)')
                index_azi                   = numexpr.evaluate('(1*(del_aziALL<20)+1*(del_aziALL>340))*validALL4')
                weightALL                   = numexpr.evaluate('sum(index_azi, 0)')
                weightALL[reason_nALL!=0]   = 0
                weightALL[weightALL!=0]     = 1./weightALL[weightALL!=0]
                weightsum                   = np.sum(weightALL, axis=0)
            #-----------------------------------------------
            # reduce large weight to some value.
            #-----------------------------------------------
            avgArr                          = np.zeros((Nlat-2*nlat_grad, Nlon-2*nlon_grad))
            avgArr[Nmeasure!=0]             = weightsum[Nmeasure!=0]/Nmeasure[Nmeasure!=0]
            # bug fixed, 02/07/2018
            signALL                         = weightALL.copy()
            signALL[signALL!=0]             = 1.
            stdArr                          = np.sum( signALL*(weightALL-avgArr)**2, axis=0)
            stdArr[Nmeasure!=0]             = stdArr[Nmeasure!=0]/Nmeasure[Nmeasure!=0]
            stdArr                          = np.sqrt(stdArr)
            threshhold                      = np.broadcast_to(avgArr+3.*stdArr, weightALL.shape)
            weightALL[weightALL>threshhold] = threshhold[weightALL>threshhold] # threshhold truncated weightALL
            # recompute weight arrays after large weight value reduction
            weightsum                       = np.sum(weightALL, axis=0)
            weightsumALL                    = np.broadcast_to(weightsum, weightALL.shape)
            # weight over all events, note that before this, weightALL is weight over events in azimuth bin
            weightALL[weightsumALL!=0]      = weightALL[weightsumALL!=0]/weightsumALL[weightsumALL!=0] 
            ###
            weightALL[weightALL==1.]        = 0. # data will be discarded if no other data within 20 degree
            #-----------------------------------------------
            # Compute mean/std of slowness
            #-----------------------------------------------
            slownessALL2                    = slownessALL*weightALL
            slowness_sum                    = np.sum(slownessALL2, axis=0)
            slowness_sumALL                 = np.broadcast_to(slowness_sum, weightALL.shape)
            # weighted standard deviation
            # formula: https://www.itl.nist.gov/div898/software/dataplot/refman2/ch2/weightsd.pdf
            signALL                         = weightALL.copy()
            signALL[signALL!=0]             = 1.
            MArr                            = np.sum(signALL, axis=0)
            temp                            = weightALL*(slownessALL-slowness_sumALL)**2
            temp                            = np.sum(temp, axis=0)
            slowness_std                    = np.zeros(temp.shape)
            tind                            = (weightsum!=0)*(MArr!=1)*(MArr!=0)
            slowness_std[tind]              = np.sqrt(temp[tind]/ ( weightsum[tind]*(MArr[tind]-1)/MArr[tind] ) )
            slowness_stdALL                 = np.broadcast_to(slowness_std, weightALL.shape)
            #-----------------------------------------------
            # discard outliers of slowness
            #-----------------------------------------------
            weightALLQC                     = weightALL.copy()
            index_outlier                   = (np.abs(slownessALL-slowness_sumALL))>2.*slowness_stdALL
            index_outlier                   += reason_nALL != 0
            weightALLQC[index_outlier]      = 0
            weightsumQC                     = np.sum(weightALLQC, axis=0)
            NmALL                           = np.sign(weightALLQC)
            NmeasureQC                      = np.sum(NmALL, axis=0)
            weightsumQCALL                  = np.broadcast_to(weightsumQC, weightALL.shape)
            weightALLQC[weightsumQCALL!=0]  = weightALLQC[weightsumQCALL!=0]/weightsumQCALL[weightsumQCALL!=0]
            temp                            = weightALLQC*slownessALL
            slowness_sumQC                  = np.sum(temp, axis=0)
            # new
            signALLQC                       = weightALLQC.copy()
            signALLQC[signALLQC!=0]         = 1.
            MArrQC                          = np.sum(signALLQC, axis=0)
            temp                            = weightALLQC*(slownessALL-slowness_sumQC)**2
            temp                            = np.sum(temp, axis=0)
            slowness_stdQC                  = np.zeros(temp.shape)
            tind                            = (weightsumQC!=0)*(MArrQC!=1)
            slowness_stdQC[tind]            = np.sqrt(temp[tind]/ ( weightsumQC[tind]*(MArrQC[tind]-1)/MArrQC[tind] ))
            #---------------------------------------------------------------
            # mask, velocity, and sem arrays of shape Nlat, Nlon
            #---------------------------------------------------------------
            mask                            = np.ones((Nlat, Nlon), dtype=np.bool)
            tempmask                        = (weightsumQC == 0)
            mask[nlat_grad:-nlat_grad, nlon_grad:-nlon_grad] \
                                            = tempmask
            vel_iso                         = np.zeros((Nlat, Nlon), dtype=np.float32)
            tempvel                         = slowness_sumQC.copy()
            tempvel[tempvel!=0]             = 1./ tempvel[tempvel!=0]
            vel_iso[nlat_grad:-nlat_grad, nlon_grad:-nlon_grad]\
                                            = tempvel
            #----------------------------------------------------------------------------------------
            # standard error of the mean, updated on 09/20/2018
            # formula: https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Statistical_properties
            #----------------------------------------------------------------------------------------
            slownessALL_temp                = slownessALL.copy()
            slownessALL_temp[slownessALL_temp==0.]\
                                            = 0.3
            if np.any(weightALLQC[slownessALL==0.]> 0.):
                raise ValueError('Check weight array!')
            temp                            = (weightALLQC*(1./slownessALL_temp-tempvel))**2
            temp                            = np.sum(temp, axis=0)
            tempsem                         = np.zeros(temp.shape)
            tind                            = (weightsumQC!=0)*(MArrQC!=1)
            tempsem[tind]                   = np.sqrt( temp[tind] * ( MArrQC[tind]/(weightsumQC[tind])**2/(MArrQC[tind]-1) ) ) 
            vel_sem                         = np.zeros((Nlat, Nlon), dtype=np.float32)
            vel_sem[nlat_grad:-nlat_grad, nlon_grad:-nlon_grad]\
                                            = tempsem
            #---------------------------------------
            # save isotropic velocity to database
            #---------------------------------------
            per_group_out                   = group_out.create_group( name='%g_sec'%( per ) )
            sdset                           = per_group_out.create_dataset(name='slowness', data=slowness_sumQC)
            s_stddset                       = per_group_out.create_dataset(name='slowness_std', data=slowness_stdQC)
            Nmdset                          = per_group_out.create_dataset(name='Nmeasure', data=Nmeasure)
            NmQCdset                        = per_group_out.create_dataset(name='NmeasureQC', data=NmeasureQC)
            maskdset                        = per_group_out.create_dataset(name='mask', data=mask)
            visodset                        = per_group_out.create_dataset(name='vel_iso', data=vel_iso)
            vsemdset                        = per_group_out.create_dataset(name='vel_sem', data=vel_sem)
            #----------------------------------------------------------------------------
            # determine anisotropic parameters, need benchmark and further verification
            #----------------------------------------------------------------------------
            if anisotropic:
                grid_factor                 = int(np.ceil(spacing_ani/dlat))
                gridx                       = grid_factor
                gridy                       = int(grid_factor*np.floor(dlon/dlat))
                Nx_size                     = Nlat-2*nlat_grad
                Ny_size                     = Nlon-2*nlon_grad
                NmeasureAni                 = np.zeros((Nlat-2*nlat_grad, Nlon-2*nlon_grad))
                total_near_neighbor         = Nmeasure[0:-2*gridx, 0:-2*gridy] + Nmeasure[0:-2*gridx, gridy:-gridy] + \
                                    Nmeasure[0:-2*gridx, 2*gridy:Ny_size] + Nmeasure[gridx:-gridx, 0:-2*gridy] +\
                                    Nmeasure[gridx:-gridx, gridy:-gridy] + Nmeasure[gridx:-gridx, 2*gridy:Ny_size] +\
                                    Nmeasure[2*gridx:Nx_size, 0:-2*gridy] + Nmeasure[2*gridx:Nx_size, gridy:-gridy] +\
                                    Nmeasure[2*gridx:Nx_size, 2*gridy:Ny_size]
                NmeasureAni[gridx:-gridx, gridy:-gridy]     \
                                            = total_near_neighbor # for quality control
                # initialization of anisotropic parameters
                d_bin                       = (maxazi-minazi)/N_bin
                print 'anisotropic grid factor = '+ str(gridx)+'/'+str(gridy)
                # number of measurements in each bin
                histArr                     = np.zeros((N_bin, Nlat-2*nlat_grad, Nlon-2*nlon_grad))
                histArr_cutted              = histArr[:, gridx:-gridx, gridy:-gridy]
                # slowness in each bin
                slow_sum_ani                = np.zeros((N_bin, Nlat-2*nlat_grad, Nlon-2*nlon_grad))
                slow_sum_ani_cutted         = slow_sum_ani[:, gridx:-gridx, gridy:-gridy]
                # slowness uncertainties for each bin
                slow_un                     = np.zeros((N_bin, Nlat-2*nlat_grad, Nlon-2*nlon_grad))
                slow_un_cutted              = slow_un[:, gridx:-gridx, gridy:-gridy]
                # velocity uncertainties for each bin
                vel_un                      = np.zeros((N_bin, Nlat-2*nlat_grad, Nlon-2*nlon_grad))
                vel_un_cutted               = vel_un[:, gridx:-gridx, gridy:-gridy]
                #
                index_dict                  = { 0: [0, -2*gridx, 0,         -2*gridy], \
                                                1: [0, -2*gridx, gridy,     -gridy],\
                                                2: [0, -2*gridx, 2*gridy,   Ny_size],\
                                                3: [gridx, -gridx, 0,       -2*gridy],\
                                                4: [gridx, -gridx, gridy, -gridy],\
                                                5: [gridx, -gridx, 2*gridy, Ny_size],\
                                                6: [2*gridx, Nx_size, 0,    -2*gridy],\
                                                7: [2*gridx, Nx_size, gridy,-gridy],\
                                                8: [2*gridx, Nx_size, 2*gridy, Ny_size]}
                nmin_bin                    = 2 # change
                #----------------------------------------------------------------------------------
                # Loop over azimuth bins to get slowness, velocity and number of measurements
                #----------------------------------------------------------------------------------
                for ibin in xrange(N_bin):
                    sumNbin                     = (np.zeros((Nlat-2*nlat_grad, Nlon-2*nlon_grad)))[gridx:-gridx, gridy:-gridy]
                    slowbin                     = (np.zeros((Nlat-2*nlat_grad, Nlon-2*nlon_grad)))[gridx:-gridx, gridy:-gridy]
                    slow_un_ibin                = (np.zeros((Nlat-2*nlat_grad, Nlon-2*nlon_grad)))[gridx:-gridx, gridy:-gridy]
                    velbin                      = (np.zeros((Nlat-2*nlat_grad, Nlon-2*nlon_grad)))[gridx:-gridx, gridy:-gridy]
                    vel_un_ibin                 = (np.zeros((Nlat-2*nlat_grad, Nlon-2*nlon_grad)))[gridx:-gridx, gridy:-gridy]
                    for i in range(9):
                        indarr                  = index_dict[i]
                        azi_arr                 = aziALL[:, indarr[0]:indarr[1], indarr[2]:indarr[3]]
                        ibinarr                 = np.floor((azi_arr - minazi)/d_bin)
                        weight_bin              = 1*(ibinarr==ibin)
                        index_outlier_cutted    = index_outlier[:, indarr[0]:indarr[1], indarr[2]:indarr[3]]
                        weight_bin[index_outlier_cutted] \
                                                = 0
                        slowsumQC_cutted        = slowness_sumQC[indarr[0]:indarr[1], indarr[2]:indarr[3]]
                        slownessALL_cutted      = slownessALL[:, indarr[0]:indarr[1], indarr[2]:indarr[3]]
                        # differences in slowness numexpr.evaluate('sum(index_azi, 0)')
                        temp_dslow              = numexpr.evaluate('weight_bin*(slownessALL_cutted-slowsumQC_cutted)')
                        temp_dslow              = numexpr.evaluate('sum(temp_dslow, 0)')
                        # velocities
                        temp_vel                = slownessALL_cutted.copy()
                        temp_vel[temp_vel!=0]   = 1./temp_vel[temp_vel!=0]
                        temp_vel                = numexpr.evaluate('weight_bin*temp_vel')
                        temp_vel                = numexpr.evaluate('sum(temp_vel, 0)')
                        # number of measurements in this bin
                        N_ibin                  = numexpr.evaluate('sum(weight_bin, 0)')
                        # quality control
                        ind_valid               = N_ibin >= nmin_bin
                        sumNbin[ind_valid]      += N_ibin[ind_valid]
                        slowbin[ind_valid]      += temp_dslow[ind_valid]
                        velbin[ind_valid]       += temp_vel[ind_valid]
                    vel_mean                    = velbin.copy()
                    vel_mean[sumNbin!=0]        = velbin[sumNbin!=0]/sumNbin[sumNbin!=0]
                    dslow_mean                  = slowbin.copy()
                    dslow_mean[sumNbin!=0]      = dslow_mean[sumNbin!=0]/sumNbin[sumNbin!=0]
                    # compute uncertainties
                    for i in range(9):
                        indarr                  = index_dict[i]
                        azi_arr                 = aziALL[:, indarr[0]:indarr[1], indarr[2]:indarr[3]]
                        ibinarr                 = np.floor((azi_arr-minazi)/d_bin)
                        weight_bin              = 1*(ibinarr==ibin)
                        index_outlier_cutted    = index_outlier[:, indarr[0]:indarr[1], indarr[2]:indarr[3]]
                        weight_bin[index_outlier_cutted] \
                                                = 0
                        slowsumQC_cutted        = slowness_sumQC[indarr[0]:indarr[1], indarr[2]:indarr[3]]
                        slownessALL_cutted      = slownessALL[:, indarr[0]:indarr[1], indarr[2]:indarr[3]]
                        temp_vel                = slownessALL_cutted.copy()
                        temp_vel[temp_vel!=0]   = 1./temp_vel[temp_vel!=0]
                        vel_un_ibin             = vel_un_ibin + numexpr.evaluate('sum( (weight_bin*(temp_vel-vel_mean))**2, 0)')
                        slow_un_ibin            = slow_un_ibin + numexpr.evaluate('sum( (weight_bin*(slownessALL_cutted-slowsumQC_cutted \
                                                                - dslow_mean))**2, 0)')
                    #------------------------------------
                    vel_un_ibin[sumNbin!=0]     = np.sqrt(vel_un_ibin[sumNbin!=0]/(sumNbin[sumNbin!=0]-1)/sumNbin[sumNbin!=0])
                    vel_un_cutted[ibin, :, :]   = vel_un_ibin
                    slow_un_ibin[sumNbin!=0]    = np.sqrt(slow_un_ibin[sumNbin!=0]/(sumNbin[sumNbin!=0]-1)/sumNbin[sumNbin!=0])
                    slow_un_cutted[ibin, :, :]  = slow_un_ibin
                    histArr_cutted[ibin, :, :]  = sumNbin
                    slow_sum_ani_cutted[ibin, :, :]  \
                                                = dslow_mean
                #-------------------------------------------
                N_thresh                                = 10 # change
                slow_sum_ani_cutted[histArr_cutted<N_thresh] \
                                                        = 0
                slow_sum_ani[:, gridx:-gridx, gridy:-gridy]\
                                                        = slow_sum_ani_cutted
                # uncertainties
                slow_un_cutted[histArr_cutted<N_thresh] = 0
                slow_un[:, gridx:-gridx, gridy:-gridy]  = slow_un_cutted
                # convert sem of slowness to sem of velocity
                vel_un_cutted[histArr_cutted<N_thresh]  = 0
                vel_un[:, gridx:-gridx, gridy:-gridy]   = vel_un_cutted
                # # # return vel_un
                # near neighbor quality control
                Ntotal_thresh                           = 45 # change
                slow_sum_ani[:, NmeasureAni<Ntotal_thresh]    \
                                                        = 0 
                slow_un[:, NmeasureAni<Ntotal_thresh]   = 0
                vel_un[:, NmeasureAni<Ntotal_thresh]    = 0
                histArr[:, gridx:-gridx, gridy:-gridy]  = histArr_cutted
                # save data to database
                s_anidset       = per_group_out.create_dataset(name='slownessAni', data=slow_sum_ani)
                s_anisemdset    = per_group_out.create_dataset(name='slownessAni_sem', data=slow_un)
                v_anisemdset    = per_group_out.create_dataset(name='velAni_sem', data=vel_un)
                histdset        = per_group_out.create_dataset(name='histArr', data=histArr)
                NmAnidset       = per_group_out.create_dataset(name='NmeasureAni', data=NmeasureAni)
        return
    
    def hybrid_eikonal_stack(self, Tmin=30., Tmax=60., minazi=-180, maxazi=180, N_bin=20, threshmeasure=80, anisotropic=False, \
                spacing_ani=0.6, use_numba=True, coverage=0.1, azi_amp_tresh=0.05):
        """
        Hybridly stack gradient results to perform Eikonal Tomography
        =================================================================================================================
        ::: input parameters :::
        Tmin/Tmax       - minimum/maximum period for merging xcorr and earthquake eikonal results
        minazi/maxazi   - min/max azimuth for anisotropic parameters determination
        N_bin           - number of bins for anisotropic parameters determination
        anisotropic     - perform anisotropic parameters determination or not
        use_numba       - use numba for large array manipulation or not, faster and much less memory requirement
        -----------------------------------------------------------------------------------------------------------------
        version history:
            Oct 17th, 2018  - first version
        =================================================================================================================
        """
        # read attribute information
        pers            = self.attrs['period_array']
        minlon          = self.attrs['minlon']
        maxlon          = self.attrs['maxlon']
        minlat          = self.attrs['minlat']
        maxlat          = self.attrs['maxlat']
        dlon            = self.attrs['dlon']
        dlat            = self.attrs['dlat']
        Nlon            = int(self.attrs['Nlon'])
        Nlat            = int(self.attrs['Nlat'])
        nlat_grad       = self.attrs['nlat_grad']
        nlon_grad       = self.attrs['nlon_grad']
        nlat_lplc       = self.attrs['nlat_lplc']
        nlon_lplc       = self.attrs['nlon_lplc']
        group_xcorr     = self['xcorr_run']
        group_quake     = self['quake_run']
        try:
            group_out   = self.create_group( name = 'Eikonal_stack_0' )
        except ValueError:
            warnings.warn('Eikonal_stack_0 exists! Will be recomputed!', UserWarning, stacklevel=1)
            del self['Eikonal_stack_0']
            group_out   = self.create_group( name = 'Eikonal_stack_0')
        #
        if anisotropic:
            grid_factor                 = int(np.ceil(spacing_ani/dlat))
            gridx                       = grid_factor
            gridy                       = grid_factor
            if gridx % 2 == 0:
                gridx                   += 1
            if gridy % 2 == 0:
                gridy                   += 1
            print '--- anisotropic grid factor = '+ str(gridx)+'/'+str(gridy)
            group_out.attrs.create(name = 'gridx', data = gridx)
            group_out.attrs.create(name = 'gridy', data = gridy)
        # attributes for output group
        group_out.attrs.create(name = 'anisotropic', data = anisotropic)
        group_out.attrs.create(name = 'N_bin', data = N_bin)
        group_out.attrs.create(name = 'minazi', data = minazi)
        group_out.attrs.create(name = 'maxazi', data = maxazi)
        for per in pers:
            start               = time.time()
            stack_xcorr         = True
            stack_quake         = True
            if per < Tmin:
                stack_quake     = False
            if per > Tmax:
                stack_xcorr     = False
            try:
                per_group_xcorr = group_xcorr['%g_sec'%( per )]
                Nevent_xcorr    = len(per_group_xcorr.keys())
            except KeyError:
                stack_xcorr     = False
            try:
                per_group_quake = group_quake['%g_sec'%( per )]
                Nevent_quake    = len(per_group_quake.keys())
            except KeyError:
                stack_quake     = False
            if (not stack_xcorr) and (not stack_quake):
                print '=== Skip stacking eikonal results for: '+str(per)+' sec'
            print '=== Stacking eikonal results for: '+str(per)+' sec'
            # initialize data arrays
            Nevent              = 0
            ev_str              = ''
            if stack_xcorr:
                Nevent          += Nevent_xcorr
                ev_str          += ' Number of noise events = '+str(Nevent_xcorr)
            if stack_quake:
                Nevent          += Nevent_quake
                ev_str          += ' Number of quake events = '+str(Nevent_quake)
            ev_str              += ' Number of total events = '+str(Nevent)
            print ev_str
            Nmeasure            = np.zeros((Nlat-2*nlat_grad, Nlon-2*nlon_grad), dtype=np.int32)
            weightALL           = np.zeros((Nevent, Nlat-2*nlat_grad, Nlon-2*nlon_grad))
            slownessALL         = np.zeros((Nevent, Nlat-2*nlat_grad, Nlon-2*nlon_grad))
            aziALL              = np.zeros((Nevent, Nlat-2*nlat_grad, Nlon-2*nlon_grad), dtype='float32')
            reason_nALL         = np.zeros((Nevent, Nlat-2*nlat_grad, Nlon-2*nlon_grad))
            validALL            = np.zeros((Nevent, Nlat-2*nlat_grad, Nlon-2*nlon_grad), dtype='float32')
            #-----------------------------------------------------
            # Loop over events to get eikonal maps for each event
            #-----------------------------------------------------
            print '--- Reading data'
            for iev in range(Nevent):
                # get data
                if stack_xcorr and (not stack_quake):
                    evid                    = per_group_xcorr.keys()[iev]
                    event_group             = per_group_xcorr[evid]
                    az                      = event_group['az'].value
                    velocity                = event_group['appV'].value
                    reason_n                = event_group['reason_n'].value
                elif (not stack_xcorr) and stack_quake:
                    evid                    = per_group_quake.keys()[iev]
                    event_group             = per_group_quake[evid]
                    az                      = event_group['az'].value
                    velocity                = event_group['appV'].value
                    reason_n                = event_group['reason_n'].value
                else:
                    if iev < Nevent_xcorr:
                        evid                = per_group_xcorr.keys()[iev]
                        event_group         = per_group_xcorr[evid]
                        az                  = event_group['az'].value
                        velocity            = event_group['appV'].value
                        reason_n            = event_group['reason_n'].value
                    else:
                        evid                = per_group_quake.keys()[iev - Nevent_xcorr]
                        event_group         = per_group_quake[evid]
                        az                  = event_group['az'].value
                        velocity            = event_group['appV'].value
                        reason_n            = event_group['reason_n'].value
                oneArr                      = np.ones((Nlat-2*nlat_grad, Nlon-2*nlon_grad), dtype=np.int32)
                oneArr[reason_n!=0]         = 0
                slowness                    = np.zeros((Nlat-2*nlat_grad, Nlon-2*nlon_grad), dtype=np.float32)
                slowness[velocity!=0]       = 1./velocity[velocity!=0]                
                slownessALL[iev, :, :]      = slowness
                reason_nALL[iev, :, :]      = reason_n
                aziALL[iev, :, :]           = az
                Nmeasure                    += oneArr
                # quality control of coverage
                try:
                    Ntotal_grd              = event_group.attrs['Ntotal_grd']
                    Nvalid_grd              = event_group.attrs['Nvalid_grd']
                    if float(Nvalid_grd)/float(Ntotal_grd)< coverage:
                        reason_nALL[iev, :, :]  = np.ones((Nlat-2*nlat_grad, Nlon-2*nlon_grad))
                except:
                    pass
            print '--- Stacking data'
            if Nmeasure.max()< threshmeasure:
                print ('No enough measurements for: '+str(per)+' sec')
                continue
            # discard grid points where number of raw measurements is low, added Sep 26th, 2018
            index_discard                   = Nmeasure < 50
            reason_nALL[:, index_discard]   = 10
            #-----------------------------------------------
            # Get weight for each grid point per event
            #-----------------------------------------------
            if use_numba:
                validALL[reason_nALL==0]    = 1
                weightALL                   = _get_azi_weight(aziALL, validALL)
                weightALL[reason_nALL!=0]   = 0
                weightALL[weightALL!=0]     = 1./weightALL[weightALL!=0]
                weightsum                   = np.sum(weightALL, axis=0)
            else:
                azi_event1                  = np.broadcast_to(aziALL, (Nevent, Nevent, Nlat-2*nlat_grad, Nlon-2*nlon_grad))
                azi_event2                  = np.swapaxes(azi_event1, 0, 1)
                validALL[reason_nALL==0]    = 1
                validALL4                   = np.broadcast_to(validALL, (Nevent, Nevent, Nlat-2*nlat_grad, Nlon-2*nlon_grad))
                # use numexpr for very large array manipulations
                del_aziALL                  = numexpr.evaluate('abs(azi_event1-azi_event2)')
                index_azi                   = numexpr.evaluate('(1*(del_aziALL<20)+1*(del_aziALL>340))*validALL4')
                weightALL                   = numexpr.evaluate('sum(index_azi, 0)')
                weightALL[reason_nALL!=0]   = 0
                weightALL[weightALL!=0]     = 1./weightALL[weightALL!=0]
                weightsum                   = np.sum(weightALL, axis=0)
            #-----------------------------------------------
            # reduce large weight to some value.
            #-----------------------------------------------
            avgArr                          = np.zeros((Nlat-2*nlat_grad, Nlon-2*nlon_grad))
            avgArr[Nmeasure!=0]             = weightsum[Nmeasure!=0]/Nmeasure[Nmeasure!=0]
            # bug fixed, 02/07/2018
            signALL                         = weightALL.copy()
            signALL[signALL!=0]             = 1.
            stdArr                          = np.sum( signALL*(weightALL-avgArr)**2, axis=0)
            stdArr[Nmeasure!=0]             = stdArr[Nmeasure!=0]/Nmeasure[Nmeasure!=0]
            stdArr                          = np.sqrt(stdArr)
            threshhold                      = np.broadcast_to(avgArr+3.*stdArr, weightALL.shape)
            weightALL[weightALL>threshhold] = threshhold[weightALL>threshhold] # threshhold truncated weightALL
            # recompute weight arrays after large weight value reduction
            weightsum                       = np.sum(weightALL, axis=0)
            weightsumALL                    = np.broadcast_to(weightsum, weightALL.shape)
            # weight over all events, note that before this, weightALL is weight over events in azimuth bin
            weightALL[weightsumALL!=0]      = weightALL[weightsumALL!=0]/weightsumALL[weightsumALL!=0] 
            ###
            weightALL[weightALL==1.]        = 0. # data will be discarded if no other data within 20 degree
            #-----------------------------------------------
            # Compute mean/std of slowness
            #-----------------------------------------------
            slownessALL2                    = slownessALL*weightALL
            slowness_sum                    = np.sum(slownessALL2, axis=0)
            slowness_sumALL                 = np.broadcast_to(slowness_sum, weightALL.shape)
            # weighted standard deviation
            # formula: https://www.itl.nist.gov/div898/software/dataplot/refman2/ch2/weightsd.pdf
            signALL                         = weightALL.copy()
            signALL[signALL!=0]             = 1.
            MArr                            = np.sum(signALL, axis=0)
            temp                            = weightALL*(slownessALL-slowness_sumALL)**2
            temp                            = np.sum(temp, axis=0)
            slowness_std                    = np.zeros(temp.shape)
            tind                            = (weightsum!=0)*(MArr!=1)*(MArr!=0)
            slowness_std[tind]              = np.sqrt(temp[tind]/ ( weightsum[tind]*(MArr[tind]-1)/MArr[tind] ) )
            slowness_stdALL                 = np.broadcast_to(slowness_std, weightALL.shape)
            #-----------------------------------------------
            # discard outliers of slowness
            #-----------------------------------------------
            weightALLQC                     = weightALL.copy()
            index_outlier                   = (np.abs(slownessALL-slowness_sumALL))>2.*slowness_stdALL
            index_outlier                   += reason_nALL != 0
            weightALLQC[index_outlier]      = 0
            weightsumQC                     = np.sum(weightALLQC, axis=0)
            NmALL                           = np.sign(weightALLQC)
            NmeasureQC                      = np.sum(NmALL, axis=0)
            weightsumQCALL                  = np.broadcast_to(weightsumQC, weightALL.shape)
            weightALLQC[weightsumQCALL!=0]  = weightALLQC[weightsumQCALL!=0]/weightsumQCALL[weightsumQCALL!=0]
            temp                            = weightALLQC*slownessALL
            slowness_sumQC                  = np.sum(temp, axis=0)
            # new
            signALLQC                       = weightALLQC.copy()
            signALLQC[signALLQC!=0]         = 1.
            MArrQC                          = np.sum(signALLQC, axis=0)
            temp                            = weightALLQC*(slownessALL-slowness_sumQC)**2
            temp                            = np.sum(temp, axis=0)
            slowness_stdQC                  = np.zeros(temp.shape)
            tind                            = (weightsumQC!=0)*(MArrQC!=1)
            slowness_stdQC[tind]            = np.sqrt(temp[tind]/ ( weightsumQC[tind]*(MArrQC[tind]-1)/MArrQC[tind] ))
            #---------------------------------------------------------------
            # mask, velocity, and sem arrays of shape Nlat, Nlon
            #---------------------------------------------------------------
            mask                            = np.ones((Nlat, Nlon), dtype=np.bool)
            tempmask                        = (weightsumQC == 0)
            mask[nlat_grad:-nlat_grad, nlon_grad:-nlon_grad] \
                                            = tempmask
            vel_iso                         = np.zeros((Nlat, Nlon), dtype=np.float32)
            tempvel                         = slowness_sumQC.copy()
            tempvel[tempvel!=0]             = 1./ tempvel[tempvel!=0]
            vel_iso[nlat_grad:-nlat_grad, nlon_grad:-nlon_grad]\
                                            = tempvel
            #----------------------------------------------------------------------------------------
            # standard error of the mean, updated on 09/20/2018
            # formula: https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Statistical_properties
            #----------------------------------------------------------------------------------------
            slownessALL_temp                = slownessALL.copy()
            slownessALL_temp[slownessALL_temp==0.]\
                                            = 0.3
            if np.any(weightALLQC[slownessALL==0.]> 0.):
                raise ValueError('Check weight array!')
            temp                            = (weightALLQC*(1./slownessALL_temp-tempvel))**2
            temp                            = np.sum(temp, axis=0)
            tempsem                         = np.zeros(temp.shape)
            tind                            = (weightsumQC!=0)*(MArrQC!=1)
            tempsem[tind]                   = np.sqrt( temp[tind] * ( MArrQC[tind]/(weightsumQC[tind])**2/(MArrQC[tind]-1) ) ) 
            vel_sem                         = np.zeros((Nlat, Nlon), dtype=np.float32)
            vel_sem[nlat_grad:-nlat_grad, nlon_grad:-nlon_grad]\
                                            = tempsem
            #---------------------------------------
            # save isotropic velocity to database
            #---------------------------------------
            per_group_out                   = group_out.create_group( name='%g_sec'%( per ) )
            sdset                           = per_group_out.create_dataset(name='slowness', data=slowness_sumQC)
            s_stddset                       = per_group_out.create_dataset(name='slowness_std', data=slowness_stdQC)
            Nmdset                          = per_group_out.create_dataset(name='Nmeasure', data=Nmeasure)
            NmQCdset                        = per_group_out.create_dataset(name='NmeasureQC', data=NmeasureQC)
            maskdset                        = per_group_out.create_dataset(name='mask', data=mask)
            visodset                        = per_group_out.create_dataset(name='vel_iso', data=vel_iso)
            vsemdset                        = per_group_out.create_dataset(name='vel_sem', data=vel_sem)
            #----------------------------------------------------------------------------
            # determine anisotropic parameters, need benchmark and further verification
            #----------------------------------------------------------------------------
            if anisotropic:
                print '*** Anisotropic stacking data: '+str(per)
                # quality control
                slowness_sumQC_ALL          = np.broadcast_to(slowness_sumQC, slownessALL.shape)
                diff_slowness               = np.abs(slownessALL-slowness_sumQC_ALL)
                ind_nonzero                 = slowness_sumQC_ALL!= 0.
                diff_slowness[ind_nonzero]  = diff_slowness[ind_nonzero]/slowness_sumQC_ALL[ind_nonzero]
                index_outlier               += diff_slowness > azi_amp_tresh
                # stacking to get anisotropic parameters
                dslow_sum_ani, dslow_un, vel_un, histArr, NmeasureAni    \
                                            = eikonaltomo._anisotropic_stacking_parallel(np.int64(gridx), np.int64(gridy), np.float32(maxazi), np.float32(minazi),\
                                                np.int64(N_bin), np.float64(Nmeasure), np.float64(aziALL),\
                                                np.float64(slowness_sumQC), np.float64(slownessALL), index_outlier.astype(bool))
                #----------------------------
                # save data to database
                #----------------------------
                out_arr         = np.zeros((N_bin, Nlat-2*nlat_grad, Nlon-2*nlon_grad))
                out_arr[:, (gridx - 1)/2:-(gridx - 1)/2, (gridy - 1)/2:-(gridy - 1)/2]\
                                = dslow_sum_ani
                s_anidset       = per_group_out.create_dataset(name='slownessAni', data=out_arr)
                
                out_arr         = np.zeros((N_bin, Nlat-2*nlat_grad, Nlon-2*nlon_grad))
                out_arr[:, (gridx - 1)/2:-(gridx - 1)/2, (gridy - 1)/2:-(gridy - 1)/2]\
                                = dslow_un
                s_anisemdset    = per_group_out.create_dataset(name='slownessAni_sem', data=out_arr)
                
                out_arr         = np.zeros((N_bin, Nlat-2*nlat_grad, Nlon-2*nlon_grad))
                out_arr[:, (gridx - 1)/2:-(gridx - 1)/2, (gridy - 1)/2:-(gridy - 1)/2]\
                                = vel_un
                v_anisemdset    = per_group_out.create_dataset(name='velAni_sem', data=out_arr)
                
                out_arr         = np.zeros((N_bin, Nlat-2*nlat_grad, Nlon-2*nlon_grad))
                out_arr[:, (gridx - 1)/2:-(gridx - 1)/2, (gridy - 1)/2:-(gridy - 1)/2]\
                                = histArr
                histdset        = per_group_out.create_dataset(name='histArr', data=out_arr)
                
                out_arr         = np.zeros((Nlat-2*nlat_grad, Nlon-2*nlon_grad))
                out_arr[(gridx - 1)/2:-(gridx - 1)/2, (gridy - 1)/2:-(gridy - 1)/2]\
                                = NmeasureAni
                NmAnidset       = per_group_out.create_dataset(name='NmeasureAni', data=out_arr)
            print '=== elasped time = '+str(time.time() - start)+' sec'
        return
    
    def hybrid_eikonal_stack_mp(self, workingdir='./eik_stack_dir', Tmin=20., Tmax=60., minazi=-180, maxazi=180, N_bin=20, threshmeasure=80, anisotropic=False, \
                spacing_ani=0.3, coverage=0.1, use_numba=True, azi_amp_tresh=0.05, nprocess=None, enhanced=False, run=True):
        # read attribute information
        pers            = self.attrs['period_array']
        minlon          = self.attrs['minlon']
        maxlon          = self.attrs['maxlon']
        minlat          = self.attrs['minlat']
        maxlat          = self.attrs['maxlat']
        dlon            = self.attrs['dlon']
        dlat            = self.attrs['dlat']
        Nlon            = int(self.attrs['Nlon'])
        Nlat            = int(self.attrs['Nlat'])
        nlat_grad       = self.attrs['nlat_grad']
        nlon_grad       = self.attrs['nlon_grad']
        nlat_lplc       = self.attrs['nlat_lplc']
        nlon_lplc       = self.attrs['nlon_lplc']
        group_xcorr     = self['xcorr_run']
        group_quake     = self['quake_run']
        try:
            group_out   = self.create_group( name = 'Eikonal_stack_0' )
        except ValueError:
            warnings.warn('Eikonal_stack_0 exists! Will be recomputed!', UserWarning, stacklevel=1)
            del self['Eikonal_stack_0']
            group_out   = self.create_group( name = 'Eikonal_stack_0')
        #
        if anisotropic:
            grid_factor                 = int(np.ceil(spacing_ani/dlat))
            gridx                       = grid_factor
            gridy                       = grid_factor
            if gridx % 2 == 0:
                gridx                   += 1
            if gridy % 2 == 0:
                gridy                   += 1
            print '--- anisotropic grid factor = '+ str(gridx)+'/'+str(gridy)
            group_out.attrs.create(name = 'gridx', data = gridx)
            group_out.attrs.create(name = 'gridy', data = gridy)
        # attributes for output group
        group_out.attrs.create(name = 'anisotropic', data = anisotropic)
        group_out.attrs.create(name = 'N_bin', data = N_bin)
        group_out.attrs.create(name = 'minazi', data = minazi)
        group_out.attrs.create(name = 'maxazi', data = maxazi)
        #-----------------------
        # prepare data
        #-----------------------
        stack_lst       = []
        for per in pers:
            stack_xcorr         = True
            stack_quake         = True
            if per < Tmin:
                stack_quake     = False
            if per > Tmax:
                stack_xcorr     = False
            try:
                per_group_xcorr = group_xcorr['%g_sec'%( per )]
                Nevent_xcorr    = len(per_group_xcorr.keys())
            except KeyError:
                stack_xcorr     = False
            try:
                per_group_quake = group_quake['%g_sec'%( per )]
                Nevent_quake    = len(per_group_quake.keys())
            except KeyError:
                stack_quake     = False
            if (not stack_xcorr) and (not stack_quake):
                print '=== Skip stacking eikonal results for: '+str(per)+' sec'
            print '=== Stacking eikonal results for: '+str(per)+' sec'
            # initialize data arrays
            Nevent              = 0
            ev_str              = ''
            if stack_xcorr:
                Nevent          += Nevent_xcorr
                ev_str          += ' Number of noise events = '+str(Nevent_xcorr)
            if stack_quake:
                Nevent          += Nevent_quake
                ev_str          += ' Number of quake events = '+str(Nevent_quake)
            ev_str              += ' Number of total events = '+str(Nevent)
            print ev_str
            Nmeasure            = np.zeros((Nlat-2*nlat_grad, Nlon-2*nlon_grad), dtype=np.int32)
            weightALL           = np.zeros((Nevent, Nlat-2*nlat_grad, Nlon-2*nlon_grad))
            slownessALL         = np.zeros((Nevent, Nlat-2*nlat_grad, Nlon-2*nlon_grad))
            aziALL              = np.zeros((Nevent, Nlat-2*nlat_grad, Nlon-2*nlon_grad), dtype='float32')
            reason_nALL         = np.zeros((Nevent, Nlat-2*nlat_grad, Nlon-2*nlon_grad))
            validALL            = np.zeros((Nevent, Nlat-2*nlat_grad, Nlon-2*nlon_grad), dtype='float32')
            #-----------------------------------------------------
            # Loop over events to get eikonal maps for each event
            #-----------------------------------------------------
            print '--- Reading data'
            for iev in range(Nevent):
                # get data
                if stack_xcorr and (not stack_quake):
                    evid                    = per_group_xcorr.keys()[iev]
                    event_group             = per_group_xcorr[evid]
                    az                      = event_group['az'].value
                    velocity                = event_group['appV'].value
                    reason_n                = event_group['reason_n'].value
                elif (not stack_xcorr) and stack_quake:
                    evid                    = per_group_quake.keys()[iev]
                    event_group             = per_group_quake[evid]
                    az                      = event_group['az'].value
                    velocity                = event_group['appV'].value
                    reason_n                = event_group['reason_n'].value
                else:
                    if iev < Nevent_xcorr:
                        evid                = per_group_xcorr.keys()[iev]
                        event_group         = per_group_xcorr[evid]
                        az                  = event_group['az'].value
                        velocity            = event_group['appV'].value
                        reason_n            = event_group['reason_n'].value
                    else:
                        evid                = per_group_quake.keys()[iev - Nevent_xcorr]
                        event_group         = per_group_quake[evid]
                        az                  = event_group['az'].value
                        velocity            = event_group['appV'].value
                        reason_n            = event_group['reason_n'].value
                oneArr                      = np.ones((Nlat-2*nlat_grad, Nlon-2*nlon_grad), dtype=np.int32)
                oneArr[reason_n!=0]         = 0
                slowness                    = np.zeros((Nlat-2*nlat_grad, Nlon-2*nlon_grad), dtype=np.float32)
                slowness[velocity!=0]       = 1./velocity[velocity!=0]                
                slownessALL[iev, :, :]      = slowness
                reason_nALL[iev, :, :]      = reason_n
                aziALL[iev, :, :]           = az
                Nmeasure                    += oneArr
                # quality control of coverage
                try:
                    Ntotal_grd              = event_group.attrs['Ntotal_grd']
                    Nvalid_grd              = event_group.attrs['Nvalid_grd']
                    if float(Nvalid_grd)/float(Ntotal_grd)< coverage:
                        reason_nALL[iev, :, :]  = np.ones((Nlat-2*nlat_grad, Nlon-2*nlon_grad))
                except:
                    pass
        
            stack_lst.append(eikonaltomo.data4stack(slownessALL=slownessALL, reason_nALL=reason_nALL, aziALL=aziALL, Nmeasure=Nmeasure, \
                                        Nevent=Nevent, period=per, Nlon=Nlon, Nlat=Nlat, nlon_grad=nlon_grad, nlat_grad=nlat_grad))
            # eikonaltomo.stack4mp(stack_lst[0], workingdir=workingdir, minazi=minazi, maxazi=maxazi, N_bin=N_bin, threshmeasure=threshmeasure,\
            #         anisotropic=anisotropic, spacing_ani=spacing_ani, coverage=coverage, use_numba=use_numba,\
            #             azi_amp_tresh=azi_amp_tresh, gridx=gridx, gridy=gridy, Nlat=Nlat, Nlon=Nlon, nlat_grad=nlat_grad,\
            #                 nlon_grad=nlon_grad, enhanced=enhanced)
            # return
        ###
        # mp stacking
        ###
        if run:
            if not os.path.isdir(workingdir):
                os.makedirs(workingdir)
            print '=== eikonal stacking'
            STACK       = partial(eikonaltomo.stack4mp, workingdir=workingdir, minazi=minazi, maxazi=maxazi, N_bin=N_bin, threshmeasure=threshmeasure,\
                        anisotropic=anisotropic, spacing_ani=spacing_ani, coverage=coverage, use_numba=use_numba,\
                            azi_amp_tresh=azi_amp_tresh, gridx=gridx, gridy=gridy, Nlat=Nlat, Nlon=Nlon, nlat_grad=nlat_grad, nlon_grad=nlon_grad,\
                            enhanced=enhanced)
            pool        = multiprocessing.Pool(processes=nprocess)
            pool.map(STACK, stack_lst) #make our results with a map call
            pool.close() #we are not adding any more processes
            pool.join() #tell it to wait until all threads are done before going on
        ###
        # read stacked data
        ###
        for per in pers:
            print '--- reading eikonal stacked data : '+str(per)+' sec'
            pfx             = '%g_sec'%( per )
            infname         = workingdir+'/iso_'+pfx+'.npz'
            inarr           = np.load(infname)
            slowness_sumQC  = inarr['arr_0']
            slowness_stdQC  = inarr['arr_1']
            Nmeasure        = inarr['arr_2']
            NmeasureQC      = inarr['arr_3']
            mask            = inarr['arr_4']
            vel_iso         = inarr['arr_5']
            vel_sem         = inarr['arr_6']
            # save to database
            per_group_out   = group_out.create_group( name='%g_sec'%( per ) )
            sdset           = per_group_out.create_dataset(name='slowness', data=slowness_sumQC)
            s_stddset       = per_group_out.create_dataset(name='slowness_std', data=slowness_stdQC)
            Nmdset          = per_group_out.create_dataset(name='Nmeasure', data=Nmeasure)
            NmQCdset        = per_group_out.create_dataset(name='NmeasureQC', data=NmeasureQC)
            maskdset        = per_group_out.create_dataset(name='mask', data=mask)
            visodset        = per_group_out.create_dataset(name='vel_iso', data=vel_iso)
            vsemdset        = per_group_out.create_dataset(name='vel_sem', data=vel_sem)
            if anisotropic:
                infname         = workingdir+'/azi_'+pfx+'.npz'
                inarr           = np.load(infname)
                dslow_sum_ani   = inarr['arr_0']
                dslow_un        = inarr['arr_1']
                vel_un          = inarr['arr_2']
                histArr         = inarr['arr_3']
                NmeasureAni     = inarr['arr_4']
                # save database
                s_anidset       = per_group_out.create_dataset(name='slownessAni', data=dslow_sum_ani)
                s_anisemdset    = per_group_out.create_dataset(name='slownessAni_sem', data=dslow_un)
                v_anisemdset    = per_group_out.create_dataset(name='velAni_sem', data=vel_un)
                histdset        = per_group_out.create_dataset(name='histArr', data=histArr)
                NmAnidset       = per_group_out.create_dataset(name='NmeasureAni', data=NmeasureAni)
    
    def merge_raytomo(self, inrayfname, runid=0, Nmeasure_thresh=50, percentage=None, num_thresh=None,\
                    inrunid=0, gausspercent=1., gstd_thresh=100., Traymin=8., Traymax=50.):
        """
        Merge eikonal tomography results with ray tomography results
        Uncertainties will be extrapolated based on the resolution values yieled by the ray tomography method
        """
        # ray tomography group
        indset      = h5py.File(inrayfname)
        raydataid   = 'reshaped_qc_run_'+str(inrunid)
        raypers     = indset.attrs['period_array']
        raypers     = raypers[(raypers<=Traymax)*(raypers>=Traymin)]
        print 'RayTomo Tmin/Tmax = '+str(raypers[0])+'/'+str(raypers[-1])
        raygrp      = indset[raydataid]
        isotropic   = raygrp.attrs['isotropic']
        org_raygrp  = indset['qc_run_'+str(inrunid)]
        if isotropic:
            print 'isotropic inversion results do not output gaussian std!'
            return
        # eikonal tomography group
        pers        = self.attrs['period_array']
        dataid      = 'Eikonal_stack_'+str(runid)
        grp         = self[dataid]
        for per in raypers:
            if not per in pers:
                raise KeyError('Period array of hybrid database should contain raytomo period array!')
        self.attrs.create(name = 'period_array_ray', data=raypers)
        # check attributes
        if self.attrs['minlon'] != indset.attrs['minlon'] or \
            self.attrs['maxlon'] != indset.attrs['maxlon'] or \
                self.attrs['minlat'] != indset.attrs['minlat'] or \
                    self.attrs['maxlat'] != indset.attrs['maxlat'] or \
                        self.attrs['dlon'] != org_raygrp.attrs['dlon'] or \
                            self.attrs['dlat'] != org_raygrp.attrs['dlat']:
            raise ValueError('Incompatible input ray tomo datasets!')
        outgrp      = self.create_group(name='merged_tomo_'+str(runid))
        #------------------------------------------------------
        # determine mask for period in ray tomography database
        #------------------------------------------------------
        mask_ray        = raygrp['mask2']
        if gstd_thresh is not None:
            for per in raypers:
                pergrp  = raygrp['%g_sec'%( per )]
                mgauss  = pergrp['gauss_std'].value
                mask_ray+= mgauss > gstd_thresh
        outgrp.create_dataset(name='mask_ray', data=mask_ray)
        #------------------------------------------------------
        # determine mask for period in eikonal database
        #------------------------------------------------------
        # mask_eik        = grp['%g_sec'%( pers[0] )]['mask'].value
        # for per in pers:
        #     pergrp          = grp['%g_sec'%( per )]
        #     mask_temp       = pergrp['mask'].value
        #     Nmeasure        = np.zeros(mask_eik.shape)
        #     Nmeasure[1:-1, 1:-1]\
        #                     = pergrp['NmeasureQC'].value
        #     mask_temp[Nmeasure<Nmeasure_thresh]\
        #                     = True
        #     mask_eik        += mask_temp
        # outgrp.create_dataset(name='mask_eik', data=mask_eik)
        for per in pers:
            pergrp          = grp['%g_sec'%( per )]
            velocity        = pergrp['vel_iso'].value
            uncertainty     = pergrp['vel_sem'].value
            mask_eik        = pergrp['mask'].value
            Nmeasure        = np.zeros(mask_eik.shape)
            Nmeasure[1:-1, 1:-1]\
                            = pergrp['NmeasureQC'].value
            mask_eik[Nmeasure<Nmeasure_thresh]\
                            = True
            #-------------------------------
            # get data
            #-------------------------------
            if per in raypers:
                per_raygrp  = raygrp['%g_sec'%( per )]
                # replace velocity value outside eikonal region
                vel_ray     = per_raygrp['vel_iso'].value
                velocity[mask_eik]\
                            = vel_ray[mask_eik]
                #--------------------------------------------------
                # replace uncertainty value outside eikonal region
                #--------------------------------------------------
                # Gaussian std from ray tomo data
                mgauss      = per_raygrp['gauss_std'].value
                index_ray   = np.logical_not(mask_ray)
                mgauss2     = mgauss[index_ray]
                gstdmin     = mgauss2.min()
                ind_gstdmin = (mgauss==gstdmin*gausspercent)*index_ray
                # eikonal 
                index       = np.logical_not(mask_eik)
                Nmeasure2   = Nmeasure[index]
                if Nmeasure2.size == 0:
                    print '--- T = '+str(per)+' sec ---'
                    print 'No uncertainty, step 1'
                    print '----------------------------'
                    continue
                NMmax       = Nmeasure2.max()
                if percentage is not None and num_thresh is None:
                    NMthresh    = NMmax*percentage
                elif percentage is None and num_thresh is not None:
                    NMthresh    = num_thresh
                elif percentage is not None and num_thresh is not None:
                    NMthresh    = min(NMmax*percentage, num_thresh)
                else:
                    raise ValueError('at least one of percentage/num_thresh should be specified')
                indstd      = (Nmeasure>=NMthresh)*index
                #------------------------
                # extrapolate uncertainties
                #------------------------
                # locate the grid points where Gaussian std is small enough and Nmeasure is large enough
                index_all   = ind_gstdmin*indstd
                temp_sem    = uncertainty[index_all]
                if temp_sem.size == 0:
                    print '--- T = '+str(per)+' sec ---'
                    print 'No uncertainty, step 2'
                    print '----------------------------'
                    continue
                sem_min     = temp_sem.mean()
                print '--- T = '+str(per)+' sec ---'
                print 'min uncertainty: '+str(sem_min*1000.)+' m/s, number of grids: '+str(temp_sem.size)
                print '----------------------------'
                est_sem     = (mgauss/gstdmin)*sem_min
                # replace uncertainties
                uncertainty[mask_eik]\
                            = est_sem[mask_eik]
            # save data to database
            out_pergrp      = outgrp.create_group(name='%g_sec'%( per ))
            vdset           = out_pergrp.create_dataset(name='vel_iso', data=velocity)
            undset          = out_pergrp.create_dataset(name='vel_sem', data=uncertainty)
            maskeikdset     = out_pergrp.create_dataset(name='mask_eik', data=mask_eik)
            if per in raypers:
                maskdset    = out_pergrp.create_dataset(name='mask', data=mask_ray)
            else:
                maskdset    = out_pergrp.create_dataset(name='mask', data=mask_eik)
            Nmdset          = out_pergrp.create_dataset(name='Nmeasure', data=Nmeasure)
        return

    def interp_surface(self, Traymax=50., workingdir='./hybridtomo_interp_surface', dlon=None, dlat=None, runid=0, deletetxt=True):
        """interpolate inverted velocity maps and uncertainties to a grid for inversion of Vs
        =================================================================================================================
        ::: input parameters :::
        workingdir  - working directory
        dlon/dlat   - grid interval for interpolation
        runid       - id of run
        =================================================================================================================
        """
        self._get_lon_lat_arr()
        dataid          = 'merged_tomo_'+str(runid)
        pers            = self.attrs['period_array']
        grp             = self[dataid]
        minlon          = self.attrs['minlon']
        maxlon          = self.attrs['maxlon']
        minlat          = self.attrs['minlat']
        maxlat          = self.attrs['maxlat']
        dlon_org        = self.attrs['dlon']
        dlat_org        = self.attrs['dlat']
        if dlon is None and dlat is None:
            print 'At least one of dlon/dlat needs to be specified!'
            return
        if dlon == dlon_org and dlat == dlat_org:
            print 'No need to perform interpolation!'
            return
        self.attrs.create(name = 'dlon_interp', data=dlon)
        self.attrs.create(name = 'dlat_interp', data=dlat)
        #--------------------------------------------------
        # get the mask array for the interpolated data
        #---------------------------------------------------
        mask_ray        = grp['mask_ray']
        index_ray       = np.logical_not(mask_ray)
        lons            = np.arange(int((maxlon-minlon)/dlon)+1)*dlon+minlon
        lats            = np.arange(int((maxlat-minlat)/dlat)+1)*dlat+minlat
        Nlon            = lons.size
        Nlat            = lats.size
        lonArr, latArr  = np.meshgrid(lons, lats)
        mask_ray_interp = _get_mask_interp(mask_ray, self.lons, self.lats, lons, lats)
        grp.create_dataset(name = 'mask_ray_interp', data=mask_ray_interp)
        grp.attrs.create(name = 'T_ray_max', data=Traymax)
        for per in pers:
            working_per = workingdir+'/'+str(per)+'sec'
            if not os.path.isdir(working_per):
                os.makedirs(working_per)
            #-------------------------------
            # get data
            #-------------------------------
            try:
                pergrp      = grp['%g_sec'%( per )]
                vel_iso     = pergrp['vel_iso'].value
                vel_sem     = pergrp['vel_sem'].value
                mask_eik    = pergrp['mask_eik'].value
            except KeyError:
                print 'No data for T = '+str(per)+' sec'
                continue
            if per <= Traymax:
                index           = index_ray.copy()
            else:
                index           = np.logical_not(mask_eik)
                mask_eik_out    = _get_mask_interp(mask_eik, self.lons, self.lats, lons, lats)
                maskinterp_dset = pergrp.create_dataset(name='mask_interp', data=mask_eik_out)
            #-------------------------------
            # interpolation for velocity
            #-------------------------------
            field2d_v       = field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=dlon,
                            minlat=minlat, maxlat=maxlat, dlat=dlat, period=per, evlo=(minlon+maxlon)/2., evla=(minlat+maxlat)/2.)
            field2d_v.read_array(lonArr = self.lonArr[index], latArr = self.latArr[index], ZarrIn = vel_iso[index])
            outfname        = 'interp_vel.lst'
            field2d_v.interp_surface(workingdir=working_per, outfname=outfname)
            vinterp_dset    = pergrp.create_dataset(name='vel_iso_interp', data=field2d_v.Zarr)
            #---------------------------------
            # interpolation for uncertainties
            #---------------------------------
            field2d_un      = field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=dlon,
                            minlat=minlat, maxlat=maxlat, dlat=dlat, period=per, evlo=(minlon+maxlon)/2., evla=(minlat+maxlat)/2.)
            field2d_un.read_array(lonArr = self.lonArr[index], latArr = self.latArr[index], ZarrIn = vel_sem[index])
            outfname        = 'interp_un.lst'
            field2d_un.interp_surface(workingdir=working_per, outfname=outfname)
            uninterp_dset   = pergrp.create_dataset(name='vel_sem_interp', data=field2d_un.Zarr)
        if deletetxt:
            shutil.rmtree(workingdir)
        return
    
    
    # # # def plot_interp(self, period, datatype, runid=0, shpfx=None, clabel='', cmap='cv', projection='lambert', hillshade=False,\
    # # #          geopolygons=None, vmin=None, vmax=None, showfig=True):
    # # #     """plot HD maps from the tomographic inversion
    # # #     =================================================================================================================
    # # #     ::: input parameters :::
    # # #     period          - period of data
    # # #     runid           - id of run
    # # #     clabel          - label of colorbar
    # # #     cmap            - colormap
    # # #     projection      - projection type
    # # #     geopolygons     - geological polygons for plotting
    # # #     vmin, vmax      - min/max value of plotting
    # # #     showfig         - show figure or not
    # # #     =================================================================================================================
    # # #     """
    # # #     dataid          = 'merged_tomo_'+str(runid)
    # # #     self._get_lon_lat_arr_interp()
    # # #     pers            = self.attrs['period_array']
    # # #     grp             = self[dataid]
    # # #     Traymax         = grp.attrs['T_ray_max']
    # # #     if not period in pers:
    # # #         raise KeyError('period = '+str(period)+' not included in the database')
    # # #     pergrp          = grp['%g_sec'%( period )]
    # # #     if datatype == 'vel' or datatype=='velocity' or datatype == 'v':
    # # #         datatype    = 'vel_iso_interp'
    # # #     if datatype == 'un' or datatype=='sem' or datatype == 'vel_sem':
    # # #         datatype    = 'vel_sem_interp'
    # # #     try:
    # # #         data    = pergrp[datatype].value
    # # #     except:
    # # #         outstr      = ''
    # # #         for key in pergrp.keys():
    # # #             outstr  +=key
    # # #             outstr  +=', '
    # # #         outstr      = outstr[:-1]
    # # #         raise KeyError('Unexpected datatype: '+datatype+\
    # # #                        ', available datatypes are: '+outstr)
    # # #     if period <= Traymax:   
    # # #         mask    = grp['mask_ray_interp']
    # # #     else:
    # # #         mask    = pergrp['mask_interp']
    # # #     if datatype == 'vel_sem_interp':
    # # #         data    = data*2000.
    # # #     mdata       = ma.masked_array(data, mask=mask )
    # # #     #-----------
    # # #     # plot data
    # # #     #-----------
    # # #     m           = self._get_basemap(projection=projection, geopolygons=geopolygons)
    # # #     x, y        = m(self.lonArr, self.latArr)
    # # #     # shapefname  = '/projects/life9360/geological_maps/qfaults'
    # # #     # m.readshapefile(shapefname, 'faultline', linewidth=2, color='grey')
    # # #     # shapefname  = '/projects/life9360/AKgeol_web_shp/AKStategeolarc_generalized_WGS84'
    # # #     # m.readshapefile(shapefname, 'faultline', linewidth=1, color='grey')
    # # #     
    # # #     if cmap == 'ses3d':
    # # #         cmap        = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
    # # #                         0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
    # # #     elif cmap == 'cv':
    # # #         import pycpt
    # # #         cmap    = pycpt.load.gmtColormap('./cpt_files/cv.cpt')
    # # #     else:
    # # #         try:
    # # #             if os.path.isfile(cmap):
    # # #                 import pycpt
    # # #                 cmap    = pycpt.load.gmtColormap(cmap)
    # # #         except:
    # # #             pass
    # # #     ################################
    # # #     if hillshade:
    # # #         from netCDF4 import Dataset
    # # #         from matplotlib.colors import LightSource
    # # #     
    # # #         etopodata   = Dataset('/projects/life9360/station_map/grd_dir/ETOPO2v2g_f4.nc')
    # # #         etopo       = etopodata.variables['z'][:]
    # # #         lons        = etopodata.variables['x'][:]
    # # #         lats        = etopodata.variables['y'][:]
    # # #         ls          = LightSource(azdeg=315, altdeg=45)
    # # #         # nx          = int((m.xmax-m.xmin)/40000.)+1; ny = int((m.ymax-m.ymin)/40000.)+1
    # # #         etopo,lons  = shiftgrid(180.,etopo,lons,start=False)
    # # #         # topodat,x,y = m.transform_scalar(etopo,lons,lats,nx,ny,returnxy=True)
    # # #         ny, nx      = etopo.shape
    # # #         topodat,xtopo,ytopo = m.transform_scalar(etopo,lons,lats,nx, ny, returnxy=True)
    # # #         m.imshow(ls.hillshade(topodat, vert_exag=1., dx=1., dy=1.), cmap='gray')
    # # #         mycm1=pycpt.load.gmtColormap('/projects/life9360/station_map/etopo1.cpt')
    # # #         mycm2=pycpt.load.gmtColormap('/projects/life9360/station_map/bathy1.cpt')
    # # #         mycm2.set_over('w',0)
    # # #         # m.imshow(ls.shade(topodat, cmap=mycm1, vert_exag=1., dx=1., dy=1., vmin=0, vmax=8000))
    # # #         # m.imshow(ls.shade(topodat, cmap=mycm2, vert_exag=1., dx=1., dy=1., vmin=-11000, vmax=-0.5))
    # # #     ###################################################################
    # # # 
    # # #     if hillshade:
    # # #         im          = m.pcolormesh(x, y, mdata, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax, alpha=.5)
    # # #     else:
    # # #         im          = m.pcolormesh(x, y, mdata, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
    # # #     cb          = m.colorbar(im, "bottom", size="3%", pad='2%')
    # # #     cb.set_label(clabel, fontsize=12, rotation=0)
    # # #     plt.suptitle(str(period)+' sec', fontsize=20)
    # # #     cb.ax.tick_params(labelsize=15)
    # # #     cb.set_alpha(1)
    # # #     cb.draw_all()
    # # #     print 'plotting data from '+dataid
    # # #     # # cb.solids.set_rasterized(True)
    # # #     cb.solids.set_edgecolor("face")
    # # #     # lons            = np.array([-160., -160., -150., -140., -130.,\
    # # #     #                             -160., -150., -140., -130.,\
    # # #     #                             -160., -150., -140., -130.])
    # # #     # lats            = np.array([55., 60., 60., 60., 60.,\
    # # #     #                             65., 65., 65., 55.,\
    # # #     #                             70., 70., 70., 70.])
    # # #     # xc, yc          = m(lons, lats)
    # # #     # m.plot(xc, yc,'ko', ms=15)
    # # #     # m.shadedrelief(scale=1., origin='lower')
    # # #     if showfig:
    # # #         plt.show()
    # # #     return
    
    def write_un(self, outfname, runid=0):
        pers        = self.attrs['period_array']
        unarr       = np.zeros(pers.size)
        i           = 0
        dataid      = 'merged_tomo_'+str(runid)
        grp         = self[dataid]
        for per in pers:
            pergrp          = grp['%g_sec'%( per )]
            velocity        = pergrp['vel_iso'].value
            uncertainty     = pergrp['vel_sem'].value
            mask_eik        = pergrp['mask_eik'].value
            unarr[i]        = uncertainty[np.logical_not(mask_eik)].mean() * 2000.
            i               += 1
        outArr  = np.append(pers, unarr)
        outArr  = outArr.reshape((2, pers.size))
        outArr  = outArr.T
        np.savetxt(outfname, outArr, fmt='%g')
        
