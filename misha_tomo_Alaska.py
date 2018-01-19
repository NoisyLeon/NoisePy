import noisedbase
import numpy as np
import timeit
import GeoPolygon
import raytomo

# dbase   = noisedbase.noiseASDF('inv_Alaska.h5')
# geopoly = GeoPolygon.GeoPolygonLst()
# geopoly.read_tomoctr('contour.ctr')
# dbase.plot_stations(geopolygons=geopoly)

dset=raytomo.RayTomoDataSet('../ray_tomo_Alaska.h5')
# dset.set_input_parameters(minlon=188, maxlon=238, minlat=52, maxlat=72, data_pfx='raytomo_in_', smoothpfx='N_INIT_', qcpfx='QC_')
# # dset.run_smooth(datadir='./raytomo_input', outdir='../ray_tomo_working_dir')
# dset.run_qc(outdir='../ray_tomo_working_dir', isotropic=False, anipara=1, alphaAni4=1000)
# dset.run_qc(outdir='../ray_tomo_working_dir', isotropic=True, anipara=1, alphaAni4=1000)
# 
dset.get_data4plot(dataid='qc_run_1', period=12.)
# dset.plot_vel_iso(vmin=2.9, vmax=3.5, fastaxis=False, projection='global')
dset.plot_vel_iso(vmin=3.5, vmax=4.0)
# dset.plot_fast_axis()
# dset.generate_corrected_map(dataid='qc_run_0', glbdir='./MAPS', outdir='./REG_MAPS')
# dset.plot_global_map(period=50., inglbpfx='./MAPS/smpkolya_phv_R')


