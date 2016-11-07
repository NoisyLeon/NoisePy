import ASDFDBase
import numpy as np

dset=ASDFDBase.quakeASDF('/home/lili/code/china_data/EA_quake.h5')

# dset.read_stationtxt('./CHINA.ARRAY.station.lst')
# dset.read_stationtxt('./sta_CEA', chans=['LHZ', 'LHE', 'LHN'])
# dset.read_sac(datadir='/work3/leon/china_data')
# dset.read_sac('/home/lili/code/china_data/Sta_dat')
# dset.read_sac('/home/lili/code/china_data')
# dset.quake_prephp(outdir='/home/lili/code/china_data/PRE_PHP')
# dset.quake_aftan(prephdir='/home/lili/code/china_data/PRE_PHP_R')
# try:
#     del dset.auxiliary_data.DISPpmf2interp
# except:
#     pass
# pers=np.arange(6, 72, 2)
# dset.interp_disp(pers=pers)
# 
# try:
#     del dset.auxiliary_data.FieldDISPpmf2interp
# except:
#     pass
# dset.quake_get_field(outdir='/home/lili/code/china_data/china_field', pers=pers)