import ASDFDBase
import numpy as np

dset=ASDFDBase.quakeASDF('/work3/leon/china_data/EA_quake.h5')

# dset.read_stationtxt('/work3/leon/china_data/CHINA.ARRAY.station.lst')
# dset.read_stationtxt('/work3/leon/china_data/sta_CEA', chans=['LHZ', 'LHE', 'LHN'])
# dset.read_sac(datadir='/work3/leon/china_data')