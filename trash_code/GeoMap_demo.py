import GeoPoint as GP
import matplotlib.pyplot as plt

datadir='/projects/life9360/NewResults'
dirPFX='1_'
mapfile='/projects/life9360/NewResults/station.lst_test'
GeoLst=GP.GeoMap()
GeoLst.ReadGeoMapLst(mapfile)
GeoLst.SetAllfname()
GeoLst.LoadVProfile(datadir=datadir, dirPFX=dirPFX)
GeoLst.LoadGrDisp(datadir=datadir, dirPFX=dirPFX)
GeoLst.LoadPhDisp(datadir=datadir, dirPFX=dirPFX)
GeoLst.GetMapLimit()
GeoLst.BrowseFigures(datatype='All',datadir=datadir, dirPFX=dirPFX, depLimit=10, browseflag=False, saveflag=True)