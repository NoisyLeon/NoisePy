import os
import obspy
import glob
datadir     = '/work2/leon/temp_working_2004_2008'
start_date  ='20040101'
end_date    ='20080530'
stime       = obspy.UTCDateTime(start_date)
etime       = obspy.UTCDateTime(end_date)
monthdict   = {1: 'JAN', 2: 'FEB', 3: 'MAR', 4: 'APR', 5: 'MAY', 6: 'JUN', 7: 'JUL', 8: 'AUG', 9: 'SEP', 10: 'OCT', 11: 'NOV', 12: 'DEC'}
month       = stime.month - 1
while(stime < etime):
    # print stime
    if month < stime.month:
        month   += 1
        print 'Deleting data: '+str(stime.year)+'.'+monthdict[stime.month]
    dailydir    = datadir+'/'+str(stime.year)+'.'+monthdict[stime.month]+'/'+str(stime.year)+'.'+monthdict[stime.month]+'.'+str(stime.day)
    if os.path.isdir(dailydir):
        resplst = glob.glob(dailydir+'/RESP*')
        ftsaclst= glob.glob(dailydir+'/ft*SAC')
        for respname in resplst:
            os.remove(respname)
        for ftsacname in ftsaclst:
            os.remove(ftsacname)
    else:
        print dailydir+' NOT exists!'
    if stime.month == 12 and stime.day == 31:
        stime   = obspy.UTCDateTime(str(stime.year + 1)+'0101')
        month   = 0
    else:
        stime.julday += 1
    

        
    