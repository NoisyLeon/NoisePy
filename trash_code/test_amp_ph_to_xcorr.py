import obspy
import noisedbase
import numpy as np
import pyfftw
pfx1    = './2001.JAN.1/ft_2001.JAN.1.CN.INK.LHE.SAC'
pfx2    = './2001.JAN.1/ft_2001.JAN.1.XE.GOO.LHN.SAC'
amp1    = obspy.read(pfx1+'.am')[0].data
amp2    = obspy.read(pfx2+'.am')[0].data
ph1     = obspy.read(pfx1+'.ph')[0].data
ph2     = obspy.read(pfx2+'.ph')[0].data

N           = amp1.size
Ns          = int(2*N - 1)
# cross-spectrum, conj(sac1)*(sac2)
x_sp        = np.zeros(Ns, dtype=complex)
out         = np.zeros(Ns, dtype=complex)
fftw_plan   = pyfftw.FFTW(input_array=x_sp, output_array=out, direction='FFTW_BACKWARD',  flags=('FFTW_EXHAUSTIVE', ))

out1        = noisedbase._amp_ph_to_xcorr(amp1, amp2, ph1, ph2)
out2        = noisedbase._amp_ph_to_xcorr_fast(amp1, amp2, ph1, ph2, fftw_plan)

# frec1   = '/work2/leon/temp_working_2016_2019/2016.MAY/2016.MAY.1/ft_2016.MAY.1.SAMH.LHE.SAC_rec'
# frec2   = '/work2/leon/temp_working_2016_2019/2016.MAY/2016.MAY.1/ft_2016.MAY.1.BERG.LHE.SAC_rec'


# arr1    = np.loadtxt(frec1)
# arr2    = np.loadtxt(frec2)
# Nrec    = 84001
# cor_rec = noisedbase._CalcRecCor((np.array([0, Nrec-1])).reshape(1, 2) , (np.array([0, Nrec-1])).reshape(1, 2) , np.int32(3000))
# cor_rec = noisedbase._CalcRecCor(arr1, arr2, np.int32(3000))
