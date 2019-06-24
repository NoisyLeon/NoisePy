import numpy as np
from scipy import optimize
import random
# 
# def f( theta, A0, A1, A2):
#     return A0 + A1*np.sin(2*theta/180.*np.pi) + A2*np.cos(2*theta/180.*np.pi)
# 
# def ff(x, p):
#     return f(x, *p)
# 
# # These are the true parameters
# A0 = 4.0
# A1 = 0.02
# A2 = 0.01
# 
# # These are initial guesses for fits:
# pstart = [
#     p0 + random.random(),
#     p1 + 5.*random.random(), 
#     p2 + random.random()
# ]
# 
# # %matplotlib inline
# import matplotlib.pyplot as plt
# xvals = np.linspace(0, 170, 18)
# yvals = f(xvals, p0, p1, p2)
# 
# # Generate data with a bit of randomness
# # (the noise-less function that underlies the data is shown as a blue line)
# 
# xdata = np.array(xvals)
# np.random.seed(42)
# err_stdev = 0.2
# yvals_err =  np.random.normal(0., err_stdev, len(xdata))
# ydata = f(xdata, p0, p1, p2) + yvals_err
# 
# plt.plot(xvals, yvals)
# plt.plot(xdata, ydata, 'o', mfc='None')
# 
# def fit_curvefit(p0, datax, datay, function, yerr=err_stdev, **kwargs):
#     """
#     Note: As per the current documentation (Scipy V1.1.0), sigma (yerr) must be:
#         None or M-length sequence or MxM array, optional
#     Therefore, replace:
#         err_stdev = 0.2
#     With:
#         err_stdev = [0.2 for item in xdata]
#     Or similar, to create an M-length sequence for this example.
#     """
#     pfit, pcov = \
#          optimize.curve_fit(f, datax, datay, p0=p0,\
#                             sigma=yerr, epsfcn=0.0001, **kwargs)
#     error = [] 
#     for i in range(len(pfit)):
#         try:
#           error.append(np.absolute(pcov[i][i])**0.5)
#         except:
#           error.append( 0.00 )
#     pfit_curvefit = pfit
#     perr_curvefit = np.array(error)
#     return pfit_curvefit, perr_curvefit 
# 
# pfit, perr = fit_curvefit(pstart, xdata, ydata, ff)
# 
# print("\n# Fit parameters and parameter errors from curve_fit method :")
# print("pfit = ", pfit)
# print("perr = ", perr)


# t = np.arange(100.)
# 
# 
# def model(t, coeffs):
#    return coeffs[0] + coeffs[1] * np.exp( - ((t-coeffs[2])/coeffs[3])**2 )
# 
# waveform_1  = model(t, [3.1, 29.9, 15.3, 1.1])
# 
# def residuals(coeffs, y, t):
#     return y - model(t, coeffs)
# 
# x0 = np.array([3, 30, 15, 1], dtype=float)
# 
# from scipy.optimize import leastsq
# x, flag = leastsq(residuals, x0, args=(waveform_1, t))
# print(x)
# 


# t = np.arange(100.)
# 
# 
# def model(t, coeffs):
#    return coeffs[0] + coeffs[1] * np.sin(2.*t/180.*np.pi) + coeffs[2] * np.cos(2.*t/180.*np.pi)
# 
# waveform_1  = model(t, [4.0, 0.1, 0.2])
# 
# def residuals(coeffs, y, t, sem):
#     return (y - model(t, coeffs))**2/sem**2
# 
# x0 = np.array([3.9, 0.05, 0.21], dtype=float)
# 
# from scipy.optimize import leastsq
# x, flag = leastsq(residuals, x0, args=(waveform_1, t, np.ones(100.)*0.1))
# print(x)


def _pre_azi_aniso(m, theta):
    return m[0] + m[1]*np.sin(2.*theta/180.*np.pi) + m[2]*np.cos(2.*theta/180.*np.pi)

def errfunc(m, azarr, tvel, tsem):
    return (_pre_azi_aniso(m, azarr) - tvel)**2 / tsem**2

azArr = np.arange(100.)

vel = _pre_azi_aniso([4.0, 0.1, 0.2], azArr)
x0 = np.array([3.9, 0.05, 0.21], dtype=float)
x, flag = optimize.leastsq(errfunc, x0=x0, args=(azArr, vel, np.ones(100.)*0.1))