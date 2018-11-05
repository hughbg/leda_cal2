# This file based on 04_lmfit.py


import ephem
import pylab as plt
import hickle as hkl
import tables as tb
from leda_cal2.utils import poly_fit, timestamp_to_lst, closest
import numpy as np
import glob
import scipy.signal
import random
from scipy.interpolate import interp1d as interp
import hickle

from lmfit import minimize, Parameters, fit_report

plt.rcParams['font.size'] = 12

def calibrate(data, caldata):
    T_H = caldata['T_H']
    T_C = caldata['T_C']
    G_S = caldata['G_S']
    T_NW = caldata['T_NW']
    # S   = caldata['scale']
    # O   = caldata['offset']

    # D = S * ((T_H - T_C) * data + T_C) / G_S + O
    D = ((T_H - T_C) * data + T_C) / G_S
    return D

ovro_location = ('37.2397808', '-118.2816819', 1183.4839)
ovro = ephem.Observer(); (ovro.lat, ovro.lon, ovro.elev) = ovro_location
sun = ephem.Sun()

gal_center = ephem.FixedBody()  
gal_center._ra  =  '17 45 40.04'
gal_center._dec = '-29 00 28.1'
gal_center.name = "Galactic Center"

# Indicate which LSTs are in nighttime by returning their indexes
def get_night_lsts(lsts, usable_indexes):
    horizon_altitide = -5	#	If sun/galaxy above this then it is daytime
    usable = []
    for i in usable_indexes:		# Start with initial usable_indexes and whittle them down
        ovro.date = lsts[i]
        sun.compute(ovro)
        gal_center.compute(ovro)
        if sun.alt < horizon_altitide*np.pi/180 or gal_center.alt < horizon_altitide*np.pi/180:
          usable.append(i)

    return usable		# Indexes of good LSTs


# Fit a power law to the temperature, which is indexed by frequencies. Return the fitted signal.
def spectral_index_fit(freq, temp):
    def S(x, C, si):
        return C*x**si
        
    # Don't want masked values or NaNs or zeros. SHould use compressed()
    nan_temp = np.ma.filled(temp, np.nan)
    f = freq[np.logical_or(np.isnan(nan_temp), (nan_temp!=0))]
    s = temp[np.logical_or(np.isnan(nan_temp), (nan_temp!=0))]

    try:
        popt, pcov = scipy.optimize.curve_fit(S, f, s)
    except:
        popt = ( 0, 0 )
        
    return S(f, popt[0],popt[1])

# Models and fits for LM optimization

def residual(params, x, model, data):
    mm = model(x, params)
    return (data - mm)

def model_sin(x, params):
    PHI = params['PHI'].value
    PHI0 = params['PHI0'].value
    A_c = params['A_c'].value
    DC = params['DC'].value
    mm = A_c * np.sin(PHI * x + PHI0) + DC
    return mm

def model_damped(x, params):
    w_0 = params["w_0"].value
    w_1 = params["w_1"].value
    w_2 = params["w_2"].value
    b = params["b"].value
    c = params["c"].value
    d = params["d"].value
    e = params["e"].value
    f = params["f"].value
    mm = w_0 + w_1*(x-70) + w_2*(x-70)**2+(e**(-x))*b*np.sin(c*x+d+f/x)
    return mm

def fit_model_sin(x, data):
    params = Parameters()
    params.add('PHI', value=0.3398, vary=True)
    params.add('A_c', value=146., vary=True)
    params.add('PHI0', value=-1.44)
    params.add('DC', value=0)
    out = minimize(residual, params, args=(x, model_sin, data))
    outvals = out.params
    for param, val in out.params.items():
        print "%08s: %2.4f" % (param, val)
    return outvals

def fit_model_sin_u(x, data):
    params = Parameters()
    params.add('PHI', value=1, vary=True)
    params.add('A_c', value=np.abs(data[0]), vary=True)
    params.add('PHI0', value=0, vary=True)
    params.add('DC', value=np.mean(data), vary=True)
    out = minimize(residual, params, args=(x, model_sin, data))
    outvals = out.params
    for param, val in out.params.items():
        print "%08s: %2.4f" % (param, val)
    return outvals



def fit_model_sin_off(x, data):
    params = Parameters()
    params.add('PHI', value=0.3398, vary=True)
    params.add('A_c', value=146., vary=True)
    params.add('PHI0', value=-1.44)
    params.add('B', value=226)
    params.add('M', value=0.2)
    out = minimize(residual, params, args=(x, model_sin_off, data))
    outvals = out.params
    for param, val in out.params.items():
        print "%08s: %2.4f" % (param, val)
    return outvals

def fit_model_damped_sin(x, data):
    params = Parameters()
    params.add("w_0", vary=True, value=-1.5)
    params.add("w_1", vary=True, value=0.5)
    params.add("w_2", vary=True, value=0.07)

    params.add("b", value=np.abs(data[0]), vary=True)
    params.add("c", value=1.0, vary=True)
    params.add("d", value=0.0, vary=True)
    params.add("e", value=1.0, vary=True)
    params.add("f", value=0.0, vary=True)
    out = minimize(residual, params, args=(x, model_damped, data))
    outvals = out.params
    for param, val in out.params.items():
        print "%08s: %2.4f" % (param, val)
    return outvals


def model_sin_off(x, params):
    PHI = params['PHI'].value
    PHI0 = params['PHI0'].value
    A_c = params['A_c'].value
    B = params['B'].value
    M = params['M'].value

    mm = A_c * np.sin(PHI * x + PHI0) + B + M * x
    return mm

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description='Plot antenna spectra and residuals')
    p.add_argument('-n',  '--n_poly', help='number of terms in log-poly to fit for residuals. Default 5', type=int,  default=3)
    args = p.parse_args()

    n_poly = args.n_poly

    started = False

    # Set some important parameters in the next few lines
    ant = "254A"
    lst_min = 11		# Hard limits on LSTs, in hours
    lst_max = 12
    nighttime_only = True	# If True will also exclude daytime LSTs

    # Read all the hickle files, which must be listed in the file "file_list.txt".
    # Accumulate usable spectra in a big array.
    for line in open("file_list.txt"):
      f = line[:-1]
      data = hickle.load(f)
 
      for key in sorted(data.keys()): 
        if key == "frequencies": frequencies = data[key]
        if key == "lsts": 	# Check what LSTs to use
          use_lst_indexes = np.arange(data[key].shape[0])[np.logical_and(data[key]>=lst_min, data[key]<=lst_max)]      
    
	  if nighttime_only: use_lst_indexes = get_night_lsts(data[key], use_lst_indexes)
            
        if key == ant:
          ant_data = data[key]

      # Select the spectra that are usable (based on LST)
      ant_data = np.ma.array(ant_data[use_lst_indexes], mask=ant_data.mask[use_lst_indexes])

      # Add them to the big array
      if not started: 
	accumulated_data = ant_data   
        started = True
      else: accumulated_data = np.ma.append(accumulated_data, ant_data, axis=0)

    # Calculate diffs of all the spectra from the mean
    #mean_spectra = np.ma.mean(accumulated_data, axis=0)
    #for i in range(accumulated_data.shape[0]):
    #  print i, np.std(mean_spectra-accumulated_data[i])
    #exit()


print "Num spectra", accumulated_data.shape[0]


# Average all the spectra, so now we have 1 spectra.
aD = np.ma.mean(accumulated_data, axis=0)

# Cut frequencies below 40MHz. To cut below 58MHz use a value of 1167
aD = aD[417:]
f2 = frequencies[417:]



# ------------ Step 1: Fit a polynomial or power law and subtract to get residual rD 

# This line fits the polynomial:
#rD = aD - poly_fit(f2, aD, n_poly)	

# This line fits a power law:
rD = aD-spectral_index_fit(f2, aD)


# Need to get rid of masked values for next step
f2 = np.ma.array(f2, mask=rD.mask).compressed()
rD = rD.compressed()

# ------------ Step 2: Fit a damped sinusoid
# You can fit a sin function by using the fit_model_sin_off and model_sin_pff functions above
# instead of the damped functions.

rD_model_params = fit_model_damped_sin(f2, rD)
rD_sin_model    = model_damped(f2, rD_model_params)

# Plot the residual and damped sinusoid fit

plt.plot(f2, rD, label="Data")
plt.plot(f2, rD_sin_model, label="Fit")
plt.legend()
plt.show()

