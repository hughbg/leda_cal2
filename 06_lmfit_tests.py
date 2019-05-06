# This file based on 04_lmfit.py


import ephem
import pylab as plt
import hickle as hkl
import tables as tb
from leda_cal2.utils import poly_fit, timestamp_to_lst, closest
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import glob
import scipy.signal
import random
from scipy.interpolate import interp1d as interp
import hickle, os
from spectra import Spectra

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



# Fit a power law to the temperature, which is indexed by frequencies. Return the fitted signal.
def spectral_index_fit(freq, temp):
    def S(x, C, si):
        return C*x**si
        
    # Don't want masked values or NaNs or zeros. 
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
    a = params["a"].value
    b = params["b"].value
    c = params["c"].value
    d = params["d"].value
    f = params["f"].value
	# w params from Lincoln
    mm = w_0 + w_1*(x-70) + w_2*(x-70)**2+(a**(-x))*b*np.sin(c*x+d+f/x)   # Get better fit with e^-x where e variable
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

def fit_model_damped_sin(x, data, previous_params):
    params = Parameters()
    params.add("w_0", vary=True, value=previous_params["w_0"])
    params.add("w_1", vary=True, value=previous_params["w_1"])
    params.add("w_2", vary=True, value=previous_params["w_2"])

    params.add("a", value=previous_params["a"], vary=True)
    params.add("b", value=previous_params["b"], vary=True)
    params.add("c", value=previous_params["c"], vary=True)
    params.add("d", value=previous_params["d"], vary=True)
    params.add("f", value=previous_params["f"], vary=True)
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


def plot_spans(acc_data, fl, d):
  import sys

  print "Plotting spans"

  if acc_data.shape[0] != len(d):
    raise("Days not the same as number of LSTs")
  plt.clf()

  plt.figure(figsize=(20,14))

  plot_index = 0
  for i in range(acc_data.shape[0]):
    if i in flags: 
      plot_index += 1
      continue

    if i%100 == 0: 
      print i, "...",
      sys.stdout.flush()
    
    plt.plot([plot_index, plot_index], [np.ma.min(acc_data[i]), np.ma.max(acc_data[i])], "r", linewidth=0.5)
    #plt.plot([plot_index], [np.ma.mean(accumulated_data[i])], marker='.', markersize=0.5, color="black")
    #plt.plot([plot_index], [np.ma.std(accumulated_data[i])], marker='.', markersize=0.5, color="blue")
 
    
    if i < acc_data.shape[0]-2:
      if d[i] != d[i+1]: plot_index += 50
      if d[i][5:7] != d[i+1][5:7]: plot_index += 200
    plot_index += 1


  plt.xlabel("Time Sequence")
  plt.ylabel("Temperature")
  plt.title("Span of temperature values for each spectrum over time, many days in April/May")
  plt.ylim(ymin=0)
  plt.savefig("spans.png")

  print

def detect_rubble(data):
  sigma_factor = 3.8
  bad = []
  for i in range(data.shape[1]):
    std = np.ma.std(data[:, i])
    mean = np.ma.mean(data[:, i])
    for j in range(data.shape[0]):
      if data[j, i] < mean-float(sigma_factor)*std or mean+float(sigma_factor)*std < data[j, i]:
        if j not in bad: bad.append(j)

  print "Bad ones:",
  for b in bad: print b,
  print

# This routine generates all the images that go in the movie.
# The movie is created with shell script "make_movie"
def make_movie_spectra(acc_data, freq, d, l, indexes, ant):

  if acc_data.shape[1] != len(freq):
    raise("Invalid number of frequencies")
  if acc_data.shape[0] != len(d):
    raise("Days array not the same length as number of spectra")
  if acc_data.shape[0] != len(d):
    raise("Days array not the same length as number of spectra")
  if acc_data.shape[0] != len(indexes):
    raise("Index array not the same length as number of spectra")

  print "Making movie images"

  lowf = min(freq)
  highf = max(freq)

  mean_spectrum = np.ma.mean(acc_data, axis=0)
  np.savetxt("mean_spectrum.dat", mean_spectrum)
  
  previous_damped_params = { "w_0": -1.5, "w_1": 0.5, "w_2": 0.07, "a": 1.0, "b": -4e5, "c": 1.0, "d": 0.0, "f": 0.0 }
   
  import sys

  plt.clf()
  plt.figure(figsize=(20,14))
  plt.rcParams.update({'font.size': 22})

  spec_index_file = open("spec_index.dat", "w")
  poly_coeff_file = open("poly_coeff.txt", "w")
  damped_sin_coeff_file = open("damped_sin_coeff.txt", "w")

  # Loop through the data making a plot of each spectrum
  for i in range(acc_data.shape[0]):

    print i, indexes[i], "-----"
    spec_index_file.write(str(i)+" "+str(indexes[i])+"\n")

    #if i%100 == 0: 
    #  print i, "...",
    #  sys.stdout.flush()

    # Make sure all nans are masked
    data = acc_data[i]
    data.mask = np.logical_or(acc_data[i].mask, np.isnan(acc_data[i]))

    # Need to get rid of masked values 
    short_freq = np.ma.array(freq, mask=data.mask).compressed()
    data = data.compressed()

    try:		# Fits can fail
      poly_values, poly_coeff = poly_fit(short_freq, data, n_poly, print_fit=False)
      after_poly = data-poly_values
      damped_fit = fit_model_damped_sin(short_freq, after_poly, previous_damped_params)
      damped = model_damped(short_freq, damped_fit)
      for param, val in damped_fit.items():
        previous_damped_params[param] = float(val)
    except:
      poly_values = [ 0 for j in range(len(short_freq)) ]
      after_poly = [ 0 for j in range(len(short_freq)) ]
      damped =[ 0 for j in range(len(short_freq)) ]
      poly_coeff = damped_fit = []
   

    plt.clf()
    plt.subplot(2, 1, 1)
    plt.plot(short_freq, data, linewidth=0.5, label="This Spectrum")
    plt.plot(freq, mean_spectrum, linewidth=0.5, label="Mean Spectrum")
    plt.plot(short_freq, poly_values, linewidth=0.5, label="Polynomial Fit")
    plt.ylabel("Temperature [K]")
    plt.xlim(lowf, highf)
    plt.ylim(ymin=0, ymax=10000)
    plt.legend()
    if len(poly_coeff) == 0:
      #plt.text(58, 6000, "Poly fit failed\n")
      poly_coeff_file.write(str(i)+" "+d[i]+" "+str(indexes[i])+" Poly fit failed\n")
    else:
      pstr = "" 
      for j, p in enumerate(poly_coeff):
        pstr += "p"+str(j)+": %.2e  " % p
      #plt.text(58, 6000, "Poly coeff  "+pstr)
      poly_coeff_file.write(str(i)+" "+d[i]+" "+str(indexes[i])+" "+pstr+"\n")
 
    plt.title("#"+str(i)+"  "+d[i]+", "+str(indexes[i])+"  LST "+( "%.2f" % l[i] )+", Ant "+ant)
    
    plt.subplot(2, 1, 2)
    plt.ylabel("Temperature [K]")
    plt.xlabel("Frequency [MHz]")
    plt.xlim(lowf, highf)
    plt.ylim(-300, 300)
    plt.plot(short_freq, after_poly, linewidth=0.5, label="This spectrum minus polynomial")
    plt.plot(short_freq, damped, "g", linewidth=0.5, label="Damped Sin Fit")
    plt.legend()
    if len(damped_fit) == 0:
      #plt.text(50, -290, "Damped fit failed")
      damped_sin_coeff_file.write(str(i)+" "+d[i]+" "+str(indexes[i])+" Damped fit failed\n")
    else:
      pstr = ""
      for param, val in damped_fit.items():
        pstr += "%s: %.2e   " % (param, val)
      #plt.text(42, -290, "Damped sin coeff  "+pstr)
      damped_sin_coeff_file.write(str(i)+" "+d[i]+" "+str(indexes[i])+" "+pstr+"\n")
    
    plt.savefig("spec"+str(i)+".png"); 

    np.savetxt("spec"+str(i)+".dat", np.array(list(zip(short_freq, data))))
    np.savetxt("subtract_poly"+str(i)+".dat", np.array(list(zip(short_freq, after_poly))))
    np.savetxt("damped_model"+str(i)+".dat", np.array(list(zip(short_freq, damped))))

  print

  poly_coeff_file.close()
  damped_sin_coeff_file.close()
  spec_index_file.close()


def plot_waterfall(acc_data, freq, fl, d):
  import sys

  print "Generating waterfall"

  if acc_data.shape[0] != len(d):
    raise("Days not the same as number of LSTs")

  if acc_data.shape[1] != len(freq):
    raise("Invalid number of frequencies")

  data = np.zeros((0, acc_data.shape[1]))

  for i in range(acc_data.shape[0]):
    if i%100 == 0: 
      print i, "...",
      sys.stdout.flush()

    if i in fl:
      data = np.append(data, np.zeros((1, acc_data.shape[1])), axis=0)
      continue

    row = np.zeros((1, acc_data.shape[1]))
    row[0, :] = acc_data[i, :]
    data = np.append(data, row, axis=0)

    if i < acc_data.shape[0]-2:
      if d[i] != d[i+1]: 
        data = np.append(data, np.zeros((50, data.shape[1])), axis=0)
        
  plt.clf()
  data = np.clip(data, 0, 10000)    # WHY CLIP NECESSARY?
  plt.imshow(data, aspect="auto", extent=[ freq[0], freq[-1], data.shape[0], 0])
  plt.ylabel("Time Sequence")
  plt.xlabel("Frequency [MHz]")
  c = plt.colorbar() 
  c.set_label("Temperature")
  plt.savefig("waterfall.png")

  print



import argparse
p = argparse.ArgumentParser(description='Plot antenna spectra and residuals')
p.add_argument('-n',  '--n_poly', help='number of terms in log-poly to fit for residuals. Default 5', type=int,  default=3)
args = p.parse_args()

n_poly = args.n_poly

ant = "254A"

spectra = Spectra("file_list.txt", "flag_db.txt", ant)
print spectra.accumulated_data.shape[0], "spectra in files"
accumulated_data, frequencies, lsts, days, indexes = spectra.good_data()
print accumulated_data.shape[0], "good spectra"

# These three routines do specific things - all visualizations.
make_movie_spectra(accumulated_data, frequencies, days, lsts, indexes, ant)
spectra.poly_flatten_time()
accumulated_data, frequencies, lsts, days, indexes = spectra.good_data()
detect_rubble(accumulated_data)
#plot_spans(accumulated_data, days)
#plot_waterfall(accumulated_data, frequencies, days)
exit()

# The rest of the code does:
# 1. Average the data
# 2. Fit and subtract polynomial
# 3. Fit and subtract damped sin
# 4. Show the result
# Also saves a few things on the way.

np.savetxt("accumulated_data.dat", np.ma.filled(accumulated_data, -1))
hickle.dump(accumulated_data, "accumulated_data.hkl")


rms = np.zeros(accumulated_data.shape[1])
for i in range(accumulated_data.shape[1]):
  rms[i] = np.ma.std(accumulated_data[:, i])


# Average all the spectra, so now we have 1 spectra.
aD = np.ma.mean(accumulated_data, axis=0)


# Cut frequencies below 40MHz. To cut below 58MHz use a value of 1167
f2 = frequencies
np.savetxt("data.dat", np.array(list(zip(f2, aD))))
filt = aD.compressed()-scipy.signal.medfilt(aD.compressed(), 9)
filt = filt[9:-9]
print "Noise", np.std(filt[filt.shape[0]/2:])
np.savetxt("rms.dat", np.array(list(zip(f2, rms))))


# ------------ Step 1: Fit a polynomial or power law and subtract to get residual rD 

# This line fits the polynomial:
rD = aD - poly_fit(f2, aD, n_poly)[0]	

# This line fits a power law:
#rD = aD-spectral_index_fit(f2, aD)

# Need to get rid of masked values for next step
f2 = np.ma.array(f2, mask=rD.mask).compressed()
rD = rD.compressed()

# ------------ Step 2: Fit a damped sinusoid


rD_model_params = fit_model_damped_sin(f2, rD)
rD_sin_model    = model_damped(f2, rD_model_params)

# Plot the residual and damped sinusoid fit

plt.plot(f2, rD, label="Data")
plt.plot(f2, rD_sin_model, label="Fit")
plt.legend()
plt.show()
