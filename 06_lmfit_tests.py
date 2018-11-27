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

def make_movie_spectra(acc_data, freq, fl, d, residual=False):
  if acc_data.shape[1] != len(freq):
    raise("Invalid number of frequencies")
  if acc_data.shape[0] != len(d):
    raise("Days not the same as number of LSTs")

  print "Making movie images"

  mean_spectrum = np.ma.mean(acc_data, axis=0)


  import sys

  plt.clf()
  plt.figure(figsize=(20,14))

  for i in range(acc_data.shape[0]):
    if i in flags: continue

    if i%100 == 0: 
      print i, "...",
      sys.stdout.flush()
  
    try:
      if residual:
        data = acc_data[i]-poly_fit(freq, acc_data[i], n_poly)
        rD_model_params = fit_model_sin_off(freq, data)
        data = data-model_sin_off(freq, rD_model_params)
      else: data = acc_data[i]
    except:
      data = [ 0 for i in range(acc_data.shape[1]) ]

    plt.clf()
    plt.plot(freq, data, linewidth=0.5)
    if not residual: plt.plot(frequencies, mean_spectrum, linewidth=0.8)
    plt.ylabel("Temperature")
    plt.xlabel("Frequency [MHz]")
    if residual: plt.ylim(-200, 300)
    else: plt.ylim(ymin=0, ymax=10000)
    
    plt.title(str(i)+" of "+str(acc_data.shape[0])+".  "+d[i])
    plt.savefig("spec"+str(i)+".png")

  print

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

def load_flags():
  if os.path.exists("flagged_times"):
    # Remove times we don't want to use. Flags
    # must be the index of the data in the current run.
    flags = []
    for line in open("flagged_times.dat"):
      l = line.split("-")
      if len(l) == 2:
        flags += range(int(l[0]), int(l[1])+1)
      elif len(l) == 1: flags.append(int(l[0]))
      else:
        print "Bad format in flags\n";
        exit(1)

  else: flags = []

  return flags

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description='Plot antenna spectra and residuals')
    p.add_argument('-n',  '--n_poly', help='number of terms in log-poly to fit for residuals. Default 5', type=int,  default=3)
    args = p.parse_args()

    n_poly = args.n_poly

    # Set some important parameters in the next few lines
    ant = "254A"

    # Read all the hickle files, which must be listed in the file "file_list.txt".
    # Accumulate usable the spectra in a big array.

    days = []
    for line in open("file_list.txt"):
      f = line[:-1]
      data = hickle.load(f)
 
      for key in sorted(data.keys()): 
        if key == "frequencies": frequencies = data[key]
            
        if key == ant:
          ant_data = data[key]

      # Select the spectra that are usable (based on LST)
      print "File", line[:-1], ant_data.shape[0], "spectra"
      days += [ os.path.basename(f)[11:21] ]*ant_data.shape[0]

      # Add them to the big array
      try: 
        accumulated_data = np.ma.append(accumulated_data, ant_data, axis=0)
      except:
        accumulated_data = ant_data

    # Calculate diffs of all the spectra from the mean
    #mean_spectra = np.ma.mean(accumulated_data, axis=0)
    #for i in range(accumulated_data.shape[0]):
    #  print i, np.std(mean_spectra-accumulated_data[i])
    #exit()



print "Num spectra", accumulated_data.shape[0]

accumulated_data = accumulated_data[:, :2292]
frequencies = frequencies[:2292]

flags = load_flags()
for i in flags:
  accumulated_data.mask[i, :] = True
accumulated_data[:, 410].mask = True; accumulated_data[:, 846].mask = True

accumulated_data = accumulated_data[:, 417:-16]
frequencies = frequencies[417:-16]

#make_movie_spectra(accumulated_data, frequencies, flags, days)
#plot_spans(accumulated_data, flags, days)
#plot_waterfall(accumulated_data, frequencies, flags, days)
#exit()

np.savetxt("accumulated_data.dat", np.ma.filled(accumulated_data, -1))

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
print "Noise", np.std(filt)
np.savetxt("rms.dat", np.array(list(zip(f2, rms))))


# ------------ Step 1: Fit a polynomial or power law and subtract to get residual rD 

# This line fits the polynomial:
rD = aD - poly_fit(f2, aD, n_poly)	

# This line fits a power law:
#rD = aD-spectral_index_fit(f2, aD)

# Need to get rid of masked values for next step
f2 = np.ma.array(f2, mask=rD.mask).compressed()
rD = rD.compressed()

# ------------ Step 2: Fit a damped sinusoid


rD_model_params = fit_model_sin_off(f2, rD)
rD_sin_model    = model_sin_off(f2, rD_model_params)

# Plot the residual and damped sinusoid fit

plt.plot(f2, rD, label="Data")
plt.plot(f2, rD_sin_model, label="Fit")
plt.legend()
plt.show()
