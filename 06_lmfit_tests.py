# This file based on 04_lmfit.py


#from matplotlib import use as muse; muse('Agg')
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
import scipy.io as sio
import sys
import bottleneck as bn
from matplotlib.ticker import AutoMinorLocator


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

def interpolate_flagged(data):
  for i, masked in enumerate(data.mask):
    if masked: 
      if i == 0: data[i] = data[i+1]
      elif i == len(data)-1: data[i] = data[i-1]
      else: data[i] = (data[i+1]+data[i-1])/2

  return data


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

  plt.figure(figsize=(20,8))

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

def make_movie_spectra_time(acc_data, freq, l, d, low, high):

  if acc_data.shape[0] != len(l):
    raise("LST array not the same length as number of spectra")
  if acc_data.shape[1] != len(freq):
    raise("Invalid number of frequencies")
  if acc_data.shape[0] != len(d):
    raise("Invalid number of days")

  plt.clf()
  plt.figure(figsize=(22, 8))

  np.savetxt("step2time.dat", np.array(list(zip(np.arange(acc_data.shape[0]), d))), fmt="%s")

  print "Plotting", acc_data.shape[1], "across channels by time"
  for j in range(acc_data.shape[1]):
    print j,; sys.stdout.flush()

    ax = plt.subplot(1, 1, 1)
    plt.plot(np.arange(acc_data.shape[0]), acc_data[:, j], linewidth=0.5)
    plt.ylabel("Temperature [K]")
    plt.xlabel("Time sequence")
    plt.title("May 2018 - May 2019, night. Channel "+str(j)+", Frequency "+str(freq[j])+" MHz")
    plt.ylim(ymin=low, ymax=high)
    #plt.xticks(np.arange(0, acc_data.shape[0]+1, 5000.0))
    #ax.minorticks_on()
    #ax.xaxis.set_minor_locator(AutoMinorLocator(10))   
    plt.savefig("time_"+str(j)+".pdf", dpi=150)
    plt.clf()
    
    np.savetxt("time_"+str(j)+".dat", np.array(list(zip(np.arange(acc_data.shape[0]), acc_data[:, j]))), fmt="%s")
    
  print


# This routine generates all the images that go in the movie.
# The movie is created with shell script "make_movie"
def make_movie_spectra(acc_data, freq, d, l, indexes, ant):

  if acc_data.shape[1] != len(freq):
    raise("Invalid number of frequencies")
  if acc_data.shape[0] != len(l):
    raise("LST array not the same length as number of spectra")
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

  subtract_poly = np.zeros((acc_data.shape[0], acc_data.shape[1]))

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
    orig = np.ma.array(np.arange(data.shape[0]), mask=data.mask).compressed()   # Track the indexes of non-masked
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
      after_poly = np.array([ 0 for j in range(len(short_freq)) ])
      damped =[ 0 for j in range(len(short_freq)) ]
      poly_coeff = damped_fit = []
   
    wide_after_poly = np.zeros(acc_data.shape[1])
    for j in range(after_poly.shape[0]): 
      wide_after_poly[orig[j]] = after_poly[j]
    subtract_poly[i] = wide_after_poly

    plt.clf()
    plt.subplot(3, 1, 1)
    plt.plot(short_freq, data, linewidth=0.5, label="This Spectrum")
    plt.plot(freq, mean_spectrum, linewidth=0.5, label="Mean Spectrum")
    plt.plot(short_freq, poly_values, linewidth=0.5, label="Polynomial Fit")
    plt.ylabel("Temperature [K]")
    plt.xlim(lowf, highf)
    plt.ylim(ymin=0, ymax=10000)
    plt.legend()
    if len(poly_coeff) == 0:
      plt.text(58, 6000, "Poly fit failed\n", fontsize=14)
      poly_coeff_file.write(str(i)+" "+d[i]+" "+str(indexes[i])+" Poly fit failed\n")
    else:
      pstr = "" 
      for j, p in enumerate(poly_coeff):
        pstr += "p"+str(j)+": %.2e  " % p
      plt.text(58, 6000, "Poly coeff  "+pstr, fontsize=14)
      poly_coeff_file.write(str(i)+" "+d[i]+" "+str(indexes[i])+" "+pstr+"\n")
 
    plt.title("#"+str(i)+"  "+d[i]+", "+str(indexes[i])+"  LST "+( "%.2f" % l[i] )+", Ant "+ant)
    
    plt.subplot(3, 1, 2)
    plt.ylabel("Temperature [K]")
    plt.xlabel("Frequency [MHz]")
    plt.xlim(lowf, highf)
    plt.ylim(-400, 400)
    plt.plot(short_freq, after_poly, linewidth=0.5, label="This spectrum minus polynomial")
    plt.plot(short_freq, damped, "g", linewidth=0.5, label="Damped Sin Fit")
    plt.legend()
    if len(damped_fit) == 0:
      plt.text(50, -290, "Damped fit failed", fontsize=14)
      damped_sin_coeff_file.write(str(i)+" "+d[i]+" "+str(indexes[i])+" Damped fit failed\n")
    else:
      pstr = ""
      for param, val in damped_fit.items():
        pstr += "%s: %.2e   " % (param, val)
      plt.text(50, -290, "Damped sin coeff  "+pstr, fontsize=14)
      damped_sin_coeff_file.write(str(i)+" "+d[i]+" "+str(indexes[i])+" "+pstr+"\n")

    plt.subplot(3, 1, 3)
    plt.ylabel("Temperature [K]")
    plt.xlabel("Frequency [MHz]")
    plt.xlim(lowf, highf)
    plt.ylim(-300, 300)
    plt.plot(short_freq, after_poly-damped, linewidth=0.5)
    plt.tight_layout()
    
    plt.savefig("spec"+str(i)+".png"); 

    np.savetxt("spec"+str(i)+".dat", np.array(list(zip(short_freq, data))))
    np.savetxt("subtract_poly"+str(i)+".dat", np.array(list(zip(short_freq, after_poly))))
    np.savetxt("damped_model"+str(i)+".dat", np.array(list(zip(short_freq, damped))))
 

  print

  poly_coeff_file.close()
  damped_sin_coeff_file.close()
  spec_index_file.close()

  sio.savemat("subtract_poly.mat", { "subtract_poly" : np.transpose(subtract_poly) })


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

def interpolate_flagged(data):
  for i, masked in enumerate(data.mask):
    if masked: 
      if i == 0: data[i] = data[i+1]
      elif i == len(data)-1: data[i] = data[i-1]
      else: data[i] = (data[i+1]+data[i-1])/2

  return data

# Run a IIR filter on the data
def filter(d, btype="low"):

  b, a = scipy.signal.butter(3, 0.05, btype=btype)

  # Forward and back in one go
  y = scipy.signal.filtfilt(b, a, d)
 
  return y

def flatten_night(a, d, l, u):	# Use a polynomial to flatten a night's data
  # Strip masked vals
  x = np.ma.array(np.arange(a.shape[0]), mask=a.mask).compressed()
  b = a.compressed()

  if len(b) == 0: return np.ma.copy(a)

  z = np.polyfit(x, b, 3)

  p = np.poly1d(z)
  c = np.ma.copy(a)

  stat = np.sqrt(np.mean((c[x]-p(x))**2))
  if stat > 100:
    plt.plot(x, c[x])
    plt.plot(x, p(x))
    plt.title("Fit = "+( "%.1e" % stat )+" "+str(d)+" "+( "%.1f" % l )+" ")
    plt.savefig("fit"+( "%.2e" % stat )+".png")
    plt.clf()
    print stat

  c[x] -= p(x)
  return c
  


def bin_to_1MHz(bottom_f, filt, variance, channel_indexes):
  def calc_rms(x): return np.sqrt(np.mean(x**2))

  if len(filt) != len(variance) or len(filt) != len(channel_indexes):
    raise RuntimeError("Arrays of different length in bin_to_1MHz "+str(len(filt))+" "+str(len(variance))+" "+str(len(variance)))

  # We want to bin 1MHz of channels. That means from channel N to N+41 (inclusive). However, there
  # may be gaps in the channels, so there may be different numbers of channels binned.
  # The averaged frequencies are calculted from averaging 4 frequencies without gaps.

  nbin = 42
  chan_width = .024

  ndata = []
  nvariance = []
  i = 0
  while i < len(channel_indexes):	# Find blocks of channels and bin them. Blocks are defined by a channel sep of 42 in the indexes.
    j = i
    weighted_mean = 0.0
    D_2 = 0.0			#  https://en.wikipedia.org/wiki/Inverse-variance_weighting
    while j < len(channel_indexes) and channel_indexes[j] < channel_indexes[i]+nbin:
      weighted_mean += filt[j]/variance[j]
      D_2 += 1/variance[j]
      j += 1

    print j-i, "channels binned"
    D_2 = 1/D_2
    weighted_mean *= D_2
    
    ndata.append(weighted_mean)
    nvariance.append(D_2)
 
    i = j


  # Get frequencies for the bins, based on what was the starting frequency originally
  bottom_freq = (bottom_f+bottom_f+(nbin-1)*chan_width)/2
  print "Bottom f", bottom_f, "->", bottom_freq
  nf = [ bottom_freq+i*nbin*chan_width for i in range(len(ndata)) ]


  print "Scrunch to length", len(nf)

  #np.savetxt("filt.dat", np.array(list(zip(filt_f, filt))))
  plt.figure(figsize=(8, 6))
  plt.plot(nf, ndata)
  plt.title("Binned to 1MHz")
  plt.xlabel("Frequency [MHz]")
  plt.ylabel("Temp [K]")
  plt.savefig("bin1MHz.png")

  plt.clf()
  plt.figure(figsize=(8, 6))
  plt.plot(nf, nvariance)
  plt.title("Variance binned")
  plt.xlabel("Frequency [MHz]")
  plt.ylabel("Temp [K$^2$]")
  plt.tight_layout()
  plt.savefig("bin1MHz_var.png")


  np.savetxt("binned_frequencies.dat", nf)
  np.savetxt("binned_data.dat", ndata)
  np.savetxt("binned_variance.dat", nvariance)

  mn = (ndata-bn.move_nanmean(ndata, 9))[4:-4]
  mn = mn[mn!=np.nan]
  print mn
  print calc_rms((ndata-scipy.signal.medfilt(ndata, 9))[4:-4]), calc_rms(mn[4:]), calc_rms(ndata-filter(ndata))

  return nf, ndata, nvariance




import argparse
p = argparse.ArgumentParser(description='Plot antenna spectra and residuals')
p.add_argument('-n',  '--n_poly', help='number of terms in log-poly to fit for residuals. Default 5', type=int,  default=3)
args = p.parse_args()

n_poly = args.n_poly

ant = "254A"

spectra = Spectra("file_list.txt", "flag_db.txt", ant) #, lst_min=11.9, lst_max=11.915)
print spectra.accumulated_data.shape[0], "spectra in files"
accumulated_data, frequencies, lsts, utcs, days, indexes = spectra.good_data()

print accumulated_data.shape[0], "good spectra"
to_matlab = {
  "data" : np.transpose(np.ma.filled(accumulated_data, 0)),
  "indexes" : indexes,
  "days": days,
  "frequencies": frequencies
}
sio.savemat("spec.mat", to_matlab)


# These three routines do specific things - all visualizations.
#make_movie_spectra_time(accumulated_data, frequencies, lsts, utcs, 0, 2200); exit()
#make_movie_spectra(accumulated_data, frequencies, days, lsts, indexes, ant)
#spectra.poly_flatten_time()
#accumulated_data, frequencies, lsts, days, indexes = spectra.good_data()
#detect_rubble(accumulated_data)
#plot_spans(accumulated_data, days)
#plot_waterfall(accumulated_data, frequencies, days)
#exit()

# The rest of the code does:
# 1. Average the data
# 2. Fit and subtract polynomial
# 3. Fit and subtract damped sin
# 4. Show the result
# Also saves a few things on the way.

np.savetxt("accumulated_data.dat", np.ma.filled(accumulated_data, -1))
hickle.dump(accumulated_data, "accumulated_data.hkl")

# Variance over time, for each channel
# First have to flatten the channels which is tricky when there are multiple days,
# due to discontinuities. Flatten each day, and leave nans at the beginning/end.

day_ranges = []		# Find start/end of days
j = 0
while j < days.shape[0]:
  k = j+1
  while k < days.shape[0] and days[k] == days[j]: k += 1
  day_ranges.append((j, k, days[j], lsts[j], utcs[j]))
  j = k

variance = np.zeros(accumulated_data.shape[1])
flattened_data = np.zeros_like(accumulated_data)
fits = []
for i in range(accumulated_data.shape[1]):
  squares = 0.0
  num = 0
  for dr in day_ranges:
    #flattened = bn.move_nanmean(np.ma.filled(accumulated_data[dr[0]:dr[1], i], np.nan), 8)  # nanmean doesn't honour masked values, only nan
    flattened = flatten_night(accumulated_data[dr[0]:dr[1], i], dr[2], dr[3], dr[4])
    if np.ma.MaskedArray.count(accumulated_data[dr[0]:dr[1], i]) != np.ma.MaskedArray.count(flattened):
      raise RuntimeError("Masked values not preserved in flattening")
    if len(flattened) != dr[1]-dr[0]:
      raise RuntimeError("Flattened night not the right length")
    flattened_data[dr[0]:dr[1], i] = flattened
    flattened = flattened.compressed()
    if len(flattened) > 0:
      fits.append(np.sqrt(np.mean(flattened**2)))
      num += flattened.shape[0]
      squares += np.sum(flattened**2)
    

  if num > 0: variance[i] = squares/num/num
  else: variance[i] = 0 

#make_movie_spectra_time(flattened_data, frequencies, lsts, -400, 400); exit()


np.savetxt("flattened.dat", flattened_data[:, 1000])
np.savetxt("fits.dat", fits)
ch_indexes = np.arange(accumulated_data.shape[1])	# Indexes into the original channel array - taking out indexes means we know what frequencies taken out
bottom_frequency = frequencies[0]	# In case it gets flagged out, we must keep it

# Average all the spectra, so now we have 1 spectra. Save it
aD = np.ma.mean(accumulated_data, axis=0)


np.savetxt("integrated_spectrum.dat", np.array(list(zip(frequencies, np.ma.filled(aD, 0))))); exit()

# Get rid of flagged values. Does not alter aD which is used later
filt = aD.compressed()-scipy.signal.medfilt(aD.compressed(), 9)
filt_f = np.ma.array(frequencies, mask=aD.mask).compressed()

#aD = interpolate_flagged(aD)
#filt = aD-scipy.signal.medfilt(aD, 9)
#filt_f = frequencies

np.savetxt("filt.dat", filt)
filt = filt[9:-9]
print "Noise half", np.std(filt[len(filt)/2:])

# ------------ Step 1: Fit a polynomial or power law and subtract to get residual rD 

# This line fits the polynomial:
rD = aD #- poly_fit(frequencies, aD, n_poly)[0]	

# This line fits a power law:
#rD = aD-spectral_index_fit(f2, aD)

# Need to get rid of masked values for next step
variance = np.ma.array(variance, mask=rD.mask).compressed()
ch_indexes = np.ma.array(ch_indexes, mask=rD.mask).compressed()
f2 = np.ma.array(frequencies, mask=rD.mask).compressed()
rD = rD.compressed()

# ------------ Step 2: Fit a damped sinusoid

previous_damped_params = { "w_0": -1.5, "w_1": 0.5, "w_2": 0.07, "a": 1.0, "b": -4e5, "c": 1.0, "d": 0.0, "f": 0.0 }
rD_model_params = fit_model_damped_sin(f2, rD, previous_damped_params)
rD_sin_model = model_damped(f2, rD_model_params)

# Plot the residual and damped sinusoid fit


rD[27] = 0
rD[28] = 0
rD[190] = 0
rD[191] = 0
rD[408] = 0
rD[409] = 0
rD[559] = 0
rD[561] = 0
rD[562] = 0
rD[594] = 0
rD[1333] = 0
rD[1401] = 0
rD[1464] = 0
rD[1639:1644] = 0
rD[1797:1797+3] = 0


f2 = f2[rD!=0]
variance = variance[rD!=0]
ch_indexes = ch_indexes[rD!=0]
rD_sin_model = rD_sin_model[rD!=0]
rD = rD[rD!=0]

bin_to_1MHz(bottom_frequency, rD, variance, ch_indexes); exit()

# Plot the variance
plt.clf()
plt.plot(f2, variance)
plt.xlabel("Frequency [MHz]")
plt.ylabel("Temperature [K]")
plt.show()
plt.savefig("variance.png")


# Plot the damped sinusoid
plt.clf()
plt.plot(f2, rD, label="Data")
#plt.plot(f2, rD_sin_model, label="Fit")
plt.xlabel("Frequency [MHz]")
plt.ylabel("Temperature [K]")
plt.legend()
plt.show()
plt.savefig("residual.png")
np.savetxt("residual.dat", np.array(list(zip(f2, rD))))


#filt = (rD-scipy.signal.medfilt(rD, 9))[9:-9]
filt = (rD-bn.move_nanmean(rD, 9))[9:-9]
filt = (rD-filter(rD))[9:-9]

#f2, filt = bin_to_1MHz(f2[9:-9], filt)


print "Noise again", np.std(filt[len(filt)/2:])


plt.figure(figsize=(10,10))

lw = 0.5
plt.clf()
plt.plot(f2, rD, linewidth=lw)
plt.title("Signal")
plt.xlabel("Frequency [MHz]")
plt.ylabel("Temperature [K]")
plt.savefig("signal.png")

plt.clf()
plt.plot(f2, np.abs(np.fft.fftshift(np.fft.fft(rD))), linewidth=lw)
plt.title("Signal FFT")
plt.xlabel("Frequency [MHz]")
plt.ylabel("Temperature [K]")
plt.yscale("log")
plt.savefig("signal_fft.png")


plt.clf()
plt.plot(f2, filter(rD), linewidth=lw)
plt.title("Filtered Signal")
plt.xlabel("Frequency [MHz]")
plt.ylabel("Temperature [K]")
plt.savefig("filtered_low.png")

plt.clf()
plt.plot(f2, np.abs(np.fft.fftshift(np.fft.fft(filter(rD)))), linewidth=lw)
plt.title("Filtered Signal FFT")
plt.xlabel("Frequency [MHz]")
plt.ylabel("Temperature [K]")
plt.yscale("log")
plt.savefig("filtered_low_fft.png")


plt.clf()
plt.plot(f2, rD-filter(rD), linewidth=lw)
plt.title("Noise")
plt.xlabel("Frequency [MHz]")
plt.ylabel("Temperature [K]")
plt.savefig("noise_low.png")

plt.clf()
plt.plot(f2, np.abs(np.fft.fftshift(np.fft.fft(rD-filter(rD)))), linewidth=lw)
plt.title("Noise FFT")
plt.xlabel("Frequency [MHz]")
plt.ylabel("Temperature [K]")
plt.yscale("log")
plt.savefig("noise_low_fft.png")



plt.clf()
plt.plot(f2, filter(rD, btype="highpass"), linewidth=lw)
plt.title("Filtered Noise")
plt.xlabel("Frequency [MHz]")
plt.ylabel("Temperature [K]")
plt.savefig("filtered_noise.png")

plt.clf()
plt.plot(f2, np.abs(np.fft.fftshift(np.fft.fft(filter(rD, btype="highpass")))), linewidth=lw)
plt.title("Filtered Noise FFT")
plt.xlabel("Frequency [MHz]")
plt.ylabel("Temperature [K]")
plt.yscale("log")
plt.savefig("filtered_noise_fft.png")


plt.clf()
plt.plot(f2, rD-filter(rD, btype="highpass"), linewidth=lw)
plt.title("Signal minus Noise")
plt.xlabel("Frequency [MHz]")
plt.ylabel("Temperature [K]")
plt.savefig("signal_after_low.png")

plt.clf()
plt.plot(f2, np.abs(np.fft.fftshift(np.fft.fft(rD-filter(rD, btype="highpass")))), linewidth=lw)
plt.title("Signal minus Noise FFT")
plt.xlabel("Frequency [MHz]")
plt.ylabel("Temperature [K]")
plt.yscale("log")
plt.savefig("signal_less_fft.png")



