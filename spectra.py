import numpy as np
import hickle, os
import random
import scipy.signal
import scipy.optimize
import json

# Find the spectral index just by curve fitting
def spectral_index(freq, temp):
    def S(x, C, si):
        return C*x**si
    
    # Don't want masked values or NaNs or zeros
    nan_temp = np.ma.filled(temp, np.nan)
    f = freq[np.logical_or(np.isnan(nan_temp), (nan_temp!=0))]
    s = temp[np.logical_or(np.isnan(nan_temp), (nan_temp!=0))]

    try:
        popt, pcov = scipy.optimize.curve_fit(S, f, s)
    except:
        popt = ( 0, 0 )
   
    return popt


class Spectra(object):

  def __init__(self, fname, flag_fname, ant, lst_min=0, lst_max=24):		# Expects to use all the data it is given except for flags

    # Read all the hickle files, which must be listed in the file "file_list.txt".
    # Accumulate usable the spectra in a big array.
    random.seed()

    self.days = []
    self.indexes = []
    self.lsts = []
    for index, line in enumerate(open(fname)):
      if line[0] == "#": continue

      f = line[:-1]
      data = hickle.load(f)
 
      for key in sorted(data.keys()): 
        if key == "frequencies": self.frequencies = data[key]
        if key == "lsts": 
          day_lsts = data[key]
          use_lst_indexes = np.arange(data[key].shape[0])[np.logical_and(data[key]>=lst_min, data[key]<=lst_max)]
        if key == "indexes": day_indexes = data[key]

        if key == ant:
          ant_data = data[key]

      if len(day_indexes) != len(day_lsts):
        raise RuntimeError("Indexes and LSTs not same length in "+f)

      ant_data = np.ma.array(ant_data[use_lst_indexes], mask=ant_data.mask[use_lst_indexes])
   
      self.days += [ os.path.basename(f)[11:-4] ]*ant_data.shape[0]		# Every spectrum gets a day tag
      self.lsts = np.append(self.lsts, day_lsts[use_lst_indexes])
      self.indexes = np.append(self.indexes, day_indexes[use_lst_indexes])

      # Add them to the big array
      try: 
        self.accumulated_data = np.ma.append(self.accumulated_data, ant_data, axis=0)
      except:
        self.accumulated_data = ant_data

      print "File", index, line[:-1], ant_data.shape[0], "spectra", "starts", self.accumulated_data.shape[0]-ant_data.shape[0], "ends", self.accumulated_data.shape[0]


    # These lines chop above 85MHz because April data doesn't have that for real.
    # Uncomment these lines for April data
    self.accumulated_data = self.accumulated_data[:, :2292]
    self.frequencies = self.frequencies[:2292]

    # Load a list of flags from flagged_times.dat. These are just sequence
    # numbers relating specifically to the time order of the loaded spectra
    # e.g as you would see in the movie "6 of 1000" so use the number 6 to 
    # flag that one.
    self.flags = self.load_flags(flag_fname)

    # This removes below 40MHz
    self.accumulated_data = self.accumulated_data[:, 417:]
    self.frequencies = self.frequencies[417:]

    self.days = np.array(self.days)

    if len(self.days) != len(self.lsts) or len(self.days) != len(self.indexes):
      raise RuntimeError("Days, LSTs, indexes not all same length")


  def load_flags_old(self, fname):
    if os.path.exists(fname):
      # Remove times we don't want to use. Flags
      # must be the index of the data in the current run.
      flags = []
      for line in open("flagged_times.txt"):
        l = line.split("-")
        if len(l) == 2:
          flags += range(int(l[0]), int(l[1])+1)
        elif len(l) == 1: flags.append(int(l[0]))
        else:
          print "Bad format in flags\n"
          exit(1)

    else: flags = []

    return flags

  def load_flags(self, fname):
    # Get the flags using there global indicators: file name and index in file
    if os.path.exists(fname):
      new_flags = {}
      for line in open(fname):
        if line[0] != "#" and line[0] != "\n":
          l = line[:-1].split()
          if l[0] == "Channels":
            new_flags["Channels"] = [ int(x) for x in l[1:] ]
          else:
            if l[0] in new_flags.keys():
  	      if l[1] == "time_index" or l[1] == "channel":
                new_flags[l[0]][l[1]] = [ int(x) for x in l[2:] ]
	      else:
	        raise RuntimeError("Invalid flag type in flag file "+fname)
            else:
	      new_flags[l[0]] = {}
              new_flags[l[0]][l[1]] = [ int(x) for x in l[2:] ]

    else: new_flags = {}
    
    return new_flags


  def good_data(self):
    # Generate sequential flag indexes for the data that was loaded
    flags = []
    for i in range(len(self.days)):
      if self.days[i] in self.flags.keys():
        if "time_index" in self.flags[self.days[i]].keys():
          if int(self.indexes[i]) in self.flags[self.days[i]]["time_index"]: flags.append(i)

    flags = sorted(flags)

    # Apply global channel flags
    if "Channels" in self.flags.keys():
      for ch in self.flags["Channels"]: 
        self.accumulated_data[:, ch].mask = True

    not_flagged = np.delete(np.arange(self.accumulated_data.shape[0]), flags)
    return self.accumulated_data[not_flagged], self.frequencies, self.lsts[not_flagged], self.days[not_flagged], self.indexes[not_flagged]

  def all_data(self):
    not_flagged = np.delete(np.arange(self.accumulated_data.shape[0]), self.flags)
    return self.accumulated_data, self.frequencies, self.lsts, self.days, self.indexes

  def flatten(self):
    window = 9
    for i in range(self.accumulated_data.shape[0]):
      self.accumulated_data.data[i, :] = np.subtract(self.accumulated_data[i], scipy.signal.medfilt(self.accumulated_data[i], window)).data

  def flatten1(self):
    dummy_fit = [ 4e07,  -2.3]
    dummy_signal = dummy_fit[0]*(self.frequencies)**dummy_fit[1]
    
    for i in range(10):
      fit = spectral_index(self.frequencies, self.accumulated_data[i])
      signal = fit[0]*self.frequencies**fit[1]
      self.accumulated_data.data[i, :] = np.add(np.subtract(self.accumulated_data.data[i], signal), dummy_signal)

  def random_sample(self, num):

    not_flagged = np.delete(np.arange(self.accumulated_data.shape[0]), self.flags)
    if num > len(not_flagged):
      raise RuntimError("Asked to select more spectra than exist unflagged in the data")   
    elif num == len(not_flagged):
      selected = not_flagged
    else:
      selected = random.sample(not_flagged, num)

    return self.accumulated_data[selected]	# Preserves mask



#spectra = Spectra("file_list.txt", "flagged_times.txt", "254A")
#for i in spectra.flags:
#  print i, spectra.days[i], int(spectra.indexes[i])

