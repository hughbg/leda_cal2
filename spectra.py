import numpy as np
import hickle, os
import random
import scipy.signal
import scipy.optimize
import scipy.io
import matplotlib.pyplot as plt


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
  """
  A class used to extract radiometry spectra from multiple hickle files, organize them, 
  and remove flagged spectra. Spectra can be flagged out in two ways:

  1. By specifying them in a flag file. Spectra are uniquely identified by the h5 file
    they reside in, and their array index number within that file. This unique identification
    is used to specify spectra to ignore (flag). See flag_db.txt for current flags.
  2. Any spectrum that has been DTV flagged can be flagged out. That is done
    automatically by good_data().

  Normally you would create a Spectra object and call good_data() - that's all.
  """

  def __init__(self, fname, flag_fname, ant, lst_min=0, lst_max=24):		# Expects to use all the data it is given except for flags
    """
        fname : str
            The name of a file containing hickle file names, one hickle file per line. Those files will be loaded.
        flag_fname: str
	    The name of the flag file. See flag_db.txt for an example of flagging. 
            The name can be the empty string if there are no flags.
        ant: str
	    Which antenna to get data for. Like "254A".
        lst_min, lst_max: float
            Restrict the data to LSTs in this range.

    """

    # Read all the hickle files, which must be listed in the file "file_list.txt".
    # Accumulate usable the spectra in a big array.
    random.seed()

    self.days = []					# Generate a file name day/time for each spectrum - many spectra will have the same one (on the same day)
    self.file_indexes = np.zeros(0, dtype=np.int)	# Just sequence number indexes into the original h5 file
    self.lsts = np.zeros(0)				# LSTs
    self.utcs = np.zeros(0)				
    self.dtv_times = np.zeros(0, dtype=np.bool)		# Times where DTV was detected and flagged
    
    for index, line in enumerate(open(fname)):
      if line[0] == "#": continue

      f = line[:-1]
      data = hickle.load(f)
 
      for key in sorted(data.keys()): 
        if key == "frequencies": self.frequencies = data[key]
        if key == "lsts": 
          day_lsts = data[key]
          use_lst_indexes = np.arange(data[key].shape[0])[np.logical_and(data[key]>=lst_min, data[key]<=lst_max)]    # List of indexes of lsts within the limits
        if key == "indexes": day_indexes = data[key]
        if key == "utcs": day_utcs = data[key]
        if key == ant+"_dtv_times": 
          dtv_times = data[key]

        if key == ant:
          ant_data = data[key]

      if len(day_indexes) != len(day_lsts):
        raise RuntimeError("Indexes and LSTs not same length in "+f)
      if len(day_indexes) != len(day_utcs):
        raise RuntimeError("Indexes and UTCs not same length in "+f)



      # Turn the DTV times into an 1-D array like a mask, indicating the times, then 
      # it is the length of the data, like the other arrays: lsts, indexes etc.
      # dtv_times are a list of indexes (absolute) in the file.
      zeros = np.full(ant_data.shape[0], False, dtype=np.bool)
      zeros[[ i for i in range(len(day_indexes)) if day_indexes[i] in dtv_times ]] = True
      dtv_times = zeros

      # Strip out unwanted LST

      ant_data = np.ma.array(ant_data[use_lst_indexes], mask=ant_data.mask[use_lst_indexes])	# lst_min to lst_max
   
      self.days += [ os.path.basename(f)[11:-4] ]*ant_data.shape[0]		# Every spectrum gets a day tag
      self.lsts = np.append(self.lsts, day_lsts[use_lst_indexes])
      self.utcs = np.append(self.utcs, day_utcs[use_lst_indexes])
      self.file_indexes = np.append(self.file_indexes, day_indexes[use_lst_indexes])
      self.dtv_times = np.append(self.dtv_times, dtv_times[use_lst_indexes])

      # Add them to the big array
      try: 
        self.accumulated_data = np.ma.append(self.accumulated_data, ant_data, axis=0)
      except:
        self.accumulated_data = ant_data

      if self.lsts.shape[0] > 0:
        print "File", index, line[:-1], ant_data.shape[0], "spectra", "starts", self.accumulated_data.shape[0]-ant_data.shape[0], "ends", self.accumulated_data.shape[0], "LST", self.lsts[0], "-", self.lsts[-1]
      else:         
        print "File", index, line[:-1], ant_data.shape[0], "spectra", "starts", self.accumulated_data.shape[0]-ant_data.shape[0], "ends", self.accumulated_data.shape[0]
     

    # These lines chop above 85MHz because April data doesn't have that for real.
    # Uncomment these lines for April data
    self.accumulated_data = self.accumulated_data[:, :2292]
    self.frequencies = self.frequencies[:2292]

    # Load the flag database.
    self.flags = self.load_flags(flag_fname)

    # This removes below 40MHz
    self.accumulated_data = self.accumulated_data[:, 417:]
    self.frequencies = self.frequencies[417:]

    self.days = np.array(self.days)

    if self.accumulated_data.shape[0] != len(self.days) or len(self.days) != len(self.lsts) or len(self.days) != len(self.utcs) or len(self.days) != len(self.file_indexes) or len(self.days) != len(self.dtv_times):
      raise RuntimeError("Data, days, LSTs, UTCs, file_indexes not all same length "+str(len(self.days))+" "+str(len(self.lsts))+" "+str(len(self.utcs))+" "+str(len(self.file_indexes))+" "+str(len(self.dtv_times)))


  def load_flags_old(self, fname):
    if len(fname) > 0 and os.path.exists(fname):
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
    """
        This is the workhorse. You will be supplied with an array of spectra containing
        all the data in the hickle files, in time order, with flagging, and extra
	information like LSTs will be supplied.

        Returns 5 values
        ----------------
        accumulated_data : numpy 2-D array
            An array of all the spectra in time order. First dimension is time,
	    second dimension is the channels.
        frequencies: float array
	    An array listing all the frequencies (MHz) for the channels. Same
	    length as the second dimension of accumulated_data.
        lsts: float array
	    The LSTs for every spectra in accumulated_data. lsts[i]
	    is the LST for spectrum accumulated_data[i]. "lsts" has
	    the same length as the first dimension of accumulated_data.
        utcs: float array
        days: array of strings
            The day/time on which every spectra were recorded.
	    days[i] is the day/time for spectrum accumulated_data[i]. "days" has
            the same length as the first dimension of accumulated_data.
        file_indexes: integer array
            For all spectra, the index within the h5 file where the spectrum is located.
	    indexes[i] is the index for spectrum accumulated_data[i]. days[i] together
            with file_indexes[i] uniquely specify where the spectrum accumulated_data[i] 
            is located - the h5 file, and the array index within that file.
            "file_indexes" has the same length as the first dimension of accumulated_data.


    """

    # Generate flag indexes (from flag database) to match data that was loaded
    flags = []
    for i in range(len(self.days)):
      if self.days[i] in self.flags.keys():
        if "time_index" in self.flags[self.days[i]].keys():
          if int(self.file_indexes[i]) in self.flags[self.days[i]]["time_index"]: flags.append(i)

    # Include dtv times
    for index in np.nonzero(self.dtv_times)[0]:
      flags.append(index)

    # Apply global channel flags
    if "Channels" in self.flags.keys():
      for ch in self.flags["Channels"]: 
        self.accumulated_data[:, ch].mask = True

    not_flagged = np.delete(np.arange(self.accumulated_data.shape[0]), flags)

    data = self.accumulated_data[not_flagged]
    print ( "%.2f%% masked" % (float(data.shape[0]*data.shape[1]-np.ma.MaskedArray.count(data))/(data.shape[0]*data.shape[1])*100) )

    return self.accumulated_data[not_flagged], self.frequencies, self.lsts[not_flagged], self.utcs[not_flagged], self.days[not_flagged], self.file_indexes[not_flagged]
		

  def bad_data(self):
    # Generate flag indexes (from flag database) to match data that was loaded
    flags = []
    for i in range(len(self.days)):
      if self.days[i] in self.flags.keys():
        if "time_index" in self.flags[self.days[i]].keys():
          if int(self.indexes[i]) in self.flags[self.days[i]]["time_index"]: flags.append(i)

    # Include dtv times
    for index in np.nonzero(self.dtv_times)[0]:
      flags.append(index)

    return self.accumulated_data[flags], self.frequencies, self.lsts[flags], self.utcs[not_flagged], self.days[flags], self.file_indexes[flags]

  def all_data(self):
    return self.accumulated_data, self.frequencies, self.lsts, self.utcs, self.days, self.file_indexes


  # These flattening routines do not necessarily work if there are multiply days in the data

  def flatten(self):
    window = 9
    for i in range(self.accumulated_data.shape[0]):
      self.accumulated_data.data[i, :] = np.subtract(self.accumulated_data[i], scipy.signal.medfilt(self.accumulated_data[i], window)).data

  def flatten1(self):
    
    for i in range(self.accumulated_data.shape[0]):
      fit = spectral_index(self.frequencies, self.accumulated_data[i])
      signal = fit[0]*self.frequencies**fit[1]
      self.accumulated_data.data[i, :] = np.subtract(self.accumulated_data.data[i], signal)

  def si_fit(self):
    
    for i in range(self.accumulated_data.shape[0]):
      fit = spectral_index(self.frequencies, self.accumulated_data[i])
      signal = fit[0]*self.frequencies**fit[1]
      self.accumulated_data.data[i, :] = signal

  def median_smooth(self):
    window = 9
    for i in range(self.accumulated_data.shape[0]):
      self.accumulated_data.data[i, :] = scipy.signal.medfilt(self.accumulated_data[i], window)

  def poly_flatten_time(self):
    for i in range(self.accumulated_data.shape[1]):
      x = np.arange(self.accumulated_data.shape[0])
      z = np.polyfit(x, self.accumulated_data[:, i], 3)
      p = np.poly1d(z)     
      self.accumulated_data.data[:, i] -= p(x)


  def random_sample(self, num):

    good, freq, lsts, days, file_indexes = self.good_data()
    if num > good.shape[0]:
      raise RuntimeError("Asked to select "+str(num)+" spectra. There are only "+str(len(good))+".")   
    else:
      selected = random.sample(np.arange(good.shape[0]), num)

    return good[selected], freq, lsts[selected], utcs[selected], days[selected], file_indexes[selected]	# Preserves mask

  def single(self, daytime, index):
    for i in range(len(self.accumulated_data)):
      if self.days[i] == daytime and self.file_indexes[i] == index:
        return self.accumulated_data[i], self.frequencies, self.lsts[i], self.utcs[selected], self.days[i], self.file_indexes[i]

    raise RuntimeError("Failed to find single spectrum "+daytime+" "+str(index))
 
  def apply_matlab_mask(self):	
	
    data, frequencies, lsts, utcs, days, indexes = self.good_data()

    print ( "%.2f%% masked" % (float(data.shape[0]*data.shape[1]-np.ma.MaskedArray.count(data))/(data.shape[0]*data.shape[1])*100) )
    index = 0
    while index < len(days):
      start = index
      # Find the end
      while index < len(days) and days[index] == days[start]: index += 1
    
      if os.path.exists("../../rubble/outriggers_"+days[start]+"/mask.mat"):
        print "Apply matlab mask to", days[start], start, "to", index
        mask = np.transpose(scipy.io.loadmat("../../rubble/outriggers_"+days[start]+"/mask.mat")["Mask1"])
        freq = scipy.io.loadmat("../../rubble/outriggers_"+days[start]+"/spec.mat")["frequencies"][0]
 
        if len(frequencies) != len(freq):
          print frequencies, freq
          raise RuntimeError("Number of frequencies not the same in mask "+str(len(frequencies))+" "+str(len(freq)))
        for i in range(len(frequencies)): 
          if abs(int(frequencies[i]*1e6)-int(freq[i]*1e6)) > 1:
            raise RuntimeError("Frequencies don't match in apply_matlab_mask "+str(frequencies[i])+" "+str(freq[i]))
        if mask.shape[0] != index-start:
          raise RuntimeError("Mask is the wrong shape in apply_matlab_mask. "+str(mask.shape[0])+" "+str(index-start))

        data[start:index][mask==0] = np.ma.masked
        #plt.figure(figsize=(20,20)); plt.imshow(mask,aspect="auto"); plt.savefig("x.png"); plt.clf(); plt.imshow(data); plt.savefig("y.png"); exit()
        print ( "%.2f%% masked" % (float(data.shape[0]*data.shape[1]-np.ma.MaskedArray.count(data))/(data.shape[0]*data.shape[1])*100) )

      else: print "No matlab mask for", days[start]

    return data, frequencies, lsts, utcs, days, indexes

  def random_mask(self):
    data, frequencies, lsts, days, file_indexes = self.good_data()
    print ( "%.2f%% masked" % (float(data.shape[0]*data.shape[1]-np.ma.MaskedArray.count(data))/(data.shape[0]*data.shape[1])*100) )
    for i in range(int(float(data.shape[0])*data.shape[1]/7)):
      x = int(np.random.random()*(data.shape[0]-1))
      y = int(np.random.random()*(data.shape[1]-1))
      data[x, y] = np.ma.masked
    print ( "%.2f%% masked" % (float(data.shape[0]*data.shape[1]-np.ma.MaskedArray.count(data))/(data.shape[0]*data.shape[1])*100) )
    return data, frequencies, lsts, utcs, days, file_indexes


"""
sigma_factor = 3.5
spectra = Spectra("file_list.txt", "flag_db.txt", "254A")
spectra.poly_flatten_time()
data, f, lsts, days, indexes = spectra.good_data()
# 114 389
bad = []
for i in range(data.shape[1]):
  std = np.ma.std(data[:, i])
  mean = np.ma.mean(data[:, i])
  for j in range(data.shape[0]):
    if data[j, i] < mean-float(sigma_factor)*std or mean+float(sigma_factor)*std < data[j, i]:
      if indexes[j] not in bad: bad.append(indexes[j])
print bad; exit()

specs = []
move = "mv"
si = np.loadtxt("spec_index.dat", dtype=np.int)
for b in bad:
  for i in range(len(si)):
    if b == si[i][1]: 
      if si[i][0] not in specs:
        specs.append(si[i][0])
        move += " spec"+str(si[i][0])+".png"
move += " bad"
os.system(move)
"""
"""
import scipy.stats
import matplotlib.pyplot as plt
spectra = Spectra("file_list.txt", "flag_db.txt", "254A")
s1 = spectra.single("2019-01-05_20H24M23S", 2654)[0]
s2 = spectra.single("2019-02-10_10H00M03S", 2668)[0]

s1 = scipy.signal.medfilt(s1, 9)
s2 = scipy.signal.medfilt(s2, 9)


def flat(a):
  aa = a #a.compressed()

  flatted = np.zeros(len(aa))
  slope = (aa[-1]-aa[0])/(len(aa)-1)

  for i in range(len(aa)):  
    flatted[i] = aa[i]-(slope*i+aa[0])


  return flatted


window = 30
res1 = []
for i in range(0, len(s1)-window, 1):
  res1.append(scipy.stats.kurtosis(flat(s1[i:i+window])))
res2 = []
for i in range(0, len(s2)-window, 1):
  res2.append(scipy.stats.kurtosis(flat(s2[i:i+window])))


plt.plot(res1, linewidth=0.5, label="bad")
plt.plot(res2, linewidth=0.5, label="good")
plt.legend()
plt.title("Sliding kurtosis window, median-smoothed radiometry spectrum")
plt.xlabel("Channel")
plt.ylabel("Standard deviation")
#plt.show()
plt.savefig("slide_k.png")
"""


