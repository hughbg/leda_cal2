from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html

def find_in_list(a, val):
    for i in range(len(a)):
        if a[i] == val: return i
    return -1

def get_data():
  # Load spectrum
  data = np.loadtxt("residual.dat")
  freq = np.zeros(1874, dtype=np.int)
  bottom = int(np.round(data[0, 0]*1e6))
  for i in range(len(freq)):
    freq[i] = bottom+i*24000

  # There will be missing channels. Generate a full frequency range and
  # fill the data in that, using 0 for flagged channels.
  full_data = np.zeros(len(freq))
  for i in range(len(data)):
    f = int(np.round(data[i, 0]*1e6))
    j = find_in_list(freq, f)
    if j == -1:
        raise RuntimeError("Did not find "+str(f))
    if full_data[j] != 0:
        raise RuntimeError("full_data has value at "+str(j)+" "+str(freq[j]))
    full_data[j] = data[i, 1]
    
  # Interpolate flagged channels
  for i in range(len(freq)):
    if full_data[i] == 0.0:
        full_data[i] = (full_data[i+1]+full_data[i-1])/2
        

  return freq, full_data

# Run a filter on the data
def filter(d, N, rs, Wn):

  b, a = signal.butter(3, 0.05)
  #b, a = signal.cheby2(N, rs, Wn, 'lowpass', analog=True)

  # Forward and back in one go
  y = signal.filtfilt(b, a, d)
 
  return y


f, data = get_data()
filtered_data = filter(data, 6, 80, -1.9); 

#print filter(1, 1.0, 0); exit()    # Reproduce the signal

# Plot filtered data
plt.plot(f/1e6, filtered_data, "b", linewidth=0.5)
plt.xlabel("Frequency [MHz]")
plt.ylabel("Temperature [K]")
plt.title("Filtered spectrum")
plt.show()

# Scale the filtered spectrum to the data
filtered_data *= np.mean(data)/np.mean(filtered_data)
 
# Plot scaled filtered data on top of data
plt.clf()
plt.plot(f/1e6, data, "r", linewidth=0.2, label="Data")
plt.plot(f/1e6, filtered_data, "b", linewidth=0.5, label="Filtered")
plt.xlabel("Frequency [MHz]")
plt.ylabel("Temperature [K]")
plt.title("Rescaled filtered spectrum and Data")
plt.legend()
plt.show()

