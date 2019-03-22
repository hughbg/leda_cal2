from datetime import datetime

import ephem
import numpy as np
import pylab as plt
from scipy.interpolate import interp1d as interp
from numpy import fft

def db20(x):
    return 20*np.log10(x)

def lin20(x):
    return 10**(x / 20.0)

def db10(x):
    return 10*np.log10(x)

def lin10(x):
    return 10.0**(x/ 10.0)

def to_complex(data, linear=True, radians=False):
    """ Convert amp / phase data to complex."""
    r = data[:, 0]
    q = data[:, 1]

    if linear is False:
        r = 10**(r / 20)
    if radians is False:
        q = np.deg2rad(q)

    x = r * np.cos(q)
    y = r * np.sin(q)
    return x + 1j* y

def poly_fit(x, y, n=5, log=True, print_fit=False):
    """ Fit a polynomial to x, y data

    x (np.array): x-axis of data (e.g. frequency)
    y (np.array): y-axis of data (e.g temperature)
    n (int): number of terms in polynomial (defaults to 5)
    """

    x, y = np.ma.array(x), np.ma.array(y)

    x_g = x
    x = np.ma.array(x, mask=y.mask).compressed()
    y = y.compressed()
    if log:
        yl = np.log10(y)
    else:
        yl = y

    fit = np.polyfit(x, yl, n)
    if print_fit:
        print fit
    p = np.poly1d(fit)

    if log:
        return 10**(p(x_g)), fit
    else:
        return p(x_g), fit

def fourier_fit(x, n_predict, n_harmonics):
    """ Fit a Fourier series to data

    Args:
        x: data to fit
        n_predict: next N data points to predict
        n_harmonics: number of harmonics to compute

    Notes:
    From github gist https://gist.github.com/tartakynov/83f3cd8f44208a1856ce
    """
    n = x.size
    n_harm = n_harmonics            # number of harmonics in model
    t = np.arange(0, n)
    p = np.polyfit(t, x, 1)         # find linear trend in x
    x_notrend = x - p[0] * t        # detrended x
    x_freqdom = fft.fft(x_notrend)  # detrended x in frequency domain
    f = fft.fftfreq(n)              # frequencies
    indexes = range(n)
    # sort indexes by frequency, lower -> higher
    indexes.sort(key = lambda i: np.absolute(f[i]))

    t = np.arange(0, n + n_predict)
    restored_sig = np.zeros(t.size)
    for i in indexes[:1 + n_harm * 2]:
        ampli = np.absolute(x_freqdom[i]) / n   # amplitude
        phase = np.angle(x_freqdom[i])          # phase
        restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
    return (restored_sig + p[0] * t)


def timestamp_to_lst(tstamps):
    (latitude, longitude, elevation) = ('37.2397808', '-118.2816819', 1183.4839)

    ov = ephem.Observer()
    ov.lon = longitude
    ov.lat = latitude
    ov.elev = elevation

    lst_stamps = np.zeros_like(tstamps)
    utc_stamps = []
    for ii, tt in enumerate(tstamps):
        utc = datetime.utcfromtimestamp(tt)
        ov.date = utc
        lst_stamps[ii] = ov.sidereal_time() * 12.0 / np.pi
    utc_stamps.append(utc)
    return lst_stamps


def closest(x, x0):
    return np.argmin(np.abs(x - x0))
