from leda_cal2.utils import *

class Spectrum(object):
    """ Class for reading spectrometer dumps

    Provides interpolation function and basic plotting of data
    """
    def __init__(self, filename):
        d = np.loadtxt(filename)
        self.filename = filename
        self.f = d[:, 0]
        self.d = interp(d[:,0], d[:, 1])
    def plot(self, log=True):
        if log:
            plt.plot(self.f, 10*np.log10(self.d(self.f)))
        else:
            plt.plot(self.f, self.d(self.f))
        plt.xlabel("Frequency [MHz]")

def read_spectrum(filename):
    return Spectrum(filename)

class Sparam(object):
    """ Basic class for interpolating and plotting S-parameter data """

    def __init__(self, f=None, s11=None, s21=None, s12=None, s22=None):
        self.f    = f
        self.s11  = s11
        self.s21  = s21
        self.s22  = s22
        self.s12  = s12

    def plot_s11(self):
        """ Plot S11 data """
        s11 = self.s11(self.f)
        s11_mag = 20*np.log10(np.abs(s11))
        s11_phs = np.rad2deg(np.angle(s11))
        plt.subplot(1,2,1)
        plt.plot(self.f, s11_mag)
        plt.subplot(1,2,2)
        plt.plot(self.f, s11_phs)

    def plot_s21(self):
        """ Plot S21 data """
        s21 = self.s21(self.f)
        s21_mag = 20*np.log10(np.abs(s21))
        s21_phs = np.rad2deg(np.angle(s21))
        plt.subplot(1,2,1)
        plt.plot(self.f, s21_mag)
        plt.subplot(1,2,2)
        plt.plot(self.f, s21_phs)


def read_anritsu_s11(filename):
    """ Read data from Anristu VNA and return instance of Sparam class """
    d = np.genfromtxt(filename, delimiter=',', skip_header=8, skip_footer=1)
    f_mhz = d[:, 0] / 1e6
    s11 = interp(f_mhz, to_complex(d[:,1:], linear=False, radians=False))
    S = Sparam(f_mhz, s11)
    return S

def read_cable_sparams(filename):
    """ Read data from cable CSV files and return instance of Sparam class """
    d = np.genfromtxt(filename, delimiter=',', skip_header=1, skip_footer=0)
    f = d[:, 0]
    s11 = interp(f, to_complex(d[:,1:3], linear=False, radians=False))
    s21 = interp(f, to_complex(d[:,3:5], linear=False, radians=False))
    s22 = interp(f, to_complex(d[:,5:7], linear=False, radians=False))
    S = Sparam(f, s11=s11, s21=s21, s22=s22)
    return S

def read_uu_sparams(filename):
    """ Read data from cable CSV files and return instance of Sparam class """
    d = np.genfromtxt(filename, skip_header=0, skip_footer=0)
    f = d[:, 0]
    s21 = interp(f, to_complex(d[:,3:5], linear=False, radians=False))
    S = Sparam(f, s21=s21)
    return S

def read_s2p_s11(filename, s11_col=7):
    """ Read data from S2P VNA file and return instance of Sparam class.

    Only reads S11 data, defaults to column 7 (sometimes it is column 1)
    """
    c = s11_col
    with open(filename) as fh:
        data = np.loadtxt(fh.readlines()[23:])		# Load data and ignore header
    f_mhz  = data[:, 0] * 1e3 # To MHz
    s11    = to_complex(data[:, c:c+2], linear=False, radians=False)
    s11_i  = interp(f_mhz, s11)
    return Sparam(f_mhz, s11_i)
