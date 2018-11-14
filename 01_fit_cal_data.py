import hickle as hkl
from leda_cal2.fileio import *
import pylab as plt

for ant_id in ['252A', '254A', '255A', '254B', '255B']:
    a = hkl.load('cal_data_%s.hkl' % ant_id)

    print a.keys()
    # ['T_H', 'scale', 'T_C', 'G_S', 'offset', 'f_mhz']

    f   = a['f_mhz']
    T_H = a['T_H']
    T_Hf = poly_fit(f, T_H, 3)

    T_C = a['T_C']
    T_Cf = poly_fit(f, T_C, 3)

    T_NW = a['T_NW']
    T_NWf = fourier_fit(T_NW, 0, 101)

    #scale = a['scale']
    #scalef = poly_fit(f, scale, 11)
    #
    #offset = a['offset']
    #offsetf = poly_fit(f, offset, 7, log=False)

    G_S = a['G_S']
    G_Sf = poly_fit(f, G_S, 11, log=False)

    b = {'f_mhz': f,
         'T_H': T_Hf,
         'T_C': T_Cf,
         #'scale': scalef,
         #'offset': offsetf,
         'T_NW': T_NW,
         'G_S': G_Sf}

    print 'cal_data_%s_fit.hkl' % ant_id
    hkl.dump(b, 'cal_data_%s_fit.hkl' % ant_id)

    plt.figure("T_H")
    plt.subplot(211)
    plt.plot(f, T_H)
    plt.plot(f, T_Hf)
    plt.subplot(212)
    plt.plot(f, T_H - T_Hf)
    plt.savefig("img/fit_fee_nd_hot_%s.png" % ant_id)

    plt.figure("T_C")
    plt.subplot(211)
    plt.plot(f, T_C)
    plt.plot(f, T_Cf)
    plt.subplot(212)
    plt.plot(f, T_C - T_Cf)
    plt.savefig("img/fit_fee_nd_cold_%s.png" % ant_id)

    #plt.figure("scale")
    #plt.subplot(211)
    #plt.plot(f, scale)
    #plt.plot(f, scalef)
    #plt.subplot(212)
    #plt.plot(f, scale - scalef)
    #
    #plt.figure("offset")
    #plt.subplot(211)
    #plt.plot(f, offset)
    #plt.plot(f, offsetf)
    #plt.subplot(212)
    #plt.plot(f, offset - offsetf)

    plt.figure("G_S")
    plt.subplot(211)
    plt.plot(f, G_S)
    plt.plot(f, G_Sf)
    plt.subplot(212)
    plt.plot(f, G_S - G_Sf)
    plt.savefig("img/fit_fee_G_S_%s.png" % ant_id)

    plt.figure("T_NW")
    plt.subplot(211)
    plt.plot(f, T_NW)
    plt.plot(f, T_NWf)
    plt.subplot(212)
    plt.plot(f, T_NW - T_NWf)
    plt.savefig("img/fit_NW_cold_%s.png" % ant_id)

    plt.show()
