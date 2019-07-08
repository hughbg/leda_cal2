import numpy as np
import matplotlib.pyplot as plt

def plot_coeffs(fname, labels, title):

  with open(fname) as f:
    for i, l in enumerate(f):
      pass
  num_lines = i+1

  data = np.ma.zeros((len(labels), num_lines))

  with open(fname) as f:
    for i, line in enumerate(f):
      if line.find("failed") == -1:
        l = line.split()
        for j in range(len(labels)):
          data[j, i] = float(l[4+2*j])
      else: 
        for j in range(len(labels)):
          data[j, i] = np.nan


  for i in range(len(labels)):
    plt.clf()
    plt.plot(data[i], linewidth=0.5)
    plt.title(title+labels[i])
    plt.ylabel("Value") 
    plt.xlabel("Time sequence id")
    if min(data[i]) > 0: plt.ylim(ymin=0)
    if max(data[i]) < 0: plt.ylim(ymax=0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(labels[i]+".png")

  data = np.ma.masked_where(np.isnan(data), data)

  cov = np.ma.cov(data[:, :500])

  cov_s = "   "
  for l in labels: cov_s += ( "%17s   " % l )
  cov_s += "\n\n"
  for i in range(len(labels)):
    cov_s += ( "%3s" % labels[i] )+" "
    for j in range(cov.shape[1]):
      cov_s += ( "%20.10f" % cov[i, j] )
    cov_s += "\n"

  corr = np.ma.corrcoef(data[:, :500])

  corr_s = "   "
  for l in labels: corr_s += ( "%17s   " % l )
  corr_s += "\n\n"
  for i in range(len(labels)):
    corr_s += ( "%3s" % labels[i] )
    for j in range(corr.shape[1]):
      corr_s += ( "%20.10f" % corr[i, j] )
    corr_s += "\n"

  return cov_s, corr_s

d_cov, d_corr = plot_coeffs("damped_sin_coeff.txt", [ "w_0", "w_1", "w_2", "a", "b", "c", "d", "f" ], "Damped sin fit coeff: ")
p_cov, p_corr = plot_coeffs("poly_coeff.txt", [ "p0", "p1", "p2", "p3" ], "Polynomial coeff: ")

f = open("coeff_plot.html", "w")
f.write("Ant 254. The x-axis in these plots is just different times, in order by time.<h2>Log Polynomial fit coefficients</h2>This the 3rd order fit of the signal. There are four  coefficients p0, p1, p2, p3\n")
f.write("in order of decreasing powers, as by the <a href=https://docs.scipy.org/doc/numpy/reference/generated/numpy.poly1d.html>numpy poly1d documentation</a><p>.\n")
f.write("<img width=1000 src=p0.png><p><img width=1000 src=p1.png><p><img width=1000 src=p2.png><p><img width=1000 src=p3.png><p>\n\n")
f.write("<pre>\n"+p_cov+"\n\n"+p_corr+"</pre>\n")
f.write("<h2>Damped sinusoid fit coefficients</h2>The equation that I am fitting is (in Python): w_0 + w_1*(x-70) + w_2*(x-70)**2+(a**(-x))*b*np.sin(c*x+d+f/x).\n")
f.write("This is the damped sinusoid from Schinzel with e^(-a x) turned into a^(-x) which optimizes better; and the DC offset term replaced with a second order polynomial to allow for baseline wander.<p>\n")
f.write("<img width=1000 src=w_0.png><p><img width=1000 src=w_1.png><p><img width=1000 src=w_2.png><p><img width=1000 src=a.png><p><img width=1000 src=b.png><p><img width=1000 src=c.png><p><img width=1000 src=d.png><p><img width=1000 src=f.png>\n\n")
f.write("<pre>\n"+d_cov+"\n\n"+d_corr+"</pre>\n")
f.close()


