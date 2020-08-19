"""
Jun 20, 2020
Christopher Fichtlscherer (fichtlscherer@mailbox.org)
GNU General Public License

plotting the gaußian and the poisson distribution for the
presentation.
"""

import numpy as np
import matplotlib.pyplot as plt 
from scipy import signal
from scipy import stats
from scipy.special import factorial

if True:
    window = signal.gaussian(51, std=7)
    plt.plot(window)
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    frame1.axes.yaxis.set_ticklabels([])
    plt.grid(color='black', linestyle=':', linewidth=0.3)                               
    #plt.plot([25, 25], [0, 1], ':', color = "black")
    plt.show()



t = np.arange(0, 15, 0.01)
d = np.exp(-3)*np.power(3, t)/factorial(t)

frame1 = plt.gca()
frame1.axes.xaxis.set_ticklabels([])
frame1.axes.yaxis.set_ticklabels([])

plt.plot(t, d)
plt.grid(color='black', linestyle=':', linewidth=0.3)                               
#plt.plot([5., 5.], [0, 0.2], ':', color = "black")
plt.show()
