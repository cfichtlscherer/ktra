"""
Jun 02, 2020
Christopher Fichtlscherer (fichtlscherer@mailbox.org)
GNU General Public License
"""

import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit

def line_integral_disk(x_1, y_1, x_2, y_2, r, density):
    
    fin = 10**4

    def disk(x, y, r, density):
        
        if x**2 + y**2 <= r**2:
            return density

        return 0
    
    #### IMPORTANT otypes=[np.float] otherwise int64 data type
    disk_vec = np.vectorize(disk, otypes=[np.float])

    def line(t):
        
        x = (1-t) * x_1 + t * x_2
        y = (1-t) * y_1 + t * y_2

        return x, y
    
    line_vec = np.vectorize(line)

    x_v, y_v = line_vec(np.linspace(0, 1, fin))

    points_eval = disk_vec(x_v, y_v, r, density)

    len_line = ((x_1 - x_2)**2 + (y_1 - y_2)**2)**0.5

    integral = len_line * np.sum(points_eval) / fin
   
    return integral
    
rays = 1000

line_ends = np.linspace(-2.309, 2.309, rays)

line_integral_disk_vec = np.vectorize(line_integral_disk)

density = 2.97
radius = 0.6

if False:
    res = line_integral_disk_vec(-2, 0, 2, line_ends, radius, density)
    k_exp = np.sum(res) / rays
    print("density: ", density, " - radius: ", radius, " - value: ", k_exp)
    quit()

# density:  1     - radius:  1    - value:  1.510956432839113
# density:  4.35  - radius:  0.5  - value:  1.5139304335767505
# density:  27.8  - radius:  0.2  - value:  1.51734090232264
# density:  1.26  - radius:  0.9  - value:  1.5081105710314349
# density:  1.63  - radius:  0.8  - value:  1.5125269286072152
# density:  2.17  - radius:  0.7  - value:  1.5170455509600607
# density:  2.97  - radius:  0.6  - value:  1.5051444732414143

r = [0.2, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
d = [27.8, 4.35, 2.97, 2.17, 1.63, 1.26, 1]

def func(x, a, b, c):

    return a * np.exp(b * x) + c

popt, pcov = curve_fit(func, r, d)

x = np.linspace(0.1, 1, 50)
y = func(x, *popt)

print(popt)

plt.plot(r, d, "bo")
plt.plot(x,y,"-r")
plt.grid(True)
plt.show()
