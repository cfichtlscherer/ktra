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

        if x**2 + y**2 <= 0.55**2:
            return 0

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
line_ends = np.linspace(-2.25, 2.25, rays)
line_integral_disk_vec = np.vectorize(line_integral_disk)

if False:

    density = 0.5
    radius = 0.75

    res = line_integral_disk_vec(-3.75, 0, 3.25, line_ends, radius, density)
    k_transform = np.sum(np.exp(-res)) / rays
    print("Example-1 ", k_transform) # 0.724


if False:
    line_integral_disk_vec = np.vectorize(line_integral_disk)

    density = 0.2
    radius = 0.75

    res = line_integral_disk_vec(-3.75, 0, 3.25, line_ends, radius, density)
    k_transform = np.sum(np.exp(-res)) / rays
    print("Example-2 ", k_transform) # 0.869

if False:

    density = 0.5
    radius = 0.75

    res = line_integral_disk_vec(-6.05, 0, 0.95, line_ends, radius, density)
    k_transform = np.sum(np.exp(-res)) / rays
    print("Example-3 ", k_transform) # 0.831


if True:
    
    # To-Do: define the disc differently !!
    density = 1
    radius = 0.75

    res = line_integral_disk_vec(-3.75, 0, 3.25, line_ends, radius, density)
    k_transform = np.sum(np.exp(-res)) / rays
    print("Example-4 ", k_transform) # 0.724

