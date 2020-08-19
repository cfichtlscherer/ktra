"""
Jun 02, 2020
Christopher Fichtlscherer (fichtlscherer@mailbox.org)
GNU General Public License
"""

import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit

def line_integral_box(x_1, y_1, x_2, y_2, density):
    
    fin = 10**4

    def box(x, y, density):
        
        if (0 <= x <= 1) and (0 <= y <= 1):
            return density

        return 0
    
    #### IMPORTANT otypes=[np.float] otherwise int64 data type
    box_vec = np.vectorize(box, otypes=[np.float])

    def line(t):
        
        x = (1-t) * x_1 + t * x_2
        y = (1-t) * y_1 + t * y_2

        return x, y
    
    line_vec = np.vectorize(line)

    x_v, y_v = line_vec(np.linspace(0, 1, fin))

    points_eval = box_vec(x_v, y_v, density)

    len_line = ((x_1 - x_2)**2 + (y_1 - y_2)**2)**0.5

    integral = len_line * np.sum(points_eval) / fin
   
    return integral
    
rays = 4000
radius = 6

line_ends_x_list = [np.cos(2*np.pi/rays*x)*radius for x in range(rays+1)]
line_ends_x = np.asarray(line_ends_x_list)

line_ends_y_list = [(np.sin(2*np.pi/rays*x)*radius + 0.5) for x in range(rays+1)]
line_ends_y = np.asarray(line_ends_y_list)

#line_ends = np.linspace(-2.75, 2.75, rays)

line_integral_box_vec = np.vectorize(line_integral_box)

density = 0.5

res = line_integral_box_vec(-3, 0.5, line_ends_x, line_ends_y, density)
k_exp = np.exp(- np.sum(res) / rays)
print("K - value: ", k_exp)

