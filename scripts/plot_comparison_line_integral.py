"""
Jul 10, 2020
Christopher Fichtlscherer (fichtlscherer@mailbox.org)
GNU General Public License

In this plot the 
"""

import matplotlib
import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt

import time as time

from ktra.k_transform import create_rays
from ktra.k_transform import generate_rays_d                                
from ktra.k_transform import generate_all_line_integral_array_matrix        
from ktra.k_transform import coordinate_discretization                      
from ktra.k_transform import discretization_of_model                        
from ktra.k_transform import k_trafo_one_dim_all                            
from ktra.k_transform import create_line_matrix                             
                                                                            
from ktra.k_transform import distance_creater                               
from ktra.k_transform import dist_array_maker                               
                                                                            
from ktra.tv_denoising_condat import tv_denoising_algorithm                 
                                                                            
from ktra.optimization_process import create_con_results                    
from ktra.optimization_process import compare_values_original_function      
from ktra.optimization_process import important_mini_cuboids                
from ktra.optimization_process import threed_to_oned                        
from ktra.optimization_process import v_step_function                       
from ktra.optimization_process import how_many_got_hit                      
from ktra.optimization_process import generate_everything                   
from ktra.optimization_process import make_threed_back                      
                                                                            
from ktra.optimization_process import line_integral_cont

from define_colors import *

matplotlib.use("pgf")           
matplotlib.rcParams.update({    
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',     
    'text.usetex': True,        
    'pgf.rcfonts': False,       
})                              

figures_path = '/home/cpf/Desktop/k_transform_tomography/figures/'

color1 = colors[3]
color2 = colors[4]

# plot normal
# plot with jit
# plot vectorization

n_steps = 101

if False:
    def inner_of_box(x, y, z):
        if x**2 + y**2 + z**2 < 0.2: return 1
        if x**2 + y**2 + z**2 < 0.4: return 0.7
        if x**2 + y**2 + z**2 < 0.6: return 0.5
        if x**2 + y**2 + z**2 < 0.7: return 0.2
        return 0
    
    source_point_d = {"0": np.array([3,0,0])}
    
    list_1 = []
    list_2 = []
    
    for steps in range(1, n_steps):
        print(steps)                                                      
        cuboid_coordinates = {'x1': -1, 'x2': 1, 'y1': -1, 'y2': 1, 'z1': -1, 'z2': 1}
                                                                                  
        x_cor, y_cor, z_cor = coordinate_discretization(cuboid_coordinates, steps)    
        dist = distance_creater(x_cor, y_cor, z_cor, steps)                           
                                                                                  
        number_rays = {'dim_1': 1, 'dim_2': 1}                                        
        fineness = 10**4 
        rays_d = generate_rays_d(source_point_d, cuboid_coordinates, number_rays)
        
        t1 = time.time()
        for i in range(200):
            line_integral_cont(rays_d[0][0], inner_of_box, fineness=10**4)
        t2 = time.time()
    
        liam_all = generate_all_line_integral_array_matrix(rays_d, cuboid_coordinates, steps,
                fineness)
    
        discrete_model = discretization_of_model(x_cor, y_cor, z_cor, inner_of_box, steps)
        one_d_model = threed_to_oned(discrete_model, cuboid_coordinates, steps)           
        
        threedma = make_threed_back(liam_all[0][0], steps, cuboid_coordinates)

        t3 = time.time()
        for i in range(200):
            np.sum(threedma * discrete_model)
        t4 = time.time()
    
        list_1 += [(t2-t1)/200.]
        list_2 += [(t4-t3)/200.]

    if (steps % 5 == 0):   
       np.save("for_loop_times.npy", np.asarray(list_1)) 
       np.save("matric_times.npy", np.asarray(list_2))

for_loop_times = np.load("for_loop_times.npy")
matrix_times = np.load("matric_times.npy")

for_loop_average = np.average(for_loop_times) * np.ones(for_loop_times.size)

plt.figure(figsize=(0.7 * 6, 0.7 * 3.4))

#plt.axhline(y=values[48], color='red', linestyle='--')
plt.plot(np.arange(1, n_steps), for_loop_average/10*10**3, "-", marker="", label="Calculation via loop")
plt.plot(np.arange(1, n_steps), matrix_times*10**3, "-", marker="", label="Calculation via :")
plt.grid(color='black', linestyle=':', linewidth=0.3)
plt.legend(loc=(0.45, 0.65))
plt.xlabel('Level of Discretization [n]')
plt.ylabel('Calculation Time [ms]')

#positionsy = (0.000, 0.002, 0.004, 0.006, 0.008, np.round(for_loop_average[0]/10, 4))
positionsy = (0.0, 2.0, 4.0, 6.0, 8.0, 8.9)
plt.yticks(positionsy, positionsy)

plt.xticks((0, 16, 24, 40, 60, 80, 100))

plt.axvline(x=16, linestyle=":", color="black")
plt.axvline(x=24, linestyle=":", color="black")

plt.savefig(figures_path + 'line_integral_calculation_times.pgf', bbox_inches='tight')

# PLOT WITH TWO AXIS

#
#fig, ax1 = plt.subplots(figsize=(6,3.4))
#
#ax1.grid(color='black', linestyle=':', linewidth=0.3)
#
#ax1.set_xlabel('Discretization Level')
#ax1.set_ylabel('Calculation time loop [s]', color="blue")
#ax1.plot(np.arange(1,n_steps), for_loop_times, marker = ".", color="blue")
#ax1.tick_params(axis='y', labelcolor="blue")
#
#ax2 = ax1.twinx()
#
#ax2.set_xlabel('Discretization Level')
#ax2.set_ylabel('Calculation time matrices [s]', color="green")
#ax2.plot(np.arange(1,n_steps), matrix_times, marker = ".", color="green")
#ax2.tick_params(axis='y', labelcolor="green")
#




























