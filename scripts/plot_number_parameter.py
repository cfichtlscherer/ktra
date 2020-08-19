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

from numba import jit

matplotlib.use("pgf")           
matplotlib.rcParams.update({    
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',     
    'text.usetex': True,        
    'pgf.rcfonts': False,       
})                              

figures_path = '/home/cpf/Desktop/k_transform_tomography/figures/'

even = [1, 4, 9, 15, 26, 36, 49, 66, 85, 103, 130, 152, 180, 214, 247, 276, 320, 362,
        399, 444, 495, 542, 601, 652, 705]
odd = [1, 4, 10, 19, 32, 45, 67, 88, 116, 145, 179, 212, 260, 300, 347, 402, 464, 517,
        592, 649, 727, 803, 886, 953, 1057]

plt.figure(figsize=(6,3.4))
plt.plot(np.arange(1, 50, 2), odd, "-", marker=".", label="Level of discretization is an odd number")
plt.plot(np.arange(2, 51, 2), even, "-", marker=".", label="Level of discretization is an even number")
plt.grid(color='black', linestyle=':', linewidth=0.3)
plt.legend(loc=(0.03, 0.74))
plt.xlabel('Level of discretization [$n$]')
plt.ylabel('Number of voxels 1D model [$N$]')


plt.savefig(figures_path + 'number_variable.pgf', bbox_inches='tight')
