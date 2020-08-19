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


# plot normal
# plot with jit
# plot vectorization

values = np.load("results_different_number_arrays.npy")

abs_dif = [ abs(values[i] - values[i+1]) for i in range(48)]

plt.figure(figsize=(0.7 * 6, 0.7 * 3.4))
plt.plot(np.arange(1, 26), 25*[values[48]], marker="", label="Continuous K-transform")
plt.plot(np.arange(1, 26), values[:25], marker=".", label="Discretized K-transform")
#plt.axhline(y=values[48], color='red', linestyle='--')#, label"Continuous K-transform measurement value")
plt.grid(color='black', linestyle=':', linewidth=0.3)
plt.legend()
plt.xlabel('Number of rays')
plt.ylabel('Value')
positionsx = (5, 10, 15, 20, 25)
labelsx = (r"$5 \times 5$", r"$10 \times 10$", r"$15 \times 15$",
        r"$20 \times 20$", r"$25 \times 25$")
plt.xticks(positionsx, labelsx)

positionsy = (0.4, 0.6, 0.8, values[48], 1.0)
labelsy = (0.4, 0.6, 0.8, np.round(values[48],2), 1.0)
plt.yticks(positionsy, labelsy)



plt.savefig(figures_path + 'number_rays_change.pgf', bbox_inches='tight')
