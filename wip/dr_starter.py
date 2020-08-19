"""
May 14, 2020
Christopher Fichtlscherer (fichtlscherer@mailbox.org)
GNU General Public License

DR is now a function, with this file we can start this function multiple times with the purpose
to find good parameters for alpha and gamma
"""

import numpy as np                                                     
import scipy.optimize                                                  
import matplotlib.pyplot as plt                                        
from tqdm import tqdm                                                  
import os                                                                   
from multiprocessing import Pool                                                

from ktra.k_transform import create_source_point_d                     
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
                                                                       
from douglas_rachford import *

gamma, alpha, perc_noise = 10**4, 0.0000, 0

dr_process(gamma, alpha, perc_noise)
