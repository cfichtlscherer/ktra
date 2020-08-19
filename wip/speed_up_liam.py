"""
Apr 23, 2020
Christopher Fichtlscherer (fichtlscherer@mailbox.org)
GNU General Public License

we try to figure out ways how liam can be computed faster

"""

import sympy
import numpy as np

from ktra.k_transform import create_source_point_d                     
from ktra.k_transform import generate_rays_d                           
from ktra.k_transform import generate_all_line_integral_array_matrix   
from ktra.k_transform import coordinate_discretization                 
from ktra.k_transform import discretization_of_model                   
from ktra.k_transform import k_trafo_one_dim_all                       
                                                                       
from ktra.tv_denoising_condat import tv_denoising_algorithm            
                                                                       
from ktra.optimization_process import create_con_results               
from ktra.optimization_process import compare_values_original_function 
from ktra.optimization_process import important_mini_cuboids           
from ktra.optimization_process import threed_to_oned                   
from ktra.optimization_process import v_step_function                  



steps = 4
number_rays = {'dim_1': 4, 'dim_2': 4}
number_ktrans = 10

cuboid_coordinates = {'x1': -1, 'x2': 1, 'y1': -1, 'y2': 1, 'z1': -1, 'z2': 1}
fineness = 10**3
radius = 3                           # from which the source points are took

source_point_d = create_source_point_d(radius, number_ktrans)
rays_d = generate_rays_d(source_point_d, cuboid_coordinates, number_rays)
liam_all = generate_all_line_integral_array_matrix(rays_d, cuboid_coordinates, steps, fineness)

