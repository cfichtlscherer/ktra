"""
Aug 24, 2021
Christopher Fichtlscherer (fichtlscherer@mailbox.org)
GNU General Public License

Here we calculate the derivation of the K-transform for different examples numerical.
We start by calculating the change of the K-transform value as a function of the change
of the object value.
"""

import numpy as np                                                           
import scipy.optimize                                                        
from tqdm import tqdm                                                        
                                                                             
from ktra.k_transform import coordinate_discretization                       
from ktra.k_transform import discretization_of_model                         
from ktra.k_transform import create_source_point_d                           
from ktra.k_transform import generate_rays_d                                 
#from ktra.optimization_process import create_con_results 

from ktra.optimization_process import k_trafo_cont
from ktra.optimization_process import k_trafo_derivation

def inner_of_box(x, y, z):                                                                         
    """ Different shells around the origin."""                                                       
                                                                                                     
    rad_pos = (x**2 + y**2 + z**2)**0.5                                                              
                                                                                                     
    if rad_pos <= 0.3: return 0.9                                                                    
    if rad_pos <= 0.7: return 0.6                                                                    
    if rad_pos <= 0.9: return 0.3                                                                    
                                                                                                     
    return 0                                                                                         

def direction(x, y, z):                                                                         
    """ Different shells around the origin."""                                                       
                                                                                                     
    rad_pos = (x**2 + y**2 + z**2)**0.5                                                              
                                                                                                     
    if rad_pos <= 0.3: return -0.5
                                                                                                     
    return 0                                                                                         

number_ktrans = 1 
number_rays = {'dim_1': 5, 'dim_2': 5}                                                               
fineness = 10**3                                                                                     
radius = 3                                                                                       
cuboid_coordinates = {'x1': -1, 'x2': 1, 'y1': -1, 'y2': 1, 'z1': -1, 'z2': 1}                   
                                                                                                 
source_point_d = create_source_point_d(radius, number_ktrans, perc_circle=0.125)                 
rays_d = generate_rays_d(source_point_d, cuboid_coordinates, number_rays)                        

rays = rays_d[0]

da = k_trafo_derivation(rays, inner_of_box, direction, fineness)
print("derivation analytical: ", da)

h = 0.01

def sum_objects(x,y,z):

    return inner_of_box(x, y, z) + h * direction(x, y, z)

dn = (k_trafo_cont(rays, sum_objects, fineness) - k_trafo_cont(rays, inner_of_box, fineness)) / h
print("derivation numerical: ", dn)


