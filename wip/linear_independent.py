"""
Apr 23, 2020
Christopher Fichtlscherer (fichtlscherer@mailbox.org)
GNU General Public License

here I try to construct a method to get only linear independent values in 
the liam_all term

vielleicht einfacher erstmal die doppelten löschen
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


steps = 8
number_rays = {'dim_1': 1, 'dim_2': 1}
number_ktrans = 400
number_alphas = 20
number_betas = 20

cuboid_coordinates = {'x1': -1, 'x2': 1, 'y1': -1, 'y2': 1, 'z1': -1, 'z2': 1}
fineness = 10**3
radius = 3                           # from which the source points are took

def prime_numpy(start, n):
    """returns an array of the first n prime numbers but only prime numbers under 10**6"""
    
    prime_array = np.array([])
    x = np.arange(start, 10**6)
    
    while True:
        prime_array = np.append(prime_array, x[0])
        x = x[(x % x[0] != 0)]
        if len(prime_array) == n:
            return(prime_array)

def create_source_point_sphere_d(radius, number_ktrans, number_alphas, number_betas):                                         
    """ creates a dictionary of source points, instead of create_source_point_d this 
    function places the source_points of a sphere instead of only a circle with the 
    hope that we gain by that more linear independent liam arrays
    
    spherical coordinates:

    alpha in [0, pi]
    beta in [0, 2pi]
    
    x = r sin(alpha) * cos(beta)
    y = r sin(alpha) * cos(alpha)
    z = r cos(alpha)
    """
    
    if number_alphas * number_betas != number_ktrans:
        print("ERROR: number_alphas * number_betas != number_ktrans")
        quit()

    source_point_d = {}                                                                                  
    
    counter = 0

    alpha_array = prime_numpy(2, number_alphas)
    beta_array = prime_numpy(541, number_betas)

    for i in alpha_array:
        for j in beta_array:
            alpha = i * np.pi / alpha_array[-1]
            beta = j * 2 * np.pi / beta_array[-1] 

            x = radius * np.sin(alpha) * np.cos(beta)
            y = radius * np.sin(alpha) * np.sin(beta)
            z = radius * np.cos(alpha)

            source_point_d[str(counter)] = np.array([x, y, z])                                                  
            counter += 1

    return source_point_d


def create_source_points_random(radius, number_ktrans):
    """ creates the source points, but random"""

    source_point_d = {}

    for i in range(number_ktrans):

        x = np.random.random() * 3
        y = np.random.random() * 3
        z = (radius**2 - x**2 - y**2)**0.5

        source_point_d[str(i)] = np.array([x,y,z])

    return source_point_d

    
"""
#print(create_source_points_random(radius, number_ktrans))

source_point_d = create_source_points_random(radius, number_ktrans)
rays_d = generate_rays_d(source_point_d, cuboid_coordinates, number_rays)
liam_all = generate_all_line_integral_array_matrix(rays_d, cuboid_coordinates, steps, fineness)

liam_all_flatten = liam_all.flatten()
liam_all_reshaped = liam_all_flatten.reshape(number_ktrans, 15)

data = np.round(liam_all_reshaped, 5)

sorted_idx = np.lexsort(data.T)
sorted_data =  data[sorted_idx,:]
row_mask = np.append([True],np.any(np.diff(sorted_data,axis=0),1))
out = sorted_data[row_mask]

print(data)
print("----------------------------------------------------------")
print(out)
print("number_ktrans: ", number_ktrans)
print("linear independent: ", len(out))
#print(liam_all_reshaped)
"""
"""
print(liam_all)
print("------------------------------------------")
first_mat = liam_all[0]
print(first_mat)
print("------------------------------------------")
sortedmat = first_mat[np.argsort(first_mat[:, 0])]
print(sortedmat)


"""
#_, inds = sympy.Matrix(liam_all_reshaped).T.rref()
#print(inds)

"""
mat = np.array([[0,1,0,0],[0,0,1,0],[0,1,1,0],[1,0,0,1]])
_, inds = sympy.Matrix(mat).T.rref()
print(inds)
print(mat[list(inds)])
"""
