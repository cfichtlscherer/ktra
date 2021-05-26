"""
Apr 14, 2020
Christopher Fichtlscherer (fichtlscherer@mailbox.org)
GNU General Public License

Douglas-Rachford-Verfahren (Classon Skript p.69)
"""

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from tqdm import tqdm
import time as time

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

# print numpy arrays in one line     
np.set_printoptions(linewidth=np.inf)

def get_tv(array_oned):                                                
    """ calculates the 1d tv of a one-dimensional array"""             
                                                                       
    edges = abs(array_oned - np.roll(array_oned, 1))[1:]               
    tv_value = np.sum(edges)                                           
                                                                       
    return tv_value                                                    


def inner_of_box(x, y, z):
    """ Different shells around the origin."""

    rad_pos = (x**2 + y**2 + z**2)**0.5
    
    if rad_pos <= 0.8: return 0.8
   
    return 0


def y_function(values_1d, *args):                                       
    """the function we are optimizing - y = values_1d"""                                     
                                                                             
    liam_all, cont_results, gamma, x, z = args         

    value_results = k_trafo_one_dim_all(liam_all, values_1d)           
    value_difference_norm = np.linalg.norm(value_results - cont_results)**2  
    
    difference = np.linalg.norm(2 * x - z - values_1d)**2                                
    
    y =  0.5 * difference + 0.5 * gamma * value_difference_norm
    
    return y


def x_step(x, y, z, alpha, gamma):

    x = tv_denoising_algorithm(z, gamma * alpha)

    return x


def y_step(x, z, y, gamma, liam_all, cont_results_noise):

    y = scipy.optimize.minimize(y_function,
                                y,
                                args = (liam_all, cont_results_noise, gamma, x, z),
                                method = 'L-BFGS-B',
                                bounds = scipy.optimize.Bounds(0, 1),
                                options = {'disp': False})
    
    return y.x


def z_step(x, y, z):

    z = z + y - x
    
    return z


def reconstruction(steps, alpha, letter="a"):
    cuboid_coordinates = {'x1': -1, 'x2': 1, 'y1': -1, 'y2': 1, 'z1': -1, 'z2': 1}
    
    x_cor, y_cor, z_cor = coordinate_discretization(cuboid_coordinates, steps)
    dist = distance_creater(x_cor, y_cor, z_cor, steps)
    dist_array = dist_array_maker(dist)
    different_lengths = len(dist_array)
    
    number_ktrans = int(different_lengths*10)
    number_rays = {'dim_1': 9, 'dim_2': 9}
    fineness = 10**3
    radii = np.linspace(2,30, different_lengths) #[2.1, 2.3, 2.5, 2.7, 2.9, 3.1, 3.3, 3.5]
    
    start_index = [i * int(number_ktrans / len(radii)) for i in range(len(radii))]
    end_index = [i * int(number_ktrans / len(radii)) for i in range(1, len(radii) + 1)]
    
    source_point_d = {}
    
    ################################################################################
    for i, r in enumerate(radii):
        new_source_points = create_source_point_d(r, end_index[i], perc_circle=0.125, start_index=start_index[i])
        source_point_d.update(new_source_points)
    
    number_ktrans = len(source_point_d) 
    print("1/8")
    
    rays_d = generate_rays_d(source_point_d, cuboid_coordinates, number_rays)
    print("2/8")
    
    if True:
        liam_all = generate_all_line_integral_array_matrix(rays_d, cuboid_coordinates, steps, fineness)
        np.save("liam_all", liam_all)
        print("3/8")
    
    if True:
        cont_results = create_con_results(rays_d, inner_of_box, fineness)
        np.save("cont_results", cont_results)
    print("4/8")
    
    #compare_values = compare_values_original_function(cuboid_coordinates, steps, inner_of_box)
    print("5/8")
    
    x_cor, y_cor, z_cor = coordinate_discretization(cuboid_coordinates, steps)
    print("6/8")
    
    values = discretization_of_model(x_cor, y_cor, z_cor, inner_of_box, steps)
    print("7/8")
    
    x = np.ones(different_lengths)
    y = np.ones(different_lengths)
    z = np.ones(different_lengths)
   
    alpha_not_normed = alpha
    alpha /= x.size
    gamma = 1 
    perc_noise = 0.01
    
    cont_results = np.load("cont_results.npy")  
    liam_all = np.load("liam_all.npy")
    
    print("8/8")
    v1d = threed_to_oned(values, cuboid_coordinates, steps)
    
    cont_results_noise = cont_results + perc_noise * np.max(cont_results) * 2 * (np.random.random(number_ktrans) - 0.5)
    

    ################################################################################
    ####################### Make wD data for comparing##############################
    ################################################################################
    x_comp = np.arange(-1, 1, 0.01)                  
    y_comp = np.arange(-1, 1, 0.01)                  
                                                
    cont = np.zeros((x_comp.size, y_comp.size))           
                                                
    for i in range(x_comp.size):                     
        for j in range(y_comp.size):                 
            cont[i,j] = inner_of_box(x_comp[i], y_comp[j], 0)
    
    ################################################################################
    
    for i in range(5000):
        
        x_old, y_old, z_old = x, y, z
    
        x = x_step(x, y, z, alpha, gamma)
        y = y_step(x, z, y, gamma, liam_all, cont_results_noise)
        z = z_step(x, y, z)
        
        tv_reg_value = get_tv(y)                                                                 
        f = open("logbook" + "-" + str(steps) + "-" + letter + "-" + str(alpha_not_normed) + ".txt", "a+")     
        f.write(str(i).ljust(8) + str(tv_reg_value).ljust(20) + str(y) + "\n" )                  
       
        print(i, "--------", np.linalg.norm(y - y_old))
        
        if i % 100 == 0:
            results3d = make_threed_back(x, steps, cuboid_coordinates)
    
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7,3.5))
            ax[0].imshow(cont, vmin=0, vmax=1.2)                     
            ax[0].axis(False)                                        
            ax[1].imshow(results3d[int(steps/2)], vmin=0, vmax=1.2)             
            ax[1].axis(False)                                        
            plt.savefig(letter + "-" + str(steps)+ "-" + str(alpha_not_normed) + "-" + str(i) + ".png")
            plt.close()


alphas =  [0.0, 0.1, 0.15, 0.2, 0.25, 0.3]
letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "l", "m"]

for steps in [20]:                                                                                  
    for i in range(5):
        reconstruction(steps, alphas[i], letters[i])                                                                   
