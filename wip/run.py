"""
Apr 14, 2020
Christopher Fichtlscherer (fichtlscherer@mailbox.org)
GNU General Public License

In the prerun the ten objects:
source_point_d, rays_d, liam_all, cont_results, compare_values, x_cor, y_cor, z_cor, values, touched
are created and saved with pickle
"""

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from tqdm import tqdm


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

from linear_independent import prime_numpy
from linear_independent import create_source_point_sphere_d

from linear_independent import create_source_points_random

# print numpy arrays in one line
np.set_printoptions(linewidth=np.inf)


def change_in_solution(old_solution, new_solution):
    """ prints the 1-norm / 2-norm / inf-norm between the solution and the solution of the last
    step in the iterative splitting method """

    one_norm = np.linalg.norm(old_solution - new_solution, 1)
    two_norm = np.linalg.norm(old_solution - new_solution, 2)
    inf_norm = np.linalg.norm(old_solution - new_solution, np.inf)
    
    print(str(one_norm).ljust(25), str(two_norm).ljust(25), str(inf_norm).ljust(25))


def inner_of_box(x, y, z):
    """ Different shells around the origin."""

    rad_pos = (x**2 + y**2 + z**2)**0.5
  
    if rad_pos <= 0.8: return 0.0
    if rad_pos <= 1.0: return 0.8 
   
    return 0


steps = 20

cuboid_coordinates = {'x1': -1, 'x2': 1, 'y1': -1, 'y2': 1, 'z1': -1, 'z2': 1}

x_cor, y_cor, z_cor = coordinate_discretization(cuboid_coordinates, steps)
dist = distance_creater(x_cor, y_cor, z_cor, steps)
dist_array = dist_array_maker(dist)
different_lengths = len(dist_array)

number_ktrans = different_lengths * 2
number_rays = {'dim_1': 7, 'dim_2': 7}
fineness = 10**3
radius = 3

################################################################################
source_point_d = create_source_point_d(radius, number_ktrans, perc_circle=0.125)
print(source_point_d)
quit()
print("1/8")

rays_d = generate_rays_d(source_point_d, cuboid_coordinates, number_rays)
print(rays_d)
quit()
print("2/8")

liam_all = generate_all_line_integral_array_matrix(rays_d, cuboid_coordinates, steps, fineness)
np.save("liam_all", liam_all)
#liam_all = np.load("liam_all.npy")
print("3/8")

cont_results = create_con_results(rays_d, inner_of_box, fineness)
np.save("cont_results", cont_results)
# cont_results = np.load("cont_results.npy")
# cont_results = np.average(cont_results) * np.ones(cont_results.size)
print("4/8")

compare_values = compare_values_original_function(cuboid_coordinates, steps, inner_of_box)
print("5/8")

x_cor, y_cor, z_cor = coordinate_discretization(cuboid_coordinates, steps)
print("6/8")

values = discretization_of_model(x_cor, y_cor, z_cor, inner_of_box, steps)
print("7/8")

# discrete_model = discretization_of_model(x_cor, y_cor, z_cor, inner_of_box, steps)             
# one_d_model = threed_to_oned(discrete_model, cuboid_coordinates, steps)                                                                                                                       
# dis_results = k_trafo_one_dim_all(liam_all, one_d_model)                                       

#cont_results = dis_results
#np.save("cont_results", cont_results)

perc_noise = 0.0001
lamb = 0.001
rho = 0.001

cont_results_noise = (np.load("cont_results.npy") +
                        2 * perc_noise * (np.random.random(number_ktrans) - 0.5))



################################################################################
u = 0.5 * np.ones(different_lengths)
for i in range(1000):
################################################################################
#########################   U STEP #############################################
################################################################################
    
    if i > 0:
        u = tv_denoising_algorithm(v.x, lamb)

################################################################################
#########################   V STEP #############################################
################################################################################

    v = scipy.optimize.minimize(
                    v_step_function,
                    u,
                    args = (liam_all,
                            cont_results_noise,
                            steps,
                            cuboid_coordinates,
                            rho,
                            u),
                    method = 'L-BFGS-B',
                    bounds = scipy.optimize.Bounds(0,1),
                    options = {'disp': False,
                               'maxcor': 10,
                               'ftol': 10**-20,
                               'gtol': 10**-20,
                               'eps': 10**-8, # if to high stops fast
                               'maxiter': 10**7,
                               'maxls': 100,
                               'maxfun': 10**7})
    
    if i > 0:
        change_in_solution(old_solution, v.x)
        print(np.linalg.norm(cont_results_noise - k_trafo_one_dim_all(liam_all, v.x)))
            #np.linalg.norm(cont_results_noise - dis_results),


    old_solution = v.x # + np.average(v.x) * 0.05 * (np.random.random(len(v.x)) - 0.5)

    
    v1d = threed_to_oned(values, cuboid_coordinates, steps)
    plt.plot(v1d, "-o")
    plt.plot(v.x, "-o")
    plt.savefig(str(i) + ".png")
    plt.close()


"""
#while dist > 3 * 10**-13:
while dist > 0.0118:

    u = threed_to_oned(values, cuboid_coordinates, steps) * 0
    v = scipy.optimize.minimize(
                v_step_function,
                start,
                args = (liam_all, cont_results, steps, cuboid_coordinates, rho, u),
                method = 'L-BFGS-B',
                bounds = scipy.optimize.Bounds(0,1),
                options = {'disp': False,
                           'maxcor': 10,
                           'ftol': 10**-20,
                           'gtol': 10**-20,
                           'eps': 10**-8, # if to high stops fast
                           'maxiter': 10**7,
                           'maxls': 100,
                           'maxfun': 10**7})

    dist = v.fun/number_ktrans
    start = v.x + np.average(v.x) * 0.05 * (np.random.random(32) - 0.5)
    print(dist)
"""
