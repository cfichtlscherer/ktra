"""
Apr 18, 2020
Christopher Fichtlscherer (fichtlscherer@mailbox.org)
GNU General Public License

The functions needed for the optimization process
"""


import numpy as np
import datetime as dt
import scipy.optimize
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

from ktra.k_transform import k_trafo_one_dim_all
from ktra.k_transform import coordinate_discretization
from ktra.k_transform import distance_creater
from ktra.k_transform import dist_array_maker
from ktra.k_transform import ray_tracer
from ktra.k_transform import middle_of_cuboid
from ktra.k_transform import length_main_diagonal
from ktra.k_transform import shadow_plane_maker
from ktra.k_transform import get_parameter_plane_vectors
from ktra.k_transform import length_vector_for_diag
from ktra.k_transform import divide_the_factor
from ktra.k_transform import create_ray_points
from ktra.k_transform import create_rays
from ktra.k_transform import discretization_of_model
from ktra.k_transform import lineintegral_matrices_maker
from ktra.k_transform import k_transform


def line_integral_cont(ray, inner_of_box, fineness=10**4):
    """
    The continuous line integral over the function inner of the box
    """
    total_sum = 0

    for i in range(fineness):
        t = 2 * (i / fineness)
        x, y, z = ray(t)
        total_sum += inner_of_box(x, y, z)

    length = np.linalg.norm(ray(0) - ray(2))

    integral = (total_sum / fineness) * length

    return integral


def k_trafo_cont(rays, inner_of_box, fineness=10**4):
    """
    Returns the value of the k-transform for the continuous case.
    """

    k_trafo_value = 0

    for i in range(len(rays)):
        k_trafo_value += np.exp(- line_integral_cont(rays[i], inner_of_box, fineness))

    k_trafo_value_normed = k_trafo_value / len(rays)

    return k_trafo_value_normed


def k_trafo_derivation(rays, inner_of_box, direction, fineness=10**4):
    """
    24.08.2021 returns the derivation as a function of the change of the object.
    Implementation of Christinas calculation.
    continuous case / non rotational symmetric
    """

    derivation_value = 0

    for i in range(len(rays)):
        derivation_value += np.exp(- line_integral_cont(rays[i], inner_of_box, fineness)) * line_integral_cont(rays[i], direction, fineness)

    derivation_value_normed = -derivation_value / len(rays)

    return derivation_value_normed


def callback_fun(values, *args):
    """this callback function is used to display the value of the function we
    are optimizing"""

    global iteration

    liam_all, cont_results, steps, cuboid_coordinates, alpha = args

    iteration += 1

    a = k_trafo_one_dim_all(liam_all, values) - cont_results

    print(str(iteration) + " : Value of the function: " + str(np.sum(a**2)**0.5))


def make_threed_back(array_oned, steps, cuboid_coordinates):
    """
    create the 3D matrix again from the 1D array.
    """

    x_cor, y_cor, z_cor = coordinate_discretization(cuboid_coordinates, steps)

    dist = np.round(distance_creater(x_cor, y_cor, z_cor, steps), 6)
    dist_fru = dist_array_maker(dist)

    model_threed = np.zeros((steps, steps, steps))
    
    for i in range(len(dist_fru)):
        model_threed += (dist == dist_fru[i]) * array_oned[i]

    return model_threed


def important_mini_cuboids(x_cor, y_cor, z_cor, steps, rays_d, values):
    """
    Gives back an array with the shape of values in which every entry is 1 if it is hit by any
    ray of a k-transform, otherwise 0.
    """

    touched = np.zeros_like(values)
    # for i in tqdm(range(len(rays_d))):
    for i in range(len(rays_d)):
        for j in range(len(rays_d[0])):
            ray = rays_d[i][j]
            points_cuboid = ray_tracer(ray, steps, x_cor, y_cor, z_cor, 10**3)
            non_z = (points_cuboid != 0).astype(int)
            touched += non_z

    return touched


def set_non_hit_cuboids_zero(solution, x_cor, y_cor, z_cor, steps, rays_d, values):
    """
    Sets the values of every minicube which is not hit any mini cuboid.
    """

    touched = important_mini_cuboids(x_cor, y_cor, z_cor, steps, rays_d, values)

    # solution is the scipy object with .x we get to the 1D results we reshape
    solution_a = solution.reshape(steps, steps, steps)

    solution_a[touched == 0] = 0

    return solution_a


def generate_everything(number_ktrans, cuboid_coordinates, number_rays, steps,
                        inner_of_box):
    """
    Generates the source points and k-tra values by simply moving the
    source around them.
    we firmly assume here that the k-transforms lie at the same distance
    in a circle (with radius 2) around the box
    """

    source_point_d = {}
    pv_d = {}
    nv_d = {}
    vx_d = {}
    vy_d = {}
    factor_x_d = {}
    factor_y_d = {}
    factor_x_array_d = {}
    factor_y_array_d = {}
    points_d = {}
    rays_d = {}

    ktra_v = np.zeros(number_ktrans)

    middle = middle_of_cuboid(cuboid_coordinates)
    diag = length_main_diagonal(cuboid_coordinates)

    for i in np.arange(number_ktrans):
        source_point = 2 * np.array([np.sin(2 * np.pi * i / number_ktrans),
                                     np.cos(2 * np.pi * i / number_ktrans), 0])
        source_point_d[i] = source_point
        pv, nv = shadow_plane_maker(source_point, cuboid_coordinates)
        pv_d[i] = pv
        nv_d[i] = nv
        vx, vy = get_parameter_plane_vectors(nv)
        vx_d[i] = vx
        vy_d[i] = vy

        factor_x = length_vector_for_diag(vx, diag)
        factor_y = length_vector_for_diag(vy, diag)
        factor_x_d[i] = factor_x
        factor_y_d[i] = factor_y

        factor_array_x = divide_the_factor(factor_x, number_rays['dim_1'])
        factor_array_y = divide_the_factor(factor_y, number_rays['dim_2'])
        factor_x_array_d[i] = factor_array_x
        factor_y_array_d[i] = factor_array_y

        points = create_ray_points(factor_array_x, factor_array_y, vx, vy, middle)
        points_d[i] = points

        rays = create_rays(source_point, points)
        rays_d[i] = rays
        x_cor, y_cor, z_cor = coordinate_discretization(cuboid_coordinates, steps)
        values = discretization_of_model(x_cor, y_cor, z_cor, inner_of_box, steps)

    line_matrices_d = lineintegral_matrices_maker(rays_d, steps, x_cor, y_cor, z_cor)

    big_line_matrix_array = np.array(
        [np.array(list(line_matrices_d[i].values())) for i in range(number_ktrans)])

    ktra_v = k_transform(values, big_line_matrix_array)

    touched = important_mini_cuboids(x_cor, y_cor, z_cor, steps, rays_d, values)

    return big_line_matrix_array, ktra_v, touched, values, x_cor, y_cor, z_cor, rays_d


def compare_plot_opt(values, solution, steps, x_cor, y_cor, z_cor, rays_d, filtered=False):
    """
    Compare the solution with the reconstruction by plotting the slices.
    """

    if filtered == True:
        solution_filtered = set_non_hit_cuboids_zero(solution, x_cor, y_cor, z_cor, steps, rays_d,
                                                     values)

    if filtered == False:
        solution_filtered = solution

    filename = str(dt.datetime.now().strftime("%y%m%d_%H:%M:%S_"))

    plotname = str(dt.datetime.now().strftime("%d.%m.%Y, %H:%M:%S"))

    for i in range(steps):
        fig, (ax1, ax2) = plt.subplots(ncols=2)
        fig.suptitle(plotname)

        img1 = ax1.imshow(solution_filtered[i], vmin=0, vmax=1)
        ax1.autoscale(False)
        ax1.set_title('Approximation')
        divider = make_axes_locatable(ax1)
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(img1, cax=cax1)

        img2 = ax2.imshow(values[i], vmin=0, vmax=1)
        ax2.autoscale(False)
        ax2.set_title('Object')
        divider = make_axes_locatable(ax2)
        cax2 = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(img2, cax=cax2)

        plt.tight_layout(h_pad=1)
        plt.savefig(filename + str(i) + ".png")
        plt.close()


def how_many_got_hit(touched):
    """
    calculates how many of the small cubes got touched by any ray from any
    k-transformation.
    """

    all_c = touched.size
    hit = len(touched[touched != 0])

    perc_hit = hit / all_c

    return all_c, hit, perc_hit


def compare_values_original_function(cuboid_coordinates, steps, inner_of_box):
    """
    creates an array of the original value of the inner_of_box function for
    comparisson
    """

    x_cor, y_cor, z_cor = coordinate_discretization(cuboid_coordinates, steps)
    dist = distance_creater(x_cor, y_cor, z_cor, steps)
    dist_fru = dist_array_maker(dist)
    compare_values = np.zeros(dist_fru.size)

    for i in range(len(compare_values)):
        compare_values[i] = inner_of_box(dist_fru[i], 0, 0)

    return compare_values


def threed_to_oned(values, cuboid_coordinates, steps):
    """takes symmetric 3d value array and creates 1d array from it
    this function was completly reworked in commit 53, 20.04.20
    """

    x_cor, y_cor, z_cor = coordinate_discretization(cuboid_coordinates, steps)
    dist = distance_creater(x_cor, y_cor, z_cor, steps)
    dist_fru = dist_array_maker(dist)

    values_oned = 0 * dist_fru

    for j, i in enumerate(dist_fru):
        x, y, z = np.where(np.round(dist, 6) == i)
        values_oned[j] = values[x[0], y[0], z[0]]

    return values_oned


def eq_system_for_opt(values, *args):
    """the function we are optimizing"""

    liam_all, cont_results, steps, cuboid_coordinates, alpha = args

    value_results = k_trafo_one_dim_all(liam_all, values)
    value_difference = value_results - cont_results
    value_difference_norm = (np.sum(value_difference**2) / len(value_difference))**0.5

    return value_difference_norm


def create_con_results(rays_d, inner_of_box, fineness):
    """
    creates array of the ktra values of the continuous (non discretized) object
    """

    cont_results = np.zeros(len(rays_d))

    # for i in tqdm(range(len(cont_results))):
    for i in range(len(cont_results)):
        cont_results[i] = k_trafo_cont(rays_d[i], inner_of_box, fineness)

    return cont_results


def dif(u, v, rho):
    """(rho/2)*||u-v||_2^{2}"""

    dif_value = (rho / 2) * np.linalg.norm(u - v)**2

    return dif_value


def prox_operator(function, z, x0):
    """ returns the prox of the function at position z
    x0 is the start guess for the minimization problem"""

    # fun: the function which is minimized for the prox operator

    fun = lambda x: function(z) + 0.5 * np.linalg.norm(z - x)**2

    minimization_output = scipy.optimize.minimize(fun, x0)
    result = float(minimization_output.x)

    return result


def v_step_function(values_1d, *args):
    """the function we are optimizing"""

    liam_all, cont_results, steps, cuboid_coordinates, rho, u = args
    value_results = k_trafo_one_dim_all(liam_all, values_1d)

    value_difference_norm = np.linalg.norm(value_results - cont_results)**2
    uv_dif = np.linalg.norm(u - values_1d)

    return value_difference_norm + (rho / 2) * uv_dif


def td_to_od_nonsymmetric(td, steps, cuboid_coordinates):
    """sums up all the values with the same distance"""
    x_cor, y_cor, z_cor = coordinate_discretization(cuboid_coordinates, steps)
    dist = distance_creater(x_cor, y_cor, z_cor, steps)
    dist_fru = dist_array_maker(dist)

    integral_array = np.zeros(dist_fru.size)
    #   need to round since we did it in the dist_array_maker
    dist_r = np.round(dist, 6)
    for i, j in enumerate(dist_fru):
        d = (dist_r == j) * td
        integral_array[i] = np.sum(d)
    return integral_array


def spxt_cont_measurement(rays, inner_of_box, fineness=10**4):                                  
    """ a continuous spxt measurement"""
                                                                                   
    spxt_value = 0                                                                  
                                                                                       
    for i in range(len(rays)):                                                         
        spxt_value +=  line_integral_cont(rays[i], inner_of_box, fineness)
    
    spxt_value_normed = spxt_value / len(rays)                                   
                                                                                       
    return spxt_value_normed                                                        

