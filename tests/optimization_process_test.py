"""
Apr 19, 2020
Christopher Fichtlscherer (fichtlscherer@mailbox.org)
GNU General Public License

the pytest functions for the functions in optimization_process
"""

import numpy as np
#   import pytest

from ktra.k_transform import middle_of_cuboid
from ktra.k_transform import length_main_diagonal
from ktra.k_transform import shadow_plane_maker
from ktra.k_transform import get_parameter_plane_vectors
from ktra.k_transform import length_vector_for_diag
from ktra.k_transform import divide_the_factor
from ktra.k_transform import create_ray_points
from ktra.k_transform import create_rays
from ktra.k_transform import coordinate_discretization
from ktra.k_transform import discretization_of_model
from ktra.k_transform import create_source_point_d
from ktra.k_transform import generate_rays_d
from ktra.k_transform import generate_all_line_integral_array_matrix
from ktra.k_transform import k_trafo_one_dim_all
from ktra.k_transform import generate_line_integral_array_matrix
from ktra.k_transform import k_trafo_one_dim


from ktra.optimization_process import line_integral_cont
from ktra.optimization_process import k_trafo_cont
from ktra.optimization_process import make_threed_back
from ktra.optimization_process import dif
from ktra.optimization_process import threed_to_oned
from ktra.optimization_process import create_con_results


def test_line_integral_cont():

    def ray(t):
        return np.array([2.5 * t, 0, 0])

    def inner_of_box_1(x, y, z):
        if (-1 < x < 2.5) and (-1 < y < 3.5) and (-1 < z < 7):
            return 4.5
        return 0

    def inner_of_box_2(x, y, z):
        rad_pos = (x**2 + y**2 + z**2)**0.5
        if rad_pos <= 0.2:
            return 1
        if rad_pos <= 0.5:
            return 0.5
        if rad_pos <= 0.8:
            return 0.1
        return 0

    integral_1 = line_integral_cont(ray, inner_of_box_1)
    integral_2 = line_integral_cont(ray, inner_of_box_2)

    np.testing.assert_almost_equal(integral_1, 11.25, decimal=2)
    np.testing.assert_almost_equal(integral_2, 0.38, decimal=2)


def test_line_int_cont_2():
    """ now we let also create the ray """

    def inner_of_box(x, y, z):

        if (0.5 > x > -0.5) and (0.5 > y > -0.5) and (0.5 > z > -0.5):
            return 1

        return 0

    number_rays = {'dim_1': 1, 'dim_2': 1}
    cuboid_coordinates = {'x1': -1, 'x2': 1, 'y1': -1, 'y2': 1, 'z1': -1, 'z2': 1}
    radius = 2
    fineness = 10**4
    number_ktrans = 1

    source_point_d = create_source_point_d(radius, number_ktrans)
    rays_d = generate_rays_d(source_point_d, cuboid_coordinates, number_rays)
    integral = line_integral_cont(rays_d[0][0], inner_of_box, fineness)

    np.testing.assert_almost_equal(integral, 1, decimal=3)


def test_k_trafo_cont():

    def inner_of_box_1(x, y, z):
        if (-1 < x < 1) and (-1 < y < 1) and (-1 < z < 1):
            return 4.5
        return 0

    def inner_of_box_2(x, y, z):
        rad_pos = (x**2 + y**2 + z**2)**0.5
        if rad_pos <= 0.2:
            return 1
        if rad_pos <= 0.5:
            return 0.5
        if rad_pos <= 0.8:
            return 0.1
        return 0

    cuboid_coordinates = {'x1': -1, 'x2': 1, 'y1': -1, 'y2': 1, 'z1': -1, 'z2': 1}
    number_rays = {'dim_1': 2, 'dim_2': 2}
    source_point = np.array([5, 0, 0])

    middle = middle_of_cuboid(cuboid_coordinates)
    diag = length_main_diagonal(cuboid_coordinates)

    pv, nv = shadow_plane_maker(source_point, cuboid_coordinates)
    vx, vy = get_parameter_plane_vectors(nv)

    factor_x = length_vector_for_diag(vx, diag)
    factor_y = length_vector_for_diag(vy, diag)

    factor_array_x = divide_the_factor(factor_x, number_rays['dim_1'])
    factor_array_y = divide_the_factor(factor_y, number_rays['dim_2'])

    points = create_ray_points(factor_array_x, factor_array_y, vx, vy, middle)
    rays = create_rays(source_point, points)

    k_trafo_cont_value_1 = k_trafo_cont(rays, inner_of_box_1)
    k_trafo_cont_value_2 = k_trafo_cont(rays, inner_of_box_2)

    # for comparison
    l1 = line_integral_cont(rays[0], inner_of_box_1)
    l2 = line_integral_cont(rays[0], inner_of_box_2)
    # since the functions are symmetric for the 4 rays we can just take exp(-l1)
    # since we would sum up 4 times the same value and divide it by 4 afterwards
    np.testing.assert_almost_equal(k_trafo_cont_value_1, np.exp(-l1), decimal=2)
    np.testing.assert_almost_equal(k_trafo_cont_value_2, np.exp(-l2), decimal=2)


def test_k_trafo_cont_2():

    def inner_of_box(x, y, z):

        if (0.5 > x > -0.5) and (0.5 > y > -0.5) and (0.5 > z > -0.5):
            return 1

        return 0

    number_rays = {'dim_1': 1, 'dim_2': 1}
    cuboid_coordinates = {'x1': -1, 'x2': 1, 'y1': -1, 'y2': 1, 'z1': -1, 'z2': 1}
    radius = 2
    fineness = 10**4
    number_ktrans = 1
    steps = 4

    source_point_d = create_source_point_d(radius, number_ktrans)
    rays_d = generate_rays_d(source_point_d, cuboid_coordinates, number_rays)

    k_cont = k_trafo_cont(rays_d[0], inner_of_box, fineness)
    print("k_kont", k_cont)
    x_cor, y_cor, z_cor = coordinate_discretization(cuboid_coordinates, steps)
    liam = generate_line_integral_array_matrix(rays_d[0], cuboid_coordinates, steps, fineness)

    values = discretization_of_model(x_cor, y_cor, z_cor, inner_of_box, steps)
    values_1d = threed_to_oned(values, cuboid_coordinates, steps)

    result_1d = k_trafo_one_dim(liam, values_1d)
    print(result_1d)

    np.testing.assert_almost_equal(k_cont, result_1d, decimal=2)


def test_make_threed_back():

    array_oned = np.array([10, 5, 1, 0])
    steps = 3
    cuboid_coordinates = {'x1': -1, 'x2': 1, 'y1': -1, 'y2': 1, 'z1': -1, 'z2': 1}
    threed_reconstruction = make_threed_back(array_oned, steps, cuboid_coordinates)

    threed_compare = np.array([[[0, 1, 0], [1, 5, 1], [0, 1, 0]],
                               [[1, 5, 1], [5, 10, 5], [1, 5, 1]],
                               [[0, 1, 0], [1, 5, 1], [0, 1, 0]]])

    np.testing.assert_array_equal(threed_reconstruction, threed_compare)


def test_dif():

    u = np.array([1, 2, 3, 4, 5])
    v = np.array([1, 1, 1, 1, 1])

    rho = 0.5

    dif_value = dif(u, v, rho)

    # 0.5/2 * (1**2 + 2**2 + 3**2 + 4**2 + 5**2) = 7.5

    assert dif_value == 7.5


def test_threed_to_oned():

    values = np.array([[[0, 1, 0], [1, 5, 1], [0, 1, 0]],
                      [[1, 5, 1], [5, 10, 5], [1, 5, 1]],
                      [[0, 1, 0], [1, 5, 1], [0, 1, 0]]])

    steps = 3
    cuboid_coordinates = {'x1': -1, 'x2': 1, 'y1': -1, 'y2': 1, 'z1': -1, 'z2': 1}

    oned = threed_to_oned(values, cuboid_coordinates, steps)

    assert (oned == np.array([10, 5, 1, 0])).all()


def test_threed_to_oned_2():
    """here it is tested if make_threed_back is exactly the opposite to threed_to_oned"""

    array_oned = np.arange(10)
    steps = 5
    cuboid_coordinates = {'x1': -1, 'x2': 1, 'y1': -1, 'y2': 1, 'z1': -1, 'z2': 1}
    threed_reconstruction = make_threed_back(array_oned, steps, cuboid_coordinates)
    oned_back = threed_to_oned(threed_reconstruction, cuboid_coordinates, steps)

    np.testing.assert_array_equal(array_oned, oned_back)


def test_threed_to_oned_3():

    array_oned = np.arange(85)
    steps = 18
    cuboid_coordinates = {'x1': -1, 'x2': 1, 'y1': -1, 'y2': 1, 'z1': -1, 'z2': 1}
    threed_reconstruction = make_threed_back(array_oned, steps, cuboid_coordinates)
    oned_back = threed_to_oned(threed_reconstruction, cuboid_coordinates, steps)

    np.testing.assert_array_equal(array_oned, oned_back)


def test_threed_to_oned_4():
    """ here a function is discretized in a threed model, transformed to a oned array and
    transformed back to a threed model, the 3d models are gonna be compared"""

    def inner_of_box(x, y, z):
        rad_pos = (x**2 + y**2 + z**2)**0.5
        if rad_pos <= 0.2:
            return 1
        if rad_pos <= 0.5:
            return 0.5
        if rad_pos <= 0.8:
            return 0.1
        return 0

    steps = 20
    cuboid_coordinates = {'x1': -1, 'x2': 1, 'y1': -1, 'y2': 1, 'z1': -1, 'z2': 1}

    x_cor, y_cor, z_cor = coordinate_discretization(cuboid_coordinates, steps)
    values = discretization_of_model(x_cor, y_cor, z_cor, inner_of_box, steps)

    oned = threed_to_oned(values, cuboid_coordinates, steps)
    threed_reconstruction = make_threed_back(oned, steps, cuboid_coordinates)

    np.testing.assert_array_equal(values, threed_reconstruction)


def test_create_con_results():
    """ testing of the cont results of the k_transform are the same as the one for the
    1d k_transform, if the object is discrete """

    def inner_of_box(x, y, z):

        if (0.5 > x > -0.5) and (0.5 > y > -0.5) and (0.5 > z > -0.5):
            return 1

        return 0

    number_rays = {'dim_1': 1, 'dim_2': 1}
    cuboid_coordinates = {'x1': -1, 'x2': 1, 'y1': -1, 'y2': 1, 'z1': -1, 'z2': 1}
    steps = 4
    radius = 2
    fineness = 10**4
    number_ktrans = 5

    source_point_d = create_source_point_d(radius, number_ktrans)
    rays_d = generate_rays_d(source_point_d, cuboid_coordinates, number_rays)
    cont_results = create_con_results(rays_d, inner_of_box, fineness)

    liam_all = generate_all_line_integral_array_matrix(rays_d, cuboid_coordinates, steps, fineness)
    x_cor, y_cor, z_cor = coordinate_discretization(cuboid_coordinates, steps)
    values = discretization_of_model(x_cor, y_cor, z_cor, inner_of_box, steps)
    values_1d = threed_to_oned(values, cuboid_coordinates, steps)
    result_1d = k_trafo_one_dim_all(liam_all, values_1d)

    print(cont_results)
    print(result_1d)

    np.testing.assert_array_almost_equal(cont_results, result_1d, decimal=2)
