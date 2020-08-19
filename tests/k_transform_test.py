"""
Apr 17, 2020
Christopher Fichtlscherer (fichtlscherer@mailbox.org)
GNU General Public License

The pytest tests for the functions in k_transform
"""

import numpy as np
import pytest

from ktra.k_transform import intersection_line_plane
from ktra.k_transform import middle_of_cuboid
from ktra.k_transform import shadow_plane_maker
from ktra.k_transform import length_main_diagonal
from ktra.k_transform import get_parameter_plane_vectors
from ktra.k_transform import length_vector_for_diag
from ktra.k_transform import divide_the_factor
from ktra.k_transform import create_ray_points
from ktra.k_transform import create_rays
from ktra.k_transform import coordinate_discretization
from ktra.k_transform import discretization_of_model
from ktra.k_transform import line_integral
from ktra.k_transform import k_transform
from ktra.k_transform import check_mini_cuboid
from ktra.k_transform import ray_tracer
from ktra.k_transform import distance_to_center
from ktra.k_transform import distance_creater
from ktra.k_transform import dist_array_maker
from ktra.k_transform import one_dimensional_line_integral_maker
from ktra.k_transform import generate_line_integral_array_matrix
from ktra.k_transform import generate_all_line_integral_array_matrix
from ktra.k_transform import create_source_point_d
from ktra.k_transform import generate_rays_d
from ktra.k_transform import k_trafo_one_dim
from ktra.k_transform import k_trafo_one_dim_all

from ktra.optimization_process import threed_to_oned
from ktra.optimization_process import generate_everything
#   from ktra.k_transform import create_line_matrix
#   from ktra.k_transform import create_single_matrix_dic
#   from ktra.k_transform import lineintegral_matrices_maker
from ktra.optimization_process import how_many_got_hit


def test_intersection__line_plane_1():
    result = intersection_line_plane(np.array([1, 0, 0]),
                                     np.array([0, 0, 0]),
                                     np.array([1, 1, 1]),
                                     np.array([0, 0, 0]))
    np.testing.assert_array_equal(np.array([0, 0, 0]), result)


def test_intersection_line_plane_2():
    result = intersection_line_plane(np.array([1, 0, 0]),
                                     np.array([3, 0, 0]),
                                     np.array([1, 1, 1]),
                                     np.array([0, 0, 0]))
    np.testing.assert_array_equal(np.array([3, 3, 3]), result)


def test_intersection_line_plane_parallel():
    with pytest.raises(RuntimeError) as errorinfo:
        intersection_line_plane(np.array([1, 0, 0]),
                                np.array([3, 0, 0]),
                                np.array([0, 0, 1]),
                                np.array([0, 0, 0]))
        np.testing.assert_string_equal(errorinfo.value, "they are parallel")


def test_middle_of_cuboid():
    cuboid_coordinates = {'x1': 1, 'x2': 2,
                          'y1': -3, 'y2': 1,
                          'z1': 1, 'z2': 0}
    result = middle_of_cuboid(cuboid_coordinates)
    np.testing.assert_array_equal(np.array([1.5, -1, 0.5]), result)


def test_shadow_plane():
    cuboid_coordinates = {'x1': 1, 'x2': 2,
                          'y1': 1, 'y2': 2,
                          'z1': 1, 'z2': 2}
    source_pos = np.array([0, 0, 0])
    pos_vec, norm_vec = shadow_plane_maker(source_pos, cuboid_coordinates)
    np.testing.assert_array_equal(np.array([1.5, 1.5, 1.5]), pos_vec)
    np.testing.assert_array_equal(np.array([1.5, 1.5, 1.5]), norm_vec)

    cuboid_coordinates = {'x1': 2, 'x2': 3,
                          'y1': 2, 'y2': 3,
                          'z1': 2, 'z2': 3}
    source_pos = np.array([1, 0, 0])
    pos_vec, norm_vec = shadow_plane_maker(source_pos, cuboid_coordinates)
    np.testing.assert_array_equal(np.array([2.5, 2.5, 2.5]), pos_vec)
    np.testing.assert_array_equal(np.array([1.5, 2.5, 2.5]), norm_vec)


def test_length_main_diagonal():
    cuboid_coordinates = {'x1': 2, 'x2': 3,
                          'y1': 2, 'y2': 3,
                          'z1': 2, 'z2': 3}
    diag = length_main_diagonal(cuboid_coordinates)
    np.testing.assert_equal(3**0.5, diag)

    cuboid_coordinates = {'x1': 4, 'x2': -3,
                          'y1': 0, 'y2': 1,
                          'z1': -5, 'z2': -3}
    diag = length_main_diagonal(cuboid_coordinates)
    np.testing.assert_equal(54**0.5, diag)


def test_parameter_plane_vectors():
    v1, v2 = get_parameter_plane_vectors(np.array([1, 0, 0]))
    np.testing.assert_array_equal(np.array([0, 1, 0]), v1)
    np.testing.assert_array_equal(np.array([0, 0, 1]), v2)

    v1, v2 = get_parameter_plane_vectors(np.array([1, 1, 0]))
    np.testing.assert_array_equal(np.array([0, 0, 1]), v1)
    np.testing.assert_array_equal(np.array([-1, 1, 0]), v2)

    v1, v2 = get_parameter_plane_vectors(np.array([1, 3, 0]))
    np.testing.assert_array_equal(np.array([0, 0, 1]), v1)
    np.testing.assert_array_equal(np.array([-3, 1, 0]), v2)

    v1, v2 = get_parameter_plane_vectors(np.array([4, 4, 4]))
    np.testing.assert_array_equal(np.array([1, -1, 0]), v1)
    np.testing.assert_array_equal(np.array([0.5, 0.5, -1]), v2)

    v1, v2 = get_parameter_plane_vectors(np.array([1, 2, 3]))
    np.testing.assert_array_equal(np.array([-3, 0, 1]), v1)
    np.testing.assert_array_equal(np.array([1, -5, 3]), v2)

    v1, v2 = get_parameter_plane_vectors(np.array([1, 1, 2]))
    np.testing.assert_array_equal(np.array([-2, 0, 1]), v1)
    np.testing.assert_array_equal(np.array([1, -5, 2]), v2)


def test_length_vector_for_diag():
    factor = length_vector_for_diag(np.array([0, -4, 7]), 10)
    np.testing.assert_equal(0.5 * 10 / (65**0.5), factor)

    factor = length_vector_for_diag(np.array([1, 5, 8]), 7)
    np.testing.assert_equal(0.5 * 7 / (90**0.5), factor)


def test_divide_the_factor():
    factor_array = divide_the_factor(10, 3)
    np.testing.assert_equal(np.array([-5, 0, 5]), factor_array)

    factor_array = divide_the_factor(1, 1)
    np.testing.assert_equal(np.array([0]), factor_array)


def test_create_ray_points():
    factor_array_x = np.array([-3, 0, 3])
    vector_x = np.array([1, 2, 0])

    factor_array_y = np.array([1, -1])
    vector_y = np.array([-2, 1, 0])

    middle_of_cuboid = np.array([0, 0, 0])

    points = create_ray_points(factor_array_x, factor_array_y, vector_x, vector_y, middle_of_cuboid)

    np.testing.assert_array_equal(np.array([-5, -5, 0]), points[0])
    np.testing.assert_array_equal(np.array([-1, -7, 0]), points[1])
    np.testing.assert_array_equal(np.array([-2, 1, 0]), points[2])
    np.testing.assert_array_equal(np.array([2, -1, 0]), points[3])
    np.testing.assert_array_equal(np.array([1, 7, 0]), points[4])
    np.testing.assert_array_equal(np.array([5, 5, 0]), points[5])


def test_creat_rays():
    points = [np.array([2, 7, 0]), np.array([-3, -4, 2])]
    source_pos = np.array([11, 12, 13])

    rays = create_rays(source_pos, points)

    np.testing.assert_array_almost_equal(np.array([-52, -23, -78]), rays[0](7))
    np.testing.assert_array_almost_equal(np.array([53, 60, 46]), rays[1](-3))


def test_line_integral():

    steps = 10
    cuboid_coordinates = {'x1': 1, 'x2': 4,
                          'y1': -3, 'y2': 3,
                          'z1': -3, 'z2': 3}

    def ray(t):
        return np.array([2.5 * t, 0, 0])

    def inner_of_box(x, y, z):
        return -1

    x_cor, y_cor, z_cor = coordinate_discretization(cuboid_coordinates, steps)
    values = discretization_of_model(x_cor, y_cor, z_cor, inner_of_box, steps)
    line_value = line_integral(ray, x_cor, y_cor, z_cor, values)
    np.testing.assert_almost_equal(line_value, -3, decimal=2)

    steps = 20
    cuboid_coordinates = {'x1': 1, 'x2': 2,
                          'y1': 1, 'y2': 2,
                          'z1': 1, 'z2': 2}

    def ray(t):
        return np.array([1.5 * t, 1.5 * t, 1.5 * t])

    def inner_of_box(x, y, z):
        return 2

    x_cor, y_cor, z_cor = coordinate_discretization(cuboid_coordinates, steps)
    values = discretization_of_model(x_cor, y_cor, z_cor, inner_of_box, steps)
    line_value = line_integral(ray, x_cor, y_cor, z_cor, values)
    np.testing.assert_almost_equal(line_value, 2 * 3**0.5, decimal=2)


def test_line_integral_2():

    def inner_of_box(x, y, z):

        if (0.5 > x > -0.5) and (0.5 > y > -0.5) and (0.5 > z > -0.5):
            return 1

        return 0

    cuboid_coordinates = {'x1': -1, 'x2': 1, 'y1': -1, 'y2': 1, 'z1': -1, 'z2': 1}
    number_rays = {'dim_1': 1, 'dim_2': 1}
    number_ktrans = 1
    radius = 10
    steps = 4

    x_cor, y_cor, z_cor = coordinate_discretization(cuboid_coordinates, steps)
    values = discretization_of_model(x_cor, y_cor, z_cor, inner_of_box, steps)

    source_point_d = create_source_point_d(radius, number_ktrans)

    rays_d = generate_rays_d(source_point_d, cuboid_coordinates, number_rays)

    line_value = line_integral(rays_d[0][0], x_cor, y_cor, z_cor, values)

    np.testing.assert_almost_equal(line_value, 1, decimal=2)


def test_k_transform():

    def inner_of_box(x, y, z):
        if (1 > x > -1) and (1 > y > -1) and (1 > z > -1):
            return 13
        return 0

    steps = 4
    cuboid_coordinates = {'x1': -1, 'x2': 1,
                          'y1': -1, 'y2': 1,
                          'z1': -1, 'z2': 1}
    number_rays = {'dim_1': 1, 'dim_2': 1}
    number_ktrans = 1
    radius = 3
    fineness = 10**3

    x_cor, y_cor, z_cor = coordinate_discretization(cuboid_coordinates, steps)
    source_point_d = create_source_point_d(radius, number_ktrans)
    rays_d = generate_rays_d(source_point_d, cuboid_coordinates, number_rays)
    liam = generate_line_integral_array_matrix(rays_d[0], cuboid_coordinates, steps, fineness)

    values = discretization_of_model(x_cor, y_cor, z_cor, inner_of_box, steps)
    values_1d = threed_to_oned(values, cuboid_coordinates, steps)
    result_k_1d = k_trafo_one_dim(liam, values_1d)

    res_c = np.exp(-line_integral(rays_d[0][0], x_cor, y_cor, z_cor, values))

    np.testing.assert_almost_equal(result_k_1d, res_c, decimal=2)


def test_check_mini_cuboids():

    x_cor = np.array([0, 1, 2, 3, 4, 5])
    y_cor = np.array([0, 1, 2, 3, 4, 5])
    z_cor = np.array([0, 1, 2, 3, 4, 5])

    point1 = np.array([0.5, 2, 4.2])
    point2 = np.array([0, 0, 0])
    point4 = np.array([0.1, 1.5, 2.9])

    i1, j1, k1 = check_mini_cuboid(x_cor, y_cor, z_cor, point1)
    p1_c = np.array([i1, j1, k1])

    i2, j2, k2 = check_mini_cuboid(x_cor, y_cor, z_cor, point2)
    p2_c = np.array([i2, j2, k2])

    i4, j4, k4 = check_mini_cuboid(x_cor, y_cor, z_cor, point4)
    p4_c = np.array([i4, j4, k4])

    np.testing.assert_array_equal(np.array([0, 1, 4]), p1_c)
    np.testing.assert_array_equal(np.array([0, 0, 0]), p2_c)
    np.testing.assert_array_equal(np.array([0, 1, 2]), p4_c)


def test_ray_tracer():

    def ray(t):
        return np.array([1.5 * t, 0, 0])

    def ray_2(t):
        return np.array([t, 5, 5])

    x_cor = np.array([1, 1.2, 1.4, 1.6, 1.8, 2])
    y_cor = np.array([-0.5, -0.3, -0.1, 0.1, 0.3, 0.5])
    z_cor = np.array([-0.5, -0.3, -0.1, 0.1, 0.3, 0.5])

    steps = 5

    res_array = ray_tracer(ray, steps, x_cor, y_cor, z_cor, 300)
    res_array_non_hit = ray_tracer(ray_2, steps, x_cor, y_cor, z_cor, 300)

    compare = np.zeros([steps, steps, steps])
    compare[:, 2, 2] = 20

    empty_compare = np.zeros([steps, steps, steps])

    np.testing.assert_allclose(res_array, compare, atol=1)
    np.testing.assert_array_equal(res_array_non_hit, empty_compare)


def test_how_many_got_hit():

    touched = np.zeros((10, 10, 10))

    touched[1, 5, 7] = 9
    touched[2, :, :] = 6
    touched[:, 1, :] = 4
    touched[:, :, 9] = 5

    all_c, hit, perc_hit = how_many_got_hit(touched)

    np.testing.assert_equal(all_c, 1000)
    np.testing.assert_equal(hit, 272)
    np.testing.assert_equal(perc_hit, 0.272)


def test_distance_to_center():

    dist1 = distance_to_center(3, 4, 0)
    assert dist1 == 5

    dist2 = distance_to_center(13.3, -6, 0.3)
    assert dist2 == pytest.approx(14.5938343145316, 10**-8)


def test_distance_creater():

    cuboid_coordinates = {'x1': -1.5, 'x2': 1.5, 'y1': -1.5, 'y2': 1.5, 'z1': -1.5, 'z2': 1.5}
    steps = 3
    x_cor, y_cor, z_cor = coordinate_discretization(cuboid_coordinates, steps)
    dist = distance_creater(x_cor, y_cor, z_cor, steps)

    dist_c = np.array([[[np.sqrt(3), np.sqrt(2), np.sqrt(3)],
                        [np.sqrt(2), 1, np.sqrt(2)],
                        [np.sqrt(3), np.sqrt(2), np.sqrt(3)]],
                       [[np.sqrt(2), 1, np.sqrt(2)],
                        [1, 0, 1],
                        [np.sqrt(2), 1, np.sqrt(2)]],
                       [[np.sqrt(3), np.sqrt(2), np.sqrt(3)],
                        [np.sqrt(2), 1, np.sqrt(2)],
                        [np.sqrt(3), np.sqrt(2), np.sqrt(3)]]])

    np.testing.assert_array_almost_equal(dist, dist_c)


def test_dist_array_maker():

    cuboid_coordinates = {'x1': -1.5, 'x2': 1.5, 'y1': -1.5, 'y2': 1.5, 'z1': -1.5, 'z2': 1.5}
    steps = 3
    x_cor, y_cor, z_cor = coordinate_discretization(cuboid_coordinates, steps)

    dist = distance_creater(x_cor, y_cor, z_cor, steps)
    dist_fru = dist_array_maker(dist)

    dist_fru_c = np.array([0, 1, np.sqrt(2), np.sqrt(3)])

    np.testing.assert_array_almost_equal(dist_fru, dist_fru_c)


def test_dist_array_maker_2():

    cuboid_coordinates2 = {'x1': -2, 'x2': 2, 'y1': -2, 'y2': 2, 'z1': -2, 'z2': 2}
    steps2 = 4
    x_cor2, y_cor2, z_cor2 = coordinate_discretization(cuboid_coordinates2, steps2)
    dist2 = distance_creater(x_cor2, y_cor2, z_cor2, steps2)
    dist_fru2 = dist_array_maker(dist2)

    dist_fru_c2 = np.array([np.sqrt(0.75), np.sqrt(2.75), np.sqrt(4.75), np.sqrt(6.75)])

    np.testing.assert_array_almost_equal(dist_fru2, dist_fru_c2)


def test_one_dimensional_line_integral_maker():

    cuboid_coordinates = {'x1': -1.5, 'x2': 1.5, 'y1': -1.5, 'y2': 1.5, 'z1': -1.5, 'z2': 1.5}
    steps = 3
    points_cuboid = np.array([[[1, 6, 5], [4, 13, 19], [2.5, 7.3, 2.1]],
                              [[1.5, 60, 51], [32, 16, 19], [1.1, 1.3, 1.6]],
                              [[21, 0.9, 2.9], [1.6, 11.1, 0.2], [2.8, 4.4, 7]]])

    x_cor, y_cor, z_cor = coordinate_discretization(cuboid_coordinates, steps)
    dist = distance_creater(x_cor, y_cor, z_cor, steps)
    dist_fru = dist_array_maker(dist)

    integral_array = one_dimensional_line_integral_maker(dist, dist_fru, points_cuboid)
    integral_array_c = np.array([16, 136.4, 98.6, 44.3])

    np.testing.assert_array_almost_equal(integral_array, integral_array_c)


def test_generate_line_integral_matrix_array():

    cuboid_coordinates = {'x1': -1.5, 'x2': 1.5, 'y1': -1.5, 'y2': 1.5, 'z1': -1.5, 'z2': 1.5}
    steps = 3
    fineness = 10**4
    number_rays = {'dim_1': 1, 'dim_2': 1}
    number_ktrans = 1
    radius = 3

    x_cor, y_cor, z_cor = coordinate_discretization(cuboid_coordinates, steps)
    source_point_d = create_source_point_d(radius, number_ktrans)
    rays_d = generate_rays_d(source_point_d, cuboid_coordinates, number_rays)
    liam = generate_line_integral_array_matrix(rays_d[0], cuboid_coordinates, steps, fineness)

    liam_control = np.array([[1, 2, 0, 0]])

    np.testing.assert_array_almost_equal(liam, liam_control, decimal=2)


def test_generate_all_line_integral_array_matrix():

    cuboid_coordinates = {'x1': -1, 'x2': 1, 'y1': -1, 'y2': 1, 'z1': -1, 'z2': 1}
    steps = 4
    fineness = 10**4
    number_rays = {'dim_1': 1, 'dim_2': 1}
    number_ktrans = 2
    radius = 3

    x_cor, y_cor, z_cor = coordinate_discretization(cuboid_coordinates, steps)
    source_point_d = create_source_point_d(radius, number_ktrans)
    rays_d = generate_rays_d(source_point_d, cuboid_coordinates, number_rays)

    liam_all = generate_all_line_integral_array_matrix(rays_d, cuboid_coordinates, steps, fineness)

    liam_all_controll = np.array([[[1, 1, 0, 0]],
                                  [[1, 1, 0, 0]]])

    np.testing.assert_array_almost_equal(liam_all, liam_all_controll, decimal=2)


def test_create_source_point_d():

    radius = 10
    number_ktrans = 4

    source_point_d = create_source_point_d(radius, number_ktrans)

    source_point_d_compare = {"0": np.array([0, 10, 0]),
                              "1": np.array([10, 0, 0]),
                              "2": np.array([0, -10, 0]),
                              "3": np.array([-10, 0, 0])}

    for i in range(number_ktrans):
        np.testing.assert_array_almost_equal(source_point_d[str(i)],
                                             source_point_d_compare[str(i)])


def test_k_transform_oned_vs_threed():
    """ test if the three dimensional K-transform leads to the same result as the one dimensional
    K transform for symmetric inner_of_box functions """

    def inner_of_box(x, y, z):
        """
        Different shells around the origin.
        """

        rad_pos = (x**2 + y**2 + z**2)**0.5
        if rad_pos <= 0.2:
            return 1
        if rad_pos <= 0.5:
            return 0.8
        if rad_pos <= 0.8:
            return 0.5
        return 0

    steps = 20
    cuboid_coordinates = {'x1': -1, 'x2': 1,
                          'y1': -1, 'y2': 1,
                          'z1': -1, 'z2': 1}

    number_rays = {'dim_1': 5, 'dim_2': 5}
    number_ktrans = 10

    big_line_matrix_array, ktra_v, touched, values, x_cor, y_cor, z_cor, rays_d =\
        generate_everything(number_ktrans, cuboid_coordinates, number_rays, steps, inner_of_box)

    result_3d = k_transform(values, big_line_matrix_array)

    fineness = 10**3
    radius = 2

    source_point_d = create_source_point_d(radius, number_ktrans)
    rays_d = generate_rays_d(source_point_d, cuboid_coordinates, number_rays)
    liam_all = generate_all_line_integral_array_matrix(rays_d, cuboid_coordinates, steps, fineness)
    x_cor, y_cor, z_cor = coordinate_discretization(cuboid_coordinates, steps)
    values = discretization_of_model(x_cor, y_cor, z_cor, inner_of_box, steps)
    values_1d = threed_to_oned(values, cuboid_coordinates, steps)

    result_1d = k_trafo_one_dim_all(liam_all, values_1d)

    assert (result_3d == result_1d).all()
