"""
Apr 14, 2020
Christopher Fichtlscherer (fichtlscherer@mailbox.org)
GNU General Public License

In the prerun the ten objects:
source_point_d, rays_d, liam_all, cont_results, compare_values, x_cor, y_cor, z_cor, values, touched
are created and saved with pickle
"""

import numpy as np
import time as time

t1 = time.time()

from ktra.k_transform import create_ray_points
from ktra.k_transform import create_rays
from ktra.k_transform import length_main_diagonal
from ktra.k_transform import shadow_plane_maker
from ktra.k_transform import get_parameter_plane_vectors
from ktra.k_transform import length_vector_for_diag
from ktra.k_transform import divide_the_factor
from ktra.k_transform import middle_of_cuboid

from ktra.optimization_process import k_trafo_cont


def inner_of_box(x, y, z):
    """ Different shells around the origin."""

    rad_pos = (x**2 + y**2 + z**2)**0.5
  
    if rad_pos <= 0.3: return 1.0
    if rad_pos <= 0.7: return 0.5
   
    return 0

cuboid_coordinates = {'x1': -1, 'x2': 1, 'y1': -1, 'y2': 1, 'z1': -1, 'z2': 1}
fineness = 10**4
number_rays = {'dim_1': 7, 'dim_2': 7}

source_point = np.array([3, 0, 0])

diag = length_main_diagonal(cuboid_coordinates)
middle = middle_of_cuboid(cuboid_coordinates)
pv, nv = shadow_plane_maker(source_point, cuboid_coordinates)
vx, vy = get_parameter_plane_vectors(nv)
factor_x = length_vector_for_diag(vx, diag)
factor_y = length_vector_for_diag(vy, diag)
factor_array_x = divide_the_factor(factor_x, number_rays['dim_1'])
factor_array_y = divide_the_factor(factor_y, number_rays['dim_2'])

points = create_ray_points(factor_array_x, factor_array_y, vx, vy, middle)
rays = create_rays(source_point, points)
k_transform_result = k_trafo_cont(rays, inner_of_box, fineness)

print(k_transform_result)
print(time.time()-t1)
