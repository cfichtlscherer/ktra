"""
Apr 17, 2020
Christopher Fichtlscherer (fichtlscherer@mailbox.org)
GNU General Public License

Contains all the functions to perform a k-transform of a given function.
"""


import numpy as np
from tqdm import tqdm
# from numba import jit
# import line_profiler


def intersection_line_plane(plane_normal,
                            plane_point,
                            ray_direction,
                            ray_point,
                            epsilon=1e-8):
    """
    Gives the point where a ray and a plane do intersect.
    """

    dot_pro = plane_normal.dot(ray_direction)

    if abs(dot_pro) < epsilon:
        raise RuntimeError("they are parallel")

    p = ray_point - plane_point
    q = - plane_normal.dot(p) / dot_pro
    intersection_point = p + q * ray_direction + plane_point

    return intersection_point


def middle_of_cuboid(cuboid_coordinates):
    """
    Returns the middle of the cuboid.
    """

    xstart, xend = cuboid_coordinates['x1'], cuboid_coordinates['x2']
    ystart, yend = cuboid_coordinates['y1'], cuboid_coordinates['y2']
    zstart, zend = cuboid_coordinates['z1'], cuboid_coordinates['z2']

    x_middle = (xend - xstart) / 2 + xstart
    y_middle = (yend - ystart) / 2 + ystart
    z_middle = (zend - zstart) / 2 + zstart

    middle = np.array([x_middle, y_middle, z_middle])

    return middle


def shadow_plane_maker(source_pos, cuboid_coordinates):
    """
    Returns the plane on which the cuboid will be projected, going through the
    middle of the cuboid.
    """

    middle = middle_of_cuboid(cuboid_coordinates)
    normal_vector = middle - source_pos
    position_vector = source_pos + normal_vector

    return position_vector, normal_vector


def length_main_diagonal(cuboid_coordinates):
    """
    Returns the length of the diagonal of the cuboid
    """

    dx = (cuboid_coordinates['x2'] - cuboid_coordinates['x1'])**2
    dy = (cuboid_coordinates['y2'] - cuboid_coordinates['y1'])**2
    dz = (cuboid_coordinates['z2'] - cuboid_coordinates['z1'])**2

    diag = (dx + dy + dz)**0.5

    return diag


def get_parameter_plane_vectors(nv):
    """
    Transform a plane into parametric form, whereby the normal vector and
    the two direction vectors of the plane are all perpendicular to each other.
    """
    zeros_in = np.where(nv == 0)[0]

    # how often do the values in the nv array occure
    z0 = z1 = z2 = np.ones(3)
    o0 = np.sum(z0[nv == nv[0]])
    o1 = np.sum(z1[nv == nv[1]])
    o2 = np.sum(z2[nv == nv[2]])

    # case1: two zeros
    if len(zeros_in) == 2:
        v1 = np.zeros(3)
        v1[zeros_in[0]] = 1
        v2 = np.zeros(3)
        v2[zeros_in[1]] = 1

    # case2: one zero, values equal
    elif len(zeros_in) == 1 and (o0 == 2 or o1 == 2):
        if o0 == 1:
            v1 = np.array([1, 0, 0])
            v2 = np.array([0, -nv[1], nv[1]])
        if o1 == 1:
            v1 = np.array([0, 1, 0])
            v2 = np.array([-nv[0], 0, nv[0]])
        if o2 == 1:
            v1 = np.array([0, 0, 1])
            v2 = np.array([-nv[0], nv[0], 0])

    # case3: one zero, values unequal
    elif len(zeros_in) == 1:
        if nv[0] == 0:
            v1 = np.array([1, 0, 0])
            v2 = np.array([0, -nv[2], nv[1]])
        if nv[1] == 0:
            v1 = np.array([0, 1, 0])
            v2 = np.array([-nv[2], 0, nv[0]])
        if nv[2] == 0:
            v1 = np.array([0, 0, 1])
            v2 = np.array([-nv[1], nv[0], 0])

    # case4: all values != 0 and equal
    elif o0 == 3:
        v1 = np.array([1, -1, 0])
        v2 = np.array([0.5, 0.5, -1])

    # case5: all values != 0 and different
    elif o0 == 1 and o2 == 1:
        v1 = np.array([-nv[2], 0, nv[0]])
        v2 = np.array([nv[0], -(nv[0]**2 + nv[2]**2) / nv[1], nv[2]])

    # case6: all values != 0 two are equal
    else:
        if o0 == 1:
            v1 = np.array([-nv[1], nv[0], 0])
            v2 = np.array([nv[0], nv[1], - (nv[0]**2 + nv[1]**2) / nv[1]])
        if o1 == 1:
            v1 = np.array([nv[1], -nv[0], 0])
            v2 = np.array([nv[0], nv[1], - (nv[0]**2 + nv[1]**2) / nv[1]])
        if o2 == 1:
            v1 = np.array([-nv[2], 0, nv[0]])
            v2 = np.array([nv[0], -(nv[0]**2 + nv[2]**2) / nv[1], nv[2]])

    # check again
    epsilon = 10**-9
    d1 = abs(np.dot(v1, v2))
    d2 = abs(np.dot(v1, nv))
    d3 = abs(np.dot(v2, nv))
    if d1 + d2 + d3 >= epsilon:
        print("ERROR, not all arrays are perpendicular!")
        quit()
    if (nv == np.zeros(3)).all():
        print("ERROR, normal vector is zero!")
        quit()

    return v1, v2


def length_vector_for_diag(v, diag):
    """
    Which multiple of the vector you have to take until it has the length of
    the diagonal.
    """

    length = np.linalg.norm(v)
    factor = 0.5 * (diag / length)

    return factor


def divide_the_factor(factor, number_of_rays_dim):
    """
    Divides the factor, for the number of rays.
    """

    factor_array_with_end = np.linspace(-factor, factor, number_of_rays_dim + 2)
    factor_array = factor_array_with_end[1:1 + number_of_rays_dim]

    return factor_array


def create_ray_points(factor_array_x, factor_array_y, vector_x, vector_y, middle_of_cuboid):
    """
    Creates the points in the shadow plane for the rays.
    """

    number_x = len(factor_array_x)
    number_y = len(factor_array_y)

    points = []

    for i in range(number_x):
        for j in range(number_y):
            point = factor_array_x[i] * vector_x + factor_array_y[j] * vector_y + middle_of_cuboid

            points += [point]

    return points


def create_rays(source_pos, points):
    """
    Creates a dictionary containing all the rays as functions through the
    object.
    """

    rays = {}

    for i in range(len(points)):

        def ray_func(t, points=points, source_pos=source_pos, i=i):
            """Here a single ray is defined as a function."""

            return t * (points[i] - source_pos) + source_pos

        rays[i] = ray_func

    return rays


def coordinate_discretization(cuboid_coordinates, steps):
    """
    Creates discretized coordinate vectors.
    the small cuboid (discretization) is always defined between to values,
    therefore we need n+1 values for the definition of n cubes
    """

    xstart, xend = cuboid_coordinates['x1'], cuboid_coordinates['x2']
    ystart, yend = cuboid_coordinates['y1'], cuboid_coordinates['y2']
    zstart, zend = cuboid_coordinates['z1'], cuboid_coordinates['z2']

    x_cor = np.linspace(xstart, xend, steps + 1)
    y_cor = np.linspace(ystart, yend, steps + 1)
    z_cor = np.linspace(zstart, zend, steps + 1)

    return x_cor, y_cor, z_cor


def discretization_of_model(x_cor, y_cor, z_cor, inner_of_box, steps):
    """
    Discretization of the continuous function represents the inner of the box.
    Steps means we create the number of cubids in a single dimension.
    """

    values = np.zeros([steps, steps, steps])

    for i in range(steps):
        for j in range(steps):
            for k in range(steps):
                middle_x = x_cor[i] + (x_cor[i + 1] - x_cor[i]) / 2
                middle_y = y_cor[j] + (y_cor[j + 1] - y_cor[j]) / 2
                middle_z = z_cor[k] + (z_cor[k + 1] - z_cor[k]) / 2
                values[i, j, k] = inner_of_box(middle_x, middle_y, middle_z)

    return values


def cube_value(x_cor, y_cor, z_cor, values, point):
    """
    Checks in which small cube a point is and returns the value of this small
    cube. If the point is in non of this small cubes the value 0 is returned.
    new version 09.11.2019 - hopefully faster
    """

    if (x_cor[0] <= point[0] < x_cor[-1]) and\
            (y_cor[0] <= point[1] < y_cor[-1]) and\
            (z_cor[0] <= point[2] < z_cor[-1]):

        i = np.searchsorted(x_cor, point[0]) - 1
        j = np.searchsorted(y_cor, point[1]) - 1
        k = np.searchsorted(z_cor, point[2]) - 1

        return values[i, j, k]

    return 0


def line_integral(ray, x_cor, y_cor, z_cor, values, fineness=10**3):
    """
    This function gives the line integral through the cuboid.
    The fineness defines how many values we will check and sum up as an
    approach to the integral.
    For t=0 the ray is the source point, for t=1 the ray intersects
    the shadow plane.
    """

    total_sum = 0

    for i in range(fineness):
        t = 2 * (i / fineness)
        point = ray(t)
        total_sum += cube_value(x_cor, y_cor, z_cor, values, point)

    length = np.linalg.norm(ray(0) - ray(2))
    integral = (total_sum / fineness) * length

    return integral


def check_mini_cuboid(x_cor, y_cor, z_cor, point):
    """
    Checks in which mini cuboid a point is
    """

    i = np.searchsorted(x_cor, point[0]) - 1
    j = np.searchsorted(y_cor, point[1]) - 1
    k = np.searchsorted(z_cor, point[2]) - 1

    if x_cor[0] == point[0]:
        i = 0
    if y_cor[0] == point[1]:
        j = 0
    if z_cor[0] == point[2]:
        k = 0

    return i, j, k


#   @profile
#   @jit(nopython=True)
def ray_tracer(ray, steps, x_cor, y_cor, z_cor, fineness):
    """
    Returns a three dimensional array dimensions of the discretizated cuboid,
    which shows how many points where checked and added to the line integral
    in every small cuboid of the big cuboid.
    """

    points_cuboid = np.zeros([steps, steps, steps])

    for i in range(fineness):
        t = 2 * (i / fineness)
        point = ray(t)

        if (x_cor[0] <= point[0] < x_cor[-1]) and\
           (y_cor[0] <= point[1] < y_cor[-1]) and\
           (z_cor[0] <= point[2] < z_cor[-1]):

            xi, yi, zi = check_mini_cuboid(x_cor, y_cor, z_cor, point)
            points_cuboid[xi, yi, zi] += 1

    return points_cuboid


def create_line_matrix(ray, steps, x_cor, y_cor, z_cor, fineness):
    """creates the matrix for a single line integral"""

    matrix = ray_tracer(ray, steps, x_cor, y_cor, z_cor, fineness)
    length = (np.sum(matrix) / fineness) * np.linalg.norm(ray(0) - ray(2))
    if np.sum(matrix) > 0:
        matrix_normed = (matrix / np.sum(matrix)) * length
        return matrix_normed

    return matrix


def create_single_matrix_dic(ktr_num, rays_d, steps, x_cor, y_cor, z_cor, fineness):
    """creates a dic of matrices which are needed for a single k-transform"""

    matrices = {}

    for j in range(len(rays_d[0])):
        matrices[j] = create_line_matrix(rays_d[ktr_num][j], steps, x_cor, y_cor, z_cor, fineness)

    return matrices


def lineintegral_matrices_maker(rays_d, steps, x_cor, y_cor, z_cor, fineness=10**3):
    """Creates a dictionary in which all the matrices which are later used for
    the k-transforms"""

    line_matrices_d = {}

    number_ktrans = len(rays_d)

    for i in range(number_ktrans):
        line_matrices_d[i] =\
            create_single_matrix_dic(i, rays_d, steps, x_cor, y_cor, z_cor, fineness)

    return line_matrices_d


def matrix_line_integral(matrix, values):
    """calculates the line integral with a matrix multiplication, those matrices
    have been created in lineintegral_matrices_maker"""

    line_int = np.sum(matrix * values, axis=(2, 3, 4))

    return line_int


def k_transform(values, big_line_matrix_array):
    """
    The final single pixel k-transform.
    """

    integral = matrix_line_integral(big_line_matrix_array, values)
    k_transform_trafo_x_rays = np.exp(- integral)
    k_transform_trafo = np.sum(k_transform_trafo_x_rays, axis=1)
    k_transform_result = k_transform_trafo / k_transform_trafo_x_rays.shape[1]

    return k_transform_result


def distance_to_center(x, y, z):
    """
    calculates the distance of the point (x,y,z) from (0,0,0)
    """

    return (x**2 + y**2 + z**2) ** 0.5


def distance_creater(x_cor, y_cor, z_cor, steps):
    """
    returns the 3D numpy array in which every entry contains the distance to the
    center
    """

    x_cs = abs(x_cor[0] - x_cor[1]) / 2
    y_cs = abs(y_cor[0] - y_cor[1]) / 2
    z_cs = abs(z_cor[0] - z_cor[1]) / 2

    dist = np.zeros([steps, steps, steps])

    for i in range(steps):
        for j in range(steps):
            for k in range(steps):
                dist[i, j, k] = distance_to_center(x_cor[i] + x_cs,
                                                   y_cor[j] + y_cs,
                                                   z_cor[k] + z_cs)

    return dist


def dist_array_maker(dist):
    """
    takes 3D distance matrix and create a 1D array of all different distances
    """

    # .flatten() makes 1D array
    dist_f = dist.flatten()

    # round otherwise unique not working by computer precission errors
    dist_fr = np.round(dist_f, 6)

    # takes every values just ones
    dist_fru = np.unique(dist_fr)

    return dist_fru


def one_dimensional_line_integral_maker(dist, dist_fru, points_cuboid):
    """
    creates a 1D line integral array from points_cuboid by suming up the values
    of all the small cubes with the same distance to the center
    """

    integral_array = np.zeros(dist_fru.size)
    #   need to round since we did it in the dist_array_maker
    dist_r = np.round(dist, 6)
    for i, j in enumerate(dist_fru):
        d = (dist_r == j) * points_cuboid
        integral_array[i] = np.sum(d)

    return integral_array


def generate_rays(source_point, cuboid_coordinates, number_rays):
    """
    this function creates the rays in form of a dictionary, for a single
    k-trafo, those rays form together one ktra for a single source point
    """

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

    return rays


def generate_rays_d(source_point_d, cuboid_coordinates, number_rays):
    """
    generates all rays for all ktrafos in form of a dictionary
    """

    rays_d = {}
    for i in range(len(source_point_d)):
        rays_d[i] = generate_rays(source_point_d[str(i)], cuboid_coordinates, number_rays)
    
    return rays_d


def generate_line_integral_array_matrix(rays, cuboid_coordinates, steps, fineness):
    """
    Generates a matirx that conatains all the line integral values for a single
    k-trafo.
    """

    x_cor, y_cor, z_cor = coordinate_discretization(cuboid_coordinates, steps)
    dist = distance_creater(x_cor, y_cor, z_cor, steps)
    dist_fru = dist_array_maker(dist)

    liam = np.zeros((len(rays), len(dist_fru)))

    for i in range(len(rays)):
        matrix_in = create_line_matrix(rays[i], steps, x_cor, y_cor, z_cor, fineness)
        integral_array = one_dimensional_line_integral_maker(dist, dist_fru, matrix_in)
        liam[i:] = integral_array

    return liam


def generate_all_line_integral_array_matrix(rays_d, cuboid_coordinates, steps, fineness):
    """
    generates the liam for all k_trafos, the liam becomes by this from a 2D to
    a 3D array. axis = 2 are the single k-trafos.
    """
    #   only for shape
    liam = generate_line_integral_array_matrix(rays_d[0], cuboid_coordinates, steps, fineness)

    liam_all = np.zeros((len(rays_d), liam.shape[0], liam.shape[1]))
    # for i in tqdm(range(len(rays_d))):
    for i in tqdm(range(len(rays_d))):
        liam = generate_line_integral_array_matrix(rays_d[i], cuboid_coordinates, steps, fineness)
        liam_all[i, :, :] = liam

    return liam_all


def k_trafo_one_dim(liam, values):
    """
    the ktra for the one dimensional case
    Carfeful: the values are here a 1d vector
    """

    line_int = np.sum(liam * values, axis=1)
    line_int_e = np.exp(-line_int)
    k_value = np.sum(line_int_e) / len(line_int)

    return k_value


def create_source_point_d(radius, number_ktrans, perc_circle=1, start_index=0):
    """ creates a dictionary of source points, radius of the circle z=0
        update by symmetry of the box: it makes sense to create the rays
        only between 0 and 90 degrees, so we added twice the factor 0.25"""

    source_point_d = {}

    for i in np.arange(start_index, number_ktrans):
        source_point_d[str(i)] = np.array([np.sin(2 * np.pi * i * perc_circle / number_ktrans),
                                           np.cos(2 * np.pi * i * perc_circle / number_ktrans),
                                           0]) * radius

    return source_point_d


def k_trafo_one_dim_all(liam_all, values):
    """
    calculates all the ktra values simultaniously and returns them in form
    of a vector
    Carfeful: the values are here a 1d vector
    """

    number_ktra, number_rays, len_vec_values = liam_all.shape
    #   liam_all (number_ktra, rays, vector of values)
    #   reshape the values for making the multiplication working
    line_int = np.sum(liam_all * values, axis=2)
    line_int_e = np.exp(-line_int)
    ktra_array = np.sum(line_int_e, axis=1) / number_rays

    return ktra_array


def generate_spxt_matrix(liam_all):
    """generates the 2D matrix for radical symmetric single pixel transform"""
   
    number_ktra, number_rays, len_vec_values = liam_all.shape
    spxt_matrix = np.sum(liam_all, axis=1) / number_rays # measurements x object 1d
    
    return spxt_matrix


def spxt_make_measurement(spxt_matrix, object_1d):
    """ returns a measurement - a matrix multiplication """
    
    spxt_measurement = np.dot(spxt_matrix, object_1d)

    return spxt_measurement

