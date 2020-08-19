"""
Apr 14, 2020
Christopher Fichtlscherer (fichtlscherer@mailbox.org)
GNU General Public License

here we compare how much difference there is between the values of the K-Transform of the continuous
object and the discrete object and how we can bring those values very close together
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



def inner_of_box_1(x, y, z):
    """ Different shells around the origin."""

    rad_pos = (x**2 + y**2 + z**2)**0.5
                                        
    if rad_pos <= 0.3: return 0.0       
    if rad_pos <= 0.7: return 0.0       
    if rad_pos <= 0.9: return 0.8       
                                        
    return 0                            


def inner_of_box_2(x, y, z):
    """ Different shells around the origin."""

    rad_pos = (x**2 + y**2 + z**2)**0.5
    
    if rad_pos <= 0.3: return 0.9
    if rad_pos <= 0.7: return 0.6
    if rad_pos <= 0.9: return 0.3
                                        
    return 0                            


steps = 60
number_ktrans = 10
number_rays = {'dim_1': 5, 'dim_2': 5}
fineness = 10**3

def compare(steps, number_ktrans, number_rays, fineness, inner_of_box):

    radius = 3
    cuboid_coordinates = {'x1': -1, 'x2': 1, 'y1': -1, 'y2': 1, 'z1': -1, 'z2': 1}

    x_cor, y_cor, z_cor = coordinate_discretization(cuboid_coordinates, steps)

    discrete_model = discretization_of_model(x_cor, y_cor, z_cor, inner_of_box, steps)
    one_d_model = threed_to_oned(discrete_model, cuboid_coordinates, steps)

    source_point_d = create_source_point_d(radius, number_ktrans, perc_circle=0.125)
    rays_d = generate_rays_d(source_point_d, cuboid_coordinates, number_rays)
    liam_all = generate_all_line_integral_array_matrix(rays_d, cuboid_coordinates, steps, fineness)
    cont_results = create_con_results(rays_d, inner_of_box, fineness)
    dis_results = k_trafo_one_dim_all(liam_all, one_d_model)

    return cont_results, dis_results

steprange = 30


co_re_list_1 = []
dis_re_list_1 = []
perc_re_list_1 = []

for steps in range(1, steprange):
    co, dis = compare(steps, number_ktrans, number_rays, fineness, inner_of_box_1)
    co = co[0]
    dis_dif = [abs(dis - co) for dis in dis]
    dis = dis[np.argmax(dis_dif)]
    perc_dif = abs((dis-co)/co)
    print(steps, co, dis, perc_dif)
    co_re_list_1 += [co]
    dis_re_list_1 += [dis]
    perc_re_list_1 += [perc_dif]
    np.save("co_re_1", np.asarray(co_re_list_1))
    np.save("dis_re_1", np.asarray(dis_re_list_1))
    np.save("perc_re_1", np.asarray(perc_re_list_1))


co_re_list_2 = []
dis_re_list_2 = []
perc_re_list_2 = []


for steps in range(1, steprange):
    co, dis = compare(steps, number_ktrans, number_rays, fineness, inner_of_box_2)
    co = co[0]
    dis_dif = [abs(dis - co) for dis in dis]
    dis = dis[np.argmax(dis_dif)]
    perc_dif = abs((dis-co)/co)
    print(steps, co, dis, perc_dif)
    co_re_list_2 += [co]
    dis_re_list_2 += [dis]
    perc_re_list_2 += [perc_dif]
    np.save("co_re_2", np.asarray(co_re_list_2))
    np.save("dis_re_2", np.asarray(dis_re_list_2))
    np.save("perc_re_2", np.asarray(perc_re_list_2))


#plt.grid(True, color='0.9', linestyle='-')
#plt.plot(range(1, steprange), co_re_list, label="Continuously")
#plt.plot(range(1, steprange), dis_re_list, "o-", label="Discrete")
#plt.plot(range(1, steprange), perc_re_list, "o-",label="Relative Difference")
#plt.title('Comparison of the K-Transform values between the continuous and the discretized object')
#plt.xlabel('Number of steps')
#plt.ylabel('Value of the K-Transform')
#plt.legend(loc="middle right")
#plt.show()

