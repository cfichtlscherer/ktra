"""
Apr 14, 2020
Christopher Fichtlscherer (fichtlscherer@mailbox.org)
GNU General Public License

here we compare how much difference there is between the values of the K-Transform of the continuous
object and the discrete object and how we can bring those values very close together
"""

import matplotlib

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

figures_path = '/home/cpf/Desktop/k_transform_tomography/figures/'

matplotlib.use("pgf")           
matplotlib.rcParams.update({    
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',     
    'text.usetex': True,        
    'pgf.rcfonts': False,       
})                              

co_re_1 = np.load("co_re_1.npy")
dis_re_1 = np.load("dis_re_1.npy")
perc_re_1 = np.load("perc_re_1.npy")

co_re_2 = np.load("co_re_2.npy")
dis_re_2 = np.load("dis_re_2.npy")
perc_re_2 = np.load("perc_re_2.npy")

plt.figure(figsize=(6,3.4))

plt.grid(True, color='0.9', linestyle='-')
#
plt.axhline(co_re_1[0], color="darkorange")
#plt.plot(range(1, len(co_re_1) + 1), co_re_1, "-", color="red", label="Continuously")
plt.plot(range(1, len(co_re_1) + 1), dis_re_1, ".-", color="darkorange",
        label= r"$\mathcal{K}_{dis} \ $"+ "Object 1")#, linewidth=0.5)
#plt.plot(range(1, len(co_re_1) + 1), perc_re_1, ".-", color="red", label="Relative Difference")
#
plt.axhline(co_re_2[0], color="royalblue")
#plt.plot(range(1, len(co_re_1) + 1), co_re_2, "-", color="blue", label="Continuously")
plt.plot(range(1, len(co_re_1) + 1), dis_re_2, ".-", color="royalblue",
        label= r"$\mathcal{K}_{dis} \ $"+ "Object 2")#, linewidth=0.5)
#plt.plot(range(1, len(co_re_1) + 1), perc_re_2, ".-", color="blue", label="Relative Difference")
#
plt.text(23.5, 0.80, r'$\mathcal{K}_{con} = \ $' + str(np.round(co_re_1[0],5)), color="black", fontsize=10)
plt.text(23.5, 0.89, r'$\mathcal{K}_{con} = \ $' + str(np.round(co_re_2[0],5)), color="black", fontsize=10)
#
#plt.title('Comparison of the K-Transform values between the continuous and the discretized object')
plt.xlabel('Level of Discretization [$n$]')
plt.ylabel('Value of the K-Transform')
plt.legend(loc="lower right")

#y_pos = (0.4, 0.6, co_re_1[0]-0.01, co_re_2[0]+0.01, 0.8, 1.0)
#y_lab = (0.4, 0.6, co_re_1[10] , co_re_2[10], 0.8, 1.0)
#plt.yticks(y_pos, y_lab)

plt.savefig(figures_path + 'discretization_error.pgf', bbox_inches='tight') 
