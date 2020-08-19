"""
May 26, 2020
Christopher Fichtlscherer (fichtlscherer@mailbox.org)
GNU General Public License

Here we calculate for a simple example different values of the K-Transform for different
levels of discretization.
"""

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from tqdm import tqdm


from ktra.k_transform import k_transform
from ktra.k_transform import create_source_point_d
from ktra.k_transform import generate_rays_d
from ktra.optimization_process import create_con_results

cuboid_coordinates = {'x1': -0.5, 'x2': 0.5, 'y1': -0.5, 'y2': 0.5, 'z1': -0.5, 'z2': 0.5}
number_ktrans = 1
fineness = 10**6
radius = 1.5

def inner_of_box(x, y, z):
    """ Different shells around the origin."""

    if (-0.5 <= x <= 0.5) and (-0.5 <= y <= 0.5) and (-0.5 <= z <= 0.5):
        return 1

    return 0


for i in [1, 3, 5, 1000]:
    number_rays = {'dim_1': i, 'dim_2': 1}

    source_point_d = create_source_point_d(radius, number_ktrans, perc_circle=0.125)
    rays_d = generate_rays_d(source_point_d, cuboid_coordinates, number_rays)
    cont_results = create_con_results(rays_d, inner_of_box, fineness)
    print(str(i) + " rays -  ", cont_results[0])
    if i < 10:
       for j in range(i):
           print(rays_d[0][j](0), rays_d[0][j](2))

"""
results: 
1 rays -   0.36787980905106743
[0.  1.5 0. ] [ 0.  -1.5  0. ]

3 rays -   0.43379907557037906
[0.  1.5 0. ] [-0.8660254 -1.5        0.       ]
[0.  1.5 0. ] [ 0.  -1.5  0. ]
[0.  1.5 0. ] [ 0.8660254 -1.5        0.       ]

5 rays -   0.5083889832764983
[0.  1.5 0. ] [-1.15470054 -1.5         0.        ]
[0.  1.5 0. ] [-0.57735027 -1.5         0.        ]
[0.  1.5 0. ] [ 0.  -1.5  0. ]
[0.  1.5 0. ] [ 0.57735027 -1.5         0.        ]
[0.  1.5 0. ] [ 1.15470054 -1.5         0.        ]

1000 rays -   0.5912083502828241
"""
