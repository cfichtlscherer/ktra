"""
Jun 17, 2020
Christopher Fichtlscherer (fichtlscherer@mailbox.org)
GNU General Public License

plot of the results:
commit 04f173576d4f61775834a495cc278818130115e9
"""

import numpy as np
import matplotlib.pyplot as plt 

from ktra.optimization_process import make_threed_back

def inner_of_box(x, y):                    
    """ Different shells around the origin."""
                                              
    rad_pos = (x**2 + y**2)**0.5       
    
    if rad_pos <= 0.6: return 0.6
    if rad_pos <= 1.0: return 0.3
                             
    return 0                     
                                    

x = np.arange(-1, 1, 0.01)
y = np.arange(-1, 1, 0.01)

cont = np.zeros((x.size, y.size))

for i in range(x.size):
    for j in range(y.size):
        cont[i,j] = inner_of_box(x[i], y[j])

res = ([0.38705688, 0.38705688, 0.38705688, 0.38705688, 0.38705688, 0.38705688,
        0.38705688, 0.38705688, 0.38705688, 0.38705688, 0.38705688, 0.38705688,
        0.38705688, 0.38705688, 0.38705688, 0.38705688, 0.38705688, 0.38705688,
        0.38705688, 0.38705688, 0.26610687, 0.26610687, 0.26610687, 0.26610687,
        0.26610687, 0.26610687, 0.26610687, 0.26610687, 0.26610687, 0.26610687,
        0.2579235 , 0.209664  , 0.19521278, 0.11621265, 0.11621265, 0.04358677,
        0.04358677, 0.04358677, 0.04358677, 0.01065764, 0.01065764, 0.01065764,
        0.00957863, 0.00957863, 0.00957863, 0.00957863, 0.00957863, 0.00957863,
        0.00957863, 0.00957863, 0.00957863, 0.00957863, 0.00957863, 0.04000303,
        0.04000303, 0.04000303, 0.04000303, 0.01263719, 0.01134836, 0.01134836,
        0.01134836, 0.01134836, 0.01134836, 0.01134836, 0.01134836, 0.01134836])

steps = 16
cuboid_coordinates = {'x1': -1, 'x2': 1, 'y1': -1, 'y2': 1, 'z1': -1, 'z2': 1}

results3d = make_threed_back(res, steps, cuboid_coordinates)

#border_results = np.zeros((26, 26))
#border_results[3:-3,3:-3] = results3d[9]

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7,3.5))
ax[0].imshow(cont, vmin=0, vmax=1.2)
ax[0].axis(False)
ax[1].imshow(results3d[9], vmin=0, vmax=1.2)
ax[1].axis(False)
plt.show()


