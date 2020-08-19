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
    
    if rad_pos <= 0.4: return 0.7
    if rad_pos <= 0.6: return 0.4
    if rad_pos <= 0.8: return 0.2
                             
    return 0                     
                                    

x = np.arange(-1, 1, 0.01)
y = np.arange(-1, 1, 0.01)

cont = np.zeros((x.size, y.size))

for i in range(x.size):
    for j in range(y.size):
        cont[i,j] = inner_of_box(x[i], y[j])

res = ([3.34195134e-01, 3.34195134e-01, 3.34195134e-01, 3.34195134e-01,
 3.34195134e-01, 3.34195134e-01, 3.34195134e-01, 3.34195134e-01,
 3.34195134e-01, 2.78904384e-01, 2.78904384e-01, 2.48883750e-01,
 2.06035550e-01, 2.06035550e-01, 1.52170496e-01, 1.52170496e-01,
 1.52170496e-01, 1.52170496e-01, 1.52170496e-01, 1.52170496e-01,
 1.11722024e-01, 1.11722024e-01, 1.01822370e-01, 2.89274539e-02,
 2.89274539e-02, 1.13122390e-02, 1.13122390e-02, 1.31583988e-02,
 2.99037664e-02, 2.99037664e-02, 2.99037664e-02, 1.64142657e-02,
 1.64142657e-02, 1.64142657e-02, 1.64142657e-02, 1.64142657e-02,
 1.64142657e-02, 1.64142657e-02, 2.35782025e-02, 3.95651120e-02,
 3.39937817e-02, 3.20622103e-02, 2.33688306e-02, 2.33688306e-02,
 2.33688306e-02, 2.33688306e-02, 2.00539058e-02, 8.91864877e-03,
 4.12667947e-14, 4.12667947e-14, 4.12667947e-14, 4.12667947e-14,
 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
 3.43604133e-14, 3.43604133e-14, 3.43604133e-14, 3.43604133e-14,
 3.43604133e-14, 3.43604133e-14, 3.43604133e-14, 3.43604133e-14,
 3.43604133e-14, 3.43604133e-14])                             

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


