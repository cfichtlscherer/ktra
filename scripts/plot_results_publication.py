"""
Jun 15, 2021
Christopher Fichtlscherer (fichtlscherer@mailbox.org)
GNU General Public License

Plot reults of some reconstructions in a nice way for publication.
"""

import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
import pandas as pd
import scipy as scp
import os as os
import pylab as pl

from mpl_toolkits.axes_grid1 import make_axes_locatable

from ktra.optimization_process import make_threed_back

###################################################################################################

def generate_result_plot(obj):

    if obj == 1:

        path = "/home/cpf/Desktop/ktransform_simulation_results/files/calulations_missing/20-0.2-0.01-1/logbook-20-0.01-0.2.txt"

        def inner_of_box(x, y, z):                                                              
            rad_pos = (x**2 + y**2 + z**2)**0.5                                                   
            if rad_pos <= 0.8: return 0.8                
            return 0                                     


    if obj == 2:

        path = "/home/cpf/Desktop/ktransform_simulation_results/files/calulations_missing/20-0.04-0.01-3/logbook-20-0.01-0.04.txt"

        def inner_of_box(x, y, z):                                                              
            rad_pos = (x**2 + y**2 + z**2)**0.5                                                   
            if rad_pos <= 0.4: return 0.9                
            if rad_pos <= 0.6: return 0.7                
            if rad_pos <= 0.8: return 0.4                
            return 0                                     


    if obj == 3:

        path = "/home/cpf/Desktop/ktransform_simulation_results/files/calulations_missing/20-0.04-0.02-2/logbook-20-0.02-0.04.txt"

        def inner_of_box(x, y, z):                                                              
            rad_pos = (x**2 + y**2 + z**2)**0.5                                                   
            if rad_pos <= 0.7: return 0
            if rad_pos <= 0.8: return 0.8                
            return 0                                     


    x_comp = np.arange(-1, 1, 0.01)                                         
    y_comp = np.arange(-1, 1, 0.01)                                         
                                                                        
    cont = np.zeros((x_comp.size, y_comp.size))                             
                                                                        
    for i in range(x_comp.size):                                            
        for j in range(y_comp.size):                                        
            cont[i,j] = inner_of_box(x_comp[i], y_comp[j], 0)               
                                                                        
    cuboid_coordinates = {'x1': -1, 'x2': 1, 'y1': -1, 'y2': 1, 'z1': -1, 'z2': 1}      
    steps = 20 
    
    logbook = pd.read_csv(path)
    reconstruction_string = logbook.iloc[5000][5]
    reconstruction_string_cut = reconstruction_string[2:-2]
    reconstruction = np.fromstring(reconstruction_string_cut, sep =" ")

    results3d = make_threed_back(reconstruction, steps, cuboid_coordinates)                       


    fig, ax1 = plt.subplots(figsize=(7,7))
    left, bottom, width, height = [0.594, 0.579, 0.3, 0.3]
    ax2 = fig.add_axes([left, bottom, width, height])
    ax1.imshow(results3d[10], vmin=0, vmax=1)
    ax1.axis(False)

    ax2.imshow(cont, vmin=0, vmax=1)
    ax2.axis(False)

    autoAxis = ax2.axis()
    rec = Rectangle((autoAxis[0]-0.7,autoAxis[2]-0.2),(autoAxis[1]-autoAxis[0])+1,(autoAxis[3]-autoAxis[2])+0.4,fill=False,lw=2)
    rec = ax2.add_patch(rec)
    rec.set_clip_on(False)
    
    plt.savefig("/home/cpf/Desktop/publications-in-work/K-Transform-publication/plot-" + str(obj) + ".png")
    
################################################################################

if False:
    generate_result_plot(1)
    generate_result_plot(2)
    generate_result_plot(3)

################################################################################


a = np.array([[0,1]])
pl.figure(figsize=(9, 1.5))
img = pl.imshow(a)
pl.gca().set_visible(False)
cax = pl.axes([0.1, 0.2, 0.8, 0.6])
pl.colorbar(orientation="horizontal", cax=cax)
pl.savefig("/home/cpf/Desktop/publications-in-work/K-Transform-publication/colorbar.png")


