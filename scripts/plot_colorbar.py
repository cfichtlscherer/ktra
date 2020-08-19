"""
Aug 15, 2020
Christopher Fichtlscherer (fichtlscherer@mailbox.org)
GNU General Public License

Create here the colorbar for the results
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib

matplotlib.use("pgf")           
matplotlib.rcParams.update({    
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',     
    'text.usetex': True,        
    'pgf.rcfonts': False,       
})                              

figures_path = '/home/cpf/Desktop/k_transform_tomography/figures/'
fig = plt.figure(figsize=(4,1.5))
ax = fig.add_axes([0.05, 0.80, 0.9, 0.1])

cb = mpl.colorbar.ColorbarBase(ax, orientation='horizontal') 

plt.savefig(figures_path + "colorbar.pgf", bbox_inches='tight')
