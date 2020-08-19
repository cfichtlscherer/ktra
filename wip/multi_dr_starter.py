"""
May 14, 2020
Christopher Fichtlscherer (fichtlscherer@mailbox.org)
GNU General Public License

starts the file dr_starter multiple times
"""

import os
from multiprocessing import Pool                                                

number_processes = 1

alpha_list = [0]
gamma_list = [10**6]
perc_noise_list = [0]

"""
number_processes = 16

alpha_list =      [0.0, 0.0, 0.0, 0.00001, 0.00001, 0.00001, 0.00000, 0.000001, 0.0000001, 0.000000000, 0.000001, 0.000001, 0.000005, 0.000001, 0.000001, 0.000001]
gamma_list =      [0.1, 0.2, 1.0, 0.10000, 0.20000, 0.50000, 1.00000, 0.500000, 0.5000000, 1.000000000, 1.000000, 0.100000, 0.100000, 0.200000, 0.500000, 1.000000]
perc_noise_list = [0.0, 0.0, 0.0, 0.00100, 0.00100, 0.00100, 0.00100, 0.001000, 0.0010000, 0.000100000, 0.000100, 0.000100, 0.000100, 0.000100, 0.000100, 0.000100]
"""

cwd = os.getcwd()

for i in range(number_processes):
    os.mkdir(str(i))
    os.system("cp *.py " + str(i))
    with open(str(i) + '/dr_starter.py', 'r') as file :
        filedata = file.read()
        filedata = filedata.replace('alpha', str(alpha_list[i]))
        filedata = filedata.replace('gamma', str(gamma_list[i]))
        filedata = filedata.replace('perc_noise', str(perc_noise_list[i]))
    with open(str(i) + '/dr_starter.py', 'w') as file:
        file.write(filedata)

processes = []

for i in range(number_processes):
    change = 'cd ' + str(i) + '&& '
    run_command = 'python3 dr_starter.py' 
    processes += [change + run_command]


def run_process(process):                                                             
    os.system('{}'.format(process))                                       
                                                                                
pool = Pool(processes=number_processes)                                                        
pool.map(run_process, processes)          
