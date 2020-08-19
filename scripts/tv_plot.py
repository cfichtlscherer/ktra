"""
Apr 04, 2020
Christopher Fichtlscherer (fichtlscherer@mailbox.org)
GNU General Public License

This script creates a pgf plot which is demonstrating the algorithm.

A direct Algorithm for 1-D Total Variation Denoising

Input: real sequence y = np.array([1,2,3,4,5,6])
       real parameter lamb

Output: real sequence x = np.array([1,2,3,4,5,6])

x is the solution of the minimization problem (1).
"""

import matplotlib
import scipy.optimize

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2)

matplotlib.use("pgf")           
matplotlib.rcParams.update({    
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',     
    'text.usetex': True,        
    'pgf.rcfonts': False,       
})                              

figures_path = '/home/cpf/Desktop/k_transform_tomography/figures/'

def step_1(y, lamb):

    v = {}
    v["N"] = y.size
    v["x"] = np.zeros(v["N"])
    v["k"] = 0
    v["k_0"] = 0
    v["k_minus"] = 0
    v["k_plus"] = 0
    v["v_min"] = y[0] - lamb
    v["v_max"] = y[0] + lamb
    v["u_min"] = lamb
    v["u_max"] = -lamb
    v["stop"] = False

    return v


def step_2(v, y, lamb):

    if v["k"] == v["N"] - 1:
        v["x"][v["N"] - 1] = v["v_min"] + v["u_min"]
        v["stop"] = True

    return v


def step_3_4_5_6_7(v, y, lamb):

    # step 7
    while v["k"] < v["N"] - 1:
        # step 3
        if y[v["k"] + 1] + v["u_min"] < v["v_min"] - lamb:
            for i in range(v["k_0"], v["k_minus"] + 1):
                v["x"][i] = v["v_min"]
            v["k"] = v["k_0"] = v["k_minus"] = v["k_plus"] = v["k_minus"] + 1
            v["v_min"] = y[v["k"]]
            v["v_max"] = y[v["k"]] + 2 * lamb
            v["u_min"] = lamb
            v["u_max"] = -lamb

        # step 4
        elif y[v["k"] + 1] + v["u_max"] > v["v_max"] + lamb:
            for i in range(v["k_0"], v["k_plus"] + 1):
                v["x"][i] = v["v_max"]
            v["k"] = v["k_0"] = v["k_minus"] = v["k_plus"] = v["k_plus"] + 1
            v["v_min"] = y[v["k"]] - 2 * lamb
            v["v_max"] = y[v["k"]]
            v["u_min"] = lamb
            v["u_max"] = -lamb

        # step 5
        else:
            v["k"] = v["k"] + 1
            v["u_min"] = v["u_min"] + y[v["k"]] - v["v_min"]
            v["u_max"] = v["u_max"] + y[v["k"]] - v["v_max"]
        # step 6
            if v["u_min"] >= lamb:
                v["v_min"] = v["v_min"] + ((v["u_min"] - lamb) / (v["k"] - v["k_0"] + 1))
                v["u_min"] = lamb
                v["k_minus"] = v["k"]

            if v["u_max"] <= -lamb:
                v["v_max"] = v["v_max"] + ((v["u_max"] + lamb) / (v["k"] - v["k_0"] + 1))
                v["u_max"] = -lamb
                v["k_plus"] = v["k"]

    return v


def step_8_9_10(v, y, lamb):

    # step 8
    if v["u_min"] < 0:
        for i in range(v["k_0"], v["k_minus"] + 1):
            v["x"][i] = v["v_min"]
        v["k"] = v["k_0"] = v["k_minus"] = v["k_minus"] + 1
        v["v_min"] = y[v["k"]]
        v["u_min"] = lamb
        v["u_max"] = y[v["k"]] + lamb - v["v_max"]

    # step 9
    elif v["u_max"] > 0:
        for i in range(v["k_0"], v["k_plus"] + 1):
            v["x"][i] = v["v_max"]
        v["k"] = v["k_0"] = v["k_plus"] = v["k_plus"] + 1
        v["v_max"] = y[v["k"]]
        v["u_max"] = -lamb
        v["u_min"] = y[v["k"]] - lamb - v["v_min"]

    # step 10
    else:
        for i in range(v["k_0"], v["N"]):
            v["x"][i] = v["v_min"] + (v["u_min"] / (v["k"] - v["k_0"] + 1))
        v["stop"] = True

    return v


def tv_denoising_algorithm(y, lamb):

    v = step_1(y, lamb)

    while True:
        v = step_2(v, y, lamb)
        if v["stop"] == True:
            return v["x"]

        v = step_3_4_5_6_7(v, y, lamb)
        v = step_8_9_10(v, y, lamb)

        if v["stop"] == True:
            return v["x"]


def tikhonov_difference(x):
    """ calculates the 2-norm of the gradient"""

    return np.linalg.norm(x[1:] - x[:-1], 2)**2


def tik_min_function(x, y, ti_la):
    
    return np.linalg.norm(x-y,2)**2 + ti_la * tikhonov_difference(x)


n = 100

y = np.arange(n)
lamb = 1
y2 = 0.1 * np.repeat(np.array([1,1,9,9,4,4,4,2,2,2]), 10) + np.random.random(n) * 0.12

z = tv_denoising_algorithm(y2, lamb)

x = 50 * np.ones(y2.size)

tik = scipy.optimize.minimize(tik_min_function,                                            
                                x,                                                     
                                args = (y2, 4),    
                                method = 'L-BFGS-B',                                   
                                bounds = scipy.optimize.Bounds(0, 100),                  
                                options = {'disp': False}).x



plt.figure(figsize=(6,3.4))
plt.plot(y,y2, "-", label="Signal with Noise")
plt.plot(y, tik, "-", label="Tikhonov Regularization", color="green")
plt.plot(y,z, "-", label="TV Regularization", color="orange")
plt.grid(color='black', linestyle=':', linewidth=0.3)
plt.legend(loc='upper right')
plt.xlabel(r'$x$')
plt.ylabel(r'$f(x)$')

plt.savefig(figures_path + 'tv_demonstration.pgf', bbox_inches='tight')

