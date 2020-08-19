"""
Jun 12, 2020
Christopher Fichtlscherer (fichtlscherer@mailbox.org)
GNU General Public License

We want to produce several plots that show how the algorithm works.
"""

import matplotlib

import numpy as np
import matplotlib.pyplot as plt

if False:
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
    v["x"] = np.ones(v["N"]) * np.sum(y) / v["N"] # we start with average values
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


def step_2(v, y, lamb, v_memory):

    if v["k"] == v["N"] - 1:
        v["x"][v["N"] - 1] = v["v_min"] + v["u_min"]
        v_memory += [list(v["x"])]
        v["stop"] = True
    
    return v, v_memory


def step_3_4_5_6_7(v, y, lamb, v_memory):

    # step 7
    while v["k"] < v["N"] - 1:
        # step 3
        if y[v["k"] + 1] + v["u_min"] < v["v_min"] - lamb:
            for i in range(v["k_0"], v["k_minus"] + 1):
                v["x"][i] = v["v_min"]
                v_memory += [list(v["x"])]

            v["k"] = v["k_0"] = v["k_minus"] = v["k_plus"] = v["k_minus"] + 1
            v["v_min"] = y[v["k"]]
            v["v_max"] = y[v["k"]] + 2 * lamb
            v["u_min"] = lamb
            v["u_max"] = -lamb

        # step 4
        elif y[v["k"] + 1] + v["u_max"] > v["v_max"] + lamb:
            for i in range(v["k_0"], v["k_plus"] + 1):
                v["x"][i] = v["v_max"]
                v_memory += [list(v["x"])]
            
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

    return v, v_memory


def step_8_9_10(v, y, lamb, v_memory):

    # step 8
    if v["u_min"] < 0:
        for i in range(v["k_0"], v["k_minus"] + 1):
            v["x"][i] = v["v_min"]
            v_memory += [list(v["x"])]

        v["k"] = v["k_0"] = v["k_minus"] = v["k_minus"] + 1
        v["v_min"] = y[v["k"]]
        v["u_min"] = lamb
        v["u_max"] = y[v["k"]] + lamb - v["v_max"]
 

    # step 9
    elif v["u_max"] > 0:
        print("y")
        for i in range(v["k_0"], v["k_plus"] + 1):
            v["x"][i] = v["v_max"]
            v_memory += [list(v["x"])]

        v["k"] = v["k_0"] = v["k_plus"] = v["k_plus"] + 1
        v["v_max"] = y[v["k"]]
        v["u_max"] = -lamb
        v["u_min"] = y[v["k"]] - lamb - v["v_min"]
        
    # step 10
    else:
        for i in range(v["k_0"], v["N"]):
            v["x"][i] = v["v_min"] + (v["u_min"] / (v["k"] - v["k_0"] + 1))
            v_memory += [list(v["x"])]

        v["stop"] = True
    
    return v, v_memory



def tv_denoising_algorithm(y, lamb):

    v_memory = []

    v = step_1(y, lamb)
    
    v_memory += [list(v["x"])]

    while True:
        v, v_memory = step_2(v, y, lamb, v_memory)

        if v["stop"] == True:
            return v["x"], v_memory

        v, v_memory = step_3_4_5_6_7(v, y, lamb, v_memory)

        v, v_memory= step_8_9_10(v, y, lamb, v_memory)

        if v["stop"] == True:
            return v["x"], v_memory

np.random.seed(0)

n = 100

y = np.arange(n)
lamb_1 = 15
lamb_2 = 70

#y2 = 10 * np.repeat(np.array([1,1,8,8,5,5,5,2,2,2]), int(n/10)) + np.random.random(n) * 12
y2 = 10 * np.repeat(np.array([0,1,0,2,0,3,0,4,0,5]), int(n/10)) + (np.random.random(n)-0.5) * 30

#y2 = np.random.random(100) * 200

z_1, v_memory_1 = tv_denoising_algorithm(y2, lamb_1)
z_2, v_memory_2 = tv_denoising_algorithm(y2, lamb_2)

stop_1 = 21
stop_2 = 60

fig, ax = plt.subplots(2, 2, figsize=(6.5,3.5), sharex=True, sharey=True)

ax[0, 0].plot(y, y2, "-")
ax[0, 0].plot(y, z_1, "--", color="black")
ax[0, 0].plot(y[:stop_1], v_memory_1[stop_1][:stop_1], "-")
ax[0, 0].grid(color='black', linestyle=':', linewidth=0.3)

ax[1, 0].plot(y, y2, "-")
ax[1, 0].plot(y, z_2, "--", color="black")
ax[1, 0].plot(y[:stop_1], v_memory_2[stop_1][:stop_1], "-")
ax[1, 0].grid(color='black', linestyle=':', linewidth=0.3)

ax[0, 1].plot(y,y2, "-")
ax[0, 1].plot(y,z_1, "--", color="black")
ax[0, 1].plot(y[:stop_2], v_memory_1[stop_2][:stop_2], "-")
ax[0, 1].grid(color='black', linestyle=':', linewidth=0.3)

ax[1, 1].plot(y,y2, "-")
ax[1, 1].plot(y,z_2, "--", color="black")
ax[1, 1].plot(y[:stop_2], v_memory_2[stop_2][:stop_2], "-")
ax[1, 1].grid(color='black', linestyle=':', linewidth=0.3)

# plt.show()

plt.savefig(figures_path + 'how_algorithm_works.pgf')

