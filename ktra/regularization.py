"""
Apr 19, 2020
Christopher Fichtlscherer (fichtlscherer@mailbox.org)
GNU General Public License

the functions used for regularization
"""


import numpy as np


def get_all_edges(matrix):
    """returns all the hights of all the edges of the matrix
    [[1,1],[4,9]] -> [0,5,3,8]
    A N**3 matrix contains 3 * (N-1) * N**2 edges.
    """

    # for getting the edges to the corner
    matrix_pad = np.pad(matrix, pad_width=1, mode='constant', constant_values=0)

    # .ravel() makes from an ND array a 1D array
    x_dif = ((np.roll(matrix_pad, 1, axis=0) - matrix_pad)[1:, :, :]).ravel()
    y_dif = ((np.roll(matrix_pad, 1, axis=1) - matrix_pad)[:, 1:, :]).ravel()
    z_dif = ((np.roll(matrix_pad, 1, axis=2) - matrix_pad)[:, :, 1:]).ravel()

    edges = np.concatenate((x_dif, y_dif, z_dif), axis=None)
    edges_abs = abs(edges)

    return edges_abs


def smoothed_one_norm(vector, alpha):
    """
    a version of the one norm smoothed at x = 0
    """

    value = (1 / alpha) * (np.log(1 + np.exp(-alpha * vector)) + np.log(1 + np.exp(alpha * vector)))

    return np.sum(value)


def punish_the_edges(matrix, k, a, alpha=10**-7):
    """creates all the edges with the function get_all_edges and calculates
    a sigmoid function of them (arctan(kx) with amplitude a is used).
    k is the steepness of the logistic function.
    it returns the 2 Norm of all those values."""

    edges = get_all_edges(matrix)
    edges_punished = (2 / np.pi) * a * np.arctan(k * edges)

    punish_value = smoothed_one_norm(edges_punished, alpha) / len(edges)

    return punish_value


def tv_regularisation(array_oned, a, k):
    """
    calculating the height of all edges in the 1D array, punish them with a
    sigmoid function and sum the punishterm up
    k is the steepness of the logistic function
    a is the hight of the sigmoid function
    also punishing now the outerest value by the difference to 0, otherwise
    some sticking to the walls will occure
    """

    edges = abs(array_oned - np.roll(array_oned, 1))[1:]
    edges_end = np.append(edges, array_oned[-1])
    edges_punished = (2 / np.pi) * a * np.arctan(k * edges_end)
    punish_term = np.sum(edges_punished ** 2)**0.5

    return punish_term
