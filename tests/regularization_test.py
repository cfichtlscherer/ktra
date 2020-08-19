"""
Apr 19, 2020
Christopher Fichtlscherer (fichtlscherer@mailbox.org)
GNU General Public License
"""


#   import numpy as np
#   import pytest

#   from ktra.regularization import get_all_edges
#   from ktra.regularization import punish_the_edges


"""
def test_get_all_edges():

    matrix = np.array([[[1, 4], [5, 5]], [[14, 0], [2.5, 0.1]]])

    # sort the array, we are only interested if they contain the same elements
    edges = np.array([3, 0, 14, 2.4, 4, 1, 11.5, 0.1, 13, 4, 2.5, 4.9]).sort()

    edges_created = get_all_edges(matrix).sort()

    np.testing.assert_array_equal(edges, edges_created)

def test_punish_the_edges():

    matrix = np.array([[[1, 4], [5, 5]], [[14, 0], [2.5, 0.1]]])
    k = 10
    a = 0.1

    punish_value = punish_the_edges(matrix, k, a)

    edges = get_all_edges(matrix)
    print(edges)
    edges_punished = (2 / np.pi) * a * np.arctan(k * edges)
    print(edges_punished)
    print("###################")
    punish_value_control = np.sum(edges_punished**2)**0.5

    np.testing.assert_almost_equal(punish_value, punish_value_control)
"""
