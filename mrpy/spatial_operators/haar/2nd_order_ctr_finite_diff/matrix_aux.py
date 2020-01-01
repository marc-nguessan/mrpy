from __future__ import print_function, division

"""...

"""

import petsc4py.PETSc as petsc
from six.moves import range
import config as cfg
from mrpy.mr_utils import mesh
import numpy as np
import math
import importlib

def matrix_add(tree, matrix, row, value, level, index_x=0, index_y=0, index_z=0, add_to_col=0):
    """...

    """

    index = mesh.z_curve_index(tree.dimension, level, index_x, index_y, index_z)

    if index in tree.tree_nodes and tree.nisleaf[index]:
        col = tree.nindex_tree_leaves[index]
        col += add_to_col
        matrix.setValue(row, col, value, True)
        #matrix[row, col] = matrix[row, col] + value

    #elif index in tree.tree_nodes and not tree.nisleaf[index]:
    #    #the children of the nodes are leaves
    #    children_number = len(tree.nchildren[index])

    #    for child_index in tree.nchildren[index]:
    #        i = tree.nindex_x[child_index]
    #        j = tree.nindex_y[child_index]
    #        k = tree.nindex_z[child_index]
    #        matrix_add(tree, matrix, row, value*(1./children_number), level+1, i, j, k, add_to_col)

    elif index not in tree.tree_nodes:
        #the parent of the node is a leaf; we need to report the value of the node to its parent
        i = int(math.floor(index_x/2)) # parent index_x
        j = int(math.floor(index_y/2)) # parent index_y
        k = int(math.floor(index_z/2)) # parent index_z

        matrix_add(tree, matrix, row, value*1., level-1, i, j, k, add_to_col=add_to_col)
