from __future__ import print_function, division

"""...

"""

import petsc4py.PETSc as petsc
from six.moves import range
from mrpy.mr_utils import mesh
import numpy as np
import math

# I need to precise where do the coefficients come from
coef = np.zeros(shape=(6, 5), dtype=np.float)
coef[1, 0] = -1./8
coef[2, 0] = -22./128
coef[2, 1] = 3./128
coef[3, 0] = -201./1024
coef[3, 1] = 11./256
coef[3, 2] = -5./1024
coef[4, 0] = -3461./16384
coef[4, 1] = 949./16384
coef[4, 2] = -185./16384
coef[4, 3] = 35./32768
coef[5, 0] = -29011./131072
coef[5, 1] = 569./8192
coef[5, 2] = -4661./262144
coef[5, 3] = 49./16384
coef[5, 4] = -63./262144

def nonzero_list_add(tree, nonzero_list, level, index_x=0, index_y=0, index_z=0):
    """...

    """

    index = mesh.z_curve_index(tree.dimension, level, index_x, index_y, index_z)

    if index in tree.tree_nodes and tree.nisleaf[index]:
        nonzero_list[-1] += 1

    elif index in tree.tree_nodes and not tree.nisleaf[index]:
        #the children of the nodes are leaves
        children_number = len(tree.nchildren[index])

        for child_index in tree.nchildren[index]:
            i = tree.nindex_x[child_index]
            j = tree.nindex_y[child_index]
            k = tree.nindex_z[child_index]
            nonzero_list_add(tree, nonzero_list, level+1, i, j, k)

    elif index not in tree.tree_nodes:
        #the parent of the node is a leaf; we need to compute the prediction value
        s = tree.stencil_prediction

        if tree.dimension == 1:
            i = int(math.floor(index_x/2)) # parent index_x
            p = index_x % 2

            # One more nonzero element for the parent of the node
            nonzero_list_add(tree, nonzero_list, level-1, i)

            for m in range(1, s+1):
                if mesh.bc_compatible_local_indexes(tree, level-1, i+m) is not None:
                    im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i+m)
                    # One more nonzero element for every node used for the prediction
                    nonzero_list_add(tree, nonzero_list, level-1, im)

                if mesh.bc_compatible_local_indexes(tree, level-1, i-m) is not None:
                    im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i-m)
                    # One more nonzero element for every node used for the prediction
                    nonzero_list_add(tree, nonzero_list, level-1, im)

        elif tree.dimension == 2:
            i = int(math.floor(index_x/2)) # parent index_x
            j = int(math.floor(index_y/2)) # parent index_y
            p = index_x % 2
            q = index_y % 2

            # One more nonzero element for the parent of the node
            nonzero_list_add(tree, nonzero_list, level-1, i, j)

            for m in range(1, s+1):
                if mesh.bc_compatible_local_indexes(tree, level-1, i+m, j) is not None:
                    im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i+m, j)
                    # One more nonzero element for every node used for the prediction
                    nonzero_list_add(tree, nonzero_list, level-1, im, jm)

                if mesh.bc_compatible_local_indexes(tree, level-1, i-m, j) is not None:
                    im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i-m, j)
                    # One more nonzero element for every node used for the prediction
                    nonzero_list_add(tree, nonzero_list, level-1, im, jm)

                if mesh.bc_compatible_local_indexes(tree, level-1, i, j+m) is not None:
                    im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i, j+m)
                    # One more nonzero element for every node used for the prediction
                    nonzero_list_add(tree, nonzero_list, level-1, im, jm)

                if mesh.bc_compatible_local_indexes(tree, level-1, i, j-m) is not None:
                    im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i, j-m)
                    # One more nonzero element for every node used for the prediction
                    nonzero_list_add(tree, nonzero_list, level-1, im, jm)

            for a in range(1, s+1):
                for b in range(1, s+1):
                    if mesh.bc_compatible_local_indexes(tree, level-1, i+a, j+b) is not None:
                        im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i+a, j+b)
                        # One more nonzero element for every node used for the prediction
                        nonzero_list_add(tree, nonzero_list, level-1, im, jm)

                    if mesh.bc_compatible_local_indexes(tree, level-1, i-a, j+b) is not None:
                        im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i-a, j+b)
                        # One more nonzero element for every node used for the prediction
                        nonzero_list_add(tree, nonzero_list, level-1, im, jm)

                    if mesh.bc_compatible_local_indexes(tree, level-1, i+a, j-b) is not None:
                        im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i+a, j-b)
                        # One more nonzero element for every node used for the prediction
                        nonzero_list_add(tree, nonzero_list, level-1, im, jm)

                    if mesh.bc_compatible_local_indexes(tree, level-1, i-a, j-b) is not None:
                        im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i-a, j-b)
                        # One more nonzero element for every node used for the prediction
                        nonzero_list_add(tree, nonzero_list, level-1, im, jm)

        elif tree.dimension == 3:
            i = int(math.floor(index_x/2)) # parent index_x
            j = int(math.floor(index_y/2)) # parent index_y
            k = int(math.floor(index_z/2)) # parent index_z
            p = index_x % 2
            q = index_y % 2
            r = index_z % 2

            # One more nonzero element for the parent of the node
            nonzero_list_add(tree, nonzero_list, level-1, i, j, k)

            for m in range(1, s+1):
                if mesh.bc_compatible_local_indexes(tree, level-1, i+m, j, k) is not None:
                    im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i+m, j, k)
                    # One more nonzero element for every node used for the prediction
                    nonzero_list_add(tree, nonzero_list, level-1, im, jm, km)

                if mesh.bc_compatible_local_indexes(tree, level-1, i-m, j, k) is not None:
                    im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i-m, j, k)
                    # One more nonzero element for every node used for the prediction
                    nonzero_list_add(tree, nonzero_list, level-1, im, jm, km)

                if mesh.bc_compatible_local_indexes(tree, level-1, i, j+m, k) is not None:
                    im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i, j+m, k)
                    # One more nonzero element for every node used for the prediction
                    nonzero_list_add(tree, nonzero_list, level-1, im, jm, km)

                if mesh.bc_compatible_local_indexes(tree, level-1, i, j-m, k) is not None:
                    im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i, j-m, k)
                    # One more nonzero element for every node used for the prediction
                    nonzero_list_add(tree, nonzero_list, level-1, im, jm, km)

                if mesh.bc_compatible_local_indexes(tree, level-1, i, j, k+m) is not None:
                    im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i, j, k+m)
                    # One more nonzero element for every node used for the prediction
                    nonzero_list_add(tree, nonzero_list, level-1, im, jm, km)

                if mesh.bc_compatible_local_indexes(tree, level-1, i, j, k-m) is not None:
                    im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i, j, k-m)
                    # One more nonzero element for every node used for the prediction
                    nonzero_list_add(tree, nonzero_list, level-1, im, jm, km)

            for a in range(1, s+1):
                for b in range(1, s+1):
                    if mesh.bc_compatible_local_indexes(tree, level-1, i+a, j+b, k) is not None:
                        im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i+a, j+b, k)
                        # One more nonzero element for every node used for the prediction
                        nonzero_list_add(tree, nonzero_list, level-1, im, jm, km)

                    if mesh.bc_compatible_local_indexes(tree, level-1, i-a, j+b, k) is not None:
                        im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i-a, j+b, k)
                        # One more nonzero element for every node used for the prediction
                        nonzero_list_add(tree, nonzero_list, level-1, im, jm, km)

                    if mesh.bc_compatible_local_indexes(tree, level-1, i+a, j-b, k) is not None:
                        im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i+a, j-b, k)
                        # One more nonzero element for every node used for the prediction
                        nonzero_list_add(tree, nonzero_list, level-1, im, jm, km)

                    if mesh.bc_compatible_local_indexes(tree, level-1, i-a, j-b, k) is not None:
                        im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i-a, j-b, k)
                        # One more nonzero element for every node used for the prediction
                        nonzero_list_add(tree, nonzero_list, level-1, im, jm, km)

            for a in range(1, s+1):
                for b in range(1, s+1):
                    if mesh.bc_compatible_local_indexes(tree, level-1, i+a, j, k+b) is not None:
                        im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i+a, j, k+b)
                        # One more nonzero element for every node used for the prediction
                        nonzero_list_add(tree, nonzero_list, level-1, im, jm, km)

                    if mesh.bc_compatible_local_indexes(tree, level-1, i-a, j, k+b) is not None:
                        im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i-a, j, k+b)
                        # One more nonzero element for every node used for the prediction
                        nonzero_list_add(tree, nonzero_list, level-1, im, jm, km)

                    if mesh.bc_compatible_local_indexes(tree, level-1, i+a, j, k-b) is not None:
                        im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i+a, j, k-b)
                        # One more nonzero element for every node used for the prediction
                        nonzero_list_add(tree, nonzero_list, level-1, im, jm, km)

                    if mesh.bc_compatible_local_indexes(tree, level-1, i-a, j, k-b) is not None:
                        im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i-a, j, k-b)
                        # One more nonzero element for every node used for the prediction
                        nonzero_list_add(tree, nonzero_list, level-1, im, jm, km)

            for a in range(1, s+1):
                for b in range(1, s+1):
                    if mesh.bc_compatible_local_indexes(tree, level-1, i, j+a, k+b) is not None:
                        im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i, j+a, k+b)
                        # One more nonzero element for every node used for the prediction
                        nonzero_list_add(tree, nonzero_list, level-1, im, jm, km)

                    if mesh.bc_compatible_local_indexes(tree, level-1, i, j-a, k+b) is not None:
                        im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i, j-a, k+b)
                        # One more nonzero element for every node used for the prediction
                        nonzero_list_add(tree, nonzero_list, level-1, im, jm, km)

                    if mesh.bc_compatible_local_indexes(tree, level-1, i, j+a, k-b) is not None:
                        im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i, j+a, k-b)
                        # One more nonzero element for every node used for the prediction
                        nonzero_list_add(tree, nonzero_list, level-1, im, jm, km)

                    if mesh.bc_compatible_local_indexes(tree, level-1, i, j-a, k-b) is not None:
                        im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i, j-a, k-b)
                        # One more nonzero element for every node used for the prediction
                        nonzero_list_add(tree, nonzero_list, level-1, im, jm, km)

            for a in range(1, s+1):
                for b in range(1, s+1):
                    for c in range(1, s+1):
                        if mesh.bc_compatible_local_indexes(tree, level-1, i+a, j+b, k+c) is not None:
                            im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i+a, j+b, k+c)
                            # One more nonzero element for every node used for the prediction
                            nonzero_list_add(tree, nonzero_list, level-1, im, jm, km)

                        if mesh.bc_compatible_local_indexes(tree, level-1, i-a, j+b, k+c) is not None:
                            im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i-a, j+b, k+c)
                            # One more nonzero element for every node used for the prediction
                            nonzero_list_add(tree, nonzero_list, level-1, im, jm, km)

                        if mesh.bc_compatible_local_indexes(tree, level-1, i+a, j-b, k+c) is not None:
                            im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i+a, j-b, k+c)
                            # One more nonzero element for every node used for the prediction
                            nonzero_list_add(tree, nonzero_list, level-1, im, jm, km)

                        if mesh.bc_compatible_local_indexes(tree, level-1, i-a, j-b, k+c) is not None:
                            im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i-a, j-b, k+c)
                            # One more nonzero element for every node used for the prediction
                            nonzero_list_add(tree, nonzero_list, level-1, im, jm, km)

                        if mesh.bc_compatible_local_indexes(tree, level-1, i+a, j+b, k-c) is not None:
                            im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i+a, j+b, k-c)
                            # One more nonzero element for every node used for the prediction
                            nonzero_list_add(tree, nonzero_list, level-1, im, jm, km)

                        if mesh.bc_compatible_local_indexes(tree, level-1, i-a, j+b, k-c) is not None:
                            im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i-a, j+b, k-c)
                            # One more nonzero element for every node used for the prediction
                            nonzero_list_add(tree, nonzero_list, level-1, im, jm, km)

                        if mesh.bc_compatible_local_indexes(tree, level-1, i+a, j-b, k-c) is not None:
                            im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i+a, j-b, k-c)
                            # One more nonzero element for every node used for the prediction
                            nonzero_list_add(tree, nonzero_list, level-1, im, jm, km)

                        if mesh.bc_compatible_local_indexes(tree, level-1, i-a, j-b, k-c) is not None:
                            im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i-a, j-b, k-c)
                            # One more nonzero element for every node used for the prediction
                            nonzero_list_add(tree, nonzero_list, level-1, im, jm, km)

def matrix_add(tree, matrix, row, value, level, index_x=0, index_y=0, index_z=0, add_to_col=0):
    """...

    """

    index = mesh.z_curve_index(tree.dimension, level, index_x, index_y, index_z)

    if index in tree.tree_nodes and tree.nisleaf[index]:
        col = tree.nindex_tree_leaves[index]
        col += add_to_col
        matrix.setValue(row, col, value, True)
        #matrix[row, col] = matrix[row, col] + value

    elif index in tree.tree_nodes and not tree.nisleaf[index]:
        #the children of the nodes are leaves
        children_number = len(tree.nchildren[index])

        for child_index in tree.nchildren[index]:
            i = tree.nindex_x[child_index]
            j = tree.nindex_y[child_index]
            k = tree.nindex_z[child_index]
            matrix_add(tree, matrix, row, value*(1./children_number), level+1, i, j, k, add_to_col)

    elif index not in tree.tree_nodes:
        #the parent of the node is a leaf; we need to compute the prediction value
        s = tree.stencil_prediction

        if tree.dimension == 1:
            i = int(math.floor(index_x/2)) # parent index_x
            p = index_x % 2

            matrix_add(tree, matrix, row, value*1., level-1, i, add_to_col=add_to_col)

            for m in range(1, s+1):
                if mesh.bc_compatible_local_indexes(tree, level-1, i+m) is not None:
                    im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i+m)
                    matrix_add(tree, matrix, row, value*(-1)**p*coef[s, m-1], level-1, im, add_to_col=add_to_col)

                if mesh.bc_compatible_local_indexes(tree, level-1, i-m) is not None:
                    im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i-m)
                    matrix_add(tree, matrix, row, value*(-1)*(-1)**p*coef[s, m-1], level-1, im, add_to_col=add_to_col)

        elif tree.dimension == 2:
            i = int(math.floor(index_x/2)) # parent index_x
            j = int(math.floor(index_y/2)) # parent index_y
            p = index_x % 2
            q = index_y % 2

            matrix_add(tree, matrix, row, value*1., level-1, i, j, add_to_col=add_to_col)

            for m in range(1, s+1):
                if mesh.bc_compatible_local_indexes(tree, level-1, i+m, j) is not None:
                    im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i+m, j)
                    matrix_add(tree, matrix, row, value*(-1)**p*coef[s, m-1], level-1, im, jm, add_to_col=add_to_col)

                if mesh.bc_compatible_local_indexes(tree, level-1, i-m, j) is not None:
                    im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i-m, j)
                    matrix_add(tree, matrix, row, value*(-1)*(-1)**p*coef[s, m-1], level-1, im, jm, add_to_col=add_to_col)

                if mesh.bc_compatible_local_indexes(tree, level-1, i, j+m) is not None:
                    im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i, j+m)
                    matrix_add(tree, matrix, row, value*(-1)**q*coef[s, m-1], level-1, im, jm, add_to_col=add_to_col)

                if mesh.bc_compatible_local_indexes(tree, level-1, i, j-m) is not None:
                    im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i, j-m)
                    matrix_add(tree, matrix, row, value*(-1)*(-1)**q*coef[s, m-1], level-1, im, jm, add_to_col=add_to_col)

            for a in range(1, s+1):
                for b in range(1, s+1):
                    if mesh.bc_compatible_local_indexes(tree, level-1, i+a, j+b) is not None:
                        im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i+a, j+b)
                        matrix_add(tree, matrix, row, value*(-1)*(-1)**(p+q)*coef[s, a-1]*coef[s, b-1], level-1, im, jm, add_to_col=add_to_col)

                    if mesh.bc_compatible_local_indexes(tree, level-1, i-a, j+b) is not None:
                        im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i-a, j+b)
                        matrix_add(tree, matrix, row, value*(-1)*(-1)**(p+q)*coef[s, a-1]*coef[s, b-1], level-1, im, jm, add_to_col=add_to_col)

                    if mesh.bc_compatible_local_indexes(tree, level-1, i+a, j-b) is not None:
                        im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i+a, j-b)
                        matrix_add(tree, matrix, row, value*(-1)*(-1)**(p+q)*coef[s, a-1]*coef[s, b-1], level-1, im, jm, add_to_col=add_to_col)

                    if mesh.bc_compatible_local_indexes(tree, level-1, i-a, j-b) is not None:
                        im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i-a, j-b)
                        matrix_add(tree, matrix, row, value*(-1)*(-1)**(p+q)*coef[s, a-1]*coef[s, b-1], level-1, im, jm, add_to_col=add_to_col)

        elif tree.dimension == 3:
            i = int(math.floor(index_x/2)) # parent index_x
            j = int(math.floor(index_y/2)) # parent index_y
            k = int(math.floor(index_z/2)) # parent index_z
            p = index_x % 2
            q = index_y % 2
            r = index_z % 2

            matrix_add(tree, matrix, row, value*1., level-1, i, j, k)

            for m in range(1, s+1):
                if mesh.bc_compatible_local_indexes(tree, level-1, i+m, j, k) is not None:
                    im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i+m, j, k)
                    matrix_add(tree, matrix, row, value*(-1)**p*coef[s, m-1], level-1, im, jm, km, add_to_col=add_to_col)

                if mesh.bc_compatible_local_indexes(tree, level-1, i-m, j, k) is not None:
                    im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i-m, j, k)
                    matrix_add(tree, matrix, row, value*(-1)*(-1)**p*coef[s, m-1], level-1, im, jm, km, add_to_col=add_to_col)

                if mesh.bc_compatible_local_indexes(tree, level-1, i, j+m, k) is not None:
                    im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i, j+m, k)
                    matrix_add(tree, matrix, row, value*(-1)**q*coef[s, m-1], level-1, im, jm, km, add_to_col=add_to_col)

                if mesh.bc_compatible_local_indexes(tree, level-1, i, j-m, k) is not None:
                    im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i, j-m, k)
                    matrix_add(tree, matrix, row, value*(-1)*(-1)**q*coef[s, m-1], level-1, im, jm, km, add_to_col=add_to_col)

                if mesh.bc_compatible_local_indexes(tree, level-1, i, j, k+m) is not None:
                    im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i, j, k+m)
                    matrix_add(tree, matrix, row, value*(-1)**r*coef[s, m-1], level-1, im, jm, km, add_to_col=add_to_col)

                if mesh.bc_compatible_local_indexes(tree, level-1, i, j, k-m) is not None:
                    im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i, j, k-m)
                    matrix_add(tree, matrix, row, value*(-1)*(-1)**r*coef[s, m-1], level-1, im, jm, km, add_to_col=add_to_col)

            for a in range(1, s+1):
                for b in range(1, s+1):
                    if mesh.bc_compatible_local_indexes(tree, level-1, i+a, j+b, k) is not None:
                        im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i+a, j+b, k)
                        matrix_add(tree, matrix, row, value*(-1)*(-1)**(p+q)*coef[s, a-1]*coef[s, b-1], level-1, im, jm, km, add_to_col=add_to_col)

                    if mesh.bc_compatible_local_indexes(tree, level-1, i-a, j+b, k) is not None:
                        im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i-a, j+b, k)
                        matrix_add(tree, matrix, row, value*(-1)*(-1)**(p+q)*coef[s, a-1]*coef[s, b-1], level-1, im, jm, km, add_to_col=add_to_col)

                    if mesh.bc_compatible_local_indexes(tree, level-1, i+a, j-b, k) is not None:
                        im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i+a, j-b, k)
                        matrix_add(tree, matrix, row, value*(-1)*(-1)**(p+q)*coef[s, a-1]*coef[s, b-1], level-1, im, jm, km, add_to_col=add_to_col)

                    if mesh.bc_compatible_local_indexes(tree, level-1, i-a, j-b, k) is not None:
                        im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i-a, j-b, k)
                        matrix_add(tree, matrix, row, value*(-1)*(-1)**(p+q)*coef[s, a-1]*coef[s, b-1], level-1, im, jm, km, add_to_col=add_to_col)

            for a in range(1, s+1):
                for b in range(1, s+1):
                    if mesh.bc_compatible_local_indexes(tree, level-1, i+a, j, k+b) is not None:
                        im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i+a, j, k+b)
                        matrix_add(tree, matrix, row, value*(-1)*(-1)**(p+r)*coef[s, a-1]*coef[s, b-1], level-1, im, jm, km, add_to_col=add_to_col)

                    if mesh.bc_compatible_local_indexes(tree, level-1, i-a, j, k+b) is not None:
                        im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i-a, j, k+b)
                        matrix_add(tree, matrix, row, value*(-1)*(-1)**(p+r)*coef[s, a-1]*coef[s, b-1], level-1, im, jm, km, add_to_col=add_to_col)

                    if mesh.bc_compatible_local_indexes(tree, level-1, i+a, j, k-b) is not None:
                        im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i+a, j, k-b)
                        matrix_add(tree, matrix, row, value*(-1)*(-1)**(p+r)*coef[s, a-1]*coef[s, b-1], level-1, im, jm, km, add_to_col=add_to_col)

                    if mesh.bc_compatible_local_indexes(tree, level-1, i-a, j, k-b) is not None:
                        im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i-a, j, k-b)
                        matrix_add(tree, matrix, row, value*(-1)*(-1)**(p+r)*coef[s, a-1]*coef[s, b-1], level-1, im, jm, km, add_to_col=add_to_col)

            for a in range(1, s+1):
                for b in range(1, s+1):
                    if mesh.bc_compatible_local_indexes(tree, level-1, i, j+a, k+b) is not None:
                        im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i, j+a, k+b)
                        matrix_add(tree, matrix, row, value*(-1)*(-1)**(q+r)*coef[s, a-1]*coef[s, b-1], level-1, im, jm, km, add_to_col=add_to_col)

                    if mesh.bc_compatible_local_indexes(tree, level-1, i, j-a, k+b) is not None:
                        im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i, j-a, k+b)
                        matrix_add(tree, matrix, row, value*(-1)*(-1)**(q+r)*coef[s, a-1]*coef[s, b-1], level-1, im, jm, km, add_to_col=add_to_col)

                    if mesh.bc_compatible_local_indexes(tree, level-1, i, j+a, k-b) is not None:
                        im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i, j+a, k-b)
                        matrix_add(tree, matrix, row, value*(-1)*(-1)**(q+r)*coef[s, a-1]*coef[s, b-1], level-1, im, jm, km, add_to_col=add_to_col)

                    if mesh.bc_compatible_local_indexes(tree, level-1, i, j-a, k-b) is not None:
                        im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i, j-a, k-b)
                        matrix_add(tree, matrix, row, value*(-1)*(-1)**(q+r)*coef[s, a-1]*coef[s, b-1], level-1, im, jm, km, add_to_col=add_to_col)

            for a in range(1, s+1):
                for b in range(1, s+1):
                    for c in range(1, s+1):
                        if mesh.bc_compatible_local_indexes(tree, level-1, i+a, j+b, k+c) is not None:
                            im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i+a, j+b, k+c)
                            matrix_add(tree, matrix, row, value*(-1)**(p+q+r)*coef[s, a-1]*coef[s, b-1]*coef[s, c-1], level-1, im, jm, km, add_to_col=add_to_col)

                        if mesh.bc_compatible_local_indexes(tree, level-1, i-a, j+b, k+c) is not None:
                            im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i-a, j+b, k+c)
                            matrix_add(tree, matrix, row, value*(-1)**(p+q+r)*coef[s, a-1]*coef[s, b-1]*coef[s, c-1], level-1, im, jm, km, add_to_col=add_to_col)

                        if mesh.bc_compatible_local_indexes(tree, level-1, i+a, j-b, k+c) is not None:
                            im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i+a, j-b, k+c)
                            matrix_add(tree, matrix, row, value*(-1)**(p+q+r)*coef[s, a-1]*coef[s, b-1]*coef[s, c-1], level-1, im, jm, km, add_to_col=add_to_col)

                        if mesh.bc_compatible_local_indexes(tree, level-1, i-a, j-b, k+c) is not None:
                            im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i-a, j-b, k+c)
                            matrix_add(tree, matrix, row, value*(-1)**(p+q+r)*coef[s, a-1]*coef[s, b-1]*coef[s, c-1], level-1, im, jm, km, add_to_col=add_to_col)

                        if mesh.bc_compatible_local_indexes(tree, level-1, i+a, j+b, k-c) is not None:
                            im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i+a, j+b, k-c)
                            matrix_add(tree, matrix, row, value*(-1)**(p+q+r)*coef[s, a-1]*coef[s, b-1]*coef[s, c-1], level-1, im, jm, km, add_to_col=add_to_col)

                        if mesh.bc_compatible_local_indexes(tree, level-1, i-a, j+b, k-c) is not None:
                            im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i-a, j+b, k-c)
                            matrix_add(tree, matrix, row, value*(-1)**(p+q+r)*coef[s, a-1]*coef[s, b-1]*coef[s, c-1], level-1, im, jm, km, add_to_col=add_to_col)

                        if mesh.bc_compatible_local_indexes(tree, level-1, i+a, j-b, k-c) is not None:
                            im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i+a, j-b, k-c)
                            matrix_add(tree, matrix, row, value*(-1)**(p+q+r)*coef[s, a-1]*coef[s, b-1]*coef[s, c-1], level-1, im, jm, km, add_to_col=add_to_col)

                        if mesh.bc_compatible_local_indexes(tree, level-1, i-a, j-b, k-c) is not None:
                            im, jm, km = mesh.bc_compatible_local_indexes(tree, level-1, i-a, j-b, k-c)
                            matrix_add(tree, matrix, row, value*(-1)**(p+q+r)*coef[s, a-1]*coef[s, b-1]*coef[s, c-1], level-1, im, jm, km, add_to_col=add_to_col)
