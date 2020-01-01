from __future__ import print_function, division

"""...

"""

import numpy as np
from six.moves import range
import config as cfg
from mrpy.mr_utils import mesh

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


def compute_prediction_value(tree, index_parent, index_child):
    """...

    """

    s = tree.stencil_prediction

    if tree.dimension == 1:
        i = tree.nindex_x[index_parent]
        parent_level = tree.nlevel[index_parent]
        p = tree.nindex_x[index_child] % 2

        prediction_value = tree.nvalue[index_parent]
        for m in range(1, s+1):
            prediction_value = prediction_value + (-1)**p*coef[s, m-1]*(mesh.get_value(tree, parent_level, i+m) - mesh.get_value(tree, parent_level, i-m))

        return prediction_value

    if tree.dimension == 2:
        i = tree.nindex_x[index_parent]
        j = tree.nindex_y[index_parent]
        parent_level = tree.nlevel[index_parent]
        p = tree.nindex_x[index_child] % 2
        q = tree.nindex_y[index_child] % 2

        prediction_value = tree.nvalue[index_parent]
        for m in range(1, s+1):
            prediction_value = prediction_value + \
                    (-1)**p*coef[s, m-1]*(mesh.get_value(tree, parent_level, i+m, j) - mesh.get_value(tree, parent_level, i-m, j)) + \
                    (-1)**q*coef[s, m-1]*(mesh.get_value(tree, parent_level, i, j+m) - mesh.get_value(tree, parent_level, i, j-m))

        temp = 0
        for a in range(1, s+1):
            foo = 0
            for b in range(1, s+1):
                foo = foo + coef[s, b-1]*(mesh.get_value(tree, parent_level, i+a, j+b) - \
                                          mesh.get_value(tree, parent_level, i-a, j+b) - \
                                          mesh.get_value(tree, parent_level, i+a, j-b) + \
                                          mesh.get_value(tree, parent_level, i-a, j-b))
            temp = temp + coef[s, a-1]*foo

        prediction_value = prediction_value - (-1)**(p+q)*temp

        return prediction_value

    if tree.dimension == 3:
        i = tree.nindex_x[index_parent]
        j = tree.nindex_y[index_parent]
        k = tree.nindex_z[index_parent]
        parent_level = tree.nlevel[index_parent]
        p = tree.nindex_x[index_child] % 2
        q = tree.nindex_y[index_child] % 2
        r = tree.nindex_z[index_child] % 2

        prediction_value = tree.nvalue[index_parent]
        for m in range(1, s+1):
            prediction_value = prediction_value + \
                    (-1)**p*coef[s, m-1]*(mesh.get_value(tree, parent_level, i+m, j, k) - mesh.get_value(tree, parent_level, i-m, j, k)) + \
                    (-1)**q*coef[s, m-1]*(mesh.get_value(tree, parent_level, i, j+m, k) - mesh.get_value(tree, parent_level, i, j-m, k)) + \
                    (-1)**r*coef[s, m-1]*(mesh.get_value(tree, parent_level, i, j, k+m) - mesh.get_value(tree, parent_level, i, j, k-m))

        temp = 0
        for a in range(1, s+1):
            foo = 0
            for b in range(1, s+1):
                foo = foo + coef[s, b-1]*(mesh.get_value(tree, parent_level, i+a, j+b, k) - \
                                          mesh.get_value(tree, parent_level, i-a, j+b, k) - \
                                          mesh.get_value(tree, parent_level, i+a, j-b, k) + \
                                          mesh.get_value(tree, parent_level, i-a, j-b, k))
            temp = temp + coef[s, a-1]*foo

        prediction_value = prediction_value - (-1)**(p+q)*temp

        temp = 0
        for a in range(1, s+1):
            foo = 0
            for b in range(1, s+1):
                foo = foo + coef[s, b-1]*(mesh.get_value(tree, parent_level, i+a, j, k+b) - \
                                          mesh.get_value(tree, parent_level, i-a, j, k+b) - \
                                          mesh.get_value(tree, parent_level, i+a, j, k-b) + \
                                          mesh.get_value(tree, parent_level, i-a, j, k-b))
            temp = temp + coef[s, a-1]*foo

        prediction_value = prediction_value - (-1)**(p+r)*temp

        temp = 0
        for a in range(1, s+1):
            foo = 0
            for b in range(1, s+1):
                foo = foo + coef[s, b-1]*(mesh.get_value(tree, parent_level, i, j+a, k+b) - \
                                          mesh.get_value(tree, parent_level, i, j-a, k+b) - \
                                          mesh.get_value(tree, parent_level, i, j+a, k-b) + \
                                          mesh.get_value(tree, parent_level, i, j-a, k-b))
            temp = temp + coef[s, a-1]*foo

        prediction_value = prediction_value - (-1)**(q+r)*temp

        temp = 0
        for a in range(1, s+1):
            bar = 0
            for b in range(1, s+1):
                foo = 0
                for c in range(1, s+1):
                    foo = foo + coef[s, c-1]*(mesh.get_value(tree, parent_level, i+a, j+b, k+c) - \
                                              mesh.get_value(tree, parent_level, i-a, j+b, k+c) - \
                                              mesh.get_value(tree, parent_level, i+a, j-b, k+c) - \
                                              mesh.get_value(tree, parent_level, i+a, j+b, k-c) + \
                                              mesh.get_value(tree, parent_level, i-a, j-b, k+c) + \
                                              mesh.get_value(tree, parent_level, i-a, j+b, k-c) + \
                                              mesh.get_value(tree, parent_level, i+a, j-b, k-c) - \
                                              mesh.get_value(tree, parent_level, i-a, j-b, k-c))
                bar = bar + coef[s, b-1]*foo
            temp = temp + coef[s, a-1]*bar

        prediction_value = prediction_value + (-1)**(p+q+r)*temp

        return prediction_value
