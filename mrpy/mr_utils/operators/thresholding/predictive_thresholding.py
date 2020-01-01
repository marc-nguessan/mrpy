from __future__ import print_function, division

"""...

"""

from six.moves import range
from mrpy.mr_utils import mesh

def compute_level_threshold_parameter(tree, index, threshold_parameter):
    """...

    """

    return threshold_parameter*2**(tree.dimension*(tree.nlevel[index] - tree.max_level))

def compute_predictive_level_threshold_parameter(tree, index, threshold_parameter):
    """...

    """

    if tree.dimension == 1:
        p = 1
    else:
        p = 2*tree.stencil_prediction + 2

    return 2**(2*p)*threshold_parameter*2**(tree.dimension*(tree.nlevel[index] - tree.max_level))

def run_thresholding(threshold_parameter, *trees):
    """...

    """

    tree_indexes = trees[0].tree_nodes.keys()
    dimension = trees[0].dimension
    max_level = trees[0].max_level

    for index in list(tree_indexes):

        keep_children = False

        for tree in trees:

            #if not tree.nisleaf[index]:
            if (tree.nisleaf[index] is False) and (tree.max_norm_details != 0.):
                if (abs(tree.nnorm_details[index]/tree.max_norm_details) >= \
                        compute_level_threshold_parameter(tree, index, threshold_parameter) \
                        or tree.nlevel[index] <= tree.min_level):
                    keep_children = True

        for tree in trees:
            if not tree.nisleaf[index]:
                tree.nkeep_children[index] = keep_children

        keep_grandchildren = False

        for tree in trees:

            if not tree.nisleaf[index] and tree.nlevel[index] < tree.max_level-1 and (tree.max_norm_details != 0.):
                if (abs(tree.nnorm_details[index]/tree.max_norm_details) >= \
                        compute_predictive_level_threshold_parameter(tree, index, threshold_parameter)):
                    keep_grandchildren = True

        for tree in trees:

            if keep_grandchildren:
                tree.nkeep_children[index] = True
                for child_index in tree.nchildren[index]:
                    tree.nkeep_children[child_index] = True
                    temp = tree.nchildren[child_index][0]
                    if not temp in tree.tree_nodes: #the grandchildren of the current node aren't in the tree and need to be created
                        #print(temp)
                        i = tree.nindex_x[child_index]
                        j = tree.nindex_y[child_index]
                        k = tree.nindex_z[child_index]
                        level = tree.nlevel[child_index]

                        if dimension == 1:
                            for m in range(2):
                                mesh.create_node(tree, dimension, level + 1, 2*i + m)
                                # Children pointers creation
                                if level + 1 != max_level:
                                    mesh.create_children_pointers(tree, dimension, level + 1, 2*i + m)

                        elif dimension == 2:
                            for n in range(2):
                                for m in range(2):
                                    #print child_index
                                    #print mesh.z_curve_index(dimension, level + 1, 2*i + m, 2*j + n)
                                    #print ""
                                    mesh.create_node(tree, dimension, level + 1, 2*i + m, 2*j + n)
                                    # Children pointers creation
                                    if level + 1 != max_level:
                                        mesh.create_children_pointers(tree, dimension, level + 1, 2*i + m, 2*j + n)

                        elif dimension == 3:
                            for o in range(2):
                                for n in range(2):
                                    for m in xrrange(2):
                                        mesh.create_node(tree, dimension, level + 1, 2*i + m, 2*j + n, 2*k + o)
                                        # Children pointers creation
                                        if level + 1 != max_level:
                                            mesh.create_children_pointers(tree, dimension, level + 1, 2*i + m, 2*j + n, 2*k + o)
