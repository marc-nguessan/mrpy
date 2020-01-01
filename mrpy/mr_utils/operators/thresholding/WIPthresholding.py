"""...

"""

import config as cfg
import math
import mr_mesh as mesh

def compute_level_threshold_parameter(tree, index, threshold_parameter):
    """...

    """

    return threshold_parameter*2**(tree.dimension*(tree.nlevel[index] - tree.max_level))

def thresholding_information_propagation(tree, index, axis):
    """...

    """

    i = tree.nindex_x[index]
    j = tree.nindex_y[index]
    k = tree.nindex_z[index]

    if axis == 0:
        p = i % 2

        if p == 0:
            for m in xrange(1, cfg.threshold_speed_propagation+1):
                if mesh.bc_compatible_local_indexes(tree.nlevel[index], i-m, j, k) is not None:
                    a, b, c = mesh.bc_compatible_local_indexes(tree.nlevel[index], i-m, j, k)
                    index_child = mesh.z_curve_index(tree.dimension, tree.nlevel[index], a, b, c)
                    if index_child in tree.tree_nodes:
                        index_parent = tree.nparent[index_child]
                        tree.nkeep_children[index_parent] = True
                    else:
                        if tree.nlevel[index] != 0:
                            index_x_parent = int(math.floor(a/2))
                            index_y_parent = int(math.floor(b/2))
                            index_z_parent = int(math.floor(c/2))
                            index_parent = mesh.z_curve_index(tree.dimension, tree.nlevel[index] - 1, index_x_parent, index_y_parent, index_z_parent)
                            parent = tree.tree_nodes[index_parent]
                            parent.keep_children = True
                            mesh.create_children_of_node(parent)

        elif p == 1:
            for m in xrange(1, cfg.threshold_speed_propagation+1):
                if tree.bc_compatible_local_indexes(tree.nlevel[index], i+m, j, k) is not None:
                    a, b, c = tree.bc_compatible_local_indexes(tree.nlevel[index], i+m, j, k)
                    index_child = mesh.z_curve_index(tree.dimension, tree.nlevel[index], a, b, c)
                    if index_child in tree.tree_nodes:
                        index_parent = tree.nparent[index_child]
                        tree.nkeep_children[index_parent] = True
                    else:
                        if tree.nlevel[index] != 0:
                            index_x_parent = int(math.floor(a/2))
                            index_y_parent = int(math.floor(b/2))
                            index_z_parent = int(math.floor(c/2))
                            index_parent = mesh.z_curve_index(tree.dimension, tree.nlevel[index] - 1, index_x_parent, index_y_parent, index_z_parent)
                            parent = tree.tree_nodes[index_parent]
                            parent.keep_children = True
                            tree.create_children_of_node(parent)

    elif axis == 1:
        p = j % 2

        if p == 0:
            for m in xrange(1, cfg.threshold_speed_propagation+1):
                if tree.bc_compatible_local_indexes(tree.nlevel[index], i, j-m, k) is not None:
                    a, b, c = tree.bc_compatible_local_indexes(tree.nlevel[index], i, j-m, k)
                    index_child = mesh.z_curve_index(tree.dimension, tree.nlevel[index], a, b, c)
                    if index_child in tree.tree_nodes:
                        index_parent = tree.nparent[index_child]
                        tree.nkeep_children[index_parent] = True
                    else:
                        if tree.nlevel[index] != 0:
                            index_x_parent = int(math.floor(a/2))
                            index_y_parent = int(math.floor(b/2))
                            index_z_parent = int(math.floor(c/2))
                            index_parent = mesh.z_curve_index(tree.dimension, tree.nlevel[index] - 1, index_x_parent, index_y_parent, index_z_parent)
                            parent = tree.tree_nodes[index_parent]
                            parent.keep_children = True
                            tree.create_children_of_node(parent)

        elif p == 1:
            for m in xrange(1, cfg.threshold_speed_propagation+1):
                if tree.bc_compatible_local_indexes(tree.nlevel[index], i, j+m, k) is not None:
                    a, b, c = tree.bc_compatible_local_indexes(tree.nlevel[index], i, j+m, k)
                    index_child = mesh.z_curve_index(tree.dimension, tree.nlevel[index], a, b, c)
                    if index_child in tree.tree_nodes:
                        index_parent = tree.nparent[index_child]
                        tree.nkeep_children[index_parent] = True
                    else:
                        if tree.nlevel[index] != 0:
                            index_x_parent = int(math.floor(a/2))
                            index_y_parent = int(math.floor(b/2))
                            index_z_parent = int(math.floor(c/2))
                            index_parent = mesh.z_curve_index(tree.dimension, tree.nlevel[index] - 1, index_x_parent, index_y_parent, index_z_parent)
                            parent = tree.tree_nodes[index_parent]
                            parent.keep_children = True
                            tree.create_children_of_node(parent)

    elif axis == 2:
        p = k % 2

        if p == 0:
            for m in xrange(1, cfg.threshold_speed_propagation+1):
                if tree.bc_compatible_local_indexes(tree.nlevel[index], i, j, k-m) is not None:
                    a, b, c = tree.bc_compatible_local_indexes(tree.nlevel[index], i, j, k-m)
                    index_child = mesh.z_curve_index(tree.dimension, tree.nlevel[index], a, b, c)
                    if index_child in tree.tree_nodes:
                        index_parent = tree.nparent[index_child]
                        tree.nkeep_children[index_parent] = True
                    else:
                        if tree.nlevel[index] != 0:
                            index_x_parent = int(math.floor(a/2))
                            index_y_parent = int(math.floor(b/2))
                            index_z_parent = int(math.floor(c/2))
                            index_parent = mesh.z_curve_index(tree.dimension, tree.nlevel[index] - 1, index_x_parent, index_y_parent, index_z_parent)
                            parent = tree.tree_nodes[index_parent]
                            parent.keep_children = True
                            tree.create_children_of_node(parent)

        elif p == 1:
            for m in xrange(1, cfg.threshold_speed_propagation+1):
                if tree.bc_compatible_local_indexes(tree.nlevel[index], i, j, k+m) is not None:
                    a, b, c = tree.bc_compatible_local_indexes(tree.nlevel[index], i, j, k+m)
                    index_child = mesh.z_curve_index(tree.dimension, tree.nlevel[index], a, b, c)
                    if index_child in tree.tree_nodes:
                        index_parent = tree.nparent[index_child]
                        tree.nkeep_children[index_parent] = True
                    else:
                        if tree.nlevel[index] != 0:
                            index_x_parent = int(math.floor(a/2))
                            index_y_parent = int(math.floor(b/2))
                            index_z_parent = int(math.floor(c/2))
                            index_parent = mesh.z_curve_index(tree.dimension, tree.nlevel[index] - 1, index_x_parent, index_y_parent, index_z_parent)
                            parent = tree.tree_nodes[index_parent]
                            parent.keep_children = True
                            tree.create_children_of_node(parent)

def run_thresholding(threshold_parameter, *trees):
    """...

    """

    max_tree_nodes_index = max(trees[0].tree_nodes.keys())

    for index in xrange(max_tree_nodes_index+1):
        if index in trees[0].tree_nodes:

            keep_children = False

            for tree in trees:

                parent = tree.tree_nodes[index]

                if not parent.isleaf:

                    if (abs(parent.norm_details/tree.max_norm_details) >= \
                            compute_level_threshold_parameter(tree, parent, threshold_parameter) \
                            or parent.level <= tree.min_level):

                        keep_children = True

            for tree in trees:

                parent = tree.tree_nodes[index]

                if not parent.isleaf:

                    parent.keep_children = keep_children

    temp =[]
    for index in xrange(1, max_tree_nodes_index+1):
        if index in trees[0].tree_nodes:
            node = trees[0].tree_nodes[index]
            if trees[0].tree_nodes[node.parent].keep_children:
                temp.append(index)

    for index in temp:
        for tree in trees:

            node = tree.tree_nodes[index]
            for axis in xrange(tree.dimension):
                thresholding_information_propagation(tree, node, axis)
