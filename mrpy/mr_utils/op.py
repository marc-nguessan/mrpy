from __future__ import print_function, division
from six.moves import range

"""...

"""

import importlib
import config as cfg
import math
from mrpy.mr_utils import mesh

prediction_op = importlib.import_module(cfg.prediction_operator_module)
thresholding_op = importlib.import_module(cfg.thresholding_operator_module)


def compute_projection_value(tree, index):
    """...

    """

    temp = 0
    for index_child in tree.nchildren[index]:
        temp = temp + tree.nvalue[index_child]

    tree.nvalue[index] = temp / len(tree.nchildren[index])

def run_projection(*trees):
    """...

    """

    max_tree_nodes_index = max(trees[0].tree_nodes.keys())

    for index in range(int(max_tree_nodes_index), -1, -1):
        if index in trees[0].tree_nodes:
            for tree in trees:
                if not tree.nisleaf[index]:
                    compute_projection_value(tree, index)

def compute_prediction_value(tree, index_parent, index_child):
    """...

    """

    return prediction_op.compute_prediction_value(tree, index_parent, index_child)

def encode_details(*trees):
    """...
#expliquer l'utilisation du foo

    """

    def single_tree_function(tree):
        tree_indexes = tree.tree_nodes.keys()

        for index_parent in tree_indexes:
            if not tree.nisleaf[index_parent]:

                tree.nnorm_details[index_parent] = 0
                foo = []
                for index_child in tree.nchildren[index_parent]:
                    temp = tree.nvalue[index_child] - compute_prediction_value(tree, index_parent, index_child)
                    foo.append(temp)
                    # Norm L2 of the details
                    tree.nnorm_details[index_parent] += temp**2

                tree.ndetails[index_parent] = foo
                tree.nnorm_details[index_parent] = math.sqrt(tree.nnorm_details[index_parent] / len(tree.nchildren[index_parent]))

                tree.max_norm_details = max(tree.max_norm_details, tree.nnorm_details[index_parent])

    for tree in trees:
        single_tree_function(tree)

def insert_in_graded_tree(tree, level, index_x=0, index_y=0, index_z=0):
    """...

    """

    index = mesh.z_curve_index(tree.dimension, level, index_x, index_y, index_z)
    if index in tree.tree_nodes:
        index_parent = tree.nparent[index]
        tree.nkeep_children[index_parent] = True
    else:
        mesh.create_node(tree, tree.dimension, level, index_x, index_y, index_z)
        index_parent = tree.nparent[index]
        mesh.create_children_of_node(tree, index_parent)
        tree.nkeep_children[index_parent] = True
        #if level != tree.max_level:
        #    mesh.create_children_pointers(tree, tree.dimension, level, index_x, index_y, index_z)

def local_grading_along_axis(tree, index, stencil, axis):
    """...

    """

    level = tree.nlevel[index]
    index_x = tree.nindex_x[index]
    index_y = tree.nindex_y[index]
    index_z = tree.nindex_z[index]

    if tree.nkeep_children[index] and not tree.ngraded[index]:

        for m in range(1, stencil+1):

            if axis == 0:

                local_indexes = mesh.bc_compatible_local_indexes(tree, level, index_x-m, index_y, index_z)
                if local_indexes is not None:
                    insert_in_graded_tree(tree, level, local_indexes[0], local_indexes[1], local_indexes[2])

                local_indexes = mesh.bc_compatible_local_indexes(tree, level, index_x+m, index_y, index_z)
                if local_indexes is not None:
                    insert_in_graded_tree(tree, level, local_indexes[0], local_indexes[1], local_indexes[2])

            elif axis == 1:
                local_indexes = mesh.bc_compatible_local_indexes(tree, level, index_x, index_y-m, index_z)
                if local_indexes is not None:
                    insert_in_graded_tree(tree, level, local_indexes[0], local_indexes[1], local_indexes[2])

                local_indexes = mesh.bc_compatible_local_indexes(tree, level, index_x, index_y+m, index_z)
                if local_indexes is not None:
                    insert_in_graded_tree(tree, level, local_indexes[0], local_indexes[1], local_indexes[2])

            elif axis == 2:
                local_indexes = mesh.bc_compatible_local_indexes(tree, level, index_x, index_y, index_z-m)
                if local_indexes is not None:
                    insert_in_graded_tree(tree, level, local_indexes[0], local_indexes[1], local_indexes[2])

                local_indexes = mesh.bc_compatible_local_indexes(tree, level, index_x, index_y, index_z+m)
                if local_indexes is not None:
                    insert_in_graded_tree(tree, level, local_indexes[0], local_indexes[1], local_indexes[2])

def local_grading_diagonal(tree, index, stencil, dimension):
    """...

    """

    if tree.nkeep_children[index] and not tree.ngraded[index]:

        if dimension == 1:
            pass

        elif dimension == 2:

            index_x = tree.nindex_x[index]
            index_y = tree.nindex_y[index]
            level = tree.nlevel[index]
            for m in range(1, stencil+1):
                for n in range(1, stencil+1):

                    local_indexes = mesh.bc_compatible_local_indexes(tree, level, index_x-m, index_y-n)
                    if local_indexes is not None:
                        insert_in_graded_tree(tree, level, local_indexes[0], local_indexes[1], local_indexes[2])

                    local_indexes = mesh.bc_compatible_local_indexes(tree, level, index_x-m, index_y+n)
                    if local_indexes is not None:
                        insert_in_graded_tree(tree, level, local_indexes[0], local_indexes[1], local_indexes[2])

                    local_indexes = mesh.bc_compatible_local_indexes(tree, level, index_x+m, index_y-n)
                    if local_indexes is not None:
                        insert_in_graded_tree(tree, level, local_indexes[0], local_indexes[1], local_indexes[2])

                    local_indexes = mesh.bc_compatible_local_indexes(tree, level, index_x+m, index_y+n)
                    if local_indexes is not None:
                        insert_in_graded_tree(tree, level, local_indexes[0], local_indexes[1], local_indexes[2])

        elif dimension == 3:
            index_x = tree.nindex_x[index]
            index_y = tree.nindex_y[index]
            index_z = tree.nindex_z[index]
            level = tree.nlevel[index]
            for m in range(1, stencil+1):
                for n in range(1, stencil+1):

                    local_indexes = mesh.bc_compatible_local_indexes(tree, level, index_x-m, index_y-n, index_z)
                    if local_indexes is not None:
                        insert_in_graded_tree(tree, level, local_indexes[0], local_indexes[1], local_indexes[2])

                    local_indexes = mesh.bc_compatible_local_indexes(tree, level, index_x-m, index_y+n, index_z)
                    if local_indexes is not None:
                        insert_in_graded_tree(tree, level, local_indexes[0], local_indexes[1], local_indexes[2])

                    local_indexes = mesh.bc_compatible_local_indexes(tree, level, index_x+m, index_y-n, index_z)
                    if local_indexes is not None:
                        insert_in_graded_tree(tree, level, local_indexes[0], local_indexes[1], local_indexes[2])

                    local_indexes = mesh.bc_compatible_local_indexes(tree, level, index_x+m, index_y+n, index_z)
                    if local_indexes is not None:
                        insert_in_graded_tree(tree, level, local_indexes[0], local_indexes[1], local_indexes[2])

                    local_indexes = mesh.bc_compatible_local_indexes(tree, level, index_x-m, index_y, index_z-n)
                    if local_indexes is not None:
                        insert_in_graded_tree(tree, level, local_indexes[0], local_indexes[1], local_indexes[2])

                    local_indexes = mesh.bc_compatible_local_indexes(tree, level, index_x-m, index_y, index_z+n)
                    if local_indexes is not None:
                        insert_in_graded_tree(tree, level, local_indexes[0], local_indexes[1], local_indexes[2])

                    local_indexes = mesh.bc_compatible_local_indexes(tree, level, index_x+m, index_y, index_z-n)
                    if local_indexes is not None:
                        insert_in_graded_tree(tree, level, local_indexes[0], local_indexes[1], local_indexes[2])

                    local_indexes = mesh.bc_compatible_local_indexes(tree, level, index_x+m, index_y, index_z+n)
                    if local_indexes is not None:
                        insert_in_graded_tree(tree, level, local_indexes[0], local_indexes[1], local_indexes[2])

                    local_indexes = mesh.bc_compatible_local_indexes(tree, level, index_x, index_y-m, index_z-n)
                    if local_indexes is not None:
                        insert_in_graded_tree(tree, level, local_indexes[0], local_indexes[1], local_indexes[2])

                    local_indexes = mesh.bc_compatible_local_indexes(tree, level, index_x, index_y-m, index_z+n)
                    if local_indexes is not None:
                        insert_in_graded_tree(tree, level, local_indexes[0], local_indexes[1], local_indexes[2])

                    local_indexes = mesh.bc_compatible_local_indexes(tree, level, index_x, index_y+m, index_z-n)
                    if local_indexes is not None:
                        insert_in_graded_tree(tree, level, local_indexes[0], local_indexes[1], local_indexes[2])

                    local_indexes = mesh.bc_compatible_local_indexes(tree, level, index_x, index_y+m, index_z+n)
                    if local_indexes is not None:
                        insert_in_graded_tree(tree, level, local_indexes[0], local_indexes[1], local_indexes[2])

                    for o in range(1, stencil+1):

                        local_indexes = mesh.bc_compatible_local_indexes(tree, level, index_x-m, index_y-n, index_z+o)
                        if local_indexes is not None:
                            insert_in_graded_tree(tree, level, local_indexes[0], local_indexes[1], local_indexes[2])

                        local_indexes = mesh.bc_compatible_local_indexes(tree, level, index_x-m, index_y+n, index_z+o)
                        if local_indexes is not None:
                            insert_in_graded_tree(tree, level, local_indexes[0], local_indexes[1], local_indexes[2])

                        local_indexes = mesh.bc_compatible_local_indexes(tree, level, index_x+m, index_y-n, index_z+o)
                        if local_indexes is not None:
                            insert_in_graded_tree(tree, level, local_indexes[0], local_indexes[1], local_indexes[2])

                        local_indexes = mesh.bc_compatible_local_indexes(tree, level, index_x+m, index_y+n, index_z+o)
                        if local_indexes is not None:
                            insert_in_graded_tree(tree, level, local_indexes[0], local_indexes[1], local_indexes[2])

                        local_indexes = mesh.bc_compatible_local_indexes(tree, level, index_x-m, index_y-n, index_z-o)
                        if local_indexes is not None:
                            insert_in_graded_tree(tree, level, local_indexes[0], local_indexes[1], local_indexes[2])

                        local_indexes = mesh.bc_compatible_local_indexes(tree, level, index_x-m, index_y+n, index_z-o)
                        if local_indexes is not None:
                            insert_in_graded_tree(tree, level, local_indexes[0], local_indexes[1], local_indexes[2])

                        local_indexes = mesh.bc_compatible_local_indexes(tree, level, index_x+m, index_y-n, index_z-o)
                        if local_indexes is not None:
                            insert_in_graded_tree(tree, level, local_indexes[0], local_indexes[1], local_indexes[2])

                        local_indexes = mesh.bc_compatible_local_indexes(tree, level, index_x+m, index_y+n, index_z-o)
                        if local_indexes is not None:
                            insert_in_graded_tree(tree, level, local_indexes[0], local_indexes[1], local_indexes[2])

def compute_missing_values(*trees):
    """...

    """

    max_tree_nodes_index = max(trees[0].tree_nodes.keys())

    def single_tree_function(tree):
        for index in range(max_tree_nodes_index+1):
            if index in tree.tree_nodes and tree.nvalue[index] is None:
                index_parent = tree.nparent[index]
                tree.nvalue[index] = compute_prediction_value(tree, index_parent, index)

    for tree in trees:
        single_tree_function(tree)

def run_grading(*trees):
    """...

    """

    stencil_graduation = trees[0].stencil_graduation
    max_tree_nodes_index = max(trees[0].tree_nodes.keys())
    dimension = trees[0].dimension

    #We do not need to grade the root
    for index in range(max_tree_nodes_index, 2**dimension, -1):
        if index in trees[0].tree_nodes:

            for tree in trees:
                index_parent = tree.nparent[index]
                for axis in range(dimension):
                    local_grading_along_axis(tree, index_parent, stencil_graduation, axis)

                local_grading_diagonal(tree, index_parent, stencil_graduation, dimension)
                # the parent is graded, so we do not need to do the same thing for the brothers
                tree.ngraded[index_parent] = True

#Could be changed in the future
    # Computing of prediction value in the newly created nodes
    for tree in trees:
        compute_missing_values(tree)
        #single_tree_function(tree)

    for index in trees[0].tree_nodes.keys():
        for tree in trees:
            # We need to reinitialize this variable for every node
            tree.ngraded[index] = False

def run_thresholding(*trees, threshold_parameter=cfg.threshold_parameter):
    """...

    """

    thresholding_op.run_thresholding(threshold_parameter, *trees)

def run_pruning(*trees):
    """...

    """

    max_tree_nodes_index = max(trees[0].tree_nodes.keys())

    for index in range(int(max_tree_nodes_index), -1, -1):
        if index in trees[0].tree_nodes:

            for tree in trees:

                if not tree.nkeep_children[index] and \
                       tree.nchildren[index] != []  and \
                       tree.nchildren[index][0] in tree.tree_nodes:
                    for child_index in tree.nchildren[index]:
                        mesh.delete_node(tree, child_index)

def set_to_same_grading(target, *trees):
    """...

    """

    def single_tree_function(tree):
        for index in tree.tree_nodes:
            tree.nkeep_children[index] = False # we reset the flag to False before setting to the grading of the target

        for index in target.tree_nodes:
            keep_children_target = target.nkeep_children[index]
            tree.nkeep_children[index] = keep_children_target
            if keep_children_target == True:
                if tree.nchildren[index] == [] or tree.nchildren[index][0] not in tree.tree_nodes:
                    #print index
                    mesh.create_children_of_node(tree, index)

    #    # Computing of prediction value in the newly created nodes
    #    compute_missing_values(tree)

    for tree in trees:
        single_tree_function(tree)

def project_to_finest_grid(*trees):
    """...

    """

    def single_tree_function(tree):

        for level in range(tree.max_level):

            if tree.dimension == 1:
                for i in range(2**level):
                    index = mesh.z_curve_index(tree.dimension, level, i)
                    if tree.nchildren[index] == [] or tree.nchildren[index][0] not in tree.tree_nodes:
                        mesh.create_children_of_node(tree, index)

            elif tree.dimension == 2:
                for i in range(2**level):
                    for j in range(2**level):
                        index = mesh.z_curve_index(tree.dimension, level, i, j)
                        if tree.nchildren[index] == [] or tree.nchildren[index][0] not in tree.tree_nodes:
                            mesh.create_children_of_node(tree, index)

            elif tree.dimension == 3:
                for i in range(2**level):
                    for j in range(2**level):
                        for k in range(2**level):
                            index = mesh.z_curve_index(tree.dimension, level, i, j, k)
                            if tree.nchildren[index] == [] or tree.nchildren[index][0] not in tree.tree_nodes:
                                mesh.create_children_of_node(tree, index)

        # Computing of prediction value in the newly created nodes
        compute_missing_values(tree)

    for tree in trees:
        single_tree_function(tree)

def global_error_to_finest_grid(tree_adapted, tree_finest):
    """Computes the L2 error of the adapted solution compared to the
    solution on the finest grid.

    The tree_adapted must have been projected to the finest grid first."""

    error = 0
    for index in tree_finest.tree_leaves:
        error += (tree_finest.nvalue[index] - tree_adapted.nvalue[index])**2

    error = math.sqrt(error)

    if tree_finest.dimension == 1:
        dx = mesh.space_step(tree_finest, tree_finest.max_level, 0)
        return error*dx

    elif tree_finest.dimension == 2:
        dx = mesh.space_step(tree_finest, tree_finest.max_level, 0)
        dy = mesh.space_step(tree_finest, tree_finest.max_level, 1)
        return error*dx*dy

    elif tree_finest.dimension == 3:
        dx = mesh.space_step(tree_finest, tree_finest.max_level, 0)
        dy = mesh.space_step(tree_finest, tree_finest.max_level, 1)
        dz = mesh.space_step(tree_finest, tree_finest.max_level, 2)
        return error*dx*dy*dz

if __name__ == "__main__":

    module = importlib.import_module(cfg.prediction_operator_module)
    module.test()
    print("done")
