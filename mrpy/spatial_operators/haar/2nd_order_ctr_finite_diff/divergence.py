from __future__ import print_function, division

"""This module is used to compute the divergence operator in the x-direction.

The procedure "create_matrix" returns the matrix representing the linear
combination of this operation on a cartesian grid representation of a variable.
Since the spatial operator depends on the specific boundary conditions applied
to the computed variable, this matrix depends on the boundary conditions.
The procedure mesh.bc_compatbile_local_indexes is used to return the right
indexes depending on the boundary conditions. It returns "None" if there is no
real node corresponding to the input indexes. Since we loop on the real leaves,
if this procedure returns "None" at a specific boundary, we know in which part
of the space we are.

The procedure "create_bc_scalar" returns an array of the values needed to
complete the computation of the spatial operation on the meshes located at the boundary
of the domain. We assume that the type and the values of the variable at the
boundray do not change with time, so that this array is built with the type of
boundary condition applied to the varialbe computed, and the values at the
north, south, east and west boundaries of the variable.
...
...

"""

import petsc4py.PETSc as petsc
from six.moves import range
import config as cfg
from mrpy.mr_utils import mesh
from mrpy.mr_utils import op
import numpy as np
import math
import importlib
from .matrix_aux import matrix_add

#!!!!!!! penser a rajouter un mr_bc_scalar !!!!!!!!
def create_matrix(tree, axis):

    matrix = petsc.Mat().create()
    number_of_rows = tree.number_of_leaves
    size_row = (number_of_rows, number_of_rows)
    size_col = (number_of_rows, number_of_rows)
    matrix.setSizes((size_row, size_col))
    matrix.setUp()
#    matrix = np.zeros(shape=(number_of_rows, number_of_rows), dtype=np.float)

    if axis == 0:

        for row in range(number_of_rows):
            index = tree.tree_leaves[row]
            level = tree.nlevel[index]
            i = tree.nindex_x[index]
            j = tree.nindex_y[index]
            k = tree.nindex_z[index]
            dx = tree.ndx[index]
            dy = tree.ndy[index]
            dz = tree.ndz[index]

            # left flux
            if mesh.bc_compatible_local_indexes(tree, level, i-1, j, k) is not None:
                i_left, j_left, k_left = mesh.bc_compatible_local_indexes(tree, level, i-1, j, k)
                index_left = mesh.z_curve_index(tree.dimension, level, i_left, j_left, k_left)

                if index_left in tree.tree_nodes and tree.nisleaf[index_left] \
                    or index_left not in tree.tree_nodes:
                # the finest level for the left flux is the node's level
                    if tree.dimension == 2:
                        matrix_add(tree, matrix, row, -(dy)/(2), level, i, j, k)
                        matrix_add(tree, matrix, row, -(dy)/(2), level, i_left, j_left, k_left)

                    elif tree.dimension == 3:
                        matrix_add(tree, matrix, row, -(dy*dz)/(2), level, i, j, k)
                        matrix_add(tree, matrix, row, -(dy*dz)/(2), level, i_left, j_left, k_left)

                else:
                # the finest level for the left flux is the level of the node's children
                    #if tree.dimension == 1:
                    #    matrix_add(tree, matrix, row, -1./((dx/2.)*2.), level+1, 2*i, 2*j, 2*k)
                    #    matrix_add(tree, matrix, row, -1./((dx/2.)*2.), level+1, 2*i_left+1, 2*j, 2*k)
#REVOIR LA DIMENSION 1 !!!

                    if tree.dimension == 2:
                        for n in range(2):
                            matrix_add(tree, matrix, row, -(dy/2)/(2), level+1, 2*i, 2*j+n, 2*k)
                            matrix_add(tree, matrix, row, -(dy/2)/(2), level+1, 2*i_left+1, 2*j+n, 2*k)

                    elif tree.dimension == 3:
                        for o in range(2):
                            for n in range(2):
                                matrix_add(tree, matrix, row, -((dy/2)*(dz/2))/(2), level+1, 2*i, 2*j+n, 2*k+o)
                                matrix_add(tree, matrix, row, -((dy/2)*(dz/2))/(2), level+1, 2*i_left+1, 2*j+n, 2*k+o)

            elif tree.bc["west"][0] == "dirichlet":
            #the left flux depends only on the boundary condition scalar

                pass

            elif tree.bc["west"][0] == "neumann":
            # the finest level for the left flux is the node's level; this node
            # receives a second contribution because of the boundary condition
                if tree.dimension == 2:
                    matrix_add(tree, matrix, row, -(dy), level, i, j, k)

                elif tree.dimension == 3:
                    matrix_add(tree, matrix, row, -(dy*dz), level, i, j, k)

            # right flux
            if mesh.bc_compatible_local_indexes(tree, level, i+1, j, k) is not None:
                i_right, j_right, k_right = mesh.bc_compatible_local_indexes(tree, level, i+1, j, k)
                index_right = mesh.z_curve_index(tree.dimension, level, i_right, j_right, k_right)

                if index_right in tree.tree_nodes and tree.nisleaf[index_right] \
                    or index_right not in tree.tree_nodes:
                # the finest level for the right flux is the node's level
                    if tree.dimension == 2:
                        matrix_add(tree, matrix, row, dy/(2), level, i, j, k)
                        matrix_add(tree, matrix, row, dy/(2), level, i_right, j_right, k_right)

                    elif tree.dimension == 3:
                        matrix_add(tree, matrix, row, dy*dz/(2), level, i, j, k)
                        matrix_add(tree, matrix, row, dy*dz/(2), level, i_right, j_right, k_right)

                else:
                # the finest level for the right flux is the level of the node's children
                    #if tree.dimension == 1:
                    #    matrix_add(tree, matrix, row, 1./((dx/2.)*2.), level+1, 2*i+1, 2*j, 2*k)
                    #    matrix_add(tree, matrix, row, 1./((dx/2.)*2.), level+1, 2*i_right, 2*j, 2*k)
#REVOIR LA DIMENSION 1 !!!

                    if tree.dimension == 2:
                        for n in range(2):
                            matrix_add(tree, matrix, row, (dy/2)/(2), level+1, 2*i+1, 2*j+n, 2*k)
                            matrix_add(tree, matrix, row, (dy/2)/(2), level+1, 2*i_right, 2*j+n, 2*k)

                    elif tree.dimension == 3:
                        for o in range(2):
                            for n in range(2):
                                matrix_add(tree, matrix, row, (dy/2)*(dz/2)/(2), level+1, 2*i+1, 2*j+n, 2*k+o)
                                matrix_add(tree, matrix, row, (dy/2)*(dz/2)/(2), level+1, 2*i_right, 2*j+n, 2*k+o)

            elif tree.bc["east"][0] == "dirichlet":
            #the right flux depends only on the boundary condition scalar

                pass

            elif tree.bc["east"][0] == "neumann":
            # the finest level for the right flux is the node's level; this node
            # receives a second contribution because of the boundary condition
                if tree.dimension == 2:
                    matrix_add(tree, matrix, row, (dy), level, i, j, k)

                elif tree.dimension == 3:
                    matrix_add(tree, matrix, row, (dy*dz), level, i, j, k)

        matrix.assemble()

        return matrix

    elif axis == 1:

        for row in range(number_of_rows):
            index = tree.tree_leaves[row]
            i = tree.nindex_x[index]
            j = tree.nindex_y[index]
            k = tree.nindex_z[index]
            level = tree.nlevel[index]
            dx = tree.ndx[index]
            dy = tree.ndy[index]
            dz = tree.ndz[index]

            # left flux
            if mesh.bc_compatible_local_indexes(tree, level, i, j-1, k) is not None:
                i_left, j_left, k_left = mesh.bc_compatible_local_indexes(tree, level, i, j-1, k)
                index_left = mesh.z_curve_index(tree.dimension, level, i_left, j_left, k_left)

                if index_left in tree.tree_nodes and tree.nisleaf[index_left] \
                    or index_left not in tree.tree_nodes:
                # the finest level for the left flux is the node's level
                    if tree.dimension == 2:
                        matrix_add(tree, matrix, row, -(dx)/(2), level, i, j, k)
                        matrix_add(tree, matrix, row, -(dx)/(2), level, i_left, j_left, k_left)

                    elif tree.dimension == 3:
                        matrix_add(tree, matrix, row, -(dx*dz)/(2), level, i, j, k)
                        matrix_add(tree, matrix, row, -(dx*dz)/(2), level, i_left, j_left, k_left)

                else:
                # the finest level for the left flux is the level of the node's children
                    #if tree.dimension == 1:
                    #    matrix_add(tree, matrix, row, -1./((dy/2.)*2.), level+1, 2*i, 2*j, 2*k)
                    #    matrix_add(tree, matrix, row, -1./((dy/2.)*2.), level+1, 2*i, 2*j_left+1, 2*k)
#REVOIR LA DIMENSION 1 !!!

                    if tree.dimension == 2:
                        for m in range(2):
                            matrix_add(tree, matrix, row, -(dx/2)/(2), level+1, 2*i+m, 2*j, 2*k)
                            matrix_add(tree, matrix, row, -(dx/2)/(2), level+1, 2*i+m, 2*j_left+1, 2*k)

                    elif tree.dimension == 3:
                        for o in range(2):
                            for m in range(2):
                                matrix_add(tree, matrix, row, -(dx/2)*(dz/2)/(2), level+1, 2*i+m, 2*j, 2*k+o)
                                matrix_add(tree, matrix, row, -(dx/2)*(dz/2)/(2), level+1, 2*i+m, 2*j_left+1, 2*k+o)

            elif tree.bc["south"][0] == "dirichlet":
            #the left flux depends only on the boundary condition scalar

                pass

            elif tree.bc["south"][0] == "neumann":
            # the finest level for the left flux is the node's level; this node
            # receives a second contribution because of the boundary condition
                if tree.dimension == 2:
                    matrix_add(tree, matrix, row, -(dx), level, i, j, k)

                elif tree.dimension == 3:
                    matrix_add(tree, matrix, row, -(dx*dz), level, i, j, k)

            # right flux
            if mesh.bc_compatible_local_indexes(tree, level, i, j+1, k) is not None:
                i_right, j_right, k_right = mesh.bc_compatible_local_indexes(tree, level, i, j+1, k)
                index_right = mesh.z_curve_index(tree.dimension, level, i_right, j_right, k_right)

                if index_right in tree.tree_nodes and tree.nisleaf[index_right] \
                    or index_right not in tree.tree_nodes:
                # the finest level for the right flux is the node's level
                    if tree.dimension == 2:
                        matrix_add(tree, matrix, row, dx/(2), level, i, j, k)
                        matrix_add(tree, matrix, row, dx/(2), level, i_right, j_right, k_right)

                    elif tree.dimension == 3:
                        matrix_add(tree, matrix, row, dx*dz/(2), level, i, j, k)
                        matrix_add(tree, matrix, row, dx*dz/(2), level, i_right, j_right, k_right)

                else:
                # the finest level for the right flux is the level of the node's children
                    #if tree.dimension == 1:
                    #    matrix_add(tree, matrix, row, 1./((dy/2.)*2.), level+1, 2*i, 2*j+1, 2*k)
                    #    matrix_add(tree, matrix, row, 1./((dy/2.)*2.), level+1, 2*i, 2*j_right, 2*k)
#REVOIR LA DIMENSION 1 !!!

                    if tree.dimension == 2:
                        for m in range(2):
                            matrix_add(tree, matrix, row, (dx/2)/(2), level+1, 2*i+m, 2*j+1, 2*k)
                            matrix_add(tree, matrix, row, (dx/2)/(2), level+1, 2*i+m, 2*j_right, 2*k)

                    elif tree.dimension == 3:
                        for o in range(2):
                            for m in range(2):
                                matrix_add(tree, matrix, row, (dx/2)*(dz/2)/(2), level+1, 2*i+m, 2*j+1, 2*k+o)
                                matrix_add(tree, matrix, row, (dx/2)*(dz/2)/(2), level+1, 2*i+m, 2*j_right, 2*k+o)

            elif tree.bc["north"][0] == "dirichlet":
            #the right flux depends only on the boundary condition scalar

                pass

            elif tree.bc["north"][0] == "neumann":
            # the finest level for the left flux is the node's level; this node
            # receives a second contribution because of the boundary condition
                if tree.dimension == 2:
                    matrix_add(tree, matrix, row, (dx), level, i, j, k)

                elif tree.dimension == 3:
                    matrix_add(tree, matrix, row, (dx*dz), level, i, j, k)

        matrix.assemble()

        return matrix

    elif axis == 2:

        for row in range(number_of_rows):
            index = tree.tree_leaves[row]
            i = tree.nindex_x[index]
            j = tree.nindex_y[index]
            k = tree.nindex_z[index]
            level = tree.nlevel[index]
            dx = tree.ndx[index]
            dy = tree.ndy[index]
            dz = tree.ndz[index]

            # left flux
            if mesh.bc_compatible_local_indexes(tree, level, i, j, k-1) is not None:
                i_left, j_left, k_left = mesh.bc_compatible_local_indexes(tree, level, i, j, k-1)
                index_left = mesh.z_curve_index(tree.dimension, level, i_left, j_left, k_left)

                if index_left in tree.tree_nodes and tree.nisleaf[index_left] \
                    or index_left not in tree.tree_nodes:
                # the finest level for the left flux is the node's level
                    matrix_add(tree, matrix, row, -(dx*dy)/(2), level, i, j, k)
                    matrix_add(tree, matrix, row, -(dx*dy)/(2), level, i_left, j_left, k_left)

                else:
                # the finest level for the left flux is the level of the node's children
                    for n in range(2):
                        for m in range(2):
                            matrix_add(tree, matrix, row, -(dx/2)*(dy/2)/(2), level+1, 2*i+m, 2*j+n, 2*k)
                            matrix_add(tree, matrix, row, -(dx/2)*(dy/2)/(2), level+1, 2*i+m, 2*j+n, 2*k_left+1)

            elif tree.bc["back"][0] == "dirichlet":
            #the left flux depends only on the boundary condition scalar

                pass

            elif tree.bc["back"][0] == "neumann":
            # the finest level for the left flux is the node's level; this node
            # receives a second contribution because of the boundary condition
                matrix_add(tree, matrix, row, -(dx*dz), level, i, j, k)

            # right flux
            if mesh.bc_compatible_local_indexes(tree, level, i, j, k+1) is not None:
                i_right, j_right, k_right = mesh.bc_compatible_local_indexes(tree, level, i, j, k+1)
                index_right = mesh.z_curve_index(tree.dimension, level, i_right, j_right, k_right)

                if index_right in tree.tree_nodes and tree.nisleaf[index_right] \
                    or index_right not in tree.tree_nodes:
                # the finest level for the right flux is the node's level
                    matrix_add(tree, matrix, row, (dx*dy)/(2), level, i, j, k)
                    matrix_add(tree, matrix, row, (dx*dy)/(2), level, i_right, j_right, k_right)

                else:
                # the finest level for the right flux is the level of the node's children
                    for n in range(2):
                        for m in range(2):
                            matrix_add(tree, matrix, row, (dx/2)*(dy/2)/(2), level+1, 2*i+m, 2*j+n, 2*k+1)
                            matrix_add(tree, matrix, row, (dx/2)*(dy/2)/(2), level+1, 2*i+m, 2*j+n, 2*k_right)

            elif tree.bc["forth"][0] == "dirichlet":
            #the right flux depends only on the boundary condition scalar

                pass

            elif tree.bc["forth"][0] == "neumann":
            # the finest level for the left flux is the node's level; this node
            # receives a second contribution because of the boundary condition
                matrix_add(tree, matrix, row, (dx*dz), level, i, j, k)

        matrix.assemble()

        return matrix

def create_bc_scalar(tree, axis, north=None, south=None, east=None, west=None, forth=None, back=None):

    scalar = petsc.Vec().create()
    number_of_rows = tree.number_of_leaves
    scalar.setSizes(number_of_rows, number_of_rows)
    scalar.setUp()

    if north is None and south is None and east is None and west is None and forth is None and back is None:
        north = tree.bc["north"][1]
        south = tree.bc["south"][1]
        west = tree.bc["west"][1]
        east = tree.bc["east"][1]
        forth = tree.bc["forth"][1]
        back = tree.bc["back"][1]

    if axis == 0:

        for row in range(number_of_rows):
            index = tree.tree_leaves[row]
            level = tree.nlevel[index]
            i = tree.nindex_x[index]
            j = tree.nindex_y[index]
            k = tree.nindex_z[index]
            dx = tree.ndx[index]
            dy = tree.ndy[index]
            dz = tree.ndz[index]

            #left flux
            if i == 0:

                if tree.bc["west"][0] == "periodic":
                    pass

                elif tree.bc["west"][0] == "dirichlet":
                    if tree.dimension == 2:
                        coords = mesh.boundary_coords(tree, "west", level, 0, j, k)
                        scalar.setValue(row, -west(coords)*dy, True)

                    elif tree.dimension == 3:
                        coords = mesh.boundary_coords(tree, "west", level, 0, j, k)
                        scalar.setValue(row, -west(coords)*dy*dz, True)

                elif tree.bc["west"][0] == "neumann":
                    if tree.dimension == 2:
                        coords = mesh.boundary_coords(tree, "west", level, 0, j, k)
                        scalar.setValue(row, -(west(coords)*dx/2)*dy, True)

                    elif tree.dimension == 3:
                        coords = mesh.boundary_coords(tree, "west", level, 0, j, k)
                        scalar.setValue(row, -(west(coords)*dx/2)*dy*dz, True)

            #right flux
            if i == 2**level-1:

                if tree.bc["east"][0] == "periodic":
                    pass

                elif tree.bc["east"][0] == "dirichlet":
                    if tree.dimension == 2:
                        coords = mesh.boundary_coords(tree, "east", level, 0, j, k)
                        scalar.setValue(row, east(coords)*dy, True)

                    elif tree.dimension == 3:
                        coords = mesh.boundary_coords(tree, "east", level, 0, j, k)
                        scalar.setValue(row, east(coords)*dy*dz, True)

                elif tree.bc["east"][0] == "neumann":
                    if tree.dimension == 2:
                        coords = mesh.boundary_coords(tree, "east", level, 0, j, k)
                        scalar.setValue(row, (east(coords)*dx/2)*dy, True)

                    elif tree.dimension == 3:
                        coords = mesh.boundary_coords(tree, "east", level, 0, j, k)
                        scalar.setValue(row, (east(coords)*dx/2)*dy*dz, True)

        return scalar

    elif axis == 1:

        for row in range(number_of_rows):
            index = tree.tree_leaves[row]
            level = tree.nlevel[index]
            i = tree.nindex_x[index]
            j = tree.nindex_y[index]
            k = tree.nindex_z[index]
            dx = tree.ndx[index]
            dy = tree.ndy[index]
            dz = tree.ndz[index]

            #left flux
            if j == 0:

                if tree.bc["south"][0] == "periodic":
                    pass

                elif tree.bc["south"][0] == "dirichlet":
                    if tree.dimension == 2:
                        coords = mesh.boundary_coords(tree, "south", level, i, 0, k)
                        scalar.setValue(row, -south(coords)*dx, True)

                    elif tree.dimension == 3:
                        coords = mesh.boundary_coords(tree, "south", level, i, 0, k)
                        scalar.setValue(row, -south(coords)*dx*dz, True)

                elif tree.bc["south"][0] == "neumann":
                    if tree.dimension == 2:
                        coords = mesh.boundary_coords(tree, "south", level, i, 0, k)
                        scalar.setValue(row, -(south(coords)*dy/2)*dx, True)

                    elif tree.dimension == 3:
                        coords = mesh.boundary_coords(tree, "south", level, i, 0, k)
                        scalar.setValue(row, -(south(coords)*dy/2)*dx*dz, True)

            #right flux
            if j == 2**level-1:

                if tree.bc["north"][0] == "periodic":
                    pass

                elif tree.bc["north"][0] == "dirichlet":
                    if tree.dimension == 2:
                        coords = mesh.boundary_coords(tree, "north", level, i, 0, k)
                        scalar.setValue(row, north(coords)*dx, True)

                    elif tree.dimension == 3:
                        coords = mesh.boundary_coords(tree, "north", level, i, 0, k)
                        scalar.setValue(row, north(coords)*dx*dz, True)

                elif tree.bc["north"][0] == "neumann":
                    if tree.dimension == 2:
                        coords = mesh.boundary_coords(tree, "north", level, i, 0, k)
                        scalar.setValue(row, (north(coords)*dy/2)*dx, True)

                    elif tree.dimension == 3:
                        coords = mesh.boundary_coords(tree, "north", level, i, 0, k)
                        scalar.setValue(row, (north(coords)*dy/2)*dx*dz, True)

        return scalar

    elif axis == 2:

        for row in range(number_of_rows):
            index = tree.tree_leaves[row]
            level = tree.nlevel[index]
            i = tree.nindex_x[index]
            j = tree.nindex_y[index]
            k = tree.nindex_z[index]
            dx = tree.ndx[index]
            dy = tree.ndy[index]
            dz = tree.ndz[index]

            #left flux
            if k == 0:

                if tree.bc["back"][0] == "periodic":
                    pass

                elif tree.bc["back"][0] == "dirichlet":
                    coords = mesh.boundary_coords(tree, "back", level, i, j, 0)
                    scalar.setValue(row, -back(coords)*dx*dy, True)

                elif tree.bc["back"][0] == "neumann":
                    coords = mesh.boundary_coords(tree, "back", level, i, j, 0)
                    scalar.setValue(row, -(back(coords)*dz/2)*dx*dy, True)

            #right flux
            if k == 2**level-1:

                if tree.bc["forth"][0] == "periodic":
                    pass

                elif tree.bc["forth"][0] == "dirichlet":
                    coords = mesh.boundary_coords(tree, "back", level, i, j, 0)
                    scalar.setValue(row, forth(coords)*dx*dy, True)

                elif tree.bc["forth"][0] == "neumann":
                    coords = mesh.boundary_coords(tree, "back", level, i, j, 0)
                    scalar.setValue(row, (forth(coords)*dz/2)*dx*dy, True)

        return scalar


if __name__ == "__main__":

    output_module = importlib.import_module(cfg.output_module_name)

    tree = mesh.create_new_tree(cfg.dimension, cfg.min_level, cfg.max_level, cfg.stencil_graduation, cfg.stencil_prediction)

    tree.tag = "u"

    mesh.listing_of_leaves(tree)

    print(tree.number_of_leaves)
    print("")
    divergence_matrix = create_matrix(tree, 0)
    divergence_matrix.view()
    print("")

    for index in tree.tree_leaves:
        tree.nvalue[index] = cfg.function(tree.ncoord_x[index], tree.ncoord_y[index])

    output_module.write(tree, "finest_grid.dat")

    op.run_projection(tree)

    op.encode_details(tree)

    op.run_thresholding(tree)

    op.run_grading(tree)

    op.run_pruning(tree)

    mesh.listing_of_leaves(tree)

    print(tree.number_of_leaves)
    print("")

    output_module.write(tree, "test_adapted_grid.dat")

    divergence_matrix = create_matrix(tree, 0)
    divergence_matrix.view()
    print("")

    divergence_bc = create_bc_scalar(tree, 0)
    divergence_bc.view()
    print("")
