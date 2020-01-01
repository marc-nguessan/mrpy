from __future__ import print_function, division
#!!!!!!!!!!! DEPRECATED. SHOULD BE SUPRESSED !!!!!!!!!!!!!

"""This module is used to compute the laplacian operator in the x-direction.

The procedure "create_matrix" returns the matrix representing the linear
combination of this operation on a cartesian grid representation of a variable.
Since the spatial operator depends on the specific boundary conditions applied
to the computed variable, this matrix depends on the boundary conditions.

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
    boundary_conditions = tree.bc[0]
#    matrix = np.zeros(shape=(number_of_rows, number_of_rows), dtype=np.float)

    if tree.dimension == 2:

        for row in range(number_of_rows):
            index = tree.tree_leaves[row]
            i = tree.nindex_x[index]
            j = tree.nindex_y[index]
            k = tree.nindex_z[index]
            level = tree.nlevel[index]
            dx = tree.ndx[index]
            dy = tree.ndy[index]
            dz = tree.ndz[index]

            # left flux for axis 0
            if mesh.bc_compatible_local_indexes(tree, level, i-1, j, k) is not None:
                i_left, j_left, k_left = mesh.bc_compatible_local_indexes(tree, level, i-1, j, k)
                index_left = mesh.z_curve_index(tree.dimension, level, i_left, j_left, k_left)

                if index_left in tree.tree_nodes and tree.nisleaf[index_left] \
                    or index_left not in tree.tree_nodes:
                # the finest level for the left flux is the node's level
                    matrix_add(tree, matrix, row, -(dy)/(dx), level, i, j, k)
                    matrix_add(tree, matrix, row, (dy)/(dx), level, i_left, j_left, k_left)

                else:
                    for n in range(2):
                        matrix_add(tree, matrix, row, -(dy/2.)/((dx/2.)), level+1, 2*i, 2*j+n, 2*k)
                        matrix_add(tree, matrix, row, (dy/2.)/((dx/2.)), level+1, 2*i_left+1, 2*j+n, 2*k)

            elif boundary_conditions == "dirichlet":
                # the finest level for the left flux is the node's level; this node
                # receives a second contribution because of the boundary condition
                matrix_add(tree, matrix, row, -2*(dy)/(dx), level, i, j, k)

            elif boundary_conditions == "neumann":
            #the left flux depends only on the boundary condition scalar

                pass

            # right flux for axis 0
            if mesh.bc_compatible_local_indexes(tree, level, i+1, j, k) is not None:
                i_right, j_right, k_right = mesh.bc_compatible_local_indexes(tree, level, i+1, j, k)
                index_right = mesh.z_curve_index(tree.dimension, level, i_right, j_right, k_right)

                if index_right in tree.tree_nodes and tree.nisleaf[index_right] \
                    or index_right not in tree.tree_nodes:
                # the finest level for the right flux is the node's level
                    matrix_add(tree, matrix, row, -dy/(dx), level, i, j, k)
                    matrix_add(tree, matrix, row, dy/(dx), level, i_right, j_right, k_right)

                else:
                    for n in range(2):
                        matrix_add(tree, matrix, row, -(dy/2.)/((dx/2.)), level+1, 2*i+1, 2*j+n, 2*k)
                        matrix_add(tree, matrix, row, (dy/2.)/((dx/2.)), level+1, 2*i_right, 2*j+n, 2*k)

            elif boundary_conditions == "dirichlet":
                # the finest level for the right flux is the node's level; this node
                # receives a second contribution because of the boundary condition
                matrix_add(tree, matrix, row, -2*(dy)/(dx), level, i, j, k)

            elif boundary_conditions == "neumann":
            #the left flux depends only on the boundary condition scalar

                pass

            # left flux for axis 1
            if mesh.bc_compatible_local_indexes(tree, level, i, j-1, k) is not None:
                i_left, j_left, k_left = mesh.bc_compatible_local_indexes(tree, level, i, j-1, k)
                index_left = mesh.z_curve_index(tree.dimension, level, i_left, j_left, k_left)

                if index_left in tree.tree_nodes and tree.nisleaf[index_left] \
                    or index_left not in tree.tree_nodes:
                # the finest level for the left flux is the node's level
                    matrix_add(tree, matrix, row, -(dx)/(dy), level, i, j, k)
                    matrix_add(tree, matrix, row, (dx)/(dy), level, i_left, j_left, k_left)

                else:
                    for m in range(2):
                        matrix_add(tree, matrix, row, -(dx/2.)/((dy/2.)), level+1, 2*i+m, 2*j, 2*k)
                        matrix_add(tree, matrix, row, (dx/2.)/((dy/2.)), level+1, 2*i+m, 2*j_left+1, 2*k)

            elif boundary_conditions == "dirichlet":
                # the finest level for the left flux is the node's level; this node
                # receives a second contribution because of the boundary condition
                matrix_add(tree, matrix, row, -2*(dx)/(dy), level, i, j, k)

            elif boundary_conditions == "neumann":
            #the left flux depends only on the boundary condition scalar

                pass

            # right flux for axis 1
            if mesh.bc_compatible_local_indexes(tree, level, i, j+1, k) is not None:
                i_right, j_right, k_right = mesh.bc_compatible_local_indexes(tree, level, i, j+1, k)
                index_right = mesh.z_curve_index(tree.dimension, level, i_right, j_right, k_right)

                if index_right in tree.tree_nodes and tree.nisleaf[index_right] \
                    or index_right not in tree.tree_nodes:
                # the finest level for the right flux is the node's level
                    matrix_add(tree, matrix, row, -dx/(dy), level, i, j, k)
                    matrix_add(tree, matrix, row, dx/(dy), level, i_right, j_right, k_right)

                else:
                    for m in range(2):
                        matrix_add(tree, matrix, row, -(dx/2.)/((dy/2.)), level+1, 2*i+m, 2*j+1, 2*k)
                        matrix_add(tree, matrix, row, (dx/2.)/((dy/2.)), level+1, 2*i+m, 2*j_right, 2*k)

            elif boundary_conditions == "dirichlet":
                # the finest level for the right flux is the node's level; this node
                # receives a second contribution because of the boundary condition
                matrix_add(tree, matrix, row, -2*(dx)/(dy), level, i, j, k)

            elif boundary_conditions == "neumann":
            #the left flux depends only on the boundary condition scalar

                pass

        matrix.assemble()

        return matrix

    if tree.dimension == 3:

        for row in range(number_of_rows):
            index = tree.tree_leaves[row]
            i = tree.nindex_x[index]
            j = tree.nindex_y[index]
            k = tree.nindex_z[index]
            level = tree.nlevel[index]
            dx = tree.ndx[index]
            dy = tree.ndy[index]
            dz = tree.ndz[index]

            node = tree.tree_leaves[row]
            i = node.index_x
            j = node.index_y
            k = node.index_z

            # left flux for axis 0
            if mesh.bc_compatible_local_indexes(tree, level, i-1, j, k) is not None:
                i_left, j_left, k_left = mesh.bc_compatible_local_indexes(tree, level, i-1, j, k)
                index_left = mesh.z_curve_index(tree.dimension, level, i_left, j_left, k_left)

                if index_left in tree.tree_nodes and tree.nisleaf[index_left] \
                    or index_left not in tree.tree_nodes:
                # the finest level for the left flux is the node's level
                    matrix_add(tree, matrix, row, -(dy*dz)/(dx), level, i, j, k)
                    matrix_add(tree, matrix, row, (dy*dz)/(dx), level, i_left, j_left, k_left)

                else:
                    for o in range(2):
                        for n in range(2):
                            matrix_add(tree, matrix, row, -((dy/2.)*(dz/2.))/((dx/2)), level+1, 2*i, 2*j+n, 2*k+o)
                            matrix_add(tree, matrix, row, ((dy/2.)*(dz/2.))/((dx/2)), level+1, 2*i_left+1, 2*j+n, 2*k+o)

            elif boundary_conditions == "dirichlet":
                # the finest level for the left flux is the node's level; this node
                # receives a second contribution because of the boundary condition
                matrix_add(tree, matrix, row, -2*(dy*dz)/(dx), level, i, j, k)

            elif boundary_conditions == "neumann":
            #the left flux depends only on the boundary condition scalar

                pass

            # right flux for axis 0
            if mesh.bc_compatible_local_indexes(tree, level, i+1, j, k) is not None:
                i_right, j_right, k_right = mesh.bc_compatible_local_indexes(tree, level, i+1, j, k)
                index_right = mesh.z_curve_index(tree.dimension, level, i_right, j_right, k_right)

                if index_right in tree.tree_nodes and tree.nisleaf[index_right] \
                    or index_right not in tree.tree_nodes:
                # the finest level for the right flux is the node's level
                    matrix_add(tree, matrix, row, -dy*dz/(dx), level, i, j, k)
                    matrix_add(tree, matrix, row, dy*dz/(dx), level, i_right, j_right, k_right)

                else:
                    for o in range(2):
                        for n in range(2):
                            matrix_add(tree, matrix, row, -(dy/2.)*(dz/2.)/((dx/2.)), level+1, 2*i+1, 2*j+n, 2*k+o)
                            matrix_add(tree, matrix, row, (dy/2)*(dz/2.)/((dx/2.)), level+1, 2*i_right, 2*j+n, 2*k+o)

            elif boundary_conditions == "dirichlet":
                # the finest level for the right flux is the node's level; this node
                # receives a second contribution because of the boundary condition
                matrix_add(tree, matrix, row, -2*(dy*dz)/(dx), level, i, j, k)

            elif boundary_conditions == "neumann":
            #the left flux depends only on the boundary condition scalar

                pass

            # left flux for axis 1
            if mesh.bc_compatible_local_indexes(tree, level, i, j-1, k) is not None:
                i_left, j_left, k_left = mesh.bc_compatible_local_indexes(tree, level, i, j-1, k)
                index_left = mesh.z_curve_index(tree.dimension, level, i_left, j_left, k_left)

                if index_left in tree.tree_nodes and tree.nisleaf[index_left] \
                    or index_left not in tree.tree_nodes:
                # the finest level for the left flux is the node's level
                    matrix_add(tree, matrix, row, -(dx*dz)/(dy), level, i, j, k)
                    matrix_add(tree, matrix, row, (dx*dz)/(dy), level, i_left, j_left, k_left)

                else:
                    for o in range(2):
                        for m in range(2):
                            matrix_add(tree, matrix, row, -(dx/2.)*(dz/2.)/((dy/2.)), level+1, 2*i+m, 2*j, 2*k+o)
                            matrix_add(tree, matrix, row, (dx/2.)*(dz/2.)/((dy/2.)), level+1, 2*i+m, 2*j_left+1, 2*k+o)

            elif boundary_conditions == "dirichlet":
                # the finest level for the left flux is the node's level; this node
                # receives a second contribution because of the boundary condition
                matrix_add(tree, matrix, row, -2*(dx*dz)/(dy), level, i, j, k)

            elif boundary_conditions == "neumann":
            #the left flux depends only on the boundary condition scalar

                pass

            # right flux for axis 1
            if mesh.bc_compatible_local_indexes(tree, level, i, j+1, k) is not None:
                i_right, j_right, k_right = mesh.bc_compatible_local_indexes(tree, level, i, j+1, k)
                index_right = mesh.z_curve_index(tree.dimension, level, i_right, j_right, k_right)

                if index_right in tree.tree_nodes and tree.nisleaf[index_right] \
                    or index_right not in tree.tree_nodes:
                # the finest level for the right flux is the node's level
                    matrix_add(tree, matrix, row, -dx*dz/(dy), level, i, j, k)
                    matrix_add(tree, matrix, row, dx*dz/(dy), level, i_right, j_right, k_right)

                else:
                    for o in range(2):
                        for m in range(2):
                            matrix_add(tree, matrix, row, -(dx/2.)*(dz/2.)/((dy/2.)), level+1, 2*i+m, 2*j, 2*k+o)
                            matrix_add(tree, matrix, row, (dx/2.)*(dz/2.)/((dy/2.)), level+1, 2*i+m, 2*j_right, 2*k+o)

            elif boundary_conditions == "dirichlet":
                # the finest level for the right flux is the node's level; this node
                # receives a second contribution because of the boundary condition
                matrix_add(tree, matrix, row, -2*(dx*dz)/(dy), level, i, j, k)

            elif boundary_conditions == "neumann":
            #the left flux depends only on the boundary condition scalar

                pass

            # left flux for axis 2
            if mesh.bc_compatible_local_indexes(tree, level, i, j, k-1) is not None:
                i_left, j_left, k_left = mesh.bc_compatible_local_indexes(tree, level, i, j, k-1)
                index_left = mesh.z_curve_index(tree.dimension, level, i_left, j_left, k_left)

                if index_left in tree.tree_nodes and tree.nisleaf[index_left] \
                    or index_left not in tree.tree_nodes:
                # the finest level for the left flux is the node's level
                    matrix_add(tree, matrix, row, -(dx*dy)/(dz), level, i, j, k)
                    matrix_add(tree, matrix, row, (dx*dy)/(dz), level, i_left, j_left, k_left)

                else:
                    for n in range(2):
                        for m in range(2):
                            matrix_add(tree, matrix, row, -(dx/2.)*(dy/2.)/((dz/2.)), level+1, 2*i+m, 2*j+n, 2*k)
                            matrix_add(tree, matrix, row, (dx/2.)*(dy/2.)/((dz/2.)), level+1, 2*i+m, 2*j+n, 2*k_left+1)

            elif boundary_conditions == "dirichlet":
                # the finest level for the left flux is the node's level; this node
                # receives a second contribution because of the boundary condition
                matrix_add(tree, matrix, row, -2*(dx*dy)/(dz), level, i, j, k)

            elif boundary_conditions == "neumann":
            #the left flux depends only on the boundary condition scalar

                pass

            # right flux for axis 2
            if mesh.bc_compatible_local_indexes(tree, level, i, j, k+1) is not None:
                i_right, j_right, k_right = mesh.bc_compatible_local_indexes(tree, level, i, j, k+1)
                index_right = mesh.z_curve_index(tree.dimension, level, i_right, j_right, k_right)

                if index_right in tree.tree_nodes and tree.nisleaf[index_right] \
                    or index_right not in tree.tree_nodes:
                # the finest level for the right flux is the node's level
                    matrix_add(tree, matrix, row, -(dx*dy)/(dz), level, i, j, k)
                    matrix_add(tree, matrix, row, (dx*dy)/(dz), level, i_right, j_right, k_right)

                else:
                    for n in range(2):
                        for m in range(2):
                            matrix_add(tree, matrix, row, -(dx/2.)*(dy/2.)/((dz/2.)), level+1, 2*i+m, 2*j+n, 2*k+1)
                            matrix_add(tree, matrix, row, (dx/2.)*(dy/2.)/((dz/2.)), level+1, 2*i+m, 2*j+n, 2*k_right)

            elif boundary_conditions == "dirichlet":
                # the finest level for the right flux is the node's level; this node
                # receives a second contribution because of the boundary condition
                matrix_add(tree, matrix, row, -2*(dx*dy)/(dz), level, i, j, k)

            elif boundary_conditions == "neumann":
            #the left flux depends only on the boundary condition scalar

                pass

        matrix.assemble()

        return matrix

def create_bc_scalar(tree, axis, north=None, south=None, east=None, west=None, front=None, back=None):

    scalar = petsc.Vec().create()
    number_of_rows = tree.number_of_leaves
    scalar.setSizes(number_of_rows, number_of_rows)
    scalar.setUp()

    boundary_conditions = tree.bc[0]
    if north is None and south is None and east is None and west is None and front is None and back is None:
        north=tree.bc[1][0]
        south=tree.bc[1][1]
        west=tree.bc[1][2]
        east=tree.bc[1][3]
        front=tree.bc[1][4]
        back=tree.bc[1][5]

    if boundary_conditions == "periodic":
        return scalar

    elif boundary_conditions == "dirichlet":

        if tree.dimension == 2:

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
                    scalar.setValue(row, 2*west*dy/(dx*(dx*dy)), True)

                if j == 0:
                    scalar.setValue(row, 2*south*dx/(dy*(dx*dy)), True)

                #right flux
                if i == 2**level-1:
                    scalar.setValue(row, 2*east*dy/(dx*(dx*dy)), True)

                if j == 2**level-1:
                    scalar.setValue(row, 2*north*dx/(dy*(dx*dy)), True)

            return scalar

        elif tree.dimension == 3:

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
                    scalar.setValue(row, 2*west*dy*dz/(dx*(dx*dy*dz)), True)

                if j == 0:
                    scalar.setValue(row, 2*south*dx*dz/(dy*(dx*dy*dz)), True)

                if k == 0:
                    scalar.setValue(row, 2*back*dx*dy/(dz*(dx*dy*dz)), True)

                #right flux
                if i == 2**level-1:
                    scalar.setValue(row, 2*east*dy*dz/(dx*(dx*dy*dz)), True)

                if j == 2**level-1:
                    scalar.setValue(row, 2*north*dx*dz/(dy*(dx*dy*dz)), True)

                if k == 0:
                    scalar.setValue(row, 2*front*dx*dy/(dz*(dx*dy*dz)), True)

            return scalar

    elif boundary_conditions == "neumann":

        if tree.dimension == 2:

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
                    scalar.setValue(row, west*dy/(dx*dy), True)

                if j == 0:
                    scalar.setValue(row, south*dx/(dx*dy), True)

                #right flux
                if i == 2**level-1:
                    scalar.setValue(row, east*dy/(dx*dy), True)

                if j == 2**level-1:
                    scalar.setValue(row, north*dx/(dx*dy), True)

            return scalar

        elif tree.dimension == 3:

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
                    scalar.setValue(row, west*dy*dz/(dx*dy*dz), True)

                if j == 0:
                    scalar.setValue(row, south*dx*dz/(dx*dy*dz), True)

                if k == 0:
                    scalar.setValue(row, back*dx*dy/(dx*dy*dz), True)

                #right flux
                if i == 2**level-1:
                    scalar.setValue(row, east*dy*dz/(dx*dy*dz), True)

                if j == 2**level-1:
                    scalar.setValue(row, north*dx*dz/(dx*dy*dz), True)

                if k == 2**level-1:
                    scalar.setValue(row, front*dx*dy/(dx*dy*dz), True)

            return scalar

def create_stokes_part_matrix(tree_x=None, tree_y=None, tree_z=None):

    matrix = petsc.Mat().create()

    if cfg.dimension == 2:
        number_of_rows = tree_x.number_of_leaves
        boundary_conditions_x = cfg.bc_dict[tree_x.tag][0]
        boundary_conditions_y = cfg.bc_dict[tree_y.tag][0]
        size_row = (cfg.dimension*number_of_rows, cfg.dimension*number_of_rows)
        size_col = (cfg.dimension*number_of_rows, cfg.dimension*number_of_rows)
        matrix.setSizes((size_row, size_col))
        matrix.setUp()
#    matrix = np.zeros(shape=(number_of_rows, number_of_rows), dtype=np.float)

        # x-component
        for row in range(number_of_rows):
            index = tree_x.tree_leaves[row]
            i = tree_x.nindex_x[index]
            j = tree_x.nindex_y[index]
            k = tree_x.nindex_z[index]
            level = tree_x.nlevel[index]
            dx = tree_x.ndx[index]
            dy = tree_x.ndy[index]
            dz = tree_x.ndz[index]

            # left flux for axis 0
            if mesh.bc_compatible_local_indexes(tree_x, level, i-1, j, k) is not None:
                i_left, j_left, k_left = mesh.bc_compatible_local_indexes(tree_x, level, i-1, j, k)
                index_left = mesh.z_curve_index(tree_x.dimension, level, i_left, j_left, k_left)

                if index_left in tree_x.tree_nodes and tree_x.nisleaf[index_left] \
                    or index_left not in tree_x.tree_nodes:
                # the finest level for the left flux is the node's level
                    matrix_add(tree_x, matrix, row, -(dy)/(dx*(dx*dy)), level, i, j, k)
                    matrix_add(tree_x, matrix, row, (dy)/(dx*(dx*dy)), level, i_left, j_left, k_left)

                else:
                    for n in range(2):
                        matrix_add(tree_x, matrix, row, -(dy/2)/((dx/2)*(dx*dy)), level+1, 2*i, 2*j+n, 2*k)
                        matrix_add(tree_x, matrix, row, (dy/2)/((dx/2)*(dx*dy)), level+1, 2*i_left+1, 2*j+n, 2*k)

            elif boundary_conditions_x == "dirichlet":
                # the finest level for the left flux is the node's level; this node
                # receives a second contribution because of the boundary condition
                matrix_add(tree_x, matrix, row, -2*(dy)/(dx*(dx*dy)), level, i, j, k)

            elif boundary_conditions_x == "neumann":
            #the left flux depends only on the boundary condition scalar

                pass

            # right flux for axis 0
            if mesh.bc_compatible_local_indexes(tree_x, level, i+1, j, k) is not None:
                i_right, j_right, k_right = mesh.bc_compatible_local_indexes(tree_x, level, i+1, j, k)
                index_right = mesh.z_curve_index(tree_x.dimension, level, i_right, j_right, k_right)

                if index_right in tree_x.tree_nodes and tree_x.nisleaf[index_right] \
                    or index_right not in tree_x.tree_nodes:
                # the finest level for the right flux is the node's level
                    matrix_add(tree_x, matrix, row, -dy/(dx*(dx*dy)), level, i, j, k)
                    matrix_add(tree_x, matrix, row, dy/(dx*(dx*dy)), level, i_right, j_right, k_right)

                else:
                    for n in range(2):
                        matrix_add(tree_x, matrix, row, -(dy/2)/((dx/2)*(dx*dy)), level+1, 2*i+1, 2*j+n, 2*k)
                        matrix_add(tree_x, matrix, row, (dy/2)/((dx/2)*(dx*dy)), level+1, 2*i_right, 2*j+n, 2*k)

            elif boundary_conditions_x == "dirichlet":
                # the finest level for the right flux is the node's level; this node
                # receives a second contribution because of the boundary condition
                matrix_add(tree_x, matrix, row, -2*(dy)/(dx*(dx*dy)), level, i, j, k)

            elif boundary_conditions_x == "neumann":
            #the left flux depends only on the boundary condition scalar

                pass

            # left flux for axis 1
            if mesh.bc_compatible_local_indexes(tree_x, level, i, j-1, k) is not None:
                i_left, j_left, k_left = mesh.bc_compatible_local_indexes(tree_x, level, i, j-1, k)
                index_left = mesh.z_curve_index(tree_x.dimension, level, i_left, j_left, k_left)

                if index_left in tree_x.tree_nodes and tree_x.nisleaf[index_left] \
                    or index_left not in tree_x.tree_nodes:
                # the finest level for the left flux is the node's level
                    matrix_add(tree_x, matrix, row, -(dx)/(dy*(dx*dy)), level, i, j, k)
                    matrix_add(tree_x, matrix, row, (dx)/(dy*(dx*dy)), level, i_left, j_left, k_left)

                else:
                    for m in range(2):
                        matrix_add(tree_x, matrix, row, -(dx/2)/((dy/2)*(dx*dy)), level+1, 2*i+m, 2*j, 2*k)
                        matrix_add(tree_x, matrix, row, (dx/2)/((dy/2)*(dx*dy)), level+1, 2*i+m, 2*j_left+1, 2*k)

            elif boundary_conditions_x == "dirichlet":
                # the finest level for the left flux is the node's level; this node
                # receives a second contribution because of the boundary condition
                matrix_add(tree_x, matrix, row, -2*(dx)/(dy*(dx*dy)), level, i, j, k)

            elif boundary_conditions_x == "neumann":
            #the left flux depends only on the boundary condition scalar

                pass

            # right flux for axis 1
            if mesh.bc_compatible_local_indexes(tree_x, level, i, j+1, k) is not None:
                i_right, j_right, k_right = mesh.bc_compatible_local_indexes(tree_x, level, i, j+1, k)
                index_right = mesh.z_curve_index(tree_x.dimension, level, i_right, j_right, k_right)

                if index_right in tree_x.tree_nodes and tree_x.nisleaf[index_right] \
                    or index_right not in tree_x.tree_nodes:
                # the finest level for the right flux is the node's level
                    matrix_add(tree_x, matrix, row, -dx/(dy*(dx*dy)), level, i, j, k)
                    matrix_add(tree_x, matrix, row, dx/(dy*(dx*dy)), level, i_right, j_right, k_right)

                else:
                    for m in range(2):
                        matrix_add(tree_x, matrix, row, -(dx/2)/((dy/2)*(dx*dy)), level+1, 2*i+m, 2*j+1, 2*k)
                        matrix_add(tree_x, matrix, row, (dx/2)/((dy/2)*(dx*dy)), level+1, 2*i+m, 2*j_right, 2*k)

            elif boundary_conditions_x == "dirichlet":
                # the finest level for the right flux is the node's level; this node
                # receives a second contribution because of the boundary condition
                matrix_add(tree_x, matrix, row, -2*(dx)/(dy*(dx*dy)), level, i, j, k)

            elif boundary_conditions_x == "neumann":
            #the left flux depends only on the boundary condition scalar

                pass

        # y-component
        for row in range(number_of_rows):
            index = tree_y.tree_leaves[row]
            i = tree_y.nindex_x[index]
            j = tree_y.nindex_y[index]
            k = tree_y.nindex_z[index]
            level = tree_y.nlevel[index]
            dx = tree_y.ndx[index]
            dy = tree_y.ndy[index]
            dz = tree_y.ndz[index]

            row_y = row + number_of_rows

            # left flux for axis 0
            if mesh.bc_compatible_local_indexes(tree_y, level, i-1, j, k) is not None:
                i_left, j_left, k_left = mesh.bc_compatible_local_indexes(tree_y, level, i-1, j, k)
                index_left = mesh.z_curve_index(tree_y.dimension, level, i_left, j_left, k_left)

                if index_left in tree_y.tree_nodes and tree_y.nisleaf[index_left] \
                    or index_left not in tree_y.tree_nodes:
                # the finest level for the left flux is the node's level
                    matrix_add(tree_y, matrix, row_y, -(dy)/(dx*(dx*dy)), level, i, j, k, number_of_rows)
                    matrix_add(tree_y, matrix, row_y, (dy)/(dx*(dx*dy)), level, i_left, j_left, k_left, number_of_rows)

                else:
                    for n in range(2):
                        matrix_add(tree_y, matrix, row_y, -(dy/2)/((dx/2)*(dx*dy)), level+1, 2*i, 2*j+n, 2*k, number_of_rows)
                        matrix_add(tree_y, matrix, row_y, (dy/2)/((dx/2)*(dx*dy)), level+1, 2*i_left+1, 2*j+n, 2*k, number_of_rows)

            elif boundary_conditions_y == "dirichlet":
                # the finest level for the left flux is the node's level; this node
                # receives a second contribution because of the boundary condition
                matrix_add(tree_y, matrix, row_y, -2*(dy)/(dx*(dx*dy)), level, i, j, k, number_of_rows)

            elif boundary_conditions_y == "neumann":
            #the left flux depends only on the boundary condition scalar

                pass

            # right flux for axis 0
            if mesh.bc_compatible_local_indexes(tree_y, level, i+1, j, k) is not None:
                i_right, j_right, k_right = mesh.bc_compatible_local_indexes(tree_y, level, i+1, j, k)
                index_right = mesh.z_curve_index(tree_y.dimension, level, i_right, j_right, k_right)

                if index_right in tree_y.tree_nodes and tree_y.nisleaf[index_right] \
                    or index_right not in tree_y.tree_nodes:
                # the finest level for the right flux is the node's level
                    matrix_add(tree_y, matrix, row_y, -dy/(dx*(dx*dy)), level, i, j, k, number_of_rows)
                    matrix_add(tree_y, matrix, row_y, dy/(dx*(dx*dy)), level, i_right, j_right, k_right, number_of_rows)

                else:
                    for n in range(2):
                        matrix_add(tree_y, matrix, row_y, -(dy/2)/((dx/2)*(dx*dy)), level+1, 2*i+1, 2*j+n, 2*k, number_of_rows)
                        matrix_add(tree_y, matrix, row_y, (dy/2)/((dx/2)*(dx*dy)), level+1, 2*i_right, 2*j+n, 2*k, number_of_rows)

            elif boundary_conditions_y == "dirichlet":
                # the finest level for the right flux is the node's level; this node
                # receives a second contribution because of the boundary condition
                matrix_add(tree_y, matrix, row_y, -2*(dy)/(dx*(dx*dy)), level, i, j, k, number_of_rows)

            elif boundary_conditions_y == "neumann":
            #the left flux depends only on the boundary condition scalar

                pass

            # left flux for axis 1
            if mesh.bc_compatible_local_indexes(tree_y, level, i, j-1, k) is not None:
                i_left, j_left, k_left = mesh.bc_compatible_local_indexes(tree_y, level, i, j-1, k)
                index_left = mesh.z_curve_index(tree_y.dimension, level, i_left, j_left, k_left)

                if index_left in tree_y.tree_nodes and tree_y.nisleaf[index_left] \
                    or index_left not in tree_y.tree_nodes:
                # the finest level for the left flux is the node's level
                    matrix_add(tree_y, matrix, row_y, -(dx)/(dy*(dx*dy)), level, i, j, k, number_of_rows)
                    matrix_add(tree_y, matrix, row_y, (dx)/(dy*(dx*dy)), level, i_left, j_left, k_left, number_of_rows)

                else:
                    for m in range(2):
                        matrix_add(tree_y, matrix, row_y, -(dx/2)/((dy/2)*(dx*dy)), level+1, 2*i+m, 2*j, 2*k, number_of_rows)
                        matrix_add(tree_y, matrix, row_y, (dx/2)/((dy/2)*(dx*dy)), level+1, 2*i+m, 2*j_left+1, 2*k, number_of_rows)

            elif boundary_conditions_y == "dirichlet":
                # the finest level for the left flux is the node's level; this node
                # receives a second contribution because of the boundary condition
                matrix_add(tree_y, matrix, row_y, -2*(dx)/(dy*(dx*dy)), level, i, j, k, number_of_rows)

            elif boundary_conditions_y == "neumann":
            #the left flux depends only on the boundary condition scalar

                pass

            # right flux for axis 1
            if mesh.bc_compatible_local_indexes(tree_y, level, i, j+1, k) is not None:
                i_right, j_right, k_right = mesh.bc_compatible_local_indexes(tree_y, level, i, j+1, k)
                index_right = mesh.z_curve_index(tree_y.dimension, level, i_right, j_right, k_right)

                if index_right in tree_y.tree_nodes and tree_y.nisleaf[index_right] \
                    or index_right not in tree_y.tree_nodes:
                # the finest level for the right flux is the node's level
                    matrix_add(tree_y, matrix, row_y, -dx/(dy*(dx*dy)), level, i, j, k, number_of_rows)
                    matrix_add(tree_y, matrix, row_y, dx/(dy*(dx*dy)), level, i_right, j_right, k_right, number_of_rows)

                else:
                    for m in range(2):
                        matrix_add(tree_y, matrix, row_y, -(dx/2)/((dy/2)*(dx*dy)), level+1, 2*i+m, 2*j+1, 2*k, number_of_rows)
                        matrix_add(tree_y, matrix, row_y, (dx/2)/((dy/2)*(dx*dy)), level+1, 2*i+m, 2*j_right, 2*k, number_of_rows)

            elif boundary_conditions_y == "dirichlet":
                # the finest level for the right flux is the node's level; this node
                # receives a second contribution because of the boundary condition
                matrix_add(tree_y, matrix, row_y, -2*(dx)/(dy*(dx*dy)), level, i, j, k, number_of_rows)

            elif boundary_conditions_y == "neumann":
            #the left flux depends only on the boundary condition scalar

                pass

        matrix.assemble()

        return matrix

    if cfg.dimension == 3:
        number_of_rows = tree_x.number_of_leaves
        boundary_conditions_x = cfg.bc_dict[tree_x.tag][0]
        boundary_conditions_y = cfg.bc_dict[tree_y.tag][0]
        boundary_conditions_z = cfg.bc_dict[tree_z.tag][0]
        size_row = (cfg.dimension*number_of_rows, cfg.dimension*number_of_rows)
        size_col = (cfg.dimension*number_of_rows, cfg.dimension*number_of_rows)
        matrix.setSizes((size_row, size_col))
        matrix.setUp()
#    matrix = np.zeros(shape=(number_of_rows, number_of_rows), dtype=np.float)

        # x-component
        for row in range(number_of_rows):
            index = tree_x.tree_leaves[row]
            i = tree_x.nindex_x[index]
            j = tree_x.nindex_y[index]
            k = tree_x.nindex_z[index]
            level = tree_x.nlevel[index]
            dx = tree_x.ndx[index]
            dy = tree_x.ndy[index]
            dz = tree_x.ndz[index]

            # left flux for axis 0
            if mesh.bc_compatible_local_indexes(tree_x, level, i-1, j, k) is not None:
                i_left, j_left, k_left = mesh.bc_compatible_local_indexes(tree_x, level, i-1, j, k)
                index_left = mesh.z_curve_index(tree_x.dimension, level, i_left, j_left, k_left)

                if index_left in tree_x.tree_nodes and tree_x.nisleaf[index_left] \
                    or index_left not in tree_x.tree_nodes:
                # the finest level for the left flux is the node's level
                    matrix_add(tree_x, matrix, row, -(dy*dz)/(dx*(dx*dy*dz)), level, i, j, k)
                    matrix_add(tree_x, matrix, row, (dy*dz)/(dx*(dx*dy*dz)), level, i_left, j_left, k_left)

                else:
                    for o in range(2):
                        for n in range(2):
                            matrix_add(tree_x, matrix, row, -((dy/2)*(dz/2))/((dx/2)*(dx*dy*dz)), level+1, 2*i, 2*j+n, 2*k+o)
                            matrix_add(tree_x, matrix, row, ((dy/2)*(dz/2))/((dx/2)*(dx*dy*dz)), level+1, 2*i_left+1, 2*j+n, 2*k+o)

            elif boundary_conditions == "dirichlet":
                # the finest level for the left flux is the node's level; this node
                # receives a second contribution because of the boundary condition
                matrix_add(tree_x, matrix, row, -2*(dy*dz)/(dx*(dx*dy*dz)), level, i, j, k)

            elif boundary_conditions == "neumann":
            #the left flux depends only on the boundary condition scalar

                pass

            # right flux for axis 0
            if mesh.bc_compatible_local_indexes(tree_x, level, i+1, j, k) is not None:
                i_right, j_right, k_right = mesh.bc_compatible_local_indexes(tree_x, level, i+1, j, k)
                index_right = mesh.z_curve_index(tree_x.dimension, level, i_right, j_right, k_right)

                if index_right in tree_x.tree_nodes and tree_x.nisleaf[index_right] \
                    or index_right not in tree_x.tree_nodes:
                # the finest level for the right flux is the node's level
                    matrix_add(tree_x, matrix, row, -dy*dz/(dx*(dx*dy*dz)), level, i, j, k)
                    matrix_add(tree_x, matrix, row, dy*dz/(dx*(dx*dy*dz)), level, i_right, j_right, k_right)

                else:
                    for o in range(2):
                        for n in range(2):
                            matrix_add(tree_x, matrix, row, -(dy/2)*(dz/2)/((dx/2)*(dx*dy*dz)), level+1, 2*i+1, 2*j+n, 2*k+o)
                            matrix_add(tree_x, matrix, row, (dy/2)*(dz/2)/((dx/2)*(dx*dy*dz)), level+1, 2*i_right, 2*j+n, 2*k+o)

            elif boundary_conditions == "dirichlet":
                # the finest level for the right flux is the node's level; this node
                # receives a second contribution because of the boundary condition
                matrix_add(tree_x, matrix, row, -2*(dy*dz)/(dx*(dx*dy*dz)), level, i, j, k)

            elif boundary_conditions == "neumann":
            #the left flux depends only on the boundary condition scalar

                pass

            # left flux for axis 1
            if mesh.bc_compatible_local_indexes(tree_x, level, i, j-1, k) is not None:
                i_left, j_left, k_left = mesh.bc_compatible_local_indexes(tree_x, level, i, j-1, k)
                index_left = mesh.z_curve_index(tree_x.dimension, level, i_left, j_left, k_left)

                if index_left in tree_x.tree_nodes and tree_x.nisleaf[index_left] \
                    or index_left not in tree_x.tree_nodes:
                # the finest level for the left flux is the node's level
                    matrix_add(tree_x, matrix, row, -(dx*dz)/(dy*(dx*dy*dz)), level, i, j, k)
                    matrix_add(tree_x, matrix, row, (dx*dz)/(dy*(dx*dy*dz)), level, i_left, j_left, k_left)

                else:
                    for o in range(2):
                        for m in range(2):
                            matrix_add(tree_x, matrix, row, -(dx/2)*(dz/2)/((dy/2)*(dx*dy*dz)), level+1, 2*i+m, 2*j, 2*k+o)
                            matrix_add(tree_x, matrix, row, (dx/2)*(dz/2)/((dy/2)*(dx*dy*dz)), level+1, 2*i+m, 2*j_left+1, 2*k+o)

            elif boundary_conditions == "dirichlet":
                # the finest level for the left flux is the node's level; this node
                # receives a second contribution because of the boundary condition
                matrix_add(tree_x, matrix, row, -2*(dx*dz)/(dy*(dx*dy*dz)), level, i, j, k)

            elif boundary_conditions == "neumann":
            #the left flux depends only on the boundary condition scalar

                pass

            # right flux for axis 1
            if mesh.bc_compatible_local_indexes(tree_x, level, i, j+1, k) is not None:
                i_right, j_right, k_right = mesh.bc_compatible_local_indexes(tree_x, level, i, j+1, k)
                index_right = mesh.z_curve_index(tree_x.dimension, level, i_right, j_right, k_right)

                if index_right in tree_x.tree_nodes and tree_x.nisleaf[index_right] \
                    or index_right not in tree_x.tree_nodes:
                # the finest level for the right flux is the node's level
                    matrix_add(tree_x, matrix, row, -dx*dz/(dy*(dx*dy*dz)), level, i, j, k)
                    matrix_add(tree_x, matrix, row, dx*dz/(dy*(dx*dy*dz)), level, i_right, j_right, k_right)

                else:
                    for o in range(2):
                        for m in range(2):
                            matrix_add(tree_x, matrix, row, -(dx/2)*(dz/2)/((dy/2)*(dx*dy*dz)), level+1, 2*i+m, 2*j, 2*k+o)
                            matrix_add(tree_x, matrix, row, (dx/2)*(dz/2)/((dy/2)*(dx*dy*dz)), level+1, 2*i+m, 2*j_left+1, 2*k+o)

            elif boundary_conditions == "dirichlet":
                # the finest level for the right flux is the node's level; this node
                # receives a second contribution because of the boundary condition
                matrix_add(tree_x, matrix, row, -2*(dx*dz)/(dy*(dx*dy*dz)), level, i, j, k)

            elif boundary_conditions == "neumann":
            #the left flux depends only on the boundary condition scalar

                pass

            # left flux for axis 2
            if mesh.bc_compatible_local_indexes(tree_x, level, i, j, k-1) is not None:
                i_left, j_left, k_left = mesh.bc_compatible_local_indexes(tree_x, level, i, j, k-1)
                index_left = mesh.z_curve_index(tree_x.dimension, level, i_left, j_left, k_left)

                if index_left in tree_x.tree_nodes and tree_x.nisleaf[index_left] \
                    or index_left not in tree_x.tree_nodes:
                # the finest level for the left flux is the node's level
                    matrix_add(tree_x, matrix, row, -(dx*dy)/(dz*(dx*dy*dz)), level, i, j, k)
                    matrix_add(tree_x, matrix, row, (dx*dy)/(dz*(dx*dy*dz)), level, i_left, j_left, k_left)

                else:
                    for n in range(2):
                        for m in range(2):
                            matrix_add(tree_x, matrix, row, -(dx/2)*(dy/2)/((dz/2)*(dx*dy*dz)), level+1, 2*i+m, 2*j+n, 2*k)
                            matrix_add(tree_x, matrix, row, (dx/2)*(dy/2)/((dz/2)*(dx*dy*dz)), level+1, 2*i+m, 2*j+n, 2*k_left+1)

            elif boundary_conditions == "dirichlet":
                # the finest level for the left flux is the node's level; this node
                # receives a second contribution because of the boundary condition
                matrix_add(tree_x, matrix, row, -2*(dx*dy)/(dz*(dx*dy*dz)), level, i, j, k)

            elif boundary_conditions == "neumann":
            #the left flux depends only on the boundary condition scalar

                pass

            # right flux for axis 2
            if mesh.bc_compatible_local_indexes(tree_x, level, i, j, k+1) is not None:
                i_right, j_right, k_right = mesh.bc_compatible_local_indexes(tree_x, level, i, j, k+1)
                index_right = mesh.z_curve_index(tree_x.dimension, level, i_right, j_right, k_right)

                if index_right in tree_x.tree_nodes and tree_x.nisleaf[index_right] \
                    or index_right not in tree_x.tree_nodes:
                # the finest level for the right flux is the node's level
                    matrix_add(tree_x, matrix, row, -(dx*dy)/(dz*(dx*dy*dz)), level, i, j, k)
                    matrix_add(tree_x, matrix, row, (dx*dy)/(dz*(dx*dy*dz)), level, i_right, j_right, k_right)

                else:
                    for n in range(2):
                        for m in range(2):
                            matrix_add(tree_x, matrix, row, -(dx/2)*(dy/2)/((dz/2)*(dx*dy*dz)), level+1, 2*i+m, 2*j+n, 2*k+1)
                            matrix_add(tree_x, matrix, row, (dx/2)*(dy/2)/((dz/2)*(dx*dy*dz)), level+1, 2*i+m, 2*j+n, 2*k_right)

            elif boundary_conditions == "dirichlet":
                # the finest level for the right flux is the node's level; this node
                # receives a second contribution because of the boundary condition
                matrix_add(tree_x, matrix, row, -2*(dx*dy)/(dz*(dx*dy*dz)), level, i, j, k)

            elif boundary_conditions == "neumann":
            #the left flux depends only on the boundary condition scalar

                pass

        # y-component
        for row in range(number_of_rows):
            index = tree_y.tree_leaves[row]
            i = tree_y.nindex_x[index]
            j = tree_y.nindex_y[index]
            k = tree_y.nindex_z[index]
            level = tree_y.nlevel[index]
            dx = tree_y.ndx[index]
            dy = tree_y.ndy[index]
            dz = tree_y.ndz[index]

            row_y = row + number_of_rows

            # left flux for axis 0
            if mesh.bc_compatible_local_indexes(tree_y, level, i-1, j, k) is not None:
                i_left, j_left, k_left = mesh.bc_compatible_local_indexes(tree_y, level, i-1, j, k)
                index_left = mesh.z_curve_index(tree_y.dimension, level, i_left, j_left, k_left)

                if index_left in tree_y.tree_nodes and tree_y.nisleaf[index_left] \
                    or index_left not in tree_y.tree_nodes:
                # the finest level for the left flux is the node's level
                    matrix_add(tree_y, matrix, row_y, -(dy*dz)/(dx*(dx*dy*dz)), level, i, j, k, number_of_rows)
                    matrix_add(tree_y, matrix, row_y, (dy*dz)/(dx*(dx*dy*dz)), level, i_left, j_left, k_left, number_of_rows)

                else:
                    for o in range(2):
                        for n in range(2):
                            matrix_add(tree_y, matrix, row_y, -((dy/2)*(dz/2))/((dx/2)*(dx*dy*dz)), level+1, 2*i, 2*j+n, 2*k+o, number_of_rows)
                            matrix_add(tree_y, matrix, row_y, ((dy/2)*(dz/2))/((dx/2)*(dx*dy*dz)), level+1, 2*i_left+1, 2*j+n, 2*k+o, number_of_rows)

            elif boundary_conditions == "dirichlet":
                # the finest level for the left flux is the node's level; this node
                # receives a second contribution because of the boundary condition
                matrix_add(tree_y, matrix, row_y, -2*(dy*dz)/(dx*(dx*dy*dz)), level, i, j, k, number_of_rows)

            elif boundary_conditions == "neumann":
            #the left flux depends only on the boundary condition scalar

                pass

            # right flux for axis 0
            if mesh.bc_compatible_local_indexes(tree_y, level, i+1, j, k) is not None:
                i_right, j_right, k_right = mesh.bc_compatible_local_indexes(tree_y, level, i+1, j, k)
                index_right = mesh.z_curve_index(tree_y.dimension, level, i_right, j_right, k_right)

                if index_right in tree_y.tree_nodes and tree_y.nisleaf[index_right] \
                    or index_right not in tree_y.tree_nodes:
                # the finest level for the right flux is the node's level
                    matrix_add(tree_y, matrix, row_y, -dy*dz/(dx*(dx*dy*dz)), level, i, j, k, number_of_rows)
                    matrix_add(tree_y, matrix, row_y, dy*dz/(dx*(dx*dy*dz)), level, i_right, j_right, k_right, number_of_rows)

                else:
                    for o in range(2):
                        for n in range(2):
                            matrix_add(tree_y, matrix, row_y, -(dy/2)*(dz/2)/((dx/2)*(dx*dy*dz)), level+1, 2*i+1, 2*j+n, 2*k+o, number_of_rows)
                            matrix_add(tree_y, matrix, row_y, (dy/2)*(dz/2)/((dx/2)*(dx*dy*dz)), level+1, 2*i_right, 2*j+n, 2*k+o, number_of_rows)

            elif boundary_conditions == "dirichlet":
                # the finest level for the right flux is the node's level; this node
                # receives a second contribution because of the boundary condition
                matrix_add(tree_y, matrix, row_y, -2*(dy*dz)/(dx*(dx*dy*dz)), level, i, j, k, number_of_rows)

            elif boundary_conditions == "neumann":
            #the left flux depends only on the boundary condition scalar

                pass

            # left flux for axis 1
            if mesh.bc_compatible_local_indexes(tree_y, level, i, j-1, k) is not None:
                i_left, j_left, k_left = mesh.bc_compatible_local_indexes(tree_y, level, i, j-1, k)
                index_left = mesh.z_curve_index(tree_y.dimension, level, i_left, j_left, k_left)

                if index_left in tree_y.tree_nodes and tree_y.nisleaf[index_left] \
                    or index_left not in tree_y.tree_nodes:
                # the finest level for the left flux is the node's level
                    matrix_add(tree_y, matrix, row_y, -(dx*dz)/(dy*(dx*dy*dz)), level, i, j, k, number_of_rows)
                    matrix_add(tree_y, matrix, row_y, (dx*dz)/(dy*(dx*dy*dz)), level, i_left, j_left, k_left, number_of_rows)

                else:
                    for o in range(2):
                        for m in range(2):
                            matrix_add(tree_y, matrix, row_y, -(dx/2)*(dz/2)/((dy/2)*(dx*dy*dz)), level+1, 2*i+m, 2*j, 2*k+o, number_of_rows)
                            matrix_add(tree_y, matrix, row_y, (dx/2)*(dz/2)/((dy/2)*(dx*dy*dz)), level+1, 2*i+m, 2*j_left+1, 2*k+o, number_of_rows)

            elif boundary_conditions == "dirichlet":
                # the finest level for the left flux is the node's level; this node
                # receives a second contribution because of the boundary condition
                matrix_add(tree_y, matrix, row_y, -2*(dx*dz)/(dy*(dx*dy*dz)), level, i, j, k, number_of_rows)

            elif boundary_conditions == "neumann":
            #the left flux depends only on the boundary condition scalar

                pass

            # right flux for axis 1
            if mesh.bc_compatible_local_indexes(tree_y, level, i, j+1, k) is not None:
                i_right, j_right, k_right = mesh.bc_compatible_local_indexes(tree_y, level, i, j+1, k)
                index_right = mesh.z_curve_index(tree_y.dimension, level, i_right, j_right, k_right)

                if index_right in tree_y.tree_nodes and tree_y.nisleaf[index_right] \
                    or index_right not in tree_y.tree_nodes:
                # the finest level for the right flux is the node's level
                    matrix_add(tree_y, matrix, row_y, -dx*dz/(dy*(dx*dy*dz)), level, i, j, k, number_of_rows)
                    matrix_add(tree_y, matrix, row_y, dx*dz/(dy*(dx*dy*dz)), level, i_right, j_right, k_right, number_of_rows)

                else:
                    for o in range(2):
                        for m in range(2):
                            matrix_add(tree_y, matrix, row_y, -(dx/2)*(dz/2)/((dy/2)*(dx*dy*dz)), level+1, 2*i+m, 2*j, 2*k+o, number_of_rows)
                            matrix_add(tree_y, matrix, row_y, (dx/2)*(dz/2)/((dy/2)*(dx*dy*dz)), level+1, 2*i+m, 2*j_left+1, 2*k+o, number_of_rows)

            elif boundary_conditions == "dirichlet":
                # the finest level for the right flux is the node's level; this node
                # receives a second contribution because of the boundary condition
                matrix_add(tree_y, matrix, row_y, -2*(dx*dz)/(dy*(dx*dy*dz)), level, i, j, k, number_of_rows)

            elif boundary_conditions == "neumann":
            #the left flux depends only on the boundary condition scalar

                pass

            # left flux for axis 2
            if mesh.bc_compatible_local_indexes(tree_y, level, i, j, k-1) is not None:
                i_left, j_left, k_left = mesh.bc_compatible_local_indexes(tree_y, level, i, j, k-1)
                index_left = mesh.z_curve_index(tree_y.dimension, level, i_left, j_left, k_left)

                if index_left in tree_y.tree_nodes and tree_y.nisleaf[index_left] \
                    or index_left not in tree_y.tree_nodes:
                # the finest level for the left flux is the node's level
                    matrix_add(tree_y, matrix, row_y, -(dx*dy)/(dz*(dx*dy*dz)), level, i, j, k, number_of_rows)
                    matrix_add(tree_y, matrix, row_y, (dx*dy)/(dz*(dx*dy*dz)), level, i_left, j_left, k_left, number_of_rows)

                else:
                    for n in range(2):
                        for m in range(2):
                            matrix_add(tree_y, matrix, row_y, -(dx/2)*(dy/2)/((dz/2)*(dx*dy*dz)), level+1, 2*i+m, 2*j+n, 2*k, number_of_rows)
                            matrix_add(tree_y, matrix, row_y, (dx/2)*(dy/2)/((dz/2)*(dx*dy*dz)), level+1, 2*i+m, 2*j+n, 2*k_left+1, number_of_rows)

            elif boundary_conditions == "dirichlet":
                # the finest level for the left flux is the node's level; this node
                # receives a second contribution because of the boundary condition
                matrix_add(tree_y, matrix, row_y, -2*(dx*dy)/(dz*(dx*dy*dz)), level, i, j, k, number_of_rows)

            elif boundary_conditions == "neumann":
            #the left flux depends only on the boundary condition scalar

                pass

            # right flux for axis 2
            if mesh.bc_compatible_local_indexes(tree_y, level, i, j, k+1) is not None:
                i_right, j_right, k_right = mesh.bc_compatible_local_indexes(tree_y, level, i, j, k+1)
                index_right = mesh.z_curve_index(tree_y.dimension, level, i_right, j_right, k_right)

                if index_right in tree_y.tree_nodes and tree_y.nisleaf[index_right] \
                    or index_right not in tree_y.tree_nodes:
                # the finest level for the right flux is the node's level
                    matrix_add(tree_y, matrix, row_y, -(dx*dy)/(dz*(dx*dy*dz)), level, i, j, k, number_of_rows)
                    matrix_add(tree_y, matrix, row_y, (dx*dy)/(dz*(dx*dy*dz)), level, i_right, j_right, k_right, number_of_rows)

                else:
                    for n in range(2):
                        for m in range(2):
                            matrix_add(tree_y, matrix, row_y, -(dx/2)*(dy/2)/((dz/2)*(dx*dy*dz)), level+1, 2*i+m, 2*j+n, 2*k+1, number_of_rows)
                            matrix_add(tree_y, matrix, row_y, (dx/2)*(dy/2)/((dz/2)*(dx*dy*dz)), level+1, 2*i+m, 2*j+n, 2*k_right, number_of_rows)

            elif boundary_conditions == "dirichlet":
                # the finest level for the right flux is the node's level; this node
                # receives a second contribution because of the boundary condition
                matrix_add(tree_y, matrix, row_y, -2*(dx*dy)/(dz*(dx*dy*dz)), level, i, j, k, number_of_rows)

            elif boundary_conditions == "neumann":
            #the left flux depends only on the boundary condition scalar

                pass

        # z_component
        for row in range(number_of_rows):
            index = tree_z.tree_leaves[row]
            i = tree_z.nindex_x[index]
            j = tree_z.nindex_y[index]
            k = tree_z.nindex_z[index]
            level = tree_z.nlevel[index]
            dx = tree_z.ndx[index]
            dy = tree_z.ndy[index]
            dz = tree_z.ndz[index]

            row_z = row + 2*number_of_rows

            # left flux for axis 0
            if mesh.bc_compatible_local_indexes(tree_z, level, i-1, j, k) is not None:
                i_left, j_left, k_left = mesh.bc_compatible_local_indexes(tree_z, level, i-1, j, k)
                index_left = mesh.z_curve_index(tree_z.dimension, level, i_left, j_left, k_left)

                if index_left in tree_z.tree_nodes and tree_z.nisleaf[index_left] \
                    or index_left not in tree_z.tree_nodes:
                # the finest level for the left flux is the node's level
                    matrix_add(tree_z, matrix, row_z, -(dy*dz)/(dx*(dx*dy*dz)), level, i, j, k, 2*number_of_rows)
                    matrix_add(tree_z, matrix, row_z, (dy*dz)/(dx*(dx*dy*dz)), level, i_left, j_left, k_left, 2*number_of_rows)

                else:
                    for o in range(2):
                        for n in range(2):
                            matrix_add(tree_z, matrix, row_z, -((dy/2)*(dz/2))/((dx/2)*(dx*dy*dz)), level+1, 2*i, 2*j+n, 2*k+o, 2*number_of_rows)
                            matrix_add(tree_z, matrix, row_z, ((dy/2)*(dz/2))/((dx/2)*(dx*dy*dz)), level+1, 2*i_left+1, 2*j+n, 2*k+o, 2*number_of_rows)

            elif boundary_conditions == "dirichlet":
                # the finest level for the left flux is the node's level; this node
                # receives a second contribution because of the boundary condition
                matrix_add(tree_z, matrix, row_z, -2*(dy*dz)/(dx*(dx*dy*dz)), level, i, j, k, 2*number_of_rows)

            elif boundary_conditions == "neumann":
            #the left flux depends only on the boundary condition scalar

                pass

            # right flux for axis 0
            if mesh.bc_compatible_local_indexes(tree_z, level, i+1, j, k) is not None:
                i_right, j_right, k_right = mesh.bc_compatible_local_indexes(tree_z, level, i+1, j, k)
                index_right = mesh.z_curve_index(tree_z.dimension, level, i_right, j_right, k_right)

                if index_right in tree_z.tree_nodes and tree_z.nisleaf[index_right] \
                    or index_right not in tree_z.tree_nodes:
                # the finest level for the right flux is the node's level
                    matrix_add(tree_z, matrix, row_z, -dy*dz/(dx*(dx*dy*dz)), level, i, j, k, 2*number_of_rows)
                    matrix_add(tree_z, matrix, row_z, dy*dz/(dx*(dx*dy*dz)), level, i_right, j_right, k_right, 2*number_of_rows)

                else:
                    for o in range(2):
                        for n in range(2):
                            matrix_add(tree_z, matrix, row_z, -(dy/2)*(dz/2)/((dx/2)*(dx*dy*dz)), level+1, 2*i+1, 2*j+n, 2*k+o, 2*number_of_rows)
                            matrix_add(tree_z, matrix, row_z, (dy/2)*(dz/2)/((dx/2)*(dx*dy*dz)), level+1, 2*i_right, 2*j+n, 2*k+o, 2*number_of_rows)

            elif boundary_conditions == "dirichlet":
                # the finest level for the right flux is the node's level; this node
                # receives a second contribution because of the boundary condition
                matrix_add(tree_z, matrix, row_z, -2*(dy*dz)/(dx*(dx*dy*dz)), level, i, j, k, 2*number_of_rows)

            elif boundary_conditions == "neumann":
            #the left flux depends only on the boundary condition scalar

                pass

            # left flux for axis 1
            if mesh.bc_compatible_local_indexes(tree_z, level, i, j-1, k) is not None:
                i_left, j_left, k_left = mesh.bc_compatible_local_indexes(tree_z, level, i, j-1, k)
                index_left = mesh.z_curve_index(tree_z.dimension, level, i_left, j_left, k_left)

                if index_left in tree_z.tree_nodes and tree_z.nisleaf[index_left] \
                    or index_left not in tree_z.tree_nodes:
                # the finest level for the left flux is the node's level
                    matrix_add(tree_z, matrix, row_z, -(dx*dz)/(dy*(dx*dy*dz)), level, i, j, k, 2*number_of_rows)
                    matrix_add(tree_z, matrix, row_z, (dx*dz)/(dy*(dx*dy*dz)), level, i_left, j_left, k_left, 2*number_of_rows)

                else:
                    for o in range(2):
                        for m in range(2):
                            matrix_add(tree_z, matrix, row_z, -(dx/2)*(dz/2)/((dy/2)*(dx*dy*dz)), level+1, 2*i+m, 2*j, 2*k+o, 2*number_of_rows)
                            matrix_add(tree_z, matrix, row_z, (dx/2)*(dz/2)/((dy/2)*(dx*dy*dz)), level+1, 2*i+m, 2*j_left+1, 2*k+o, 2*number_of_rows)

            elif boundary_conditions == "dirichlet":
                # the finest level for the left flux is the node's level; this node
                # receives a second contribution because of the boundary condition
                matrix_add(tree_z, matrix, row_z, -2*(dx*dz)/(dy*(dx*dy*dz)), level, i, j, k, 2*number_of_rows)

            elif boundary_conditions == "neumann":
            #the left flux depends only on the boundary condition scalar

                pass

            # right flux for axis 1
            if mesh.bc_compatible_local_indexes(tree_z, level, i, j+1, k) is not None:
                i_right, j_right, k_right = mesh.bc_compatible_local_indexes(tree_z, level, i, j+1, k)
                index_right = mesh.z_curve_index(tree_z.dimension, level, i_right, j_right, k_right)

                if index_right in tree_z.tree_nodes and tree_z.nisleaf[index_right] \
                    or index_right not in tree_z.tree_nodes:
                # the finest level for the right flux is the node's level
                    matrix_add(tree_z, matrix, row_z, -dx*dz/(dy*(dx*dy*dz)), level, i, j, k, 2*number_of_rows)
                    matrix_add(tree_z, matrix, row_z, dx*dz/(dy*(dx*dy*dz)), level, i_right, j_right, k_right, 2*number_of_rows)

                else:
                    for o in range(2):
                        for m in range(2):
                            matrix_add(tree_z, matrix, row_z, -(dx/2)*(dz/2)/((dy/2)*(dx*dy*dz)), level+1, 2*i+m, 2*j, 2*k+o, 2*number_of_rows)
                            matrix_add(tree_z, matrix, row_z, (dx/2)*(dz/2)/((dy/2)*(dx*dy*dz)), level+1, 2*i+m, 2*j_left+1, 2*k+o, 2*number_of_rows)

            elif boundary_conditions == "dirichlet":
                # the finest level for the right flux is the node's level; this node
                # receives a second contribution because of the boundary condition
                matrix_add(tree_z, matrix, row_z, -2*(dx*dz)/(dy*(dx*dy*dz)), level, i, j, k, 2*number_of_rows)

            elif boundary_conditions == "neumann":
            #the left flux depends only on the boundary condition scalar

                pass

            # left flux for axis 2
            if mesh.bc_compatible_local_indexes(tree_z, level, i, j, k-1) is not None:
                i_left, j_left, k_left = mesh.bc_compatible_local_indexes(tree_z, level, i, j, k-1)
                index_left = mesh.z_curve_index(tree_z.dimension, level, i_left, j_left, k_left)

                if index_left in tree_z.tree_nodes and tree_z.nisleaf[index_left] \
                    or index_left not in tree_z.tree_nodes:
                # the finest level for the left flux is the node's level
                    matrix_add(tree_z, matrix, row_z, -(dx*dy)/(dz*(dx*dy*dz)), level, i, j, k, 2*number_of_rows)
                    matrix_add(tree_z, matrix, row_z, (dx*dy)/(dz*(dx*dy*dz)), level, i_left, j_left, k_left, 2*number_of_rows)

                else:
                    for n in range(2):
                        for m in range(2):
                            matrix_add(tree_z, matrix, row_z, -(dx/2)*(dy/2)/((dz/2)*(dx*dy*dz)), level+1, 2*i+m, 2*j+n, 2*k, 2*number_of_rows)
                            matrix_add(tree_z, matrix, row_z, (dx/2)*(dy/2)/((dz/2)*(dx*dy*dz)), level+1, 2*i+m, 2*j+n, 2*k_left+1, 2*number_of_rows)

            elif boundary_conditions == "dirichlet":
                # the finest level for the left flux is the node's level; this node
                # receives a second contribution because of the boundary condition
                matrix_add(tree_z, matrix, row_z, -2*(dx*dy)/(dz*(dx*dy*dz)), level, i, j, k, 2*number_of_rows)

            elif boundary_conditions == "neumann":
            #the left flux depends only on the boundary condition scalar

                pass

            # right flux for axis 2
            if mesh.bc_compatible_local_indexes(tree_z, level, i, j, k+1) is not None:
                i_right, j_right, k_right = mesh.bc_compatible_local_indexes(tree_z, level, i, j, k+1)
                index_right = mesh.z_curve_index(tree_z.dimension, level, i_right, j_right, k_right)

                if index_right in tree_z.tree_nodes and tree_z.nisleaf[index_right] \
                    or index_right not in tree_z.tree_nodes:
                # the finest level for the right flux is the node's level
                    matrix_add(tree_z, matrix, row_z, -(dx*dy)/(dz*(dx*dy*dz)), level, i, j, k, 2*number_of_rows)
                    matrix_add(tree_z, matrix, row_z, (dx*dy)/(dz*(dx*dy*dz)), level, i_right, j_right, k_right, 2*number_of_rows)

                else:
                    for n in range(2):
                        for m in range(2):
                            matrix_add(tree_z, matrix, row_z, -(dx/2)*(dy/2)/((dz/2)*(dx*dy*dz)), level+1, 2*i+m, 2*j+n, 2*k+1, 2*number_of_rows)
                            matrix_add(tree_z, matrix, row_z, (dx/2)*(dy/2)/((dz/2)*(dx*dy*dz)), level+1, 2*i+m, 2*j+n, 2*k_right, 2*number_of_rows)

            elif boundary_conditions == "dirichlet":
                # the finest level for the right flux is the node's level; this node
                # receives a second contribution because of the boundary condition
                matrix_add(tree_z, matrix, row_z, -2*(dx*dy)/(dz*(dx*dy*dz)), level, i, j, k, 2*number_of_rows)

            elif boundary_conditions == "neumann":
            #the left flux depends only on the boundary condition scalar

                pass

        matrix.assemble()

        return matrix


if __name__ == "__main__":

    output_module = importlib.import_module(cfg.output_module_name)

    tree = mesh.create_new_tree(cfg.dimension, cfg.min_level, cfg.max_level, cfg.stencil_graduation, cfg.stencil_prediction)

    tree.tag = "u"

    mesh.listing_of_leaves(tree)

    print(tree.number_of_leaves)
    print("")
    laplacian_matrix = create_matrix(tree, 0)
    laplacian_matrix.view()
    print("")
    for index in tree.tree_leaves:
        tree.nvalue[index] = cfg.function(tree.ncoord_x[index], tree.ncoord_y[index])

    output_module.write(tree, "test_finest_grid.dat")

    op.run_projection(tree)

    op.encode_details(tree)

    op.run_thresholding(tree)

    op.run_grading(tree)

    op.run_pruning(tree)

    mesh.listing_of_leaves(tree)

    print(tree.number_of_leaves)
    print("")

    output_module.write(tree, "test_adapted_grid.dat")

    laplacian_matrix = create_matrix(tree, 0)
    laplacian_matrix.view()
    print("")

    laplacian_bc = create_bc_scalar(tree, 0)
    laplacian_bc.view()
    print("")
