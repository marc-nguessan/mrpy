from __future__ import print_function, division

"""This module is used to compute the mass operator in the x-direction.

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
#from mrpy.mr_utils import mesh
#from mrpy.mr_utils import op
#import numpy as np
#import math
#import importlib

#!!!!!!! penser a rajouter un mr_bc_scalar !!!!!!!!
def create_matrix(tree, axis):

    matrix = petsc.Mat().create()
    number_of_rows = tree.number_of_leaves
    size_row = (number_of_rows, number_of_rows)
    size_col = (number_of_rows, number_of_rows)
    matrix.setSizes((size_row, size_col))
    matrix.setUp()
#    matrix = np.zeros(shape=(number_of_rows, number_of_rows), dtype=np.float)

    temp = petsc.Vec().create()
    temp.setSizes(size_row)
    temp.setUp()

    if tree.dimension == 2:

        for row in range(number_of_rows):
            index = tree.tree_leaves[row]
            level = tree.nlevel[index]
            i = tree.nindex_x[index]
            j = tree.nindex_y[index]
            dx = tree.ndx[index]
            dy = tree.ndy[index]

            temp.setValue(row, 1/(dx*dy), True)

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

            temp.setValue(row, 1/(dx*dy*dz), True)

    matrix.setDiagonal(temp)

    return matrix

def create_bc_scalar(tree, axis, north=None, south=None, east=None, west=None, forth=None, back=None):

    scalar = petsc.Vec().create()
    number_of_rows = tree.number_of_leaves
    scalar.setSizes(number_of_rows, number_of_rows)
    scalar.setUp()

    return scalar


if __name__ == "__main__":

    output_module = importlib.import_module(cfg.output_module_name)

    tree = mesh.create_new_tree(cfg.dimension, cfg.min_level, cfg.max_level, cfg.stencil_graduation, cfg.stencil_prediction)

    tree.tag = "u"

    mesh.listing_of_leaves(tree)

    print(tree.number_of_leaves)
    print("")
    mass_matrix = create_matrix(tree, 0)
    mass_matrix.view()
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

    mass_matrix = create_matrix(tree, 0)
    mass_matrix.view()
    print("")

    mass_bc = create_bc_scalar(tree, 0)
    mass_bc.view()
    print("")
