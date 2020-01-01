from __future__ import print_function, division

"""...

"""

import sys, petsc4py
petsc4py.init(sys.argv)
import petsc4py.PETSc as petsc
import mpi4py.MPI as mpi
import numpy as np
import scipy.sparse as sp
from six.moves import range
import importlib
import math

from mrpy.mr_utils import mesh
from mrpy.mr_utils import op
import mrpy.discretization.spatial as sd
import config as cfg


def update_bc_velocity_div_x(tree, tree_bc, direction=None):

    scalar = petsc.Vec().create()
    number_of_rows = tree.number_of_leaves
    scalar.setSizes(number_of_rows, number_of_rows)
    scalar.setUp()

    #if north is None and south is None and east is None and west is None and forth is None and back is None:
    #    north = tree.bc["north"][1]
    #    south = tree.bc["south"][1]
    west_func = tree.bc["west"][1]
    east_func = tree.bc["east"][1]
    #    forth = tree.bc["forth"][1]
    #    back = tree.bc["back"][1]

    #if axis == 0:

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
                if direction == "west":
                    west = tree_bc.nvalue[index]
                    if tree.dimension == 2:
                        #coords = mesh.boundary_coords(tree, "west", level, 0, j, k)
                        scalar.setValue(row, -west*dy, True)

                    elif tree.dimension == 3:
                        #coords = mesh.boundary_coords(tree, "west", level, 0, j, k)
                        scalar.setValue(row, -west*dy*dz, True)

                else:
                    if tree.dimension == 2:
                        coords = mesh.boundary_coords(tree, "west", level, 0, j, k)
                        scalar.setValue(row, -west_func(coords)*dy, True)

                    elif tree.dimension == 3:
                        coords = mesh.boundary_coords(tree, "west", level, 0, j, k)
                        scalar.setValue(row, -west_func(coords)*dy*dz, True)

            elif tree.bc["west"][0] == "neumann":
                if direction == "west":
                    west = tree_bc.nvalue[index]
                    if tree.dimension == 2:
                        #coords = mesh.boundary_coords(tree, "west", level, 0, j, k)
                        scalar.setValue(row, -(west*dx/2)*dy, True)

                    elif tree.dimension == 3:
                        #coords = mesh.boundary_coords(tree, "west", level, 0, j, k)
                        scalar.setValue(row, -(west*dx/2)*dy*dz, True)

                else:
                    if tree.dimension == 2:
                        coords = mesh.boundary_coords(tree, "west", level, 0, j, k)
                        scalar.setValue(row, -(west_func(coords)*dx/2)*dy, True)

                    elif tree.dimension == 3:
                        coords = mesh.boundary_coords(tree, "west", level, 0, j, k)
                        scalar.setValue(row, -(west_func(coords)*dx/2)*dy*dz, True)

        #right flux
        if i == 2**level-1:

            if tree.bc["east"][0] == "periodic":
                pass

            elif tree.bc["east"][0] == "dirichlet":
                if direction == "east":
                    east = tree_bc.nvalue[index]
                    if tree.dimension == 2:
                        #coords = mesh.boundary_coords(tree, "east", level, 0, j, k)
                        scalar.setValue(row, east*dy, True)

                    elif tree.dimension == 3:
                        #coords = mesh.boundary_coords(tree, "east", level, 0, j, k)
                        scalar.setValue(row, east*dy*dz, True)

                else:
                    if tree.dimension == 2:
                        coords = mesh.boundary_coords(tree, "east", level, 0, j, k)
                        scalar.setValue(row, east_func(coords)*dy, True)

                    elif tree.dimension == 3:
                        coords = mesh.boundary_coords(tree, "east", level, 0, j, k)
                        scalar.setValue(row, east_func(coords)*dy*dz, True)

            elif tree.bc["east"][0] == "neumann":
                if direction == "east":
                    east = tree_bc.nvalue[index]
                    if tree.dimension == 2:
                        #coords = mesh.boundary_coords(tree, "east", level, 0, j, k)
                        scalar.setValue(row, (east*dx/2)*dy, True)

                    elif tree.dimension == 3:
                        #coords = mesh.boundary_coords(tree, "east", level, 0, j, k)
                        scalar.setValue(row, (east*dx/2)*dy*dz, True)

                else:
                    if tree.dimension == 2:
                        coords = mesh.boundary_coords(tree, "east", level, 0, j, k)
                        scalar.setValue(row, (east_func(coords)*dx/2)*dy, True)

                    elif tree.dimension == 3:
                        coords = mesh.boundary_coords(tree, "east", level, 0, j, k)
                        scalar.setValue(row, (east_func(coords)*dx/2)*dy*dz, True)

    return scalar

def update_bc_velocity_div_y(tree, tree_bc, direction=None):

    scalar = petsc.Vec().create()
    number_of_rows = tree.number_of_leaves
    scalar.setSizes(number_of_rows, number_of_rows)
    scalar.setUp()

    north_func = tree.bc["north"][1]
    south_func = tree.bc["south"][1]

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
                if direction == "south":
                    south = tree_bc.nvalue[index]
                    if tree.dimension == 2:
                        #coords = mesh.boundary_coords(tree, "south", level, i, 0, k)
                        scalar.setValue(row, -south*dx, True)

                    elif tree.dimension == 3:
                        #coords = mesh.boundary_coords(tree, "south", level, i, 0, k)
                        scalar.setValue(row, -south*dx*dz, True)

                else:
                    if tree.dimension == 2:
                        coords = mesh.boundary_coords(tree, "south", level, i, 0, k)
                        scalar.setValue(row, -south_func(coords)*dx, True)

                    elif tree.dimension == 3:
                        coords = mesh.boundary_coords(tree, "south", level, i, 0, k)
                        scalar.setValue(row, -south_func(coords)*dx*dz, True)

            elif tree.bc["south"][0] == "neumann":
                if direction == "south":
                    south = tree_bc.nvalue[index]
                    if tree.dimension == 2:
                        #coords = mesh.boundary_coords(tree, "south", level, i, 0, k)
                        scalar.setValue(row, -(south*dy/2)*dx, True)

                    elif tree.dimension == 3:
                        #coords = mesh.boundary_coords(tree, "south", level, i, 0, k)
                        scalar.setValue(row, -(south*dy/2)*dx*dz, True)

                else:
                    if tree.dimension == 2:
                        coords = mesh.boundary_coords(tree, "south", level, i, 0, k)
                        scalar.setValue(row, -(south_func(coords)*dy/2)*dx, True)

                    elif tree.dimension == 3:
                        coords = mesh.boundary_coords(tree, "south", level, i, 0, k)
                        scalar.setValue(row, -(south_func(coords)*dy/2)*dx*dz, True)

        #right flux
        if j == 2**level-1:

            if tree.bc["north"][0] == "periodic":
                pass

            elif tree.bc["north"][0] == "dirichlet":
                if direction == "north":
                    north = tree_bc.nvalue[index]
                    if tree.dimension == 2:
                        #coords = mesh.boundary_coords(tree, "north", level, i, 0, k)
                        scalar.setValue(row, north*dx, True)

                    elif tree.dimension == 3:
                        #coords = mesh.boundary_coords(tree, "north", level, i, 0, k)
                        scalar.setValue(row, north*dx*dz, True)

                else:
                    if tree.dimension == 2:
                        coords = mesh.boundary_coords(tree, "north", level, i, 0, k)
                        scalar.setValue(row, north_func(coords)*dx, True)

                    elif tree.dimension == 3:
                        coords = mesh.boundary_coords(tree, "north", level, i, 0, k)
                        scalar.setValue(row, north_func(coords)*dx*dz, True)

            elif tree.bc["north"][0] == "neumann":
                if direction == "north":
                    north = tree_bc.nvalue[index]
                    if tree.dimension == 2:
                        #coords = mesh.boundary_coords(tree, "north", level, i, 0, k)
                        scalar.setValue(row, (north*dy/2)*dx, True)

                    elif tree.dimension == 3:
                        #coords = mesh.boundary_coords(tree, "north", level, i, 0, k)
                        scalar.setValue(row, (north*dy/2)*dx*dz, True)

                else:
                    if tree.dimension == 2:
                        coords = mesh.boundary_coords(tree, "north", level, i, 0, k)
                        scalar.setValue(row, (north_func(coords)*dy/2)*dx, True)

                    elif tree.dimension == 3:
                        coords = mesh.boundary_coords(tree, "north", level, i, 0, k)
                        scalar.setValue(row, (north_func(coords)*dy/2)*dx*dz, True)

    return scalar

def update_bc_velocity_div_z(tree, tree_bc):

    scalar = petsc.Vec().create()
    number_of_rows = tree.number_of_leaves
    scalar.setSizes(number_of_rows, number_of_rows)
    scalar.setUp()

    #elif axis == 2:

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
                back = tree_bc.nvalue[index]
                #coords = mesh.boundary_coords(tree, "back", level, i, j, 0)
                scalar.setValue(row, -back*dx*dy, True)

            elif tree.bc["back"][0] == "neumann":
                back = tree_bc.nvalue[index]
                #coords = mesh.boundary_coords(tree, "back", level, i, j, 0)
                scalar.setValue(row, -(back*dz/2)*dx*dy, True)

        #right flux
        if k == 2**level-1:

            if tree.bc["forth"][0] == "periodic":
                pass

            elif tree.bc["forth"][0] == "dirichlet":
                forth = tree_bc.nvalue[index]
                #coords = mesh.boundary_coords(tree, "back", level, i, j, 0)
                scalar.setValue(row, forth*dx*dy, True)

            elif tree.bc["forth"][0] == "neumann":
                forth = tree_bc.nvalue[index]
                #coords = mesh.boundary_coords(tree, "back", level, i, j, 0)
                scalar.setValue(row, (forth*dz/2)*dx*dy, True)

    return scalar

def update_bc_velocity_lap(tree, tree_bc):

    scalar = petsc.Vec().create()
    number_of_rows = tree.number_of_leaves
    scalar.setSizes(number_of_rows, number_of_rows)
    scalar.setUp()

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
                west = tree_bc.nvalue[index]
                if tree.bc["west"][0] == "dirichlet":
                    #coords = mesh.boundary_coords(tree, "west", level, 0, j, k)
                    scalar.setValue(row, 2*west*dy/(dx), True)

                elif tree.bc["west"][0] == "neumann":
                    #coords = mesh.boundary_coords(tree, "west", level, 0, j, k)
                    scalar.setValue(row, west*dy, True)

                elif tree.bc["west"][0] == "periodic":
                    pass

            if j == 0:
                south = tree_bc.nvalue[index]
                if tree.bc["south"][0] == "dirichlet":
                    #coords = mesh.boundary_coords(tree, "south", level, i, 0, k)
                    scalar.setValue(row, 2*south*dx/(dy), True)

                elif tree.bc["south"][0] == "neumann":
                    #coords = mesh.boundary_coords(tree, "south", level, i, 0, k)
                    scalar.setValue(row, south*dx, True)

                elif tree.bc["south"][0] == "periodic":
                    pass

            #right flux
            if i == 2**level-1:
                east = tree_bc.nvalue[index]
                if tree.bc["east"][0] == "dirichlet":
                    #coords = mesh.boundary_coords(tree, "east", level, 0, j, k)
                    scalar.setValue(row, 2*east*dy/(dx), True)

                elif tree.bc["east"][0] == "neumann":
                    #coords = mesh.boundary_coords(tree, "east", level, 0, j, k)
                    scalar.setValue(row, east*dy, True)

                elif tree.bc["east"][0] == "periodic":
                    pass

            if j == 2**level-1:
                north = tree_bc.nvalue[index]
                if tree.bc["north"][0] == "dirichlet":
                    #coords = mesh.boundary_coords(tree, "north", level, i, 0, k)
                    scalar.setValue(row, 2*north*dx/(dy), True)

                elif tree.bc["north"][0] == "neumann":
                    #coords = mesh.boundary_coords(tree, "north", level, i, 0, k)
                    scalar.setValue(row, north*dx, True)

                elif tree.bc["north"][0] == "periodic":
                    pass

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
                west = tree_bc.nvalue[index]
                if tree.bc["west"][0] == "dirichlet":
                    #coords = mesh.boundary_coords(tree, "west", level, 0, j, k)
                    scalar.setValue(row, 2*west*dy*dz/(dx), True)

                elif tree.bc["west"][0] == "neumann":
                    #coords = mesh.boundary_coords(tree, "west", level, 0, j, k)
                    scalar.setValue(row, west*dy*dz, True)

                elif tree.bc["west"][0] == "periodic":
                    pass

            if j == 0:
                south = tree_bc.nvalue[index]
                if tree.bc["south"][0] == "dirichlet":
                    #coords = mesh.boundary_coords(tree, "south", level, i, 0, k)
                    scalar.setValue(row, 2*south*dx*dz/(dy), True)

                elif tree.bc["south"][0] == "neumann":
                    #coords = mesh.boundary_coords(tree, "south", level, i, 0, k)
                    scalar.setValue(row, south*dx*dz, True)

                elif tree.bc["south"][0] == "periodic":
                    pass

            if k == 0:
                back = tree_bc.nvalue[index]
                if tree.bc["back"][0] == "dirichlet":
                    #coords = mesh.boundary_coords(tree, "back", level, i, j, 0)
                    scalar.setValue(row, 2*back*dx*dy/(dz), True)

                elif tree.bc["back"][0] == "neumann":
                    #coords = mesh.boundary_coords(tree, "back", level, i, j, 0)
                    scalar.setValue(row, back*dx*dy, True)

                elif tree.bc["back"][0] == "periodic":
                    pass

            #right flux
            if i == 2**level-1:
                east = tree_bc.nvalue[index]
                if tree.bc["east"][0] == "dirichlet":
                    #coords = mesh.boundary_coords(tree, "east", level, 0, j, k)
                    scalar.setValue(row, 2*east*dy*dz/(dx), True)

                elif tree.bc["east"][0] == "neumann":
                    #coords = mesh.boundary_coords(tree, "east", level, 0, j, k)
                    scalar.setValue(row, east*dy*dz, True)

                elif tree.bc["east"][0] == "periodic":
                    pass

            if j == 2**level-1:
                north = tree_bc.nvalue[index]
                if tree.bc["north"][0] == "dirichlet":
                    #coords = mesh.boundary_coords(tree, "north", level, i, 0, k)
                    scalar.setValue(row, 2*north*dx*dz/(dy), True)

                elif tree.bc["north"][0] == "neumann":
                    #coords = mesh.boundary_coords(tree, "north", level, i, 0, k)
                    scalar.setValue(row, north*dx*dz, True)

                elif tree.bc["north"][0] == "periodic":
                    pass

            if k == 2**level-1:
                forth = tree_bc.nvalue[index]
                if tree.bc["forth"][0] == "dirichlet":
                    #coords = mesh.boundary_coords(tree, "forth", level, i, j, 0)
                    scalar.setValue(row, 2*forth*dx*dy/(dz), True)

                elif tree.bc["forth"][0] == "neumann":
                    #coords = mesh.boundary_coords(tree, "forth", level, i, j, 0)
                    scalar.setValue(row, forth*dx*dy, True)

                elif tree.bc["forth"][0] == "periodic":
                    pass

        return scalar

def input_mass_flux(tree, direction=None):

    number_of_rows = tree.number_of_leaves
    input_mass = 0.

    #if north is None and south is None and east is None and west is None and forth is None and back is None:
    north_func = tree.bc["north"][1]
    south_func = tree.bc["south"][1]
    west_func = tree.bc["west"][1]
    east_func = tree.bc["east"][1]
    forth_func = tree.bc["forth"][1]
    back_func = tree.bc["back"][1]

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
        if direction == "west":
            if i == 0:
                if tree.dimension == 2:
                    coords = mesh.boundary_coords(tree, "west", level, 0, j, k)
                    input_mass += west_func(coords)*dy

                elif tree.dimension == 3:
                    coords = mesh.boundary_coords(tree, "west", level, 0, j, k)
                    input_mass += west_func(coords)*dy*dz

        elif direction == "south":
            if j == 0:
                if tree.dimension == 2:
                    coords = mesh.boundary_coords(tree, "south", level, i, 0, k)
                    input_mass += south_func(coords)*dx

                elif tree.dimension == 3:
                    coords = mesh.boundary_coords(tree, "south", level, i, 0, k)
                    input_mass += south_func(coords)*dx*dz

        elif direction == "back":
            if k == 0:
                coords = mesh.boundary_coords(tree, "back", level, i, j, 0)
                input_mass += back_func(coords)*dx*dy

        #right flux
        elif direction == "east":
            if i == 2**level-1:
                if tree.dimension == 2:
                    coords = mesh.boundary_coords(tree, "east", level, 0, j, k)
                    input_mass += east_func(coords)*dy

                if tree.dimension == 3:
                    coords = mesh.boundary_coords(tree, "east", level, 0, j, k)
                    input_mass += east_func(coords)*dy*dz

        elif direction == "north":
            if j == 2**level-1:
                if tree.dimension == 2:
                    coords = mesh.boundary_coords(tree, "north", level, i, 0, k)
                    input_mass += north_func(coords)*dx

                if tree.dimension == 3:
                    coords = mesh.boundary_coords(tree, "north", level, i, 0, k)
                    input_mass += north_func(coords)*dx*dz

        elif direction == "forth":
            if k == 2**level-1:
                coords = mesh.boundary_coords(tree, "forth", level, i, j, 0)
                input_mass += forth_func(coords)*dx*dy

    return input_mass

def output_mass_flux(tree, direction=None):

    number_of_rows = tree.number_of_leaves
    output_mass = 0.

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
        if direction == "west":
            if i == 0:
                if tree.dimension == 2:
                    output_mass += tree.nvalue[index]*dy

                elif tree.dimension == 3:
                    output_mass += tree.nvalue[index]*dy*dz

        elif direction == "south":
            if j == 0:
                if tree.dimension == 2:
                    output_mass += tree.nvalue[index]*dx

                elif tree.dimension == 3:
                    output_mass += tree.nvalue[index]*dx*dz

        elif direction == "back":
            if k == 0:
                output_mass += tree.nvalue[index]*dx*dy

        #right flux
        elif direction == "east":
            if i == 2**level-1:
                if tree.dimension == 2:
                    output_mass += tree.nvalue[index]*dy

                elif tree.dimension == 3:
                    output_mass += tree.nvalue[index]*dy*dz

        elif direction == "north":
            if j == 2**level-1:
                if tree.dimension == 2:
                    output_mass += tree.nvalue[index]*dx

                if tree.dimension == 3:
                    output_mass += tree.nvalue[index]*dx*dz

        elif direction == "forth":
            if k == 2**level-1:
                output_mass += tree.nvalue[index]*dx*dy

    return output_mass

def update_output_mass_flux(tree, output_target, direction=None):

    number_of_rows = tree.number_of_leaves
    output_mass = output_mass_flux(tree, direction)

    if direction == "west" or direction == "east":
        if tree.dimension == 2:
            output_corr_unit = (output_target - output_mass) / (tree.ymax - tree.ymin)

        elif tree.dimension == 3:
            output_corr_unit = (output_target - output_mass) / ((tree.ymax -
                tree.ymin)*(tree.zmax - tree.zmin))

    if direction == "south" or direction == "north":
        if tree.dimension == 2:
            output_corr_unit = (output_target - output_mass) / (tree.xmax - tree.xmin)

        elif tree.dimension == 3:
            output_corr_unit = (output_target - output_mass) / ((tree.xmax -
                tree.xmin)*(tree.zmax - tree.zmin))

    if direction == "back" or direction == "forth":
        output_corr_unit = (output_target - output_mass) / ((tree.xmax -
            tree.xmin)*(tree.ymax - tree.ymin))

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
        if direction == "west":
            if i == 0:
                tree.nvalue[index] += output_corr_unit
                #if tree.dimension == 2:
                #    #tree.nvalue[index] += output_corr_unit*dy
                #    tree.nvalue[index] += output_corr_unit

                #elif tree.dimension == 3:
                #    #tree.nvalue[index] += output_corr_unit*dy*dz
                #    tree.nvalue[index] += output_corr_unit

        elif direction == "south":
            if j == 0:
                tree.nvalue[index] += output_corr_unit
                #if tree.dimension == 2:
                #    #tree.nvalue[index] += output_corr_unit*dx
                #    tree.nvalue[index] += output_corr_unit

                #elif tree.dimension == 3:
                #    #tree.nvalue[index] += output_corr_unit*dx*dz
                #    tree.nvalue[index] += output_corr_unit

        elif direction == "back":
            if k == 0:
                tree.nvalue[index] += output_corr_unit
                #tree.nvalue[index] += output_corr_unit*dx*dy

        #right flux
        elif direction == "east":
            if i == 2**level-1:
                tree.nvalue[index] += output_corr_unit
                #if tree.dimension == 2:
                #    #tree.nvalue[index] += output_corr_unit*dy
                #    tree.nvalue[index] += output_corr_unit

                #elif tree.dimension == 3:
                #    #tree.nvalue[index] += output_corr_unit*dy*dz
                #    tree.nvalue[index] += output_corr_unit

        elif direction == "north":
            if j == 2**level-1:
                tree.nvalue[index] += output_corr_unit
                #if tree.dimension == 2:
                #    #tree.nvalue[index] += output_corr_unit*dx
                #    tree.nvalue[index] += output_corr_unit

                #if tree.dimension == 3:
                #    #tree.nvalue[index] += output_corr_unit*dx*dz
                #    tree.nvalue[index] += output_corr_unit

        elif direction == "forth":
            if k == 2**level-1:
                tree.nvalue[index] += output_corr_unit
                #tree.nvalue[index] += output_corr_unit*dx*dy
                #tree.nvalue[index] += output_corr_unit

def correct_value_boundary(tree, tree_target, direction=None):

    number_of_rows = tree.number_of_leaves

    for row in range(number_of_rows):
        index = tree.tree_leaves[row]
        level = tree.nlevel[index]
        i = tree.nindex_x[index]
        j = tree.nindex_y[index]
        k = tree.nindex_z[index]

        #left flux
        if direction == "west":
            if i == 0:
                tree.nvalue[index] = tree_target.nvalue[index]

        elif direction == "south":
            if j == 0:
                tree.nvalue[index] = tree_target.nvalue[index]

        elif direction == "back":
            if k == 0:
                tree.nvalue[index] = tree_target.nvalue[index]

        #right flux
        elif direction == "east":
            if i == 2**level-1:
                tree.nvalue[index] = tree_target.nvalue[index]

        elif direction == "north":
            if j == 2**level-1:
                tree.nvalue[index] = tree_target.nvalue[index]

        elif direction == "forth":
            if k == 2**level-1:
                tree.nvalue[index] = tree_target.nvalue[index]

def check_input_eq_output(input_mass_flux, output_mass_flux, rtol=1e-01):

    if (abs(input_mass_flux - output_mass_flux) <= rtol*abs(input_mass_flux)):
        return True
    else:
        return False
