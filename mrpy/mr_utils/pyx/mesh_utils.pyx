
cpdef int z_curve_index(int dimension, int level, int index_x=0, int index_y=0, int index_z=0):
    """...

    """

    return int((((2**dimension)**(level) - 1) / (2**dimension - 1) + \
            index_x + \
            index_y * 2**level + \
            index_z * 2**level * 2**level))

cpdef int _vertex_index(int level, int index_x, int index_y, int index_z):
        """...

        """

        return int(index_x + \
                   index_y * (2**level + 1) + \
                   index_z * (2**level + 1) * (2**level + 1))

cpdef tuple bc_compatible_local_indexes(tree, int level, int index_x=0, int index_y=0, int index_z=0):
    """...

    """

    if index_x < 0:

        if tree.bc["west"][0] == "periodic":
            return bc_compatible_local_indexes(tree, level, 2**level + index_x, index_y, index_z)
        else:
            return None

    elif index_x > 2**level-1:

        if tree.bc["east"][0] == "periodic":
            return bc_compatible_local_indexes(tree, level, index_x - 2**level, index_y, index_z)
        else:
            return None

    elif index_y < 0:

        if tree.bc["south"][0] == "periodic":
            return bc_compatible_local_indexes(tree, level, index_x, 2**level + index_y, index_z)
        else:
            return None

    elif index_y > 2**level-1:

        if tree.bc["north"][0] == "periodic":
            return bc_compatible_local_indexes(tree, level, index_x, index_y - 2**level, index_z)
        else:
            return None

    elif index_z < 0:

        if tree.bc["back"][0] == "periodic":
            return bc_compatible_local_indexes(tree, level, index_x, index_y, 2**level + index_z)
        else:
            return None

    elif index_z > 2**level-1:

        if tree.bc["forth"][0] == "periodic":
            return bc_compatible_local_indexes(tree, level, index_x, index_y, index_z - 2**level)
        else:
            return None

    else:
        return index_x, index_y, index_z

cpdef float space_step(tree, int level, int axis):
    """...

    """

    if axis == 0:
        return (tree.xmax - tree.xmin) / 2**level

    elif axis == 1:
        return (tree.ymax - tree.ymin) / 2**level

    elif axis == 2:
        return (tree.zmax - tree.zmin) / 2**level

cpdef tuple boundary_coords(tree, str side, int level, int index_x=0, int index_y=0, int index_z=0):
    """...

    """

    if side == "west":

        if index_y < 0:
            return boundary_coords(tree, side, level, 0, 0, index_z)

        elif index_y > 2**level-1:
            return boundary_coords(tree, side, level, 0, 2**level - 1, index_z)

        elif index_z < 0:
            return boundary_coords(tree, side, level, 0, index_y, 0)

        elif index_z > 2**level-1:
            return boundary_coords(tree, side, level, 0, index_y, 2**level - 1)

        else:
            if tree.dimension == 1:
                return (0., 0., 0.)

            elif tree.dimension == 2:
                dy = space_step(tree, level, 1)
                return (tree.xmin, tree.ymin + (index_y + 0.5)*dy, 0.)

            elif tree.dimension == 3:
                dy = space_step(tree, level, 1)
                dz = space_step(tree, level, 2)
                return (tree.xmin, tree.ymin + (index_y + 0.5)*dy, tree.zmin + (index_z + 0.5)*dz)

    if side == "east":

        if index_y < 0:
            return boundary_coords(tree, side, level, 0, 0, index_z)

        elif index_y > 2**level-1:
            return boundary_coords(tree, side, level, 0, 2**level - 1, index_z)

        elif index_z < 0:
            return boundary_coords(tree, side, level, 0, index_y, 0)

        elif index_z > 2**level-1:
            return boundary_coords(tree, side, level, 0, index_y, 2**level - 1)

        else:
            if tree.dimension == 1:
                return (0., 0., 0.)

            elif tree.dimension == 2:
                dy = space_step(tree, level, 1)
                return (tree.xmax, tree.ymin + (index_y + 0.5)*dy, 0.)

            elif tree.dimension == 3:
                dy = space_step(tree, level, 1)
                dz = space_step(tree, level, 2)
                return (tree.xmax, tree.ymin + (index_y + 0.5)*dy, tree.zmin + (index_z + 0.5)*dz)

    if side == "south":

        if index_x < 0:
            return boundary_coords(tree, side, level, 0, 0, index_z)

        elif index_x > 2**level-1:
            return boundary_coords(tree, side, level, 2**level - 1, 0, index_z)

        elif index_z < 0:
            return boundary_coords(tree, side, level, index_x, 0, 0)

        elif index_z > 2**level-1:
            return boundary_coords(tree, side, level, index_x, 0, 2**level - 1)

        else:
            if tree.dimension == 2:
                dx = space_step(tree, level, 0)
                return (tree.xmin + (index_x + 0.5)*dx, tree.ymin, 0.)

            elif tree.dimension == 3:
                dx = space_step(tree, level, 0)
                dz = space_step(tree, level, 2)
                return (tree.xmin + (index_x + 0.5)*dx, tree.ymin, tree.zmin + (index_z + 0.5)*dz)

    if side == "north":

        if index_x < 0:
            return boundary_coords(tree, side, level, 0, 0, index_z)

        elif index_x > 2**level-1:
            return boundary_coords(tree, side, level, 2**level - 1, 0, index_z)

        elif index_z < 0:
            return boundary_coords(tree, side, level, index_x, 0, 0)

        elif index_z > 2**level-1:
            return boundary_coords(tree, side, level, index_x, 0, 2**level - 1)

        else:
            if tree.dimension == 2:
                dx = space_step(tree, level, 0)
                return (tree.xmin + (index_x + 0.5)*dx, tree.ymax, 0.)

            elif tree.dimension == 3:
                dx = space_step(tree, level, 0)
                dz = space_step(tree, level, 2)
                return (tree.xmin + (index_x + 0.5)*dx, tree.ymax, tree.zmin + (index_z + 0.5)*dz)

    if side == "back":

        if index_x < 0:
            return boundary_coords(tree, side, level, 0, index_y, 0)

        elif index_x > 2**level-1:
            return boundary_coords(tree, side, level, 2**level - 1, index_y, 0)

        elif index_y < 0:
            return boundary_coords(tree, side, level, index_x, 0, 0)

        elif index_y > 2**level-1:
            return boundary_coords(tree, side, level, index_x, 2**level - 1, 0)

        else:
            dx = space_step(tree, level, 0)
            dy = space_step(tree, level, 1)
            return (tree.xmin + (index_x + 0.5)*dx, tree.ymin + (index_y + 0.5)*dy, tree.zmin)

    if side == "forth":

        if index_x < 0:
            return boundary_coords(tree, side, level, 0, index_y, 0)

        elif index_x > 2**level-1:
            return boundary_coords(tree, side, level, 2**level - 1, index_y, 0)

        elif index_y < 0:
            return boundary_coords(tree, side, level, index_x, 0, 0)

        elif index_y > 2**level-1:
            return boundary_coords(tree, side, level, index_x, 2**level - 1, 0)

        else:
            dx = space_step(tree, level, 0)
            dy = space_step(tree, level, 1)
            return (tree.xmin + (index_x + 0.5)*dx, tree.ymin + (index_y + 0.5)*dy, tree.zmax)

cpdef float get_value(tree, int level, int index_x=0, int index_y=0, int index_z=0):
    """...

    """

    if index_x < 0:

        if tree.bc["west"][0] == "periodic":
            return get_value(tree, level, 2**level + index_x, index_y, index_z)
        elif tree.bc["west"][0] == "dirichlet":
            coords = boundary_coords(tree, "west", level, 0, index_y, index_z) 
            return tree.bc["west"][1](coords)
        elif tree.bc["west"][0] == "neumann":
            dx = space_step(tree, level, 0)
            coords = boundary_coords(tree, "west", level, 0, index_y, index_z) 
            return dx*tree.bc["west"][1](coords) + get_value(tree, level, 0, index_y, index_z)

    elif index_x > 2**level-1:

        if tree.bc["east"][0] == "periodic":
            return get_value(tree, level, index_x - 2**level, index_y, index_z)
        elif tree.bc["east"][0] == "dirichlet":
            coords = boundary_coords(tree, "east", level, 0, index_y, index_z) 
            return tree.bc["east"][1](coords)
        elif tree.bc["east"][0] == "neumann":
            dx = space_step(tree, level, 0)
            coords = boundary_coords(tree, "east", level, 0, index_y, index_z) 
            return dx*tree.bc["east"][1](coords) + get_value(tree, level, 2**level - 1, index_y, index_z)

    elif index_y < 0:

        if tree.bc["south"][0] == "periodic":
            return get_value(tree, level, index_x, 2**level + index_y, index_z)
        elif tree.bc["south"][0] == "dirichlet":
            coords = boundary_coords(tree, "south", level, index_x, 0, index_z) 
            return tree.bc["south"][1](coords)
        elif tree.bc["south"][0] == "neumann":
            dy = space_step(tree, level, 1)
            coords = boundary_coords(tree, "south", level, index_x, 0, index_z) 
            return dy*tree.bc["south"][1](coords) + get_value(tree, level, index_x, 0, index_z)

    elif index_y > 2**level-1:

        if tree.bc["north"][0] == "periodic":
            return get_value(tree, level, index_x, index_y - 2**level, index_z)
        elif tree.bc["north"][0] == "dirichlet":
            coords = boundary_coords(tree, "north", level, index_x, 0, index_z) 
            return tree.bc["north"][1](coords)
        elif tree.bc["north"][0] == "neumann":
            dy = space_step(tree, level, 1)
            coords = boundary_coords(tree, "north", level, index_x, 0, index_z) 
            return dy*tree.bc["north"][1](coords) + get_value(tree, level, index_x, 2**level - 1, index_z)

    elif index_z < 0:

        if tree.bc["back"][0] == "periodic":
            return get_value(tree, level, index_x, index_y, 2**level + index_z)
        elif tree.bc["back"][0] == "dirichlet":
            coords = boundary_coords(tree, "back", level, index_x, index_y, 0) 
            return tree.bc["back"][1](coords)
        elif tree.bc["back"][0] == "neumann":
            dz = space_step(tree, level, 2)
            coords = boundary_coords(tree, "back", level, index_x, index_y, 0) 
            return dz*tree.bc["back"][1](coords) + get_value(tree, level, index_x, index_y, 0)

    elif index_z > 2**level-1:

        if tree.bc["forth"][0] == "periodic":
            return get_value(tree, level, index_x, index_y, index_z - 2**level)
        elif tree.bc["forth"][0] == "dirichlet":
            coords = boundary_coords(tree, "forth", level, index_x, index_y, 0) 
            return tree.bc["forth"][1](coords)
        elif tree.bc["forth"][0] == "neumann":
            dz = space_step(tree, level, 2)
            coords = boundary_coords(tree, "forth", level, index_x, index_y, 0) 
            return dz*tree.bc["forth"][1](coords) + get_value(tree, level, index_x, index_y, 0)

    else:
        return tree.nvalue[z_curve_index(tree.dimension, level, index_x, index_y, index_z)]
