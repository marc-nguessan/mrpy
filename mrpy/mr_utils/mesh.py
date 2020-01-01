from __future__ import print_function, division
from six.moves import range
"""...

"""

import numpy as np
import math
import importlib
import copy
import config as cfg

from .mesh_utils import z_curve_index, bc_compatible_local_indexes, \
boundary_coords, get_value, space_step, _vertex_index

class tree(object):
    """...

    """

    def __init__(self):

#hashtables for the nodes of the tree
        self.nlevel = {}
        self.nisleaf = {}
        self.nindex_tree_leaves = {}
        self.nindex_x = {}
        self.nindex_y = {}
        self.nindex_z = {}
        self.ndx = {}
        self.ndy = {}
        self.ndz = {}
        self.ncoord_x = {}
        self.ncoord_y = {}
        self.ncoord_z = {}
        self.nparent = {}
        self.nchildren = {}
        self.nvalue = {}
        self.ndetails = {}
        self.nnorm_details = {}
        self.nkeep_children = {}
        self.ngraded = {}
        self.nvertex_a = {}
        self.nvertex_b = {}
        self.nvertex_c = {}
        self.nvertex_d = {}
        self.nvertex_e = {}
        self.nvertex_f = {}
        self.nvertex_g = {}
        self.nvertex_h = {}
        self.vindex_x = {}
        self.vindex_y = {}
        self.vindex_z = {}

        self.tree_nodes = {}

        self.tree_leaves = None

        self.number_of_leaves = 0

        self.dimension = None

        self.stencil_graduation = None

        self.stencil_prediction = None

        self.max_norm_details = 0

        self.min_level = None

        self.max_level = None

        self.tag = None

        self.bc = None

        self.xmin = None
        self.xmax = None

        self.ymin = None
        self.ymax = None

        self.zmin = None
        self.zmax = None

    def set_to_same_grading(self, tree):
        """...

        """

        for index in tree.tree_nodes:
            keep_children_target = tree.nkeep_children[index]
            self.nkeep_children[index] = keep_children_target
            if keep_children_target == True:
                if self.nchildren[index] == [] or self.nchildren[index][0] not in self.tree_nodes:
                    #print index
                    create_children_of_node(self, index)

    def vertex_index(self, index_x=0, index_y=0, index_z=0):
        """...

        """

        #return int(index_x + \
        #           index_y * (2**self.max_level + 1) + \
        #           index_z * (2**self.max_level + 1) * (2**self.max_level + 1))

        return _vertex_index(self.max_level, index_x, index_y, index_z)

    def compute_vertex_coordinates(self, index):
        """...

        """

        coordinates = []

        if self.dimension == 1:
            dx = (self.xmax - self.xmin) / 2**self.max_level
            coordinates.append(self.xmin + self.vindex_x[index] * dx)

        elif self.dimension == 2:
            dx = (self.xmax - self.xmin) / 2**self.max_level
            dy = (self.ymax - self.ymin) / 2**self.max_level
            coordinates.append(self.xmin + self.vindex_x[index] * dx)
            coordinates.append(self.ymin + self.vindex_y[index] * dy)

        elif self.dimension == 3:
            dx = (self.xmax - self.xmin) / 2**self.max_level
            dy = (self.ymax - self.ymin) / 2**self.max_level
            dz = (self.zmax - self.zmin) / 2**self.max_level
            coordinates.append(self.xmin + self.vindex_x[index] * dx)
            coordinates.append(self.ymin + self.vindex_y[index] * dy)
            coordinates.append(self.zmin + self.vindex_z[index] * dz)

        return coordinates

    def compute_node_vertices(self, index):
        """...

        """

        if self.dimension == 1:

            temp = self.nindex_x[index] # Index_x of the vertex_a of the node on its grid
            if self.nlevel[index] == self.max_level:
                foo = temp
            else:
                foo = temp * (2**(self.max_level - self.nlevel[index])) # Index_x of the vertex_a of the node on the finest grid
            self.vindex_x[self.vertex_index(foo)] = foo

            # Vertex a
            self.nvertex_a[index] = self.vertex_index(foo)

            bar = (temp + 1) * (2**(self.max_level - self.nlevel[index])) # Index_x of the vertex_b of the node on the finest grid
            self.vindex_x[self.vertex_index(bar)] = bar

            # Vertex b
            self.nvertex_b[index] = self.vertex_index(bar)

        elif self.dimension == 2:

            temp_x = self.nindex_x[index] # Index_x of the vertex_a of the node on its grid
            temp_y = self.nindex_y[index] # Index_y of the vertex_a of the node on its grid
            foo_x = temp_x * (2**(self.max_level - self.nlevel[index])) # Index_x of the vertices a and c of the node on the finest grid
            foo_y = temp_y * (2**(self.max_level - self.nlevel[index])) # Index_y of the vertices a and b of the node on the finest grid

            # Vertex a
            self.vindex_x[self.vertex_index(foo_x, foo_y)] = foo_x
            self.vindex_y[self.vertex_index(foo_x, foo_y)] = foo_y
            self.nvertex_a[index] = self.vertex_index(foo_x, foo_y)

            # Vertex b
            bar_x = (temp_x + 1) * (2**(self.max_level - self.nlevel[index])) # Index_x of the vertices b and d of the node on the finest grid
            self.vindex_x[self.vertex_index(bar_x, foo_y)] = bar_x
            self.vindex_y[self.vertex_index(bar_x, foo_y)] = foo_y
            self.nvertex_b[index] = self.vertex_index(bar_x, foo_y)

            # Vertex c
            bar_y = (temp_y + 1) * (2**(self.max_level - self.nlevel[index])) # Index_y of the vertices c and d of the node on the finest grid
            self.vindex_x[self.vertex_index(foo_x, bar_y)] = foo_x
            self.vindex_y[self.vertex_index(foo_x, bar_y)] = bar_y
            self.nvertex_c[index] = self.vertex_index(foo_x, bar_y)

            # Vertex d
            self.vindex_x[self.vertex_index(bar_x, bar_y)] = bar_x
            self.vindex_y[self.vertex_index(bar_x, bar_y)] = bar_y
            self.nvertex_d[index] = self.vertex_index(bar_x, bar_y)

        elif self.dimension == 3:

            temp_x = self.nindex_x[index] # Index_x of the vertex_a of the node on its grid
            temp_y = self.nindex_y[index] # Index_y of the vertex_a of the node on its grid
            temp_z = self.nindex_z[index] # Index_z of the vertex_a of the node on its grid
            foo_x = temp_x * (2**(self.max_level - self.nlevel[index])) # Index_x of the vertices a, c, e and g of the node on the finest grid
            foo_y = temp_y * (2**(self.max_level - self.nlevel[index])) # Index_y of the vertices a, b, e and f of the node on the finest grid
            foo_z = temp_z * (2**(self.max_level - self.nlevel[index])) # Index_z of the vertices a, b, c and d of the node on the finest grid

            # Vertex a
            self.vindex_x[self.vertex_index(foo_x, foo_y, foo_z)] = foo_x
            self.vindex_y[self.vertex_index(foo_x, foo_y, foo_z)] = foo_y
            self.vindex_z[self.vertex_index(foo_x, foo_y, foo_z)] = foo_z
            self.nvertex_a[index] = self.vertex_index(foo_x, foo_y, foo_z)

            # Vertex b
            bar_x = (temp_x + 1) * (2**(self.max_level - self.nlevel[index])) # Index_x of the vertices b, d, f and h of the node on the finest grid
            self.vindex_x[self.vertex_index(bar_x, foo_y, foo_z)] = bar_x
            self.vindex_y[self.vertex_index(bar_x, foo_y, foo_z)] = foo_y
            self.vindex_z[self.vertex_index(bar_x, foo_y, foo_z)] = foo_z
            self.nvertex_b[index] = self.vertex_index(bar_x, foo_y, foo_z)

            # Vertex c
            bar_y = (temp_y + 1) * (2**(self.max_level - self.nlevel[index])) # Index_y of the vertices c, d, g and h of the node on the finest grid
            self.vindex_x[self.vertex_index(foo_x, bar_y, foo_z)] = foo_x
            self.vindex_y[self.vertex_index(foo_x, bar_y, foo_z)] = bar_y
            self.vindex_z[self.vertex_index(foo_x, bar_y, foo_z)] = foo_z
            self.nvertex_c[index] = self.vertex_index(foo_x, bar_y, foo_z)

            # Vertex d
            self.vindex_x[self.vertex_index(bar_x, bar_y, foo_z)] = bar_x
            self.vindex_y[self.vertex_index(bar_x, bar_y, foo_z)] = bar_y
            self.vindex_z[self.vertex_index(bar_x, bar_y, foo_z)] = foo_z
            self.nvertex_d[index] = self.vertex_index(bar_x, bar_y, foo_z)

            # Vertex e
            bar_z = (temp_z + 1) * (2**(self.max_level - self.nlevel[index])) # Index_z of the vertices e, f, g and h of the node on the finest grid
            self.vindex_x[self.vertex_index(foo_x, foo_y, bar_z)] = foo_x
            self.vindex_y[self.vertex_index(foo_x, foo_y, bar_z)] = foo_y
            self.vindex_z[self.vertex_index(foo_x, foo_y, bar_z)] = bar_z
            self.nvertex_e[index] = self.vertex_index(foo_x, foo_y, bar_z)

            # Vertex f
            self.vindex_x[self.vertex_index(bar_x, foo_y, bar_z)] = bar_x
            self.vindex_y[self.vertex_index(bar_x, foo_y, bar_z)] = foo_y
            self.vindex_z[self.vertex_index(bar_x, foo_y, bar_z)] = bar_z
            self.nvertex_f[index] = self.vertex_index(bar_x, foo_y, bar_z)

            # Vertex g
            self.vindex_x[self.vertex_index(foo_x, bar_y, bar_z)] = foo_x
            self.vindex_y[self.vertex_index(foo_x, bar_y, bar_z)] = bar_y
            self.vindex_z[self.vertex_index(foo_x, bar_y, bar_z)] = bar_z
            self.nvertex_g[index] = self.vertex_index(foo_x, bar_y, bar_z)

            # Vertex h
            self.vindex_x[self.vertex_index(bar_x, bar_y, bar_z)] = bar_x
            self.vindex_y[self.vertex_index(bar_x, bar_y, bar_z)] = bar_y
            self.vindex_z[self.vertex_index(bar_x, bar_y, bar_z)] = bar_z
            self.nvertex_h[index] = self.vertex_index(bar_x, bar_y, bar_z)

def create_new_tree(dimension, min_level, max_level, stencil_graduation,
                    stencil_prediction, xmin=0, xmax=0, ymin=0, ymax=0, zmin=0,
                    zmax=0):
    """...

    """

    temp = tree()

    temp.dimension = dimension
    temp.stencil_graduation = stencil_graduation
    temp.stencil_prediction = stencil_prediction
    temp.min_level = min_level
    temp.max_level = max_level
    temp.xmin = xmin
    temp.xmax = xmax
    temp.ymin = ymin
    temp.ymax = ymax
    temp.zmin = zmin
    temp.zmax = zmax

    for level in range(max_level+1):

        if dimension == 1:
            for i in range(2**level):
                create_node(temp, dimension, level, i)

                # Children pointers creation
                if level != max_level:
                    create_children_pointers(temp, dimension, level, i)

        elif dimension == 2:
            for i in range(2**level):
                for j in range(2**level):
                    create_node(temp, dimension, level, i, j)

                # Children pointers creation
                    if level != max_level:
                        create_children_pointers(temp, dimension, level, i, j)

        elif dimension == 3:
            for i in range(2**level):
                for j in range(2**level):
                    for k in range(2**level):
                        create_node(temp, dimension, level, i, j, k)

                # Children pointers creation
                    if level != max_level:
                        create_children_pointers(temp, dimension, level, i, j, k)

        else:
            print("Error: incorrect dimension")

    return temp

#def space_step(level, axis):
#    """...
#
#    """
#
#    if axis == 0:
#        return (self.xmax - self.xmin) / 2**level
#
#    elif axis == 1:
#        return (self.ymax - self.ymin) / 2**level
#
#    elif axis == 2:
#        return (self.zmax - self.zmin) / 2**level

def create_node(tree, dimension, level, index_x=0, index_y=0, index_z=0):
    """...

    """

    index = z_curve_index(dimension, level, index_x, index_y, index_z)
    tree.tree_nodes[index] = index

    tree.nlevel[index] = level
    if level == tree.max_level:
        tree.nisleaf[index] = True
    else:
        tree.nisleaf[index] = False

    tree.nindex_tree_leaves[index] = None
    tree.nindex_x[index] = index_x
    tree.nindex_y[index] = index_y
    tree.nindex_z[index] = index_z

    # Parent pointer creation
    if level != 0:
        index_x_parent = int(math.floor(index_x/2))
        index_y_parent = int(math.floor(index_y/2))
        index_z_parent = int(math.floor(index_z/2))

        tree.nparent[index] = z_curve_index(dimension, level-1, index_x_parent,
                                    index_y_parent, index_z_parent)

    tree.nvalue[index] = None
    tree.ndetails[index] = None
    tree.nnorm_details[index] = None
    if level != tree.max_level:
        tree.nkeep_children[index] = False
        #tree.nkeep_children[index] = True
    else:
        tree.nkeep_children[index] = False
    tree.ngraded[index] = False
    tree.nchildren[index] = []

    tree.ndx[index] = None
    tree.ndy[index] = None
    tree.ndz[index] = None

    if dimension == 1:
        tree.ndx[index] = space_step(tree, level, 0)
        tree.ncoord_x[index] = tree.xmin + (index_x + 0.5) * tree.ndx[index]

    if dimension == 2:
        tree.ndx[index] = space_step(tree, level, 0)
        tree.ncoord_x[index] = tree.xmin + (index_x + 0.5) * tree.ndx[index]

        tree.ndy[index] = space_step(tree, level, 1)
        tree.ncoord_y[index] = tree.ymin + (index_y + 0.5) * tree.ndy[index]

    if dimension == 3:
        tree.ndx[index] = space_step(tree, level, 0)
        tree.ncoord_x[index] = tree.xmin + (index_x + 0.5) * tree.ndx[index]

        tree.ndy[index] = space_step(tree, level, 1)
        tree.ncoord_y[index] = tree.ymin + (index_y + 0.5) * tree.ndy[index]

        tree.ndz[index] = space_step(tree, level, 2)
        tree.ncoord_z[index] = tree.zmin + (index_z + 0.5) * tree.ndz[index]

def delete_node(tree, index):
    del tree.nlevel[index]
    del tree.nisleaf[index]
    del tree.nindex_tree_leaves[index]
    del tree.nparent[index]
    del tree.nchildren[index]
    del tree.nvalue[index]
    del tree.ndetails[index]
    del tree.nnorm_details[index]
    del tree.nkeep_children[index]
    del tree.ngraded[index]

    if tree.dimension == 1:
        del tree.nindex_x[index]
        del tree.ndx[index]
        del tree.ncoord_x[index]
    elif tree.dimension == 2:
        del tree.nindex_x[index]
        del tree.nindex_y[index]
        del tree.ndx[index]
        del tree.ndy[index]
        del tree.ncoord_x[index]
        del tree.ncoord_y[index]
    elif tree.dimension == 3:
        del tree.nindex_x[index]
        del tree.nindex_y[index]
        del tree.nindex_z[index]
        del tree.ndx[index]
        del tree.ndy[index]
        del tree.ndz[index]
        del tree.ncoord_x[index]
        del tree.ncoord_y[index]
        del tree.ncoord_z[index]

    del tree.tree_nodes[index]

def create_children_pointers(tree, dimension, level, index_x=0, index_y=0, index_z=0):
    """...

    """

    index = z_curve_index(dimension, level, index_x, index_y, index_z)

    if dimension == 1:
        foo = []
        for m in range(2):
            foo.append(z_curve_index(dimension, level + 1, 2*index_x + m))

        tree.nchildren[index] = foo

    elif dimension == 2:
        foo = []
        for n in range(2):
            for m in range(2):
                foo.append(z_curve_index(dimension, level + 1, 2*index_x + m, 2*index_y + n))

        tree.nchildren[index] = foo

    elif dimension == 3:
        foo = []
        for o in range(2):
            for n in range(2):
                for m in range(2):
                    foo.append(z_curve_index(dimension, level + 1, 2*index_x + m, 2*index_y + n, 2*index_z + o))

        tree.nchildren[index] = foo

    else:
        print("Error: incorrect dimension")

def create_children_of_node(tree, index):
    """...

    """

    dimension = tree.dimension
    level = tree.nlevel[index]
    i = tree.nindex_x[index]
    j = tree.nindex_y[index]
    k = tree.nindex_z[index]

    create_children_pointers(tree, dimension, level, i, j, k)

    if dimension == 1:
        for m in range(2):
            create_node(tree, dimension, level + 1, 2*i + m)
            # Children pointers creation
            if level < tree.max_level-1:
                create_children_pointers(tree, dimension, level + 1, 2*i + m)

    elif dimension == 2:
        for n in range(2):
            for m in range(2):
                create_node(tree, dimension, level + 1, 2*i + m, 2*j + n)
                # Children pointers creation
                if level < tree.max_level-1:
                    create_children_pointers(tree, dimension, level + 1, 2*i + m, 2*j + n)

    elif dimension == 3:
        for o in range(2):
            for n in range(2):
                for m in range(2):
                    create_node(tree, dimension, level + 1, 2*i + m, 2*j + n, 2*k + o)
                    # Children pointers creation
                    if level < tree.max_level-1:
                        create_children_pointers(tree, dimension, level + 1, 2*i + m, 2*j + n, 2*k + o)

    else:
        print("Error: incorrect dimension")

#def z_curve_index(dimension, level, index_x=0, index_y=0, index_z=0):
#    """...
#
#    """
#
#    return int((((2**dimension)**(level) - 1) / (2**dimension - 1) + \
#            index_x + \
#            index_y * 2**level + \
#            index_z * 2**level * 2**level))

#def get_value(tree, level, index_x=0, index_y=0, index_z=0):
#    """...
#
#    """
#
#    if index_x < 0:
#
#        if cfg.bc_dict[tree.tag][0] == "periodic":
#            return get_value(tree, level, 2**level + index_x, index_y, index_z)
#        elif cfg.bc_dict[tree.tag][0] == "dirichlet":
#            return cfg.bc_dict[tree.tag][1][2]
#        elif cfg.bc_dict[tree.tag][0] == "neumann":
#            dx = space_step(level, 0)
#            return dx*cfg.bc_dict[tree.tag][1][2] + get_value(tree, level, 0, index_y, index_z)
#
#    elif index_x > 2**level-1:
#
#        if cfg.bc_dict[tree.tag][0] == "periodic":
#            return get_value(tree, level, index_x - 2**level, index_y, index_z)
#        elif cfg.bc_dict[tree.tag][0] == "dirichlet":
#            return cfg.bc_dict[tree.tag][1][3]
#        elif cfg.bc_dict[tree.tag][0] == "neumann":
#            dx = space_step(level, 0)
#            return dx*cfg.bc_dict[tree.tag][1][3] + get_value(tree, level, 2**level - 1, index_y, index_z)
#
#    elif index_y < 0:
#
#        if cfg.bc_dict[tree.tag][0] == "periodic":
#            return get_value(tree, level, index_x, 2**level + index_y, index_z)
#        elif cfg.bc_dict[tree.tag][0] == "dirichlet":
#            return cfg.bc_dict[tree.tag][1][1]
#        elif cfg.bc_dict[tree.tag][0] == "neumann":
#            dy = space_step(level, 1)
#            return dy*cfg.bc_dict[tree.tag][1][1] + get_value(tree, level, index_x, 0, index_z)
#
#    elif index_y > 2**level-1:
#
#        if cfg.bc_dict[tree.tag][0] == "periodic":
#            return get_value(tree, level, index_x, index_y - 2**level, index_z)
#        elif cfg.bc_dict[tree.tag][0] == "dirichlet":
#            return cfg.bc_dict[tree.tag][1][0]
#        elif cfg.bc_dict[tree.tag][0] == "neumann":
#            dy = space_step(level, 1)
#            return dy*cfg.bc_dict[tree.tag][1][0] + get_value(tree, level, index_x, 2**level - 1, index_z)
#
#    elif index_z < 0:
#
#        if cfg.bc_dict[tree.tag][0] == "periodic":
#            return get_value(tree, level, index_x, index_y, 2**level + index_z)
#        elif cfg.bc_dict[tree.tag][0] == "dirichlet":
#            return cfg.bc_dict[tree.tag][1][5]
#        elif cfg.bc_dict[tree.tag][0] == "neumann":
#            dz = space_step(level, 2)
#            return dz*cfg.bc_dict[tree.tag][1][5] + get_value(tree, level, index_x, index_y, 0)
#
#    elif index_z > 2**level-1:
#
#        if cfg.bc_dict[tree.tag][0] == "periodic":
#            return get_value(tree, level, index_x, index_y, index_z - 2**level)
#        elif cfg.bc_dict[tree.tag][0] == "dirichlet":
#            return cfg.bc_dict[tree.tag][1][4]
#        elif cfg.bc_dict[tree.tag][0] == "neumann":
#            dz = space_step(level, 2)
#            return dz*cfg.bc_dict[tree.tag][1][4] + get_value(tree, level, index_x, index_y, 0)
#
#    else:
#        return tree.nvalue[z_curve_index(tree.dimension, level, index_x, index_y, index_z)]

#def bc_compatible_local_indexes(tree, level, index_x=0, index_y=0, index_z=0):
#    """...
#
#    """
#
#    if index_x < 0:
#
#        if cfg.bc_dict[tree.tag][0] == "periodic":
#            return bc_compatible_local_indexes(tree, level, 2**level + index_x, index_y, index_z)
#        else:
#            return None
#
#    elif index_x > 2**level-1:
#
#        if cfg.bc_dict[tree.tag][0] == "periodic":
#            return bc_compatible_local_indexes(tree, level, index_x - 2**level, index_y, index_z)
#        else:
#            return None
#
#    elif index_y < 0:
#
#        if cfg.bc_dict[tree.tag][0] == "periodic":
#            return bc_compatible_local_indexes(tree, level, index_x, 2**level + index_y, index_z)
#        else:
#            return None
#
#    elif index_y > 2**level-1:
#
#        if cfg.bc_dict[tree.tag][0] == "periodic":
#            return bc_compatible_local_indexes(tree, level, index_x, index_y - 2**level, index_z)
#        else:
#            return None
#
#    elif index_z < 0:
#
#        if cfg.bc_dict[tree.tag][0] == "periodic":
#            return bc_compatible_local_indexes(tree, level, index_x, index_y, 2**level + index_z)
#        else:
#            return None
#
#    elif index_z > 2**level-1:
#
#        if cfg.bc_dict[tree.tag][0] == "periodic":
#            return bc_compatible_local_indexes(tree, level, index_x, index_y, index_z - 2**level)
#        else:
#            return None
#
#    else:
#        return index_x, index_y, index_z

def listing_of_leaves(*trees):
    """...

    """

    def single_tree_function(tree):
        temp = []
        tree.number_of_leaves = 0
        for index in sorted(tree.tree_nodes.keys()):
            if tree.nchildren[index] == [] or tree.nchildren[index][0] not in tree.tree_nodes:
                tree.nisleaf[index] = True
                temp.append(index)
                tree.nindex_tree_leaves[index] = tree.number_of_leaves
                tree.number_of_leaves += 1
            else:
                tree.nisleaf[index] = False

        tree.tree_leaves = np.array(temp)

    for tree in trees:
        single_tree_function(tree)

def copy_tree(target):
    """Return a copy of tree, that is of course a tree object."""

    temp = tree()

#hashtables for the nodes of the tree
    temp.nlevel = copy.copy(target.nlevel)
    temp.nisleaf = copy.copy(target.nisleaf)
    temp.nindex_tree_leaves = copy.copy(target.nindex_tree_leaves)
    temp.nindex_x = copy.copy(target.nindex_x)
    temp.nindex_y = copy.copy(target.nindex_y)
    temp.nindex_z = copy.copy(target.nindex_z)
    temp.ndx = copy.copy(target.ndx)
    temp.ndy = copy.copy(target.ndy)
    temp.ndz = copy.copy(target.ndz)
    temp.ncoord_x = copy.copy(target.ncoord_x)
    temp.ncoord_y = copy.copy(target.ncoord_y)
    temp.ncoord_z = copy.copy(target.ncoord_z)
    temp.nparent = copy.copy(target.nparent)
    temp.nchildren = copy.deepcopy(target.nchildren)
    temp.nvalue = copy.copy(target.nvalue)
    temp.ndetails = copy.copy(target.ndetails)
    temp.nnorm_details = copy.copy(target.nnorm_details)
    temp.nkeep_children = copy.copy(target.nkeep_children)
    temp.ngraded = copy.copy(target.ngraded)
    temp.nvertex_a = copy.copy(target.nvertex_a)
    temp.nvertex_b = copy.copy(target.nvertex_b)
    temp.nvertex_c = copy.copy(target.nvertex_c)
    temp.nvertex_d = copy.copy(target.nvertex_d)
    temp.nvertex_e = copy.copy(target.nvertex_e)
    temp.nvertex_f = copy.copy(target.nvertex_f)
    temp.nvertex_g = copy.copy(target.nvertex_g)
    temp.nvertex_h = copy.copy(target.nvertex_h)
    temp.vindex_x = copy.copy(target.vindex_x)
    temp.vindex_y = copy.copy(target.vindex_y)
    temp.vindex_z = copy.copy(target.vindex_z)

    temp.tree_nodes = copy.copy(target.tree_nodes)

    temp.tree_leaves = np.copy(target.tree_leaves)

    temp.number_of_leaves = target.number_of_leaves

    temp.dimension = target.dimension

    temp.stencil_graduation = target.stencil_graduation

    temp.stencil_prediction = target.stencil_prediction

    temp.max_norm_details = target.max_norm_details

    temp.min_level = target.min_level

    temp.max_level = target.max_level

    temp.tag = target.tag

    temp.bc = target.bc

    temp.xmin = target.xmin
    temp.xmax = target.xmax

    temp.ymin = target.ymin
    temp.ymax = target.ymax

    temp.zmin = target.zmin
    temp.zmax = target.zmax

    return temp




if __name__ == "__main__":

    a = create_new_tree(cfg.dimension, cfg.min_level, cfg.max_level, cfg.stencil_graduation, cfg.stencil_prediction)

    b = create_new_tree(cfg.dimension, cfg.min_level, cfg.max_level, cfg.stencil_graduation, cfg.stencil_prediction)

    c = create_new_tree(cfg.dimension, cfg.min_level, cfg.max_level, cfg.stencil_graduation, cfg.stencil_prediction)
    print("done")
    quit()

