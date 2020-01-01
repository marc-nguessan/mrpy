from __future__ import print_function, division

"""...

"""

from six.moves import range

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import h5py as hp

import numpy as np


class OutputWriter(object):
    """...

    """

    def __init__(self, output_file_basename):
####### introduire des variables pour la dimension du domaine et le type de topo ########

        self.output_file_basename = output_file_basename
        self.xdmf_number_type = "NumberType=\"Float\" Precision=\"4\""
        self.xdmf_data_format = "Format=\"HDF\""

    def xmf_write_header(self, time, number_of_elements, number_of_nodes, topology_type="Quadrilateral", nodes_per_element=4):
        """...

        """

        self.xmf_file.write("<?xml version=\"1.0\" ?>\n")
        self.xmf_file.write("<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n")
        self.xmf_file.write("<Xdmf Version=\"2.0\">\n")
        self.xmf_file.write("  <Domain>\n")
        self.xmf_file.write("    <Grid Name=\"{0}\" GridType=\"Uniform\">\n".format(self.output_file_name))
        self.xmf_file.write("      <Time TimeType=\"Single\" Value=\"{0}\" />\n".format(time))

        # Connectivity
        self.xmf_file.write("      <Topology TopologyType=\"{0}\" NumberOfElements=\" {1} \">\n".format(topology_type, number_of_elements))
        self.xmf_file.write("        <DataItem Dimensions=\"{0} {1}\" DataType=\"Int\" Format=\"HDF\">\n".format(number_of_elements, nodes_per_element))
        self.xmf_file.write("         {0}.h5:/connectivity\n".format(self.output_file_name))
        self.xmf_file.write("        </DataItem>\n")
        self.xmf_file.write("      </Topology>\n")

        # Points
        self.xmf_file.write("      <Geometry GeometryType=\"XYZ\">\n")
        self.xmf_file.write("        <DataItem Dimensions=\"{0} 3\" {1} {2}>\n".format(number_of_nodes, self.xdmf_number_type, self.xdmf_data_format))
        self.xmf_file.write("         {0}.h5:/coordinates\n".format(self.output_file_name))
        self.xmf_file.write("        </DataItem>\n")
        self.xmf_file.write("      </Geometry>\n")

    def xmf_write_main_header(self):
        """...

        """

        self.xmf_main_file.write("<?xml version=\"1.0\" ?>\n")
        self.xmf_main_file.write("<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n")
        self.xmf_main_file.write("<Xdmf xmlns:xi=\"http://www.w3.org/2001/XInclude\" Version=\"2.0\">\n")
        self.xmf_main_file.write("  <Domain Name=\"MainTimeSeries\">\n");
        self.xmf_main_file.write("    <Grid Name=\"MainTimeSeries\" GridType=\"Collection\" CollectionType=\"Temporal\">\n")

    def xmf_write_main_include(self):
        """...

        """

        self.xmf_main_file.write("      <xi:include href=\"{0}\" xpointer=\"xpointer(//Xdmf/Domain/Grid)\" />\n".format(self.output_file_name + ".xmf"))

    def xmf_write_scalar_attribute(self, scalar_attribute_name, number_of_elements):
        """...

        """

        self.xmf_file.write("      <Attribute Name=\"{0}\" AttributeType=\"Scalar\" Center=\"Cell\">\n".format(scalar_attribute_name))
        self.xmf_file.write("        <DataItem Dimensions=\"{0}\" {1} {2}>\n".format(number_of_elements, self.xdmf_number_type, self.xdmf_data_format))
        self.xmf_file.write("         {0}.h5:/{1}\n".format(self.output_file_name, scalar_attribute_name))
        self.xmf_file.write("        </DataItem>\n")
        self.xmf_file.write("      </Attribute>\n")

    def xmf_write_vector_attribute(self, vector_attribute_name, number_of_elements, number_of_components):
        """...

        """

        self.xmf_file.write("      <Attribute Name=\"{0}\" AttributeType=\"Scalar\" Center=\"Cell\">\n".format(vector_attribute_name))
        self.xmf_file.write("        <DataItem Dimensions=\"{0} {1}\" {2} {3}>\n".format(number_of_elements, number_of_components, self.xdmf_number_type, self.xdmf_data_format))
        self.xmf_file.write("         {0}.h5:/{1}\n".format(self.output_file_name, vector_attribute_name))
        self.xmf_file.write("        </DataItem>\n")
        self.xmf_file.write("      </Attribute>\n")

    def xmf_write_footer(self):
        """...

        """

        self.xmf_file.write("    </Grid>\n")
        self.xmf_file.write("  </Domain>\n")
        self.xmf_file.write("</Xdmf>\n")

    def xmf_write_main_footer(self):
        """...

        """

        self.xmf_main_file.write("    </Grid>\n")
        self.xmf_main_file.write("  </Domain>\n")
        self.xmf_main_file.write("</Xdmf>\n")

    def compute_vertices_hashtable(self, tree):
        """...

        """

        vertices_hashtable = {}
        count = 0 # position counter

        # Building the list of vertices
        for index in tree.tree_leaves:

            if tree.dimension == 1:
                # adding vertex a to the list of vertices
                if tree.nvertex_a[index] not in vertices_hashtable:
                    vertices_hashtable[tree.nvertex_a[index]] = count
                    # we increment the position counter
                    count += 1

                # adding vertex b to the list of vertices
                if tree.nvertex_b[index] not in vertices_hashtable:
                    vertices_hashtable[tree.nvertex_b[index]] = count
                    # we increment the position counter
                    count += 1

            elif tree.dimension == 2:
                # adding vertex a to the list of vertices
                if tree.nvertex_a[index] not in vertices_hashtable:
                    vertices_hashtable[tree.nvertex_a[index]] = count
                    # we increment the position counter
                    count += 1

                # adding vertex b to the list of vertices
                if tree.nvertex_b[index] not in vertices_hashtable:
                    vertices_hashtable[tree.nvertex_b[index]] = count
                    # we increment the position counter
                    count += 1

                # adding vertex c to the list of vertices
                if tree.nvertex_c[index] not in vertices_hashtable:
                    vertices_hashtable[tree.nvertex_c[index]] = count
                    # we increment the position counter
                    count += 1

                # adding vertex d to the list of vertices
                if tree.nvertex_d[index] not in vertices_hashtable:
                    vertices_hashtable[tree.nvertex_d[index]] = count
                    # we increment the position counter
                    count += 1

            elif tree.dimension == 3:
                # adding vertex a to the list of vertices
                if tree.nvertex_a[index] not in vertices_hashtable:
                    vertices_hashtable[tree.nvertex_a[index]] = count
                    # we increment the position counter
                    count += 1

                # adding vertex b to the list of vertices
                if tree.nvertex_b[index] not in vertices_hashtable:
                    vertices_hashtable[tree.nvertex_b[index]] = count
                    # we increment the position counter
                    count += 1

                # adding vertex c to the list of vertices
                if tree.nvertex_c[index] not in vertices_hashtable:
                    vertices_hashtable[tree.nvertex_c[index]] = count
                    # we increment the position counter
                    count += 1

                # adding vertex d to the list of vertices
                if tree.nvertex_d[index] not in vertices_hashtable:
                    vertices_hashtable[tree.nvertex_d[index]] = count
                    # we increment the position counter
                    count += 1

                # adding vertex e to the list of vertices
                if tree.nvertex_e[index] not in vertices_hashtable:
                    vertices_hashtable[tree.nvertex_e[index]] = count
                    # we increment the position counter
                    count += 1

                # adding vertex f to the list of vertices
                if tree.nvertex_f[index] not in vertices_hashtable:
                    vertices_hashtable[tree.nvertex_f[index]] = count
                    # we increment the position counter
                    count += 1

                # adding vertex g to the list of vertices
                if tree.nvertex_g[index] not in vertices_hashtable:
                    vertices_hashtable[tree.nvertex_g[index]] = count
                    # we increment the position counter
                    count += 1

                # adding vertex h to the list of vertices
                if tree.nvertex_h[index] not in vertices_hashtable:
                    vertices_hashtable[tree.nvertex_h[index]] = count
                    # we increment the position counter
                    count += 1

        return vertices_hashtable

    def hdf_write_mesh(self, tree, vertices_hashtable):
        """...

        """
        # Building the connectivity dataset
        if tree.dimension == 1:
            connectivity_nparray = np.empty((tree.number_of_leaves, 2), dtype=np.int32)
            for row in range(tree.number_of_leaves):
                index = tree.tree_leaves[row]
                vertex_a = tree.nvertex_a[index]
                vertex_b = tree.nvertex_b[index]
                # Setting the position of the vertices a and b for the current cell
                connectivity_nparray[row, 0] = vertices_hashtable[vertex_a]
                connectivity_nparray[row, 1] = vertices_hashtable[vertex_b]

        elif tree.dimension == 2:
            connectivity_nparray = np.empty((tree.number_of_leaves, 4), dtype=np.int32)
            for row in range(tree.number_of_leaves):
                index = tree.tree_leaves[row]
                vertex_a = tree.nvertex_a[index]
                vertex_b = tree.nvertex_b[index]
                vertex_c = tree.nvertex_c[index]
                vertex_d = tree.nvertex_d[index]
                # Setting the position of the vertices a, b, c and d for the current cell
                connectivity_nparray[row, 0] = vertices_hashtable[vertex_a]
                connectivity_nparray[row, 1] = vertices_hashtable[vertex_b]
                connectivity_nparray[row, 2] = vertices_hashtable[vertex_d] # We need to give the vertices in the anticlockwise direction
                connectivity_nparray[row, 3] = vertices_hashtable[vertex_c]

        elif tree.dimension == 3:
            connectivity_nparray = np.empty((tree.number_of_leaves, 8), dtype=np.int32)
            for row in range(tree.number_of_leaves):
                index = tree.tree_leaves[row]
                vertex_a = tree.nvertex_a[index]
                vertex_b = tree.nvertex_b[index]
                vertex_c = tree.nvertex_c[index]
                vertex_d = tree.nvertex_d[index]
                vertex_e = tree.nvertex_e[index]
                vertex_f = tree.nvertex_f[index]
                vertex_g = tree.nvertex_g[index]
                vertex_h = tree.nvertex_h[index]
                # Setting the position of the vertices a, b, c, d, e, f, g and h for the current cell
                connectivity_nparray[row, 0] = vertices_hashtable[vertex_a]
                connectivity_nparray[row, 1] = vertices_hashtable[vertex_b]
                connectivity_nparray[row, 2] = vertices_hashtable[vertex_d] # We need to give the vertices in the anticlockwise direction
                connectivity_nparray[row, 3] = vertices_hashtable[vertex_c]
                connectivity_nparray[row, 4] = vertices_hashtable[vertex_e]
                connectivity_nparray[row, 5] = vertices_hashtable[vertex_f]
                connectivity_nparray[row, 6] = vertices_hashtable[vertex_h] # We need to give the vertices in the anticlockwise direction
                connectivity_nparray[row, 7] = vertices_hashtable[vertex_g]

        self.hdf_file.create_dataset("connectivity", data=connectivity_nparray)

        # Building the coordinates dataset
        coordinates_nparray = np.zeros((len(vertices_hashtable.keys()), 3), dtype=np.float)
        if tree.dimension == 1:
            for index_vertex, position in vertices_hashtable.items():
                coordinates = tree.compute_vertex_coordinates(index_vertex)
                coordinates_nparray[position, 0] = coordinates[0]

        elif tree.dimension == 2:
            for index_vertex, position in vertices_hashtable.items():
                coordinates = tree.compute_vertex_coordinates(index_vertex)
                coordinates_nparray[position, 0] = coordinates[0]
                coordinates_nparray[position, 1] = coordinates[1]

        elif tree.dimension == 3:
            for index_vertex, position in vertices_hashtable.items():
                coordinates = tree.compute_vertex_coordinates(index_vertex)
                coordinates_nparray[position, 0] = coordinates[0]
                coordinates_nparray[position, 1] = coordinates[1]
                coordinates_nparray[position, 2] = coordinates[2]

        self.hdf_file.create_dataset("coordinates", data=coordinates_nparray)

    def hdf_write_scalar_attribute(self, tree):
        """...

        """
        # Building the connectivity dataset
        attribute_nparray = np.empty((tree.number_of_leaves,), dtype=np.float)
        for row in range(tree.number_of_leaves):
            index = tree.tree_leaves[row]
            attribute_nparray[row] = tree.nvalue[index]

        self.hdf_file.create_dataset(tree.tag, data=attribute_nparray)

    def hdf_write_vector_attribute(self, vector_attribute_name, trees_of_components):
        """...

        """
        # Building the connectivity dataset
        attribute_nparray = np.empty((trees_of_components[0].number_of_leaves, len(trees_of_components)), dtype=np.float)
        for row in range(trees_of_components[0].number_of_leaves):
            for component in range(len(trees_of_components)):
                index = trees_of_components[component].tree_leaves[row]
                attribute_nparray[row, component] = trees_of_components[component].nvalue[index]

        self.hdf_file.create_dataset(vector_attribute_name, data=attribute_nparray)

    def initialize(self):
        """...

        """

        self.xmf_main_file = open(self.output_file_basename + "_main.xmf", 'w')
        self.xmf_write_main_header()

    def write(self, scalar_trees, vector_trees=None, vector_attribute_name="velocity", output_file_name="testfile", time=0):
        """...

        """

        self.output_file_name = output_file_name
        self.xmf_file = open(output_file_name + ".xmf", 'w')
        self.hdf_file = hp.File(output_file_name + ".h5", 'w')
        self.xmf_write_main_include()

        dimension = scalar_trees[0].dimension
        number_of_elements = scalar_trees[0].number_of_leaves

        for tree in scalar_trees:
            for index in tree.tree_leaves:
                tree.compute_node_vertices(index)

        if vector_trees is not None:
            for tree in vector_trees:
                for index in tree.tree_leaves:
                    tree.compute_node_vertices(index)
        vertices_hashtable = self.compute_vertices_hashtable(scalar_trees[0])

        number_of_nodes = len(vertices_hashtable.keys())

        self.hdf_write_mesh(scalar_trees[0], vertices_hashtable)

        if dimension == 1:
            self.xmf_write_header(time, number_of_elements, number_of_nodes, topology_type="Polyline", nodes_per_element=2)

        elif dimension == 2:
            self.xmf_write_header(time, number_of_elements, number_of_nodes, topology_type="Quadrilateral", nodes_per_element=4)

        elif dimension == 3:
            self.xmf_write_header(time, number_of_elements, number_of_nodes, topology_type="Hexahedron", nodes_per_element=8)

        for tree in scalar_trees:
            self.xmf_write_scalar_attribute(tree.tag, number_of_elements)
            self.hdf_write_scalar_attribute(tree)

        if vector_trees is not None:
            self.xmf_write_vector_attribute(vector_attribute_name, number_of_elements, len(vector_trees))
            self.hdf_write_vector_attribute(vector_attribute_name, vector_trees)

        self.xmf_write_footer()

    def close(self):
        """...

        """

        self.xmf_write_main_footer()

