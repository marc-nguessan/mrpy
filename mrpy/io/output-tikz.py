"""...

"""

import config as cfg

class OutputWriter(object):
    """...

    """

    def __init__(self):

        pass

    def write(self, tree, output_file_name="test"):
        """...

        """

        with open(output_file_name + ".dat", 'w') as f:
            f.write('# coord_x value\n')

            for index in tree.tree_leaves:
                f.write('{0} {1}\n'.format(
                    tree.ncoord_x[index] - tree.ndx[index]/2.,
                    tree.nvalue[index]))
                #f.write('{0} {1}\n'.format(
                #    tree.ncoord_x[index],
                #    tree.nvalue[index]))
                f.write('{0} {1}\n'.format(
                    tree.ncoord_x[index] + tree.ndx[index]/2.,
                    tree.nvalue[index]))

    def sort_file(self, output_file_name="test"):

        with open(output_file_name + ".dat") as f:
            sorted_file = sorted(f)

        with open(output_file_name + "_sorted.dat", 'w') as f:
            f.writelines(sorted_file)
