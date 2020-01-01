"""...

"""

import config as cfg

def write(tree, output_file_name="test.dat"):
    """...

    """

    with open(output_file_name, 'w') as f:
        f.write('# coord_x coord_y dx dy level value\n')

        for index in tree.tree_leaves:
            f.write('{0} {1} {2} {3} {4} {5}\n'.format(
                tree.ncoord_x[index],
                tree.ncoord_y[index],
                tree.ndx[index],
                tree.ndy[index],
                tree.nlevel[index],
                tree.nvalue[index]))
