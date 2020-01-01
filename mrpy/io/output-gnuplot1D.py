"""...

"""

import config as cfg

def write(tree, output_file_name="test.dat"):
    """...

    """

    with open(output_file_name, 'w') as f:
        f.write('# coord_x dx level value\n')

        for index in tree.tree_leaves:
            f.write('{0} {1} {2} {3}\n'.format(
                tree.ncoord_x[index],
                tree.ndx[index],
                tree.nlevel[index],
                tree.nvalue[index]))
