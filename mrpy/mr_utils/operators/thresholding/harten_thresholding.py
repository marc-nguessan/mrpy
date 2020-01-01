from __future__ import print_function, division

"""...

"""

from six.moves import range
from mrpy.mr_utils import mesh

def compute_level_threshold_parameter(tree, index, threshold_parameter):
    """...

    """

    return threshold_parameter*2**(tree.dimension*(tree.nlevel[index] - tree.max_level))

def run_thresholding(threshold_parameter, *trees):
    """...

    """

    tree_indexes = trees[0].tree_nodes.keys()
    dimension = trees[0].dimension
    max_level = trees[0].max_level

    for index in list(tree_indexes):

        keep_children = False

        for tree in trees:

            #if not tree.nisleaf[index]:
            if (tree.nisleaf[index] is False) and (tree.max_norm_details != 0.):
                if (abs(tree.nnorm_details[index]/tree.max_norm_details) >= \
                        compute_level_threshold_parameter(tree, index, threshold_parameter) \
                        or tree.nlevel[index] <= tree.min_level):
                    keep_children = True

        for tree in trees:
            if not tree.nisleaf[index]:
                tree.nkeep_children[index] = keep_children

