from __future__ import print_function, division

"""...

"""

import importlib

import config as cfg
from mrpy.mr_utils import mesh
from mrpy.mr_utils import op

output_module = importlib.import_module(cfg.output_module_name)
writer = output_module.OutputWriter()

tree = mesh.create_new_tree(cfg.dimension, cfg.min_level, cfg.max_level,
        cfg.stencil_graduation, cfg.stencil_prediction, cfg.xmin, cfg.xmax)

tree.tag = "u"

tree.bc = cfg.bc_dict[tree.tag]

mesh.listing_of_leaves(tree)

print(tree.number_of_leaves)
print("")

for index in tree.tree_leaves:
    tree.nvalue[index] = cfg.function(tree.ncoord_x[index])

writer.write(tree, output_file_name="finest_grid")
writer.sort_file(output_file_name="finest_grid")

op.run_projection(tree)

op.encode_details(tree)

#for index in tree.tree_nodes.keys():
#    print index, tree.nindex_x[index], tree.ndx[index], tree.ncoord_x[index], tree.nvalue[index]
#    if not tree.nisleaf[index]:
#        print tree.nchildren[index]
#        print tree.ndetails[index]
#        print tree.nnorm_details[index]
#    print ""

op.run_thresholding(tree)

op.run_grading(tree)

op.run_pruning(tree)

mesh.listing_of_leaves(tree)

op.encode_details(tree)

print(tree.number_of_leaves)
print("")

writer.write(tree, output_file_name="adapted_grid")
writer.sort_file(output_file_name="adapted_grid")
