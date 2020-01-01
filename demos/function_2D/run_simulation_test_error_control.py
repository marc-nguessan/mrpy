from __future__ import print_function, division

"""...

"""

import importlib

import config as cfg
from mrpy.mr_utils import mesh
from mrpy.mr_utils import op
import mrpy.discretization.temporal as td
import mrpy.discretization.spatial as sd

output_module = importlib.import_module(cfg.output_module_name)
#writer = output_module.OutputWriter("gaussian")
writer = output_module.OutputWriter("stokes")
writer.initialize()

tree_finest = mesh.create_new_tree(cfg.dimension, cfg.min_level, cfg.max_level,
        cfg.stencil_graduation, cfg.stencil_prediction, cfg.xmin, cfg.xmax,
        cfg.ymin, cfg.ymax, cfg.zmin, cfg.zmax)
tree_adapted = mesh.create_new_tree(cfg.dimension, cfg.min_level, cfg.max_level,
        cfg.stencil_graduation, cfg.stencil_prediction, cfg.xmin, cfg.xmax,
        cfg.ymin, cfg.ymax, cfg.zmin, cfg.zmax)

tree_finest.tag = "u"
tree_adapted.tag = "u"

tree_finest.bc = cfg.bc_dict[tree_finest.tag]
tree_adapted.bc = cfg.bc_dict[tree_adapted.tag]

mesh.listing_of_leaves(tree_finest)
mesh.listing_of_leaves(tree_adapted)

print(tree_finest.number_of_leaves)
print("")

for index in tree_finest.tree_leaves:
    tree_finest.nvalue[index] = cfg.function(tree_finest.ncoord_x[index], tree_finest.ncoord_y[index])
    tree_adapted.nvalue[index] = cfg.function(tree_adapted.ncoord_x[index], tree_adapted.ncoord_y[index])

#writer.write([tree_finest], output_file_name="gaussian_finest", time=0)
writer.write([tree_finest], output_file_name="stokes-finest", time=0)

op.run_projection(tree_adapted)

op.encode_details(tree_adapted)

#for index in tree_adapted.tree_nodes.keys():
#    print(index, tree_adapted.nindex_x[index], tree_adapted.ndx[index], tree_adapted.ncoord_x[index], tree_adapted.nvalue[index])
#    if not tree.nisleaf[index]:
#        print tree.nchildren[index]
#        print tree.ndetails[index]
#        print tree.nnorm_details[index]
#    print ""

op.run_thresholding(tree_adapted)

op.run_grading(tree_adapted)

op.run_pruning(tree_adapted)

mesh.listing_of_leaves(tree_adapted)

print(tree_adapted.number_of_leaves)
print("")

writer.write([tree_adapted], output_file_name="stokes-adapted", time=0)

op.project_to_finest_grid(tree_adapted)

mesh.listing_of_leaves(tree_adapted)

print(tree_adapted.number_of_leaves)
print("")

print(op.global_error_to_finest_grid(tree_adapted, tree_finest))

#writer.write([tree], output_file_name="gaussian_adapted", time=1)
#writer.close()
