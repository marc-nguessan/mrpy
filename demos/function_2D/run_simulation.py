from __future__ import print_function, division

"""...

"""
import importlib

import config as cfg
from mrpy.mr_utils import mesh
from mrpy.mr_utils import op
import mrpy.discretization.spatial as sd

output_module = importlib.import_module(cfg.output_module_name)
writer = output_module.OutputWriter("gaussian")
writer.initialize()

tree = mesh.create_new_tree(cfg.dimension, cfg.min_level, cfg.max_level,
        cfg.stencil_graduation, cfg.stencil_prediction, cfg.xmin, cfg.xmax,
        cfg.ymin, cfg.ymax, cfg.zmin, cfg.zmax)

tree.tag = "u"

tree.bc = cfg.bc_dict[tree.tag]

mesh.listing_of_leaves(tree)

print(tree.number_of_leaves)
print("")

for index in tree.tree_leaves:
    tree.nvalue[index] = cfg.function(tree.ncoord_x[index], tree.ncoord_y[index])

writer.write([tree], output_file_name="gaussian_finest", time=0)

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

writer.write([tree], output_file_name="gaussian_adapted", time=1)
writer.close()

#print("spatial operators creation begin")
#div_x = sd.Operator(tree, 0, cfg.divergence_module_name)
#print("divergence axis 0 done")
#div_y = sd.Operator(tree, 1, cfg.divergence_module_name)
#print("divergence axis 1 done")
#
#grad_x = sd.Operator(tree, 0, cfg.gradient_module_name)
#print("gradient axis 0 done")
#grad_y = sd.Operator(tree, 1, cfg.gradient_module_name)
#print("gradient axis 1 done")
#
#lap_x = sd.Operator(tree, 0, cfg.laplacian_module_name)
#print("laplacian axis 0 done")
#lap_y = sd.Operator(tree, 1, cfg.laplacian_module_name)
#print("laplacian axis 1 done")
#divgrad = sd.add_operators(
#sd.mul_operators(grad_x, div_x),
#sd.mul_operators(grad_y, div_y))
#print("div_grad done")
#print("spatial operators creation end")

#print("Is the laplacian symmetric?: " + repr(lap_x.matrix.isSymmetric()))
#lap_x.matrix.view()

#lap_transpose = sd.Operator(tree, 0, cfg.laplacian_module_name)
#lap_x.matrix.transpose(lap_transpose.matrix)
#print("Transpose of the laplacian")
#lap_transpose.matrix.view()
#
#print("Is the div-grad symmetric?: " + repr(divgrad.matrix.isSymmetric()))
##divgrad.matrix.view()
#
#divgrad_transpose = sd.Operator(tree, 0, cfg.laplacian_module_name)
#divgrad.matrix.transpose(divgrad_transpose.matrix)
#print("Transpose of the div-grad")
#divgrad_transpose.matrix.view()
