"""...

"""
import sys, petsc4py
petsc4py.init(sys.argv)
import petsc4py.PETSc as petsc
import mpi4py.MPI as mpi
import numpy as np
import math
import importlib

import config as cfg
from mrpy.mr_utils import mesh
from mrpy.mr_utils import op
import mrpy.discretization.temporal as td
import mrpy.discretization.spatial as sd

#===============================================================================
#===============================================================================
#========================== INITIALISATION =====================================
#===============================================================================

#output_module = importlib.import_module(cfg.output_module_name)
#writer = output_module.OutputWriter("lid_driven_cavity")
#writer.initialize()

tree_exact_velocity_x = mesh.create_new_tree(cfg.dimension, cfg.min_level,
        cfg.max_level, cfg.stencil_graduation, cfg.stencil_prediction, cfg.xmin,
        cfg.xmax, cfg.ymin, cfg.ymax, cfg.zmin, cfg.zmax)
tree_exact_velocity_y = mesh.create_new_tree(cfg.dimension, cfg.min_level,
        cfg.max_level, cfg.stencil_graduation, cfg.stencil_prediction, cfg.xmin,
        cfg.xmax, cfg.ymin, cfg.ymax, cfg.zmin, cfg.zmax)
tree_exact_pressure = mesh.create_new_tree(cfg.dimension, cfg.min_level,
        cfg.max_level, cfg.stencil_graduation, cfg.stencil_prediction, cfg.xmin,
        cfg.xmax, cfg.ymin, cfg.ymax, cfg.zmin, cfg.zmax)

tree_exact_velocity_x.tag = "u"
tree_exact_velocity_y.tag = "v"
tree_exact_pressure.tag = "p"

tree_exact_velocity_x.bc = cfg.bc_dict[tree_exact_velocity_x.tag]
tree_exact_velocity_y.bc = cfg.bc_dict[tree_exact_velocity_y.tag]
tree_exact_pressure.bc = cfg.bc_dict[tree_exact_pressure.tag]

mesh.listing_of_leaves(tree_exact_velocity_x, tree_exact_velocity_y, tree_exact_pressure,
        )

print("trees creation done")

time_integrator = importlib.import_module(cfg.class_scheme_name)
time_integrator = time_integrator.Scheme(tree_velocity_x=tree_exact_velocity_x,
        tree_velocity_y=tree_exact_velocity_y, tree_pressure=tree_exact_pressure)
time_integrator.uniform = True

time_integrator.compute_initial_values(tree_velocity_x=tree_exact_velocity_x,
        tree_velocity_y=tree_exact_velocity_y, tree_pressure=tree_exact_pressure)

time_integrator.setup_internal_variables(tree_exact_velocity_x, tree_exact_velocity_y, tree_exact_pressure)
time_integrator.make_operators(tree_exact_velocity_x, tree_exact_velocity_y, tree_exact_pressure)
time_integrator.make_ksps()

print("time integrator initiation done")

nsp = petsc.NullSpace().create(constant=True)
#nsp = None

print("2D lid-driven cavity")
print("  Grid informations :")
print("    - dt = " +  repr(cfg.dt))
print("    - dx = " +  repr((cfg.xmax-cfg.xmin) / 2**(cfg.max_level)))
print("    - dy = " +  repr((cfg.ymax-cfg.ymin) / 2**(cfg.max_level)))
print("")

#===============================================================================
#========================== COMPUTATION LOOP ===================================
#===============================================================================

t = cfg.t_ini

#Used for the printing of the solutions
t_print = 0.

v_x = sd.Scalar(tree_exact_velocity_x)
v_y = sd.Scalar(tree_exact_velocity_y)
p = sd.Scalar(tree_exact_pressure)

for it in range(int(cfg.nt)):
    t_previous = t
    t = time_integrator.next_time(t)

    print("t = " + repr(t))
    print("")

    time_integrator.advance(v_x=v_x, v_y=v_y, p=p,
                                t_ini=t_previous, nsp=nsp)

time_integrator.scalar_to_tree(v_x, tree_exact_velocity_x)
time_integrator.scalar_to_tree(v_y, tree_exact_velocity_y)
time_integrator.scalar_to_tree(p, tree_exact_pressure)

# We apply the MR algortihm to the exact solution and re-project it to the most
# refined grid
#op.run_projection(tree_exact_velocity_x, tree_exact_velocity_y, tree_exact_pressure)
#op.encode_details(tree_exact_velocity_x, tree_exact_velocity_y, tree_exact_pressure)
#
#op.run_thresholding(tree_exact_velocity_x, tree_exact_velocity_y)
#op.run_grading(tree_exact_velocity_x, tree_exact_velocity_y)
#
#op.set_to_same_grading(tree_exact_velocity_x, tree_exact_pressure)
#op.compute_missing_values(tree_exact_pressure)
#
#op.run_pruning(tree_exact_velocity_x, tree_exact_velocity_y, tree_exact_pressure)
#
#mesh.listing_of_leaves(tree_exact_velocity_x, tree_exact_velocity_y, tree_exact_pressure)
#
#op.project_to_finest_grid(tree_exact_velocity_x, tree_exact_velocity_y, tree_exact_pressure)
#
#mesh.listing_of_leaves(tree_exact_velocity_x, tree_exact_velocity_y, tree_exact_pressure)

def run(threshold_parameter):
    """Runs a specific configuration given the threshold_parameter."""

#===============================================================================
#===============================================================================
#========================== INITIALISATION =====================================
#===============================================================================

    #output_module = importlib.import_module(cfg.output_module_name)
    #writer = output_module.OutputWriter("lid_driven_cavity")
    #writer.initialize()

    tree_velocity_x = mesh.create_new_tree(cfg.dimension, cfg.min_level,
            cfg.max_level, cfg.stencil_graduation, cfg.stencil_prediction, cfg.xmin,
            cfg.xmax, cfg.ymin, cfg.ymax, cfg.zmin, cfg.zmax)
    tree_velocity_y = mesh.create_new_tree(cfg.dimension, cfg.min_level,
            cfg.max_level, cfg.stencil_graduation, cfg.stencil_prediction, cfg.xmin,
            cfg.xmax, cfg.ymin, cfg.ymax, cfg.zmin, cfg.zmax)
    tree_pressure = mesh.create_new_tree(cfg.dimension, cfg.min_level,
            cfg.max_level, cfg.stencil_graduation, cfg.stencil_prediction, cfg.xmin,
            cfg.xmax, cfg.ymin, cfg.ymax, cfg.zmin, cfg.zmax)

    tree_velocity_x.tag = "u"
    tree_velocity_y.tag = "v"
    tree_pressure.tag = "p"

    tree_velocity_x.bc = cfg.bc_dict[tree_velocity_x.tag]
    tree_velocity_y.bc = cfg.bc_dict[tree_velocity_y.tag]
    tree_pressure.bc = cfg.bc_dict[tree_pressure.tag]

    mesh.listing_of_leaves(tree_velocity_x, tree_velocity_y, tree_pressure)

    print("trees creation done")

    time_integrator = importlib.import_module(cfg.class_scheme_name)
    time_integrator = time_integrator.Scheme(tree_velocity_x=tree_velocity_x,
            tree_velocity_y=tree_velocity_y, tree_pressure=tree_pressure)

    time_integrator.compute_initial_values(tree_velocity_x=tree_velocity_x,
            tree_velocity_y=tree_velocity_y, tree_pressure=tree_pressure)

    time_integrator.setup_internal_variables(tree_velocity_x, tree_velocity_y, tree_pressure)
    time_integrator.make_operators(tree_velocity_x, tree_velocity_y, tree_pressure)
    time_integrator.make_ksps()

    print("time integrator initiation done")

    nsp = petsc.NullSpace().create(constant=True)

#===============================================================================
#========================== COMPUTATION LOOP ===================================
#===============================================================================

    t = cfg.t_ini

#Used for the printing of the solutions
    t_print = 0.

    for it in range(int(cfg.nt)):
        t_previous = t
        t = time_integrator.next_time(t)

        print("t = " + repr(t))
        print("")

        if (it != 0) and (it % int(cfg.mr_freq) == 0):
            #print("projection beginning")
            op.run_projection(tree_velocity_x, tree_velocity_y, tree_pressure)
            #print("projection end")
            #print("details beginning")
            op.encode_details(tree_velocity_x, tree_velocity_y, tree_pressure)
            #print("details end")

            if tree_velocity_x.max_norm_details != 0 and tree_velocity_y.max_norm_details != 0:
            #if tree_velocity_x.max_norm_details != 0 and \
                #print("thresholding beginning")
                op.run_thresholding(tree_velocity_x, tree_velocity_y, threshold_parameter=threshold_parameter)
                #op.run_thresholding(tree_velocity_x)
                #print("thresholding end")
                #op.run_thresholding(tree_velocity_x, tree_velocity_y, tree_pressure)

                #for k, v in tree_velocity_x.tree_nodes.items():
                #    print(k, v)

                #print("grading beginning")
                op.run_grading(tree_velocity_x, tree_velocity_y)
                #op.run_grading(tree_velocity_x)
                #print("grading end")

                op.set_to_same_grading(tree_velocity_x, tree_pressure)
                op.compute_missing_values(tree_pressure)
                #op.compute_missing_values(tree_velocity_y, tree_pressure)

                op.run_pruning(tree_velocity_x, tree_velocity_y, tree_pressure)

                print("listing of leaves beginning")
                mesh.listing_of_leaves(tree_velocity_x, tree_velocity_y, tree_pressure)

                time_integrator.setup_internal_variables(tree_velocity_x, tree_velocity_y, tree_pressure)
                time_integrator.make_operators(tree_velocity_x, tree_velocity_y, tree_pressure)
                time_integrator.make_ksps()

        print("Number of leaves: " + repr(tree_pressure.number_of_leaves))
        print("")

        time_integrator.advance(v_x=tree_velocity_x, v_y=tree_velocity_y, p=tree_pressure,
                            t_ini=t_previous, nsp=nsp)

        temp = time_integrator.velocity_norm_l2(tree_velocity_x, tree_velocity_y)

        print("    - eps = " +  repr(threshold_parameter))
        print("L2 Norm of the velocity field: ")
        print(temp)
        print("")

#============================= Printing solutions ==============================
        #if t >= t_print:

        #    writer.write([tree_velocity_x, tree_velocity_y, tree_pressure],
        #            output_file_name="lid_driven_cavity_t_" + repr(it).zfill(5),
        #            time=t)

        #    t_print = t_print + cfg.dt_print

#===============================================================================
#===============================================================================
#============ COMPUTING GLOBAL ERROR (SHOULD BE DONE ELSEWHERE !!!) ============
#===============================================================================

    op.project_to_finest_grid(tree_velocity_x, tree_velocity_y, tree_pressure)

    mesh.listing_of_leaves(tree_velocity_x, tree_velocity_y, tree_pressure)

    global_error_vx = op.global_error_to_finest_grid(tree_velocity_x, tree_exact_velocity_x)
    global_error_vy = op.global_error_to_finest_grid(tree_velocity_y, tree_exact_velocity_y)
    global_error_p = op.global_error_to_finest_grid(tree_pressure, tree_exact_pressure)

    with open("norm_error_to_exact_solution_eps.txt", 'a') as f:
        f.write('{0:.5f} {1} {2} {3}\n'.format(
            threshold_parameter,
            global_error_vx,
            global_error_vy,
            global_error_p))
        #f.write('# dt ||u - u_exact|| ||v - v_exact|| ||p - p_exact||\n')

#===============================================================================
#============================ TERMINATION ======================================
#===============================================================================

    #writer.close()

threshold_parameters = [3.e-2, 1.e-2, 7.e-3, 3.e-3, 1.e-3, 7.e-4, 3.e-4, 1.e-4]

for eps in threshold_parameters:
    run(eps)

