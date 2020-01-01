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
import mrpy.discretization.temporal_base as td
import mrpy.discretization.spatial as sd

#===============================================================================
#===============================================================================
#========================== INITIALISATION =====================================
#===============================================================================

output_module = importlib.import_module(cfg.output_module_name)
writer = output_module.OutputWriter("almgren_scalar")
writer_v = output_module.OutputWriter("almgren_scalar_v")
#writer_tp = output_module.OutputWriter("almgren_scalar_tp")
writer.initialize()
writer_v.initialize()
#writer_tp.initialize()

tree_velocity_x = mesh.create_new_tree(cfg.dimension, cfg.min_level,
        cfg.max_level, cfg.stencil_graduation, cfg.stencil_prediction, cfg.xmin,
        cfg.xmax, cfg.ymin, cfg.ymax, cfg.zmin, cfg.zmax)
tree_velocity_y = mesh.create_new_tree(cfg.dimension, cfg.min_level,
        cfg.max_level, cfg.stencil_graduation, cfg.stencil_prediction, cfg.xmin,
        cfg.xmax, cfg.ymin, cfg.ymax, cfg.zmin, cfg.zmax)
tree_pressure = mesh.create_new_tree(cfg.dimension, cfg.min_level,
        cfg.max_level, cfg.stencil_graduation, cfg.stencil_prediction, cfg.xmin,
        cfg.xmax, cfg.ymin, cfg.ymax, cfg.zmin, cfg.zmax)
tree_vorticity = mesh.create_new_tree(cfg.dimension, cfg.min_level,
        cfg.max_level, cfg.stencil_graduation, cfg.stencil_prediction, cfg.xmin,
        cfg.xmax, cfg.ymin, cfg.ymax, cfg.zmin, cfg.zmax)
tree_streamfunc = mesh.create_new_tree(cfg.dimension, cfg.min_level,
        cfg.max_level, cfg.stencil_graduation, cfg.stencil_prediction, cfg.xmin,
        cfg.xmax, cfg.ymin, cfg.ymax, cfg.zmin, cfg.zmax)
tree_scalar = mesh.create_new_tree(cfg.dimension, cfg.min_level_sc,
        cfg.max_level_sc, cfg.stencil_graduation, cfg.stencil_prediction, cfg.xmin,
        cfg.xmax, cfg.ymin, cfg.ymax, cfg.zmin, cfg.zmax)

tree_velocity_x.tag = "u"
tree_velocity_y.tag = "v"
tree_pressure.tag = "p"
tree_vorticity.tag = "omega"
tree_streamfunc.tag = "phi"
tree_scalar.tag = "s"

tree_velocity_x.bc = cfg.bc_dict[tree_velocity_x.tag]
tree_velocity_y.bc = cfg.bc_dict[tree_velocity_y.tag]
tree_pressure.bc = cfg.bc_dict[tree_pressure.tag]
tree_streamfunc.bc = cfg.bc_dict[tree_streamfunc.tag]
tree_scalar.bc = cfg.bc_dict[tree_scalar.tag]

mesh.listing_of_leaves(tree_velocity_x, tree_velocity_y, tree_pressure,
        tree_vorticity, tree_streamfunc, tree_scalar)

print("trees creation done")

time_integrator = importlib.import_module(cfg.class_scheme_name)
time_integrator = time_integrator.Scheme(tree_velocity_x=tree_velocity_x,
        tree_velocity_y=tree_velocity_y, tree_pressure=tree_pressure)

time_int_scalar = importlib.import_module(cfg.scalar_scheme)
#time_int_scalar = time_int_scalar.Scheme(tree_scalar=tree_scalar)
time_int_scalar = time_int_scalar.Scheme(tree_scalar=tree_scalar, diffusion=True)
time_int_scalar.dt = cfg.dt_sc

time_int_scalar.compute_initial_values(tree_scalar, cfg.sc_init)

time_integrator.setup_internal_variables(tree_velocity_x, tree_velocity_y, tree_pressure)
time_integrator.make_operators(tree_velocity_x, tree_velocity_y, tree_pressure)
time_int_scalar.make_operators(tree_scalar, tree_velocity_x, tree_velocity_y)
time_integrator.make_ksps()

#================ Computation of initial values from vorticity =================
sf_laplacian = sd.Operator(tree_streamfunc, 0, time_integrator.laplacian)
sf_dx = sd.Operator(tree_streamfunc, 0, time_integrator.divergence)
sf_dy = sd.Operator(tree_streamfunc, 1, time_integrator.divergence)

sd.finite_volume_interpolation(tree_vorticity, cfg.omega)
vorticity = sd.Scalar(tree_vorticity)
sf_rhs = sd.mul_num_scalar(-1, vorticity)
sf = time_integrator.solve(sf_laplacian, sf_rhs)
vx = sf_dy.apply(sf)
vy = sd.mul_num_scalar(-1, sf_dx.apply(sf))
time_integrator.scalar_to_tree(vx, tree_velocity_x)
time_integrator.scalar_to_tree(vy, tree_velocity_y)
sd.finite_volume_interpolation(tree_pressure, cfg.p_exact)

tree_vx = mesh.copy_tree(tree_velocity_x)
tree_vy = mesh.copy_tree(tree_velocity_y)
tree_vx.max_level = cfg.max_level_sc
tree_vx.min_level = cfg.min_level_sc
tree_vy.max_level = cfg.max_level_sc
tree_vy.min_level = cfg.min_level_sc
op.run_projection(tree_vx, tree_vy)
op.run_projection(tree_scalar)
op.encode_details(tree_scalar)
op.run_thresholding(tree_scalar, threshold_parameter=0.) # set all the keep_children flags of tree_scalar to True
op.set_to_same_grading(tree_scalar, tree_vx, tree_vy)
op.compute_missing_values(tree_vx, tree_vy)
#op.project_to_finest_grid(tree_vx, tree_vy)
#tree_vx.tree_nodes = tree_scalar.tree_nodes.copy()
#tree_vy.tree_nodes = tree_scalar.tree_nodes.copy()
#tree_vx.nchildren = tree_scalar.nchildren.copy()
#tree_vy.nchildren = tree_scalar.nchildren.copy()
#op.run_projection(tree_vx, tree_vy)
mesh.listing_of_leaves(tree_vx, tree_vy)

print("time integrator initiation done")

nsp = petsc.NullSpace().create(constant=True)
#nsp = None

print("2D almgren four-vortices")
print("  Grid informations :")
print("    - t_end = " +  repr(cfg.t_end))
print("    - nt = " +  repr(cfg.nt))
print("    - L = " +  repr(cfg.L))
print("    - dx = " +  repr((cfg.xmax-cfg.xmin) / 2**(cfg.max_level)))
print("    - dy = " +  repr((cfg.ymax-cfg.ymin) / 2**(cfg.max_level)))
print("")

#===============================================================================
#========================== COMPUTATION LOOP ===================================
#===============================================================================

t = cfg.t_ini
t_sc = cfg.t_ini
first_adapt = False

#Used for the printing of the solutions
t_print = 0.

for it in range(int(cfg.nt)):
    t_previous = t
    t = time_integrator.next_time(t)

    print("t = " + repr(t))
    print("")

    if (it != 0) and (it % int(cfg.mr_freq) == 0):
        first_adapt = True
        #print("projection beginning")
        op.run_projection(tree_velocity_x, tree_velocity_y, tree_pressure)
        op.run_projection(tree_scalar)
        #print("projection end")
        #print("details beginning")
        op.encode_details(tree_velocity_x, tree_velocity_y, tree_pressure,
                tree_scalar)
        #print("details end")

        if tree_velocity_x.max_norm_details != 0 and tree_velocity_y.max_norm_details != 0:
        #if tree_velocity_x.max_norm_details != 0 and \
            #print("thresholding beginning")
            op.run_thresholding(tree_velocity_x, tree_velocity_y)
            #op.run_thresholding(tree_velocity_x, tree_velocity_y, tree_scalar)
            op.run_thresholding(tree_scalar)

            #print("grading beginning")
            op.run_grading(tree_velocity_x, tree_velocity_y)
            #op.run_grading(tree_velocity_x, tree_velocity_y, tree_scalar)
            op.run_grading(tree_scalar)
            #op.run_grading(tree_velocity_x)
            #print("grading end")

            op.set_to_same_grading(tree_velocity_x, tree_pressure,
                    )
            #op.set_to_same_grading(tree_velocity_x, tree_velocity_y, tree_pressure)
            op.compute_missing_values(tree_pressure)
            #op.compute_missing_values(tree_velocity_y, tree_pressure)

            op.set_to_same_grading(tree_scalar, tree_vx, tree_vy)
            op.compute_missing_values(tree_vx, tree_vy)
            op.run_pruning(tree_scalar, tree_vx, tree_vy)

            op.run_pruning(tree_velocity_x, tree_velocity_y, tree_pressure,
            )

            print("listing of leaves beginning")
            mesh.listing_of_leaves(tree_velocity_x, tree_velocity_y, tree_pressure,
                    tree_scalar, tree_vx, tree_vy)

            time_integrator.setup_internal_variables(tree_velocity_x, tree_velocity_y, tree_pressure)
            time_integrator.make_operators(tree_velocity_x, tree_velocity_y, tree_pressure)
            time_int_scalar.make_operators(tree_scalar, tree_velocity_x, tree_velocity_y)
            time_integrator.make_ksps()

    print("Number of leaves: " + repr(tree_pressure.number_of_leaves))
    print("Number of leaves tree_vx: " + repr(tree_vx.number_of_leaves))
    print("Number of leaves tree_scalar: " + repr(tree_scalar.number_of_leaves))
    print("")

    count = time_integrator.dt
    while (count > 0):
        t_sc = time_int_scalar.next_time(t)
        time_int_scalar.advance(tree_scalar, tree_vx, tree_vy)
        count -= time_int_scalar.dt

    time_integrator.advance(v_x=tree_velocity_x, v_y=tree_velocity_y, p=tree_pressure,
                            t_ini=t_previous, nsp=nsp)

    tree_vx = mesh.copy_tree(tree_velocity_x)
    tree_vy = mesh.copy_tree(tree_velocity_y)
    tree_vx.max_level = cfg.max_level_sc
    tree_vx.min_level = cfg.min_level_sc
    tree_vy.max_level = cfg.max_level_sc
    tree_vy.min_level = cfg.min_level_sc
    if (first_adapt):
        op.run_projection(tree_vx, tree_vy)
        #op.project_to_finest_grid(tree_vx, tree_vy)
        op.set_to_same_grading(tree_scalar, tree_vx, tree_vy)
        op.compute_missing_values(tree_vx, tree_vy)
        op.run_pruning(tree_vx, tree_vy)
    else:
        op.run_projection(tree_vx, tree_vy)
        #op.run_thresholding(tree_scalar, threshold_parameter=0.) # set all the keep_children flags of tree_scalar to True
        op.set_to_same_grading(tree_scalar, tree_vx, tree_vy)
        op.compute_missing_values(tree_vx, tree_vy)
        op.run_pruning(tree_vx, tree_vy)
        #tree_vx.tree_nodes = tree_scalar.tree_nodes.copy()
        #tree_vy.tree_nodes = tree_scalar.tree_nodes.copy()
    mesh.listing_of_leaves(tree_vx, tree_vy)

    temp = time_integrator.velocity_norm_l2(tree_velocity_x, tree_velocity_y)

    print("L2 Norm of the velocity field: ")
    print(temp)
    print("")

#============================= Printing solutions ==============================
    if t >= t_print:

        time_integrator.make_vorticity_x(tree_velocity_y)
        time_integrator.make_vorticity_y(tree_velocity_x)

        velocity_x = sd.Scalar(tree_velocity_x)
        velocity_y = sd.Scalar(tree_velocity_y)
        vorticity = sd.Scalar(tree_velocity_x) # we copy a vorticity scalar from the velocity component
        vorticity.sc = (sd.add_scalars(
            time_integrator.vorticity_x.apply(velocity_y),
            sd.mul_num_scalar(-1., time_integrator.vorticity_y.apply(velocity_x)))).sc.copy()

        vorticity = time_integrator.velocity_inverse_mass.apply(vorticity)

        if int((it / cfg.mr_freq)) != 0:
            op.set_to_same_grading(tree_velocity_x, tree_vorticity)
            op.run_pruning(tree_vorticity)
            mesh.listing_of_leaves(tree_vorticity)

        #print(tree_vorticity.tree_leaves)
        #print(tree_velocity_x.nvalue)
        #quit()
        for index in range(tree_velocity_x.number_of_leaves):
            tree_vorticity.nvalue[tree_vorticity.tree_leaves[index]] = vorticity.sc[index]

        writer.write([tree_scalar, tree_vx, tree_vy],
                output_file_name="almgren_scalar_t_" + repr(it).zfill(5),
                time=t)

        #temp = mesh.copy_tree(tree_velocity_x)
        #op.run_projection(temp)
        #temp.tag = "tp"
        #temp.max_level = cfg.max_level_sc
        #op.project_to_finest_grid(temp)
        #op.set_to_same_grading(tree_scalar, temp)
        ##op.compute_missing_values(tree_vx, tree_vy)
        #op.run_pruning(temp)
        #mesh.listing_of_leaves(temp)
        writer_v.write([tree_velocity_x, tree_velocity_y, tree_pressure, tree_vorticity],
                output_file_name="almgren_scalar_v_t_" + repr(it).zfill(5),
                time=t)

        #writer_tp.write([tree_vx, tree_vy],
        #        output_file_name="almgren_scalar_tp_t_" + repr(it).zfill(5),
        #        time=t)

        t_print = t_print + cfg.dt_print

#===============================================================================
#============================ TERMINATION ======================================
#===============================================================================

writer.close()
writer_v.close()
#writer_tp.close()
