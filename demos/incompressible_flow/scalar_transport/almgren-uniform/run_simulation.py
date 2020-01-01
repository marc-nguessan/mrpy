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

import time

#===============================================================================
#===============================================================================
#========================== INITIALISATION =====================================
#===============================================================================

t1 = time.time()
t5 = time.time()

#output_module = importlib.import_module(cfg.output_module_name)
#writer = output_module.OutputWriter("almgren_scalar")
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
tree_vorticity = mesh.create_new_tree(cfg.dimension, cfg.min_level,
        cfg.max_level, cfg.stencil_graduation, cfg.stencil_prediction, cfg.xmin,
        cfg.xmax, cfg.ymin, cfg.ymax, cfg.zmin, cfg.zmax)
tree_streamfunc = mesh.create_new_tree(cfg.dimension, cfg.min_level,
        cfg.max_level, cfg.stencil_graduation, cfg.stencil_prediction, cfg.xmin,
        cfg.xmax, cfg.ymin, cfg.ymax, cfg.zmin, cfg.zmax)
tree_scalar = mesh.create_new_tree(cfg.dimension, cfg.min_level,
        cfg.max_level, cfg.stencil_graduation, cfg.stencil_prediction, cfg.xmin,
        cfg.xmax, cfg.ymin, cfg.ymax, cfg.zmin, cfg.zmax)

#tree_source_term_velocity_x = mesh.create_new_tree(cfg.dimension, cfg.min_level,
#        cfg.max_level, cfg.stencil_graduation, cfg.stencil_prediction, cfg.xmin,
#        cfg.xmax, cfg.ymin, cfg.ymax, cfg.zmin, cfg.zmax)
#tree_source_term_velocity_y = mesh.create_new_tree(cfg.dimension, cfg.min_level,
#        cfg.max_level, cfg.stencil_graduation, cfg.stencil_prediction, cfg.xmin,
#        cfg.xmax, cfg.ymin, cfg.ymax, cfg.zmin, cfg.zmax)

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
        #tree_source_term_velocity_x, tree_source_term_velocity_y,
        tree_vorticity, tree_streamfunc, tree_scalar)

print("trees creation done")

time_integrator = importlib.import_module(cfg.class_scheme_name)
time_integrator = time_integrator.Scheme(tree_velocity_x=tree_velocity_x,
        tree_velocity_y=tree_velocity_y, tree_pressure=tree_pressure)
time_integrator.uniform = True

time_int_scalar = importlib.import_module(cfg.scalar_scheme)
#time_int_scalar = time_int_scalar.Scheme(tree_scalar=tree_scalar)
time_int_scalar = time_int_scalar.Scheme(tree_scalar=tree_scalar, diffusion=True)
time_int_scalar.dt = cfg.dt_sc
time_int_scalar.uniform = True

sd.finite_volume_interpolation(tree_scalar, cfg.sc_init)

time_integrator.setup_internal_variables(tree_velocity_x, tree_velocity_y, tree_pressure)
time_integrator.make_operators(tree_velocity_x, tree_velocity_y, tree_pressure)
time_int_scalar.make_operators(tree_scalar, tree_velocity_x, tree_velocity_y)
time_integrator.make_ksps()

time_integrator.make_vorticity_x(tree_velocity_y)
time_integrator.make_vorticity_y(tree_velocity_x)

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

t6 = time.time()

temp = (t6 - t5)

with open("cpu_time.txt", 'a') as f:
    f.write('{0} {1}\n'.format(
        -1,
        temp))

#===============================================================================
#========================== COMPUTATION LOOP ===================================
#===============================================================================

t = cfg.t_ini
t_sc = cfg.t_ini

#Used for the printing of the solutions
t_print = 0.

v_x = sd.Scalar(tree_velocity_x)
v_y = sd.Scalar(tree_velocity_y)
p = sd.Scalar(tree_pressure)
s = sd.Scalar(tree_scalar)

for it in range(int(cfg.nt)):
    t3 = time.time()

    t_previous = t
    t = time_integrator.next_time(t)

    print("t = " + repr(t))
    print("")

    count = time_integrator.dt
    while (count > 0):
        t_sc = time_int_scalar.next_time(t)
        time_int_scalar.advance(s=s, v_x=v_x, v_y=v_y,
            t_ini=0, nsp=None)
        count -= time_int_scalar.dt

    time_integrator.advance(v_x=v_x, v_y=v_y, p=p,
                            t_ini=t_previous, nsp=nsp)

    t4 = time.time()

    temp = (t4 - t3)

    with open("cpu_time.txt", 'a') as f:
        f.write('{0} {1}\n'.format(
            it,
            temp))

    #temp = time_integrator.velocity_norm_l2(tree_velocity_x, tree_velocity_y)

    #print("L2 Norm of the velocity field: ")
    #print(temp)
    #print("")

#============================= Printing solutions ==============================
#    if t >= t_print:
#
#        vorticity.sc = (sd.add_scalars(
#            time_integrator.vorticity_x.apply(v_y),
#            sd.mul_num_scalar(-1., time_integrator.vorticity_y.apply(v_x)))).sc.copy()
#
#        vorticity = time_integrator.velocity_inverse_mass.apply(vorticity)
#
#        time_integrator.scalar_to_tree(v_x, tree_velocity_x)
#        time_integrator.scalar_to_tree(v_y, tree_velocity_y)
#        time_integrator.scalar_to_tree(p, tree_pressure)
#        time_integrator.scalar_to_tree(s, tree_scalar)
#        time_integrator.scalar_to_tree(vorticity, tree_vorticity)
#
#        #for index in range(tree_velocity_x.number_of_leaves):
#        #    tree_vorticity.nvalue[tree_vorticity.tree_leaves[index]] = vorticity.sc[index]
#
#        writer.write([tree_velocity_x, tree_velocity_y, tree_pressure, tree_scalar, tree_vorticity],
#                output_file_name="almgren_scalar_t_" + repr(it).zfill(5),
#                time=t)
#
#        t_print = t_print + cfg.dt_print

##========================= Printing last solution ==============================
#vorticity.sc = (sd.add_scalars(
#    time_integrator.vorticity_x.apply(v_y),
#    sd.mul_num_scalar(-1., time_integrator.vorticity_y.apply(v_x)))).sc.copy()
#
#vorticity = time_integrator.velocity_inverse_mass.apply(vorticity)
#
#time_integrator.scalar_to_tree(v_x, tree_velocity_x)
#time_integrator.scalar_to_tree(v_y, tree_velocity_y)
#time_integrator.scalar_to_tree(p, tree_pressure)
#time_integrator.scalar_to_tree(s, tree_scalar)
#time_integrator.scalar_to_tree(vorticity, tree_vorticity)
#
##for index in range(tree_velocity_x.number_of_leaves):
##    tree_vorticity.nvalue[tree_vorticity.tree_leaves[index]] = vorticity.sc[index]
#
#writer.write([tree_velocity_x, tree_velocity_y, tree_pressure, tree_scalar, tree_vorticity],
#        output_file_name="almgren_scalar_t_" + repr(it).zfill(5),
#        time=t)

#===============================================================================
#============================ TERMINATION ======================================
#===============================================================================

t2 = time.time()

temp = (t2 - t1)

print("Computation time:")
print(t2 - t1)

with open("cpu_time.txt", 'a') as f:
    f.write('{0} {1}\n'.format(
        200,
        temp))

#writer.close()
