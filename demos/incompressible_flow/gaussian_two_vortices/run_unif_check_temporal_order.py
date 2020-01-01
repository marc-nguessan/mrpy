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

tree_exact_velocity_x = mesh.create_new_tree(cfg.dimension, cfg.min_level,
        cfg.max_level, cfg.stencil_graduation, cfg.stencil_prediction, cfg.xmin,
        cfg.xmax, cfg.ymin, cfg.ymax, cfg.zmin, cfg.zmax)
tree_exact_velocity_y = mesh.create_new_tree(cfg.dimension, cfg.min_level,
        cfg.max_level, cfg.stencil_graduation, cfg.stencil_prediction, cfg.xmin,
        cfg.xmax, cfg.ymin, cfg.ymax, cfg.zmin, cfg.zmax)
tree_exact_pressure = mesh.create_new_tree(cfg.dimension, cfg.min_level,
        cfg.max_level, cfg.stencil_graduation, cfg.stencil_prediction, cfg.xmin,
        cfg.xmax, cfg.ymin, cfg.ymax, cfg.zmin, cfg.zmax)
tree_exact_vorticity = mesh.create_new_tree(cfg.dimension, cfg.min_level,
        cfg.max_level, cfg.stencil_graduation, cfg.stencil_prediction, cfg.xmin,
        cfg.xmax, cfg.ymin, cfg.ymax, cfg.zmin, cfg.zmax)
tree_exact_streamfunc = mesh.create_new_tree(cfg.dimension, cfg.min_level,
        cfg.max_level, cfg.stencil_graduation, cfg.stencil_prediction, cfg.xmin,
        cfg.xmax, cfg.ymin, cfg.ymax, cfg.zmin, cfg.zmax)

tree_exact_velocity_x.tag = "u"
tree_exact_velocity_y.tag = "v"
tree_exact_pressure.tag = "p"
tree_exact_vorticity.tag = "omega"
tree_exact_streamfunc.tag = "phi"

tree_exact_velocity_x.bc = cfg.bc_dict[tree_exact_velocity_x.tag]
tree_exact_velocity_y.bc = cfg.bc_dict[tree_exact_velocity_y.tag]
tree_exact_pressure.bc = cfg.bc_dict[tree_exact_pressure.tag]
tree_exact_streamfunc.bc = cfg.bc_dict[tree_exact_streamfunc.tag]

mesh.listing_of_leaves(tree_exact_velocity_x, tree_exact_velocity_y, tree_exact_pressure,
        tree_exact_vorticity, tree_exact_streamfunc)

print("trees creation done")

time_integrator = importlib.import_module(cfg.class_scheme_name)
time_integrator = time_integrator.Scheme(tree_velocity_x=tree_exact_velocity_x,
        tree_velocity_y=tree_exact_velocity_y, tree_pressure=tree_exact_pressure)
time_integrator.uniform = True

time_integrator.setup_internal_variables(tree_exact_velocity_x, tree_exact_velocity_y, tree_exact_pressure)
time_integrator.make_operators(tree_exact_velocity_x, tree_exact_velocity_y, tree_exact_pressure)
time_integrator.make_ksps()

time_integrator.make_vorticity_x(tree_exact_velocity_y)
time_integrator.make_vorticity_y(tree_exact_velocity_x)

#================ Computation of initial values from vorticity =================
sf_laplacian = sd.Operator(tree_exact_streamfunc, 0, time_integrator.laplacian)
sf_dx = sd.Operator(tree_exact_streamfunc, 0, time_integrator.divergence)
sf_dy = sd.Operator(tree_exact_streamfunc, 1, time_integrator.divergence)

sd.finite_volume_interpolation(tree_exact_vorticity, cfg.omega)
vorticity = sd.Scalar(tree_exact_vorticity)
sf_rhs = sd.mul_num_scalar(-1, vorticity)
sf = time_integrator.solve(sf_laplacian, sf_rhs)
vx = sf_dy.apply(sf)
vy = sd.mul_num_scalar(-1, sf_dx.apply(sf))
time_integrator.scalar_to_tree(vx, tree_exact_velocity_x)
time_integrator.scalar_to_tree(vy, tree_exact_velocity_y)
sd.finite_volume_interpolation(tree_exact_pressure, cfg.p_exact)

print("time integrator initiation done")

nsp = petsc.NullSpace().create(constant=True)
#nsp = None

print("2D gaussian-vortex")
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

#Used for the printing of the solutions
t_print = 0.

v_x = sd.Scalar(tree_exact_velocity_x)
v_y = sd.Scalar(tree_exact_velocity_y)
p = sd.Scalar(tree_exact_pressure)
vorticity = sd.Scalar(tree_exact_vorticity)

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

def run(nt):
    """Runs a specific configuration given the number of timesteps nt."""

#===============================================================================
#===============================================================================
#========================== INITIALISATION =====================================
#===============================================================================

    output_module = importlib.import_module(cfg.output_module_name)
    writer = output_module.OutputWriter("gaussian_two_vortices")
    writer.initialize()

    dt = (cfg.t_end - cfg.t_ini) / nt
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

    tree_velocity_x.tag = "u"
    tree_velocity_y.tag = "v"
    tree_pressure.tag = "p"
    tree_vorticity.tag = "omega"
    tree_streamfunc.tag = "phi"

    tree_velocity_x.bc = cfg.bc_dict[tree_velocity_x.tag]
    tree_velocity_y.bc = cfg.bc_dict[tree_velocity_y.tag]
    tree_pressure.bc = cfg.bc_dict[tree_pressure.tag]
    tree_streamfunc.bc = cfg.bc_dict[tree_streamfunc.tag]

    mesh.listing_of_leaves(tree_velocity_x, tree_velocity_y, tree_pressure,
            tree_vorticity, tree_streamfunc)

    print("trees creation done")

    time_integrator = importlib.import_module(cfg.class_scheme_name)
    time_integrator = time_integrator.Scheme(tree_velocity_x=tree_velocity_x,
            tree_velocity_y=tree_velocity_y, tree_pressure=tree_pressure)
    time_integrator.uniform = True
    time_integrator.dt = dt
    time_integrator.erk.dt = dt
    time_integrator.irk.dt = dt

    time_integrator.setup_internal_variables(tree_velocity_x, tree_velocity_y, tree_pressure)
    time_integrator.make_operators(tree_velocity_x, tree_velocity_y, tree_pressure)
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

    #print("2D gaussian-vortex")
    #print("  Grid informations :")
    #print("    - t_end = " +  repr(cfg.t_end))
    #print("    - nt = " +  repr(cfg.nt))
    #print("    - L = " +  repr(cfg.L))
    #print("    - dx = " +  repr((cfg.xmax-cfg.xmin) / 2**(cfg.max_level)))
    #print("    - dy = " +  repr((cfg.ymax-cfg.ymin) / 2**(cfg.max_level)))
    #print("")

#===============================================================================
#========================== COMPUTATION LOOP ===================================
#===============================================================================

    t = cfg.t_ini

#Used for the printing of the solutions
    t_print = 0.

    v_x = sd.Scalar(tree_velocity_x)
    v_y = sd.Scalar(tree_velocity_y)
    p = sd.Scalar(tree_pressure)

    for it in range(int(nt)):
        t_previous = t
        t = time_integrator.next_time(t)

        print("t = " + repr(t))
        print("")

        time_integrator.advance(v_x=v_x, v_y=v_y, p=p,
                                t_ini=t_previous, nsp=nsp)

        #time_integrator.scalar_to_tree(v_x, tree_velocity_x)
        #time_integrator.scalar_to_tree(v_y, tree_velocity_y)

        #temp = time_integrator.velocity_norm_l2(tree_velocity_x, tree_velocity_y)

        #print("    - dt = " +  repr(dt))
        #print("L2 Norm of the velocity field: ")
        #print(temp)
        #print("")

#============================= Printing solutions ==============================
        #if t >= t_print:

        #    vorticity.sc = (sd.add_scalars(
        #        time_integrator.vorticity_x.apply(v_y),
        #        sd.mul_num_scalar(-1., time_integrator.vorticity_y.apply(v_x)))).sc.copy()

        #    vorticity = time_integrator.velocity_inverse_mass.apply(vorticity)

        #    time_integrator.scalar_to_tree(v_x, tree_velocity_x)
        #    time_integrator.scalar_to_tree(v_y, tree_velocity_y)
        #    time_integrator.scalar_to_tree(p, tree_pressure)
        #    time_integrator.scalar_to_tree(vorticity, tree_vorticity)


        #    writer.write([tree_velocity_x, tree_velocity_y, tree_pressure, tree_vorticity],
        #            output_file_name="gaussian_two_vortices_t_" + repr(it).zfill(5),
        #            time=t)

        #    t_print = t_print + cfg.dt_print

#===============================================================================
#===============================================================================
#============ COMPUTING GLOBAL ERROR (SHOULD BE DONE ELSEWHERE !!!) ============
#===============================================================================

    time_integrator.scalar_to_tree(v_x, tree_velocity_x)
    time_integrator.scalar_to_tree(v_y, tree_velocity_y)
    time_integrator.scalar_to_tree(p, tree_pressure)

    #op.project_to_finest_grid(tree_velocity_x, tree_velocity_y, tree_pressure)

    mesh.listing_of_leaves(tree_velocity_x, tree_velocity_y, tree_pressure)

    global_error_vx = op.global_error_to_finest_grid(tree_velocity_x, tree_exact_velocity_x)
    global_error_vy = op.global_error_to_finest_grid(tree_velocity_y, tree_exact_velocity_y)
    global_error_p = op.global_error_to_finest_grid(tree_pressure, tree_exact_pressure)

    with open("norm_error_to_exact_solution_t=1.txt", 'a') as f:
        f.write('{0:.5f} {1} {2} {3}\n'.format(
            dt,
            global_error_vx,
            global_error_vy,
            global_error_p))
        #f.write('# dt ||u - u_exact|| ||v - v_exact|| ||p - p_exact||\n')

#===============================================================================
#============================ TERMINATION ======================================
#===============================================================================

    writer.close()

#timesteps = [10, 20, 30, 40, 50, 70, 100, 120, 150]
#timesteps = [50, 70, 80, 90, 100, 120, 150, 170, 200]
timesteps = [50, 70, 80, 90, 100, 120]
#timesteps = [100, 200, 300, 400, 500, 700]
#timesteps = [500, 700, 1000, 1200, 1500, 2000, 2500, 3000]

for nt in timesteps:
    run(nt)

