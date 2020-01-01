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

output_module = importlib.import_module(cfg.output_module_name)
writer = output_module.OutputWriter("lid_driven_cavity")
writer.initialize()

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

tree_velocity_x.tag = "u"
tree_velocity_y.tag = "v"
tree_pressure.tag = "p"
tree_vorticity.tag = "omega"

tree_velocity_x.bc = cfg.bc_dict[tree_velocity_x.tag]
tree_velocity_y.bc = cfg.bc_dict[tree_velocity_y.tag]
tree_pressure.bc = cfg.bc_dict[tree_pressure.tag]

mesh.listing_of_leaves(tree_velocity_x, tree_velocity_y, tree_pressure,  tree_vorticity)

print("trees creation done")

time_integrator = importlib.import_module(cfg.class_scheme_name)
time_integrator = time_integrator.Scheme(tree_velocity_x=tree_velocity_x,
        tree_velocity_y=tree_velocity_y, tree_pressure=tree_pressure)
time_integrator.uniform = True

time_integrator.compute_initial_values(tree_velocity_x=tree_velocity_x,
        tree_velocity_y=tree_velocity_y, tree_pressure=tree_pressure)

time_integrator.setup_internal_variables(tree_velocity_x, tree_velocity_y, tree_pressure)
time_integrator.make_operators(tree_velocity_x, tree_velocity_y, tree_pressure)
time_integrator.make_ksps()

time_integrator.make_vorticity_x(tree_velocity_y)
time_integrator.make_vorticity_y(tree_velocity_x)

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

v_x = sd.Scalar(tree_velocity_x)
v_y = sd.Scalar(tree_velocity_y)
p = sd.Scalar(tree_pressure)
vorticity = sd.Scalar(tree_vorticity)

for it in range(int(cfg.nt)):
    t_previous = t
    t = time_integrator.next_time(t)

    print("t = " + repr(t))
    print("")

    time_integrator.advance(v_x=v_x, v_y=v_y, p=p,
                            t_ini=t_previous, nsp=nsp)

#============================= Printing solutions ==============================
    if t >= t_print:

        vorticity.sc = (sd.add_scalars(
            time_integrator.vorticity_x.apply(v_y),
            sd.mul_num_scalar(-1., time_integrator.vorticity_y.apply(v_x)))).sc.copy()

        vorticity = time_integrator.velocity_inverse_mass.apply(vorticity)

        time_integrator.scalar_to_tree(v_x, tree_velocity_x)
        time_integrator.scalar_to_tree(v_y, tree_velocity_y)
        time_integrator.scalar_to_tree(p, tree_pressure)
        time_integrator.scalar_to_tree(vorticity, tree_vorticity)

        writer.write([tree_velocity_x, tree_velocity_y, tree_pressure, tree_vorticity],
                output_file_name="lid_driven_cavity_t_" + repr(it).zfill(5),
                time=t)

        t_print = t_print + cfg.dt_print

#===============================================================================
#============================ TERMINATION ======================================
#===============================================================================

writer.close()
