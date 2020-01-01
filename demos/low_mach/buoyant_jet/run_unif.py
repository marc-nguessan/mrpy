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
writer = output_module.OutputWriter("buoyant_jet")
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
tree_density = mesh.create_new_tree(cfg.dimension, cfg.min_level,
        cfg.max_level, cfg.stencil_graduation, cfg.stencil_prediction, cfg.xmin,
        cfg.xmax, cfg.ymin, cfg.ymax, cfg.zmin, cfg.zmax)
tree_mass_frac_1 = mesh.create_new_tree(cfg.dimension, cfg.min_level,
        cfg.max_level, cfg.stencil_graduation, cfg.stencil_prediction, cfg.xmin,
        cfg.xmax, cfg.ymin, cfg.ymax, cfg.zmin, cfg.zmax)
tree_mass_frac_2 = mesh.create_new_tree(cfg.dimension, cfg.min_level,
        cfg.max_level, cfg.stencil_graduation, cfg.stencil_prediction, cfg.xmin,
        cfg.xmax, cfg.ymin, cfg.ymax, cfg.zmin, cfg.zmax)

tree_source_term_velocity_x = mesh.create_new_tree(cfg.dimension, cfg.min_level,
        cfg.max_level, cfg.stencil_graduation, cfg.stencil_prediction, cfg.xmin,
        cfg.xmax, cfg.ymin, cfg.ymax, cfg.zmin, cfg.zmax)
tree_source_term_velocity_y = mesh.create_new_tree(cfg.dimension, cfg.min_level,
        cfg.max_level, cfg.stencil_graduation, cfg.stencil_prediction, cfg.xmin,
        cfg.xmax, cfg.ymin, cfg.ymax, cfg.zmin, cfg.zmax)

tree_velocity_x.tag = "u"
tree_velocity_y.tag = "v"
tree_pressure.tag = "p"
tree_vorticity.tag = "omega"
tree_density.tag = "rho"
tree_mass_frac_1.tag = "y_1"
tree_mass_frac_2.tag = "y_2"

tree_velocity_x.bc = cfg.bc_dict[tree_velocity_x.tag]
tree_velocity_y.bc = cfg.bc_dict[tree_velocity_y.tag]
tree_pressure.bc = cfg.bc_dict[tree_pressure.tag]
tree_mass_frac_1.bc = cfg.bc_dict[tree_mass_frac_1.tag]
#tree_mass_frac_2.bc = cfg.bc_dict[tree_mass_frac_2.tag]

mesh.listing_of_leaves(tree_velocity_x, tree_velocity_y, tree_pressure,
        tree_source_term_velocity_x, tree_source_term_velocity_y,
        tree_mass_frac_1, tree_mass_frac_2,
        tree_vorticity, tree_density)

print("trees creation done")

lm = False
time_integrator = importlib.import_module(cfg.class_scheme_name)
time_integrator = time_integrator.Scheme(tree_velocity_x=tree_velocity_x,
        tree_velocity_y=tree_velocity_y, tree_pressure=tree_pressure,
        low_mach=lm, st_flag_vx=True, st_flag_vy=True)
time_integrator.uniform = True
#quit()

#time_int_scalar = importlib.import_module(cfg.scalar_scheme)
#time_int_scalar = time_int_scalar.Scheme(tree_scalar=tree_mass_frac_1,
#        diffusion=True, low_mach=lm)
#time_int_scalar.dt = cfg.dt_sc
#time_int_scalar.uniform = True

time_integrator.compute_initial_values(tree_velocity_x=tree_velocity_x,
        tree_velocity_y=tree_velocity_y, tree_pressure=tree_pressure)
sd.finite_volume_interpolation(tree_density, cfg.rho_init)

sd.finite_volume_interpolation(tree_mass_frac_1, cfg.y_1_init)
sd.finite_volume_interpolation(tree_mass_frac_2, cfg.y_2_init)

time_integrator.setup_internal_variables(tree_velocity_x, tree_velocity_y, tree_pressure)
time_integrator.setup_source_terms_variables()
time_integrator.make_operators(tree_velocity_x, tree_velocity_y, tree_pressure,
        tree_density=tree_density)
#time_int_scalar.make_operators(tree_mass_frac_1, tree_velocity_x, tree_velocity_y)
time_integrator.make_ksps()

time_integrator.make_vorticity_x(tree_velocity_y)
time_integrator.make_vorticity_y(tree_velocity_x)

print("time integrator initiation done")

nsp = petsc.NullSpace().create(constant=True)
#nsp = None

print("2D buoyant jet")
print("  Grid informations :")
print("    - dt = " +  repr(cfg.dt))
print("    - dx = " +  repr((cfg.xmax-cfg.xmin) / 2**(cfg.max_level)))
print("    - dy = " +  repr((cfg.ymax-cfg.ymin) / 2**(cfg.max_level)))
print("")

def update_density(mass_frac_1, mass_frac_2):

    temp = sd.add_scalars(
        sd.mul_num_scalar(1./cfg.M_inj, mass_frac_1),
        sd.mul_num_scalar(1./cfg.M_amb, mass_frac_2))

    foo = sd.inverse_scalar(temp)
    bar = cfg.P_0/(cfg.R*cfg.T_0)

    return sd.mul_num_scalar(bar, foo)

def compute_low_mach_cont(mass_frac_1, density):

    return sd.mul_num_scalar(cfg.xi, time_int_scalar.make_divgrad(mass_frac_1,
        density))

def compute_low_mach_visc_x(mass_frac_1, density):

    temp = sd.mul_num_scalar(cfg.xi, time_int_scalar.make_divgrad(mass_frac_1,
        density))
    res = sd.Scalar()
    res.sc = temp.sc.copy()
    time_integrator.velocity_div_x.matrix.mult(temp.sc, res.sc)

    return sd.mul_num_scalar(cfg.nu/3., res)

def compute_low_mach_visc_y(mass_frac_1, density):

    temp = sd.mul_num_scalar(cfg.xi, time_int_scalar.make_divgrad(mass_frac_1,
        density))
    res = sd.Scalar()
    res.sc = temp.sc.copy()
    time_integrator.velocity_div_y.matrix.mult(temp.sc, res.sc)

    return sd.mul_num_scalar(cfg.nu/3., res)

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
vorticity = sd.Scalar(tree_vorticity)
density = sd.Scalar(tree_density)
one_over_density = sd.inverse_scalar(density)
mass_frac_1 = sd.Scalar(tree_mass_frac_1)
mass_frac_2 = sd.Scalar(tree_mass_frac_2)
one = sd.Scalar(tree_mass_frac_1)
one.sc.set(1.)

for it in range(int(cfg.nt)):
    t_previous = t
    t = time_integrator.next_time(t)

    print("t = " + repr(t))
    print("")

    #time_integrator.scalar_to_tree(density, tree_density)
    #one_over_density = sd.inverse_scalar(density)
    one_over_density = None
    #time_integrator.low_mach_update_operators(tree_velocity_x, tree_velocity_y,
    #        tree_pressure, tree_density=tree_density)
    #time_integrator.make_ksps()
    #lmvx = compute_low_mach_visc_x(mass_frac_1, density)
    #lmvy = compute_low_mach_visc_y(mass_frac_1, density)
    #lmc = compute_low_mach_cont(mass_frac_1, density)
    lmvx = None
    lmvy = None
    lmc = None
    time_integrator.advance(v_x=v_x, v_y=v_y, p=p,
                            t_ini=t_previous, nsp=nsp,  low_mach_visc_x=lmvx,
                            low_mach_visc_y=lmvy, low_mach_cont=lmc,
                            one_over_density=one_over_density)
    #quit()

    #count = time_integrator.dt
    #while (count > 0):
    #    t_sc = time_int_scalar.next_time(t)
    #    time_int_scalar.advance(s=mass_frac_1, v_x=v_x, v_y=v_y,
    #        density=density,
    #        one_over_density=one_over_density)
    #    #time_int_scalar.advance(s=mass_frac_2, v_x=v_x, v_y=v_y,
    #    #    density=density,
    #    #    one_over_density=one_over_density)
    #    count -= time_int_scalar.dt

    #mass_frac_2.sc = sd.add_scalars(one, sd.mul_num_scalar(-1.,
    #    mass_frac_1)).sc
    #density = update_density(mass_frac_1, mass_frac_2)

    #temp = time_integrator.velocity_norm_l2(tree_velocity_x, tree_velocity_y)

    #print("L2 Norm of the velocity field: ")
    #print(temp)
    #print("")

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
        #time_integrator.scalar_to_tree(density, tree_density)
        #time_integrator.scalar_to_tree(mass_frac_1, tree_mass_frac_1)

        writer.write([tree_velocity_x, tree_velocity_y, tree_pressure,
            tree_vorticity],# tree_density, tree_mass_frac_1],
                output_file_name="buoyant_jet_t_" + repr(it).zfill(5),
                time=t)

        t_print = t_print + cfg.dt_print

#===============================================================================
#============================ TERMINATION ======================================
#===============================================================================

writer.close()
