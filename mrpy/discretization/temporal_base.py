from __future__ import print_function, division

"""The temporal-modules contain functions needed to compute the advancement in time
of the physical variables simulated. We need a specific temporal scheme to
advance a system of variables. Here, each scheme is implemented in a class. The
class is supposed to be instantiated as a "time-integrator" object in the main
module used to run the simulation. This instance then uses its procedure
attributes to advance the variables defined in the main module. All of the
spatial operations on the variables are devised via the spatial_discretization
operators, so that we have a data abstraction barrier between the procedures
designed here, and the specific data implementation of the discrete variables.
This is done to increase the modularity of this code: as long as we have a valid
spatial_discretization module, we can use this module to advance variables in
time.

Each scheme class inherits from the BaseScheme class. This class is initiated
for now with the veloicty and the pressure, but may change if we need to add
more variables in our simulation. It then processes the following instance
attributes:
    - the three main linear spatial operators, divergence, gradient and
      laplacian
    - the non linear spatial operator for the advection
    - a timestep dt
Creating these attributes at the instantiation allows to have them computed once
and for all of the simulation.
The BaseScheme class also has special methods that are generic, such as:
    - a solve method that solves a linear system "Ax = b"
    - a next-time method that advances the time of the simulation, based on the
      current time and the timestep of the class
    - a compute-initial-values method that computes the initial values of the
      variables over the entire domain
    - etc.
If we feel the need for a specific method while designing a new scheme class, we
ask whether other schemes would need this method. If the answer is yes then we
implement this method in the BaseScheme class, so that we only have to modify it
in a single place.

This module contains the BaseScheme class.
"""

import sys, petsc4py
petsc4py.init(sys.argv)
import petsc4py.PETSc as petsc
import mpi4py.MPI as mpi
import numpy as np
import scipy.sparse as sp
from six.moves import range
import importlib
import math

from mrpy.mr_utils import mesh
from mrpy.mr_utils import op
import mrpy.discretization.spatial as sd
import config as cfg


class BaseScheme(object):

    def __init__(self, dimension=cfg.dimension, tree_velocity_x=None,
            tree_velocity_y=None, tree_velocity_z=None, tree_pressure=None,
            tree_vorticity=None, tree_scalar=None, uniform=False,
            st_flag_vx=False, st_flag_vy=False, st_flag_vz=False,
            st_flag_vc=False, st_flag_s=False, diffusion=False, low_mach=False):

        self.gradient = cfg.gradient_module_name
        self.divergence = cfg.divergence_module_name
        self.laplacian = cfg.laplacian_module_name
        self.mass = cfg.mass_module_name
        self.inverse_mass = cfg.inverse_mass_module_name
        self.dimension = dimension
        self.low_mach = low_mach
        if self.low_mach:
            self.density = cfg.density_module_name

        self.gradient_module = importlib.import_module(self.gradient)
        self.divergence_module = importlib.import_module(self.divergence)
        self.laplacian_module = importlib.import_module(self.laplacian)

        self.velocity_mass = None
        self.velocity_inverse_mass = None
        if self.low_mach:
            self.one_over_density = None
        self.velocity_div_x = None
        self.velocity_div_y = None
        self.velocity_div_z = None

        if tree_vorticity is not None:
            self.vorticity_x = None
            self.vorticity_y = None

        self.velocity_adv_x = None
        self.velocity_adv_y = None
        self.velocity_adv_z = None

        self.velocity_lap_x = None
        self.velocity_lap_y = None
        self.velocity_lap_z = None

        self.pressure_grad_x = None
        self.pressure_grad_y = None
        self.pressure_grad_z = None

        self.ksp = None

        self.dt = cfg.dt

        self.uniform = uniform

        self.st_flag_vx = st_flag_vx
        self.st_flag_vy = st_flag_vy
        self.st_flag_vz = st_flag_vz
        self.st_flag_vc = st_flag_vc
        self.st_flag_s = st_flag_s

        self.st_func_vx = None
        self.st_func_vy = None
        self.st_func_vz = None
        self.st_func_vc = None
        self.st_func_s = None

    def setup_source_terms_variables(self):

        if self.st_flag_vx:
            self.st_func_vx = cfg.source_term_function_velocity_x
            #we create an internal tree that will be used to compute the source terms scalar
            self.st_tree_vx = mesh.create_new_tree(cfg.dimension, cfg.min_level,
                cfg.max_level, cfg.stencil_graduation, cfg.stencil_prediction, cfg.xmin,
                cfg.xmax, cfg.ymin, cfg.ymax, cfg.zmin, cfg.zmax)

        if self.st_flag_vy:
            self.st_func_vy = cfg.source_term_function_velocity_y
            #we create an internal tree that will be used to compute the source terms scalar
            self.st_tree_vy = mesh.create_new_tree(cfg.dimension, cfg.min_level,
                cfg.max_level, cfg.stencil_graduation, cfg.stencil_prediction, cfg.xmin,
                cfg.xmax, cfg.ymin, cfg.ymax, cfg.zmin, cfg.zmax)

        if self.st_flag_vz:
            self.st_func_vz = cfg.source_term_function_velocity_z
            #we create an internal tree that will be used to compute the source terms scalar
            self.st_tree_vz = mesh.create_new_tree(cfg.dimension, cfg.min_level,
                cfg.max_level, cfg.stencil_graduation, cfg.stencil_prediction, cfg.xmin,
                cfg.xmax, cfg.ymin, cfg.ymax, cfg.zmin, cfg.zmax)

        if self.st_flag_vc:
            self.st_func_vc = cfg.source_term_function_velocity_c
            #we create an internal tree that will be used to compute the source terms scalar
            self.st_tree_vc = mesh.create_new_tree(cfg.dimension, cfg.min_level,
                cfg.max_level, cfg.stencil_graduation, cfg.stencil_prediction, cfg.xmin,
                cfg.xmax, cfg.ymin, cfg.ymax, cfg.zmin, cfg.zmax)

        if self.st_flag_s:
            self.st_func_s = cfg.source_term_function_scalar
            #we create an internal tree that will be used to compute the source terms scalar
            self.st_tree_s = mesh.create_new_tree(cfg.dimension, cfg.min_level,
                cfg.max_level, cfg.stencil_graduation, cfg.stencil_prediction, cfg.xmin,
                cfg.xmax, cfg.ymin, cfg.ymax, cfg.zmin, cfg.zmax)

    def make_velocity_mass(self, tree_velocity_x): # we assume that there will be at least a x-component for the velocity
        self.velocity_mass = sd.Operator(tree_velocity_x, 0, self.mass)

    def make_velocity_inverse_mass(self, tree_velocity_x): # we assume that there will be at least a x-component for the velocity
        self.velocity_inverse_mass = sd.Operator(tree_velocity_x, 0, self.inverse_mass)

    def make_one_over_density(self, tree_density):
        self.one_over_density = sd.Operator(tree_density, 0, self.density)

    def make_velocity_div_x(self, tree_velocity_x):
        self.velocity_div_x = sd.Operator(tree_velocity_x, 0, self.divergence)

    def make_velocity_div_y(self, tree_velocity_y):
        self.velocity_div_y = sd.Operator(tree_velocity_y, 1, self.divergence)

    def make_velocity_div_z(self, tree_velocity_z):
        self.velocity_div_y = sd.Operator(tree_velocity_z, 2, self.divergence)

    def make_vorticity_x(self, tree_velocity_y):
        self.vorticity_x = sd.Operator(tree_velocity_y, 0, self.divergence)

    def make_vorticity_y(self, tree_velocity_x):
        self.vorticity_y = sd.Operator(tree_velocity_x, 1, self.divergence)

    def make_velocity_adv_x(self, tree_velocity_x, tree_velocity_y=None, tree_velocity_z=None):
        if self.dimension == 1:
            self.velocity_adv_x = sd.AdvectionOperator(
            self.dimension, tree_velocity_x, self.divergence,
            tree_velocity_x)

        elif self.dimension == 2:
            self.velocity_adv_x = sd.AdvectionOperator(
            self.dimension, tree_velocity_x, self.divergence,
            tree_velocity_x, tree_velocity_y)

        elif self.dimension == 3:
            self.velocity_adv_x = sd.AdvectionOperator(
            self.dimension, tree_velocity_x, self.divergence,
            tree_velocity_x, tree_velocity_y, tree_velocity_z)

    def make_velocity_adv_y(self, tree_velocity_x, tree_velocity_y, tree_velocity_z=None):

        if self.dimension == 2:
            self.velocity_adv_y = sd.AdvectionOperator(
            self.dimension, tree_velocity_y, self.divergence,
            tree_velocity_x, tree_velocity_y)

        elif self.dimension == 3:
            self.velocity_adv_y = sd.AdvectionOperator(
            self.dimension, tree_velocity_y, self.divergence,
            tree_velocity_x, tree_velocity_y, tree_velocity_z)

    def make_velocity_adv_z(self, tree_velocity_x, tree_velocity_y, tree_velocity_z):
        self.velocity_adv_z = sd.AdvectionOperator(
        self.dimension, tree_velocity_z, self.divergence,
        tree_velocity_x, tree_velocity_y, tree_velocity_z)

    def make_velocity_lap_x(self, tree_velocity_x):

        if self.low_mach:
            foo = sd.Operator(tree_velocity_x, 0, self.laplacian)
            #self.velocity_lap_x = sd.Operator(tree_velocity_x, 0, self.laplacian)
            self.velocity_lap_x = sd.mul_operators(foo, self.one_over_density)
            ##### Problem with the boundary condition in the mul_operators!!!###
        else:
            self.velocity_lap_x = sd.Operator(tree_velocity_x, 0, self.laplacian)


    def make_velocity_lap_y(self, tree_velocity_y):

        if self.low_mach:
            foo = sd.Operator(tree_velocity_y, 1, self.laplacian)
            self.velocity_lap_y = sd.mul_operators(foo, self.one_over_density)
            ##### Problem with the boundary condition in the mul_operators!!!###
        else:
            self.velocity_lap_y = sd.Operator(tree_velocity_y, 1, self.laplacian)


    def make_velocity_lap_z(self, tree_velocity_z):

        if self.low_mach:
            foo = sd.Operator(tree_velocity_z, 2, self.laplacian)
            self.velocity_lap_z = sd.mul_operators(foo, self.one_over_density)
            ##### Problem with the boundary condition in the mul_operators!!!###
        else:
            self.velocity_lap_z = sd.Operator(tree_velocity_z, 2, self.laplacian)


    def make_pressure_grad_x(self, tree_pressure):

        if self.low_mach:
            temp = sd.Operator()
            number_of_rows = len(tree_pressure.tree_leaves)
            size = (number_of_rows, number_of_rows)
            temp.matrix.setSizes((size, size))
            temp.matrix.setUp()
            self.velocity_div_x.matrix.transpose(temp.matrix)
            temp.bc.setSizes(number_of_rows, number_of_rows)

            foo = sd.mul_num_operator(-1., temp)
            #self.make_one_over_density(tree_density)
            self.pressure_grad_x = sd.mul_operators(foo, self.one_over_density)
        else:
            temp = sd.Operator()
            number_of_rows = len(tree_pressure.tree_leaves)
            size = (number_of_rows, number_of_rows)
            temp.matrix.setSizes((size, size))
            temp.matrix.setUp()
            self.velocity_div_x.matrix.transpose(temp.matrix)
            temp.bc.setSizes(number_of_rows, number_of_rows)

            self.pressure_grad_x = sd.mul_num_operator(-1., temp)


    def make_pressure_grad_y(self, tree_pressure):

        if self.low_mach:
            temp = sd.Operator()
            number_of_rows = len(tree_pressure.tree_leaves)
            size = (number_of_rows, number_of_rows)
            temp.matrix.setSizes((size, size))
            temp.matrix.setUp()
            self.velocity_div_y.matrix.transpose(temp.matrix)
            temp.bc.setSizes(number_of_rows, number_of_rows)

            foo = sd.mul_num_operator(-1., temp)
            self.pressure_grad_y = sd.mul_operators(foo, self.one_over_density)
        else:
            temp = sd.Operator()
            number_of_rows = len(tree_pressure.tree_leaves)
            size = (number_of_rows, number_of_rows)
            temp.matrix.setSizes((size, size))
            temp.matrix.setUp()
            self.velocity_div_y.matrix.transpose(temp.matrix)
            temp.bc.setSizes(number_of_rows, number_of_rows)

            self.pressure_grad_y = sd.mul_num_operator(-1., temp)


    def make_pressure_grad_z(self, tree_pressure):

        if self.low_mach:
            temp = sd.Operator()
            number_of_rows = len(tree_pressure.tree_leaves)
            size = (number_of_rows, number_of_rows)
            temp.matrix.setSizes((size, size))
            temp.matrix.setUp()
            self.velocity_div_z.matrix.transpose(temp.matrix)
            temp.bc.setSizes(number_of_rows, number_of_rows)

            foo = sd.mul_num_operator(-1., temp)
            self.pressure_grad_z = sd.mul_operators(foo, self.one_over_density)
        else:
            temp = sd.Operator()
            number_of_rows = len(tree_pressure.tree_leaves)
            size = (number_of_rows, number_of_rows)
            temp.matrix.setSizes((size, size))
            temp.matrix.setUp()
            self.velocity_div_z.matrix.transpose(temp.matrix)
            temp.bc.setSizes(number_of_rows, number_of_rows)

            self.pressure_grad_z = sd.mul_num_operator(-1., temp)

    def make_pressure_divgrad(self):

        self.pressure_divgrad = sd.add_operators(
            sd.mul_operators(self.pressure_grad_x, self.velocity_inverse_mass,
                self.velocity_div_x),
            sd.mul_operators(self.pressure_grad_y, self.velocity_inverse_mass,
                self.velocity_div_y))

        self.pressure_divgrad.bc.set(0) # we need a special right-hand side for this pressure Poisson equation, and for now we set it to zero

    def make_operators(self, tree_velocity_x, tree_velocity_y, tree_pressure,
            tree_density=None):

        self.make_velocity_mass(tree_velocity_x)
        self.make_velocity_inverse_mass(tree_velocity_x)

        if self.low_mach:
            self.make_one_over_density(tree_density)

        self.make_velocity_div_x(tree_velocity_x)
        self.make_velocity_div_y(tree_velocity_y)

        self.make_velocity_adv_x(tree_velocity_x, tree_velocity_y)
        self.make_velocity_adv_y(tree_velocity_x, tree_velocity_y)

        self.make_velocity_lap_x(tree_velocity_x)
        self.make_velocity_lap_y(tree_velocity_y)

        self.make_pressure_grad_x(tree_pressure)
        self.make_pressure_grad_y(tree_pressure)

    def low_mach_update_operators(self, tree_velocity_x, tree_velocity_y, tree_pressure,
            tree_density=None):

        if self.low_mach:
            self.make_one_over_density(tree_density)

        self.make_velocity_lap_x(tree_velocity_x)
        self.make_velocity_lap_y(tree_velocity_y)

        self.make_pressure_grad_x(tree_pressure)
        self.make_pressure_grad_y(tree_pressure)

    def make_ksps(self):

        self.ksp = None # We force the default solve method to renew its own ksp

    def scalar_to_tree(self, scalar, tree):
        """Copies the values of the scalar into the leaves of the tree."""

        for index in range(tree.number_of_leaves):
            tree.nvalue[tree.tree_leaves[index]] = scalar.sc[index]

    def make_rhs_with_bc(self, rhs, *operators):
        """Returns a scalar rhs with the bcs coming from the operator."""

        new_rhs = sd.Scalar()
        new_rhs.sc = rhs.sc.copy()

        for operator in operators:
            new_rhs.sc -= operator.bc #yes, it is a minus sign here
        #new_rhs.sc = new_rhs.sc - operator.bc #yes, it is a minus sign here

        return new_rhs

    def make_rhs_pressure_equation(self, velocity_x, velocity_y,
                                   source_term_vx=None, source_term_vy=None):

        """Forms the RHS for the pressure Poisson equation."""
        # NEED UPDATE FOR LOW-MACH

        if (source_term_vx is None):
            temp_1 = sd.add_scalars(
                #sd.mul_num_scalar(cfg.nu, self.velocity_lap_x.apply(velocity_x)))
                sd.mul_num_scalar(cfg.nu, self.velocity_lap_x.apply(velocity_x)),
                sd.mul_num_scalar(-1, self.velocity_adv_x.apply(self.dimension, velocity_x,
                    velocity_x, velocity_y, low_mach=self.low_mach)))
        else:
            temp_1 = sd.add_scalars(
                self.velocity_mass.apply(source_term_vx),
                #sd.mul_num_scalar(cfg.nu, self.velocity_lap_x.apply(velocity_x)))
                sd.mul_num_scalar(cfg.nu, self.velocity_lap_x.apply(velocity_x)),
                sd.mul_num_scalar(-1, self.velocity_adv_x.apply(self.dimension, velocity_x,
                    velocity_x, velocity_y, low_mach=self.low_mach)))

        if (source_term_vx is None):
            temp_2 = sd.add_scalars(
                #sd.mul_num_scalar(cfg.nu, self.velocity_lap_y.apply(velocity_y)))
                sd.mul_num_scalar(cfg.nu, self.velocity_lap_y.apply(velocity_y)),
                sd.mul_num_scalar(-1, self.velocity_adv_y.apply(self.dimension, velocity_y,
                    velocity_x, velocity_y, low_mach=self.low_mach)))
        else:
            temp_2 = sd.add_scalars(
                self.velocity_mass.apply(source_term_vy),
                #sd.mul_num_scalar(cfg.nu, self.velocity_lap_y.apply(velocity_y)))
                sd.mul_num_scalar(cfg.nu, self.velocity_lap_y.apply(velocity_y)),
                sd.mul_num_scalar(-1, self.velocity_adv_y.apply(self.dimension, velocity_y,
                    velocity_x, velocity_y, low_mach=self.low_mach)))

        bar_1 = self.velocity_inverse_mass.apply(temp_1)
        bar_2 = self.velocity_inverse_mass.apply(temp_2)

        foo_1, foo_2 = sd.Scalar(), sd.Scalar()
        foo_1.sc, foo_2.sc = temp_1.sc.duplicate(), temp_2.sc.duplicate()
        self.velocity_div_x.matrix.mult(bar_1.sc, foo_1.sc)
        self.velocity_div_y.matrix.mult(bar_2.sc, foo_2.sc)

        return sd.add_scalars(foo_1, foo_2)

    def solve(self, operator, rhs, nsp=None):
        """Solves a linear system given the operator, the unknown we want to solve and the rhs."""

        new_rhs = rhs.sc - operator.bc #yes, it is a minus sign here

        #operator.matrix.view()

        solution = sd.Scalar()
        solution.sc = rhs.sc.copy()

        # This structure allows to use the same ksp context to solve successive
        # linear systems with the same matrix over the successive timesteps; it
        # should be changed if the matrix is supposed to change
        if self.ksp is None:
            self.ksp = petsc.KSP().create()
            self.ksp.setOperators(operator.matrix)
            #self.ksp.setType("dgmres")
            #pc = self.ksp.getPC()
            #pc.setType("ilu")
            self.ksp.setTolerances(rtol=1.e-10)

            # Set runtime options, e.g.,
            #    -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
            # These options will override those specified above as long as
            # KSPSetFromOptions() is called _after_ any other customization
            # routines.

            # Run the program with the option -help to see all the possible
            # linear solver options.
            self.ksp.setFromOptions()

        #ksp = petsc.KSP().create()
        #ksp.setOperators(operator.matrix)
        #ksp.setFromOptions()
        #ksp.setTolerances(rtol=1.e-10)

        if nsp is None:
            pass
        else:
            operator.matrix.setNullSpace(nsp)
            nsp.remove(new_rhs)

        self.ksp.solve(new_rhs, solution.sc)

        #return solution.sc.getArray()
        return solution

# !!!!! WIP !!!!!
    def solve_stokes(self, operator, rhs, nsp=None):
        """Solves a linear system given the operator, the unknown we want to solve and the rhs."""

        new_rhs = rhs.sc - operator.bc #yes, it is a minus sign here

        solution = sd.Scalar()
        solution.sc = rhs.sc.copy()

        ksp = petsc.KSP().create()
        ksp.setOperators(operator.matrix)
        ksp.setType(petsc.KSP.Type.FGMRES)
        pc = ksp.getPC()
        pc.setType(petsc.PC.Type.FIELDSPLIT)

        is_velocity, is_pressure = operator.matrix.getNestISs()
        is_pressure[0].view()
        is_pressure[1].view()
        is_velocity[0].view()
        is_velocity[1].view()
        quit()

        if nsp is None:
            pass
        else:
            operator.matrix.setNullSpace(nsp)
            nsp.remove(new_rhs)

        ksp.setFromOptions()
        pc.setFieldSplitType(petsc.PC.CompositeType.SCHUR)
        pc.setFieldSplitSchurFactType(petsc.PC.SchurFactType.LOWER)
        pc.setFieldSplitSchurPreType(petsc.PC.SchurPreType.SELFP, None)
        pc.setFieldSplitIS(('u', is_velocity[0]), ('p', is_velocity[1]))
        ksp.setUp()
        subksp = pc.getFieldSplitSubKSP()
        subksp[0].setType("gmres")
        subksp[0].getPC().setType("lu")
        subksp[1].setType("gmres")
        #subksp[1].getPC().setType("none")
        subksp[1].setTolerances(max_it=500)
        subksp[0].setTolerances(max_it=10)

        ksp.setTolerances(max_it=20)
        ksp.setGMRESRestart(100)

        #ksp.view()
        #pc.view()
        #quit()

        ksp.solve(new_rhs, solution.sc)

        return solution

    def uzawa_solver(self, velocity_x=None, velocity_y=None, pressure=None,
                     rhs_momentum_x=None, rhs_momentum_y=None,
                     rhs_continuity=None, relax_param=1e-0, max_it=50,
                     rtol=1e-08, atol=1e-10, preconditioner="Chorin-Temam",
                     right_rhs=False):
        """implementation of the uzawa alogrithm to solve the saddle-point
        problem arising from the spatial discretization of the NS equation.

        It will use the divergence, helmholtz and gradient operators, so these
        operators need to be computed first. It should be used typically in the
        advance module.
        """

        vx = velocity_x.sc.copy()
        vy = velocity_y.sc.copy()
        p = pressure.sc.copy()

        if right_rhs is False:
            new_rhs_momentum_x = rhs_momentum_x.sc - self.velocity_helmholtz_x.bc
            new_rhs_momentum_y = rhs_momentum_y.sc - self.velocity_helmholtz_y.bc
            if rhs_continuity is None:
                new_rhs_continuity = -self.velocity_div_x.bc - self.velocity_div_y.bc
                #new_rhs_continuity.view()
                #quit()
            else:
                new_rhs_continuity = rhs_continuity.sc - self.velocity_div_x.bc - self.velocity_div_y.bc
        elif right_rhs is True:
            new_rhs_momentum_x = rhs_momentum_x.sc
            new_rhs_momentum_y = rhs_momentum_y.sc
            if rhs_continuity is None:
                new_rhs_continuity = -self.velocity_div_x.bc - self.velocity_div_y.bc
            else:
                new_rhs_continuity = rhs_continuity.sc

        # This global rhs is used to compute the residual norm
        n_vx = new_rhs_momentum_x.getSizes()
        n_vy = new_rhs_momentum_y.getSizes()
        n_div = new_rhs_continuity.getSizes()
        n_vx = n_vx[0] # vec.getSizes() returns a tuple with the local and global sizes; we just want one of them (in sequential they are equal)
        n_vy = n_vy[0]
        n_div = n_div[0]

        global_rhs = petsc.Vec().create()
        global_rhs.setSizes(n_vx + n_vy + n_div)
        global_rhs.setUp()
        for i in range(n_vx):
            global_rhs[i] = new_rhs_momentum_x[i]

        for i in range(n_vy):
            global_rhs[n_vx + i] = new_rhs_momentum_y[i]

        for i in range(n_div):
            global_rhs[n_vx + n_vy + i] = new_rhs_continuity[i]

        #global_rhs[:n_vx] = new_rhs_momentum_x
        #global_rhs[n_vx:n_vx + n_vy] = new_rhs_momentum_y
        #global_rhs[n_vx + n_vy:] = new_rhs_continuity

        # Procedure to compute the residual norm
        def residual_norm_test(vx, vy, p):

            temp_vx, temp = vx.duplicate(), vx.duplicate()
            self.velocity_helmholtz_x.matrix.mult(vx, temp_vx)
            self.pressure_grad_x.matrix.mult(p, temp)
            temp_vx += temp
            n_vx = temp_vx.getSizes()

            temp_vy, temp = vy.duplicate(), vy.duplicate()
            self.velocity_helmholtz_y.matrix.mult(vy, temp_vy)
            self.pressure_grad_y.matrix.mult(p, temp)
            temp_vy += temp
            n_vy = temp_vy.getSizes()

            temp_div_x = vx.duplicate()
            temp_div_y = vy.duplicate()
            self.velocity_div_x.matrix.mult(vx, temp_div_x)
            self.velocity_div_y.matrix.mult(vy, temp_div_y)
            temp_div = temp_div_x + temp_div_y
            n_div = temp_div.getSizes()

            n_vx = n_vx[0] # vec.getSizes() returns a tuple with the local and global sizes; we just want one of them (in sequential they are equal)
            n_vy = n_vy[0]
            n_div = n_div[0]

            temp = global_rhs.duplicate()

            #for i in range(n_vx):
            #    temp[i] = temp_vx[i]
            temp[:n_vx] = temp_vx

            #for i in range(n_vy):
            #    temp[n_vx + i] = temp_vy[i]
            temp[n_vx:n_vx+n_vy] = temp_vy

            #for i in range(n_div):
            #    temp[n_vx + n_vy + i] = temp_div[i]
            temp[n_vx+n_vy:] = temp_div

            residual_vector = temp - global_rhs
            #residual_vector.view()
            residual_norm = residual_vector.norm()
            print("residual_norm: " + repr(residual_norm))
            print("rtol*global_rhs_norm: " + repr(rtol * global_rhs.norm()))

            if residual_norm <= max(rtol * global_rhs.norm(), atol):
                return True
            else:
                return False


        converged = False
        count = 0
        while ((not converged) and (count <= max_it)):

            # computation of vx_next
            temp = p.duplicate()
            self.pressure_grad_x.matrix.mult(p, temp)
            temp = new_rhs_momentum_x - temp
            vx_next = vx.duplicate()
            self.ksp_helmholtz_x.solve(temp, vx_next)

            # computation of vy_next
            temp = vy.duplicate()
            self.pressure_grad_y.matrix.mult(p, temp)
            temp = new_rhs_momentum_y - temp
            vy_next = vy.copy()
            self.ksp_helmholtz_y.solve(temp, vy_next)

            # computation of p_next
            temp_x = vx.duplicate()
            temp_y = vy.duplicate()
            self.velocity_div_x.matrix.mult(vx_next, temp_x)
            self.velocity_div_y.matrix.mult(vy_next, temp_y)
            #temp_x.view()
            #temp_y.view()
            #quit()
            if preconditioner is not None:
                delta_p = p.duplicate()
                #nsp.remove(temp_x + temp_y - new_rhs_continuity)
                self.ksp_pc.solve(temp_x + temp_y - new_rhs_continuity, delta_p)

                p_next = p + relax_param * delta_p
            else:
                p_next = p - relax_param * (temp_x + temp_y - new_rhs_continuity)

            if residual_norm_test(vx_next, vy_next, p_next): converged = True
            count += 1

            vx = vx_next.copy()
            vy = vy_next.copy()
            p = p_next.copy()

        velocity_x.sc = vx_next.copy()
        velocity_y.sc = vy_next.copy()
        pressure.sc = p_next.copy()
        #quit()

    def advance(self, v_x=None, v_y=None, v_z=None, p=None, t_ini=0, nsp=None,
            low_mach_visc_x=None, low_mach_visc_y=None, low_mach_cont=None,
            one_over_density=None):

        pass

    def setup_internal_variables(self, tree_velocity_x, tree_velocity_y,
            tree_pressure):

        pass

    def compute_initial_values(self, tree_velocity_x=None, tree_velocity_y=None,
                               tree_velocity_z=None, tree_pressure=None):

        sd.finite_volume_interpolation(tree_velocity_x, cfg.u_exact)
        sd.finite_volume_interpolation(tree_velocity_y, cfg.v_exact)
        sd.finite_volume_interpolation(tree_pressure, cfg.p_exact)

    def compute_source_term(self, tree, source_term_function, time):

        sd.finite_volume_interpolation(tree, source_term_function, time)

    def velocity_norm_l2(self, tree_velocity_x, tree_velocity_y):
        """...

        """

        norm_x = 0.0
        norm_y = 0.0

        for index in tree_velocity_x.tree_leaves:
            norm_x += (tree_velocity_x.nvalue[index]*tree_velocity_x.ndx[index]*tree_velocity_x.ndy[index])**2
            norm_y += (tree_velocity_y.nvalue[index]*tree_velocity_y.ndx[index]*tree_velocity_y.ndy[index])**2
            #norm_y += tree_velocity_y.nvalue[index]*tree_velocity_y.nvalue[index]

        #return math.sqrt(norm_x**2 + norm_y**2)
        return math.sqrt(norm_x + norm_y)

    def next_time(self, time):

        return time + self.dt

    #def local_error_to_exact_solution(self, function_sol_exact, sol_num, time):
    #    """If the simultation is reproducing an exact solution to the pde we are
    #    trying to compute, then this method can be used to compute, at each
    #    timestep, the local (scalar) error between the exact solution and the
    #    computed solution."""

    #    return sd.add_scalars(
    #        sd.finite_volume_interpolation(function_sol_exact, time),
    #        sd.mul_num_scalar(-1, sol_num))

    #def global_error_to_exact_solution(self, function_sol_exact, sol_num, time):
    #    """If the simultation is reproducing an exact solution to the pde we are
    #    trying to compute, then this method can be used to compute, at each
    #    timestep, the global error between the exact solution and the computed
    #    solution."""

    #    vec_error = sd.add_scalars(
    #        sd.finite_volume_interpolation(function_sol_exact, time),
    #        sd.mul_num_scalar(-1, sol_num))

    #    error = vec_error.sc.norm()
    #    error = error*(cfg.dx*cfg.dy)

    #    return error
