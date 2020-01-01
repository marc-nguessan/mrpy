from __future__ import print_function, division

"""The temporal-modules contain the functions needed to comute the advancement in time
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

Each scheme class has special methods to implement its specific
time-advancement. The time-advancement is enforced by the method advance, which
each class must possess, but which class-specific. This advance method should
act like a mutator: the variables are implemented as scalars in the main module,
and their local state, which their array of values over every mesh of the
domain, is changed by the call to the advance method.

This module implements the implicit explicit Euler scheme.
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
from mrpy.discretization.temporal_base import BaseScheme
import config as cfg


class Scheme(BaseScheme):
    """Implementation of an implicit-explicit order 1 Euler scheme in 2D. The
    convection terms are explicited."""

    def __init__(self, dimension=cfg.dimension, tree_velocity_x=None,
            tree_velocity_y=None, tree_velocity_z=None, tree_pressure=None,
            tree_vorticity=None, tree_scalar=None, uniform=False,
            st_flag_vx=False, st_flag_vy=False, st_flag_vz=False,
            st_flag_vc=False, st_flag_s=False, low_mach=False):

        BaseScheme.__init__(self, dimension=dimension,
            tree_velocity_x=tree_velocity_x, tree_velocity_y=tree_velocity_y,
            tree_velocity_z=tree_velocity_z, tree_pressure=tree_pressure,
            tree_vorticity=tree_vorticity, tree_scalar=tree_scalar,
            uniform=uniform, st_flag_vx=st_flag_vx, st_flag_vy=st_flag_vy,
            st_flag_vz=st_flag_vz, st_flag_vc=st_flag_vc, st_flag_s=st_flag_s,
            low_mach=low_mach)

        #if tree_vorticity is not None:
        #    BaseScheme.__init__(self, tree_velocity_x=tree_velocity_x,
        #            tree_velocity_y=tree_velocity_y,
        #            tree_pressure=tree_pressure, tree_vorticity=tree_vorticity)
        #else:
        #    BaseScheme.__init__(self, tree_velocity_x=tree_velocity_x,
        #            tree_velocity_y=tree_velocity_y,
        #            tree_pressure=tree_pressure)

        self.velocity_helmholtz_x = None
        self.velocity_helmholtz_y = None
        self.velocity_helmholtz_z = None
        self.pc = None
        self.stokes = None
        self.ksp_helmholtz_x = None
        self.ksp_helmholtz_y = None
        self.ksp_pc = None

    def make_velocity_helmholtz_x(self, tree_velocity_x):
        self.velocity_helmholtz_x = sd.add_operators(
        sd.mul_num_operator(1/self.dt, self.velocity_mass),
        sd.mul_num_operator(-cfg.nu, self.velocity_lap_x))

    def make_velocity_helmholtz_y(self, tree_velocity_y):
        self.velocity_helmholtz_y = sd.add_operators(
        sd.mul_num_operator(1/self.dt, self.velocity_mass),
        sd.mul_num_operator(-cfg.nu, self.velocity_lap_y))

    def make_velocity_helmholtz_z(self, tree_velocity_z):
        self.velocity_helmholtz_z = sd.add_operators(
        sd.mul_num_operator(1/self.dt, self.velocity_mass),
        sd.mul_num_operator(-cfg.nu, self.velocity_lap_z))

    def make_rhs_momentum_x(self, velocity_x, velocity_y, source_term_vx=None,
            low_mach_visc=None, one_over_density=None):

        if source_term_vx is None:
            #return sd.mul_num_scalar(1/self.dt, self.velocity_mass.apply(velocity_x))
            if self.low_mach:
                return sd.add_scalars(
                    sd.mul_scalars(one_over_density, low_mach_visc),
                    sd.mul_num_scalar(1/self.dt, self.velocity_mass.apply(velocity_x)),
                    sd.mul_num_scalar(-1., self.velocity_adv_x.apply(self.dimension,
                        velocity_x, velocity_x, velocity_y, low_mach=self.low_mach)))
            else:
                return sd.add_scalars(
                    sd.mul_num_scalar(1/self.dt, self.velocity_mass.apply(velocity_x)),
                    sd.mul_num_scalar(-1., self.velocity_adv_x.apply(self.dimension,
                        velocity_x, velocity_x, velocity_y, low_mach=self.low_mach)))

        else:
            if self.low_mach:
                return sd.add_scalars(
                    sd.mul_scalars(one_over_density, low_mach_visc),
                    sd.mul_num_scalar(1/self.dt, self.velocity_mass.apply(velocity_x)),
                    #self.velocity_mass.apply(source_term_vx))
                    self.velocity_mass.apply(source_term_vx),
                    sd.mul_num_scalar(-1., self.velocity_adv_x.apply(self.dimension,
                        velocity_x, velocity_x, velocity_y, low_mach=self.low_mach)))
            else:
                return sd.add_scalars(
                    sd.mul_num_scalar(1/self.dt, self.velocity_mass.apply(velocity_x)),
                    #self.velocity_mass.apply(source_term_vx))
                    self.velocity_mass.apply(source_term_vx),
                    sd.mul_num_scalar(-1., self.velocity_adv_x.apply(self.dimension,
                        velocity_x, velocity_x, velocity_y, low_mach=self.low_mach)))

    def make_rhs_momentum_y(self, velocity_x, velocity_y, source_term_vy=None,
            low_mach_visc=None, one_over_density=None):

        if source_term_vy is None:
            #return sd.mul_num_scalar(1/self.dt, self.velocity_mass.apply(velocity_y))
            if self.low_mach:
                return sd.add_scalars(
                    sd.mul_scalars(one_over_density, low_mach_visc),
                    sd.mul_num_scalar(1/self.dt, self.velocity_mass.apply(velocity_y)),
                    sd.mul_num_scalar(-1., self.velocity_adv_y.apply(self.dimension,
                        velocity_y, velocity_x, velocity_y, low_mach=self.low_mach)))
            else:
                return sd.add_scalars(
                    sd.mul_num_scalar(1/self.dt, self.velocity_mass.apply(velocity_y)),
                    sd.mul_num_scalar(-1., self.velocity_adv_y.apply(self.dimension,
                        velocity_y, velocity_x, velocity_y, low_mach=self.low_mach)))

        else:
            if self.low_mach:
                return sd.add_scalars(
                    sd.mul_scalars(one_over_density, low_mach_visc),
                    sd.mul_num_scalar(1/self.dt, self.velocity_mass.apply(velocity_y)),
                    #self.velocity_mass.apply(source_term_vy))
                    self.velocity_mass.apply(source_term_vy),
                    sd.mul_num_scalar(-1., self.velocity_adv_y.apply(self.dimension,
                        velocity_y, velocity_x, velocity_y, low_mach=self.low_mach)))
            else:
                return sd.add_scalars(
                    sd.mul_num_scalar(1/self.dt, self.velocity_mass.apply(velocity_y)),
                    #self.velocity_mass.apply(source_term_vy))
                    self.velocity_mass.apply(source_term_vy),
                    sd.mul_num_scalar(-1., self.velocity_adv_y.apply(self.dimension,
                        velocity_y, velocity_x, velocity_y, low_mach=self.low_mach)))

    def make_rhs_stokes_equation(self, velocity_x, velocity_y,
        source_term_vx=None, source_term_vy=None):

        result = sd.Scalar()
        number_of_rows = len(velocity_x.sc.getArray())
        size = (3*number_of_rows, 3*number_of_rows)
        result.sc.setSizes(size)
        if (source_term_velocity_x is None) and (source_term_velocity_y is None):
            temp_1 = sd.add_scalars(
                sd.mul_num_scalar(1/self.dt, velocity_x),
                sd.mul_num_scalar(-1., self.velocity_adv_x.apply(self.dimension,
                    velocity_x, velocity_x, velocity_y, low_mach=self.low_mach)))
            temp_2 = sd.add_scalars(
                sd.mul_num_scalar(1/self.dt, velocity_y),
                sd.mul_num_scalar(-1., self.velocity_adv_y.apply(self.dimension,
                    velocity_y, velocity_x, velocity_y, low_mach=self.low_mach)))

        if (source_term_vx is not None):
            temp_1 = sd.add_scalars(
                sd.mul_num_scalar(1/self.dt, velocity_x),
                source_term_vx,
                sd.mul_num_scalar(-1., self.velocity_adv_x.apply(self.dimension,
                    velocity_x, velocity_x, velocity_y, low_mach=self.low_mach)))

        if (source_term_vy is not None):
            temp_2 = sd.add_scalars(
                sd.mul_num_scalar(1/self.dt, velocity_y),
                source_term_vy,
                sd.mul_num_scalar(-1., self.velocity_adv_y.apply(self.dimension,
                    velocity_y, velocity_x, velocity_y, low_mach=self.low_mach)))
        for row in range(number_of_rows):
            result.sc[row] = temp_1.sc[row]
            result.sc[number_of_rows + row] = temp_2.sc[row]

        return result

    def make_stokes_operator(self, tree_velocity_x, tree_velocity_y, tree_pressure):
        """To be used in the advance module, after the update of the divergence,
        gradient, laplacian and helmholtz operators. This module needs these
        operators.
        """

        # Stokes operator assembling
        self.stokes = sd.Operator()

        number_of_rows = len(tree_velocity_x.tree_leaves)
        stokes_bc_size = (3*number_of_rows, 3*number_of_rows)
        self.stokes.bc.setSizes(stokes_bc_size)

        temp_1 = self.velocity_helmholtz_x.bc + self.pressure_grad_x.bc
        temp_2 = self.velocity_helmholtz_y.bc + self.pressure_grad_y.bc
        temp_3 = self.velocity_div_x.bc + self.velocity_div_y.bc
        for row in range(number_of_rows):
            self.stokes.bc[row] = temp_1[row]
            self.stokes.bc[number_of_rows + row] = temp_2[row]
            self.stokes.bc[2*number_of_rows + row] = temp_3[row]

        temp_operator = sd.Operator()
        size_row = (2*number_of_rows, 2*number_of_rows)
        size_col = (2*number_of_rows, 2*number_of_rows)
        temp_operator.matrix.setSizes((size_row, size_col))
        temp_operator.matrix.setUp()

        temp = petsc.Vec().create()
        temp.setSizes(size_row)
        temp.setUp()
        temp.set(1)
        temp_operator.matrix.setDiagonal(temp)
        temp_operator.bc.setSizes(size_row)

        temp_1 = sd.Operator()
        temp_1.matrix = self.laplacian_module.create_stokes_part_matrix(tree_velocity_x, tree_velocity_y)
        temp_1.bc.setSizes(size_row)

        velocity_helmholtz = sd.add_operators(
        sd.mul_num_operator(1/self.dt, temp_operator),
        sd.mul_num_operator(-cfg.nu, temp_1))

        temp_2 = self.gradient_module.create_stokes_part_matrix(tree_pressure, tree_pressure)
        nsp_schur = petsc.NullSpace().create(constant=True)
        temp_2.setNullSpace(nsp_schur)
        temp_3 = self.divergence_module.create_stokes_part_matrix(tree_velocity_x, tree_velocity_y)
        self.stokes.matrix = petsc.Mat().createNest([[velocity_helmholtz.matrix, temp_2], [temp_3, None]])

        # NullSpace creation
        temp = petsc.Vec().create()
        temp.setSizes(3*number_of_rows)
        temp.setUp()
        for row in range(number_of_rows):
            temp[2*number_of_rows + row] = 1

        temp.normalize()

        nsp = petsc.NullSpace().create(constant=False, vectors=(temp,))

    def make_operators(self, tree_velocity_x, tree_velocity_y, tree_pressure,
            tree_density=None):

        print("spatial operators creation begin")
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

        self.make_velocity_helmholtz_x(tree_velocity_x)
        self.make_velocity_helmholtz_y(tree_velocity_y)
        print("spatial operators creation end")

    def low_mach_update_operators(self, tree_velocity_x, tree_velocity_y, tree_pressure,
            tree_density=None):

        if self.low_mach:
            self.make_one_over_density(tree_density)

        self.make_velocity_lap_x(tree_velocity_x)
        self.make_velocity_lap_y(tree_velocity_y)

        self.make_pressure_grad_x(tree_pressure)
        self.make_pressure_grad_y(tree_pressure)

        self.make_velocity_helmholtz_x(tree_velocity_x)
        self.make_velocity_helmholtz_y(tree_velocity_y)
        print("spatial operators creation end")

    def make_ksps(self, preconditioner="Chorin-Temam"):
        # This structure allows to use the same ksp context to solve successive
        # linear systems with the same matrix over the successive iterations
        self.ksp_helmholtz_x = petsc.KSP().create()
        self.ksp_helmholtz_x.setOperators(self.velocity_helmholtz_x.matrix)
        self.ksp_helmholtz_x.getPC().setType("lu")
        self.ksp_helmholtz_x.setTolerances(rtol=1.e-12)

        self.ksp_helmholtz_y = petsc.KSP().create()
        self.ksp_helmholtz_y.setOperators(self.velocity_helmholtz_y.matrix)
        self.ksp_helmholtz_y.getPC().setType("lu")
        self.ksp_helmholtz_y.setTolerances(rtol=1.e-12)

        # We compute a pc operator and use it as a preconditoner for the
        # iterative solver on the pressure
        if preconditioner == "Chorin-Temam":
            self.pc = sd.mul_num_operator(self.dt, sd.add_operators(
                sd.mul_operators(self.pressure_grad_x, self.velocity_inverse_mass, self.velocity_div_x),
                sd.mul_operators(self.pressure_grad_y, self.velocity_inverse_mass, self.velocity_div_y)))

            #pc = sd.mul_num_operator(self.dt, self.velocity_lap_x)
            #pc.matrix.view()

            self.ksp_pc = petsc.KSP().create()
            # We are going to use the real laplacian as a preconditioner
            #self.ksp_pc.setOperators(pc.matrix, self.velocity_lap_x.matrix)
            self.ksp_pc.setOperators(self.pc.matrix)
            #self.ksp_pc.setType("bcgs")
            #self.ksp_pc.getPC().setType("lu")
            self.ksp_pc.setTolerances(max_it=500)

            nsp = petsc.NullSpace().create(constant=True)
            self.pc.matrix.setNullSpace(nsp)

            self.ksp_pc.setTolerances(rtol=1.e-05)

        # Run the program with the option -help to see all the possible
        # linear solver options.
        #self.ksp_helmholtz_x.setFromOptions()
        #self.ksp_helmholtz_y.setFromOptions()
        if preconditioner is not None:
            self.ksp_pc.setFromOptions()

        self.ksp = None # We force the default solve method to renew its own ksp

    def advance(self, v_x=None, v_y=None, v_z=None, p=None, t_ini=0, nsp=None,
            low_mach_visc_x=None, low_mach_visc_y=None, low_mach_cont=None,
            one_over_density=None):

        source_term_vx=None
        source_term_vy=None
        source_term_vz=None
        source_term_vc=None

        if self.uniform: #v_x, v_y, etc are scalars, and we just advance them

            if self.st_flag_vx:
                mesh.listing_of_leaves(self.st_tree_vx)
                self.compute_source_term(self.st_tree_vx, self.st_func_vx,
                        t_ini)
                source_term_vx = sd.Scalar(self.st_tree_vx)
            rhs_momentum_x = self.make_rhs_momentum_x(v_x, v_y,
                                source_term_vx, low_mach_visc_x,
                                one_over_density)

            if self.st_flag_vy:
                mesh.listing_of_leaves(self.st_tree_vy)
                self.compute_source_term(self.st_tree_vy, self.st_func_vy,
                        t_ini)
                source_term_vy = sd.Scalar(self.st_tree_vy)
            rhs_momentum_y = self.make_rhs_momentum_y(v_x, v_y,
                                source_term_vy, low_mach_visc_y,
                                one_over_density)

            if self.st_flag_vc:
                if self.low_mach:
                    mesh.listing_of_leaves(self.st_tree_vc)
                    self.compute_source_term(self.st_tree_vc, self.st_func_vc,
                            t_ini)
                    source_term_vc = sd.add_scalars(
                        low_mach_cont,
                        sd.Scalar(self.st_tree_vc))
                else:
                    mesh.listing_of_leaves(self.st_tree_vc)
                    self.compute_source_term(self.st_tree_vc, self.st_func_vc,
                            t_ini)
                    source_term_vc = sd.Scalar(self.st_tree_vc)
            else:
                if self.low_mach:
                    source_term_vc = low_mach_cont
            self.uzawa_solver(velocity_x=v_x, velocity_y=v_y,
                pressure=p, rhs_momentum_x=rhs_momentum_x,
                rhs_momentum_y=rhs_momentum_y, rhs_continuity=source_term_vc)

        else: #v_x, etc are trees
            velocity_x = sd.Scalar(v_x)
            velocity_y = sd.Scalar(v_y)
            pressure = sd.Scalar(p)
            #if (source_term_vx is not None) and (source_term_vy is not None):
            #    source_term_vx = sd.Scalar(source_term_vx)
            #    source_term_vy = sd.Scalar(source_term_vy)
            #else:
            #    source_term_vx = None
            #    source_term_vy = None

            if self.st_flag_vx: #we need to put the st_tree_vx to the same grading as v_x
                op.set_to_same_grading(v_x, self.st_tree_vx)
                op.run_pruning(self.st_tree_vx)
                mesh.listing_of_leaves(self.st_tree_vx)
                self.compute_source_term(self.st_tree_vx, self.st_func_vx,
                        t_ini)
                #mesh.listing_of_leaves(self.st_tree_vx)
                source_term_vx = sd.Scalar(self.st_tree_vx)
                #source_term_vx.sc.view()
                #quit()
            rhs_momentum_x = self.make_rhs_momentum_x(velocity_x, velocity_y,
                                source_term_vx, low_mach_visc_x,
                                one_over_density)

            if self.st_flag_vy: #we need to put the st_tree_vy to the same grading as v_y
                op.set_to_same_grading(v_y, self.st_tree_vy)
                op.run_pruning(self.st_tree_vy)
                mesh.listing_of_leaves(self.st_tree_vy)
                self.compute_source_term(self.st_tree_vy, self.st_func_vy,
                        t_ini)
                #mesh.listing_of_leaves(self.st_tree_vy)
                source_term_vy = sd.Scalar(self.st_tree_vy)
            rhs_momentum_y = self.make_rhs_momentum_y(velocity_x, velocity_y,
                                source_term_vy, low_mach_visc_y,
                                one_over_density)

            if self.st_flag_vc: #we need to put the st_tree_vc to the same grading as v_x
                if self.low_mach:
                    op.set_to_same_grading(v_x, self.st_tree_vc)
                    op.run_pruning(self.st_tree_vc)
                    mesh.listing_of_leaves(self.st_tree_vc)
                    self.compute_source_term(self.st_tree_vc, self.st_func_vc,
                            t_ini)
                    #mesh.listing_of_leaves(self.st_tree_vc)
                    source_term_vc = sd.add_scalars(
                        low_mach_cont,
                        sd.Scalar(self.st_tree_vc))
                else:
                    op.set_to_same_grading(v_x, self.st_tree_vc)
                    op.run_pruning(self.st_tree_vc)
                    self.compute_source_term(self.st_tree_vc, self.st_func_vc,
                            t_ini)
                    mesh.listing_of_leaves(self.st_tree_vc)
                    source_term_vc = sd.Scalar(self.st_tree_vc)
            else:
                if self.low_mach:
                    source_term_vc = low_mach_cont
            self.uzawa_solver(velocity_x=velocity_x, velocity_y=velocity_y,
                pressure=pressure, rhs_momentum_x=rhs_momentum_x,
                rhs_momentum_y=rhs_momentum_y, rhs_continuity=source_term_vc)

            for index in range(v_x.number_of_leaves):
                v_x.nvalue[v_x.tree_leaves[index]] = velocity_x.sc[index]

            for index in range(v_y.number_of_leaves):
                v_y.nvalue[v_y.tree_leaves[index]] = velocity_y.sc[index]

            for index in range(p.number_of_leaves):
                p.nvalue[p.tree_leaves[index]] = pressure.sc[index]

    def advance_given_rhs(self, tree_velocity_x, tree_velocity_y, tree_pressure,
                tree_rhs_momentum_x, tree_rhs_momentum_y, tree_rhs_continuity, nsp=None):
        #NEED UPDATE FOR UNIFORM COMPUTATIONS, AND LOW-MACH

        velocity_x = sd.Scalar(tree_velocity_x)
        velocity_y = sd.Scalar(tree_velocity_y)
        pressure = sd.Scalar(tree_pressure)
        rhs_momentum_x = sd.Scalar(tree_rhs_momentum_x)
        rhs_momentum_y = sd.Scalar(tree_rhs_momentum_y)
        rhs_continuity = sd.Scalar(tree_rhs_continuity)

        self.uzawa_solver(velocity_x=velocity_x, velocity_y=velocity_y,
            pressure=pressure, rhs_momentum_x=rhs_momentum_x,
            rhs_momentum_y=rhs_momentum_y,
            rhs_continuity=rhs_continuity, right_rhs=True)

        for index in range(tree_velocity_x.number_of_leaves):
            tree_velocity_x.nvalue[tree_velocity_x.tree_leaves[index]] = velocity_x.sc[index]

        for index in range(tree_velocity_y.number_of_leaves):
            tree_velocity_y.nvalue[tree_velocity_y.tree_leaves[index]] = velocity_y.sc[index]

        for index in range(tree_pressure.number_of_leaves):
            tree_pressure.nvalue[tree_pressure.tree_leaves[index]] = pressure.sc[index]
