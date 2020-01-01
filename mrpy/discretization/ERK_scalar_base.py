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


class ERKScheme(BaseScheme):
    """Base scheme for the implementation of explicit Runge-Kutta methods for
    the time advancement of transported scalar."""

    def __init__(self, tree_scalar=None, diffusion=False, low_mach=False):

        #BaseScheme.__init__(self, dimension=dimension,
        #    tree_velocity_x=tree_velocity_x, tree_velocity_y=tree_velocity_y,
        #    tree_velocity_z=tree_velocity_z, tree_pressure=tree_pressure,
        #    tree_vorticity=tree_vorticity, tree_scalar=tree_scalar,
        #    uniform=uniform, st_flag_vx=st_flag_vx, st_flag_vy=st_flag_vy,
        #    st_flag_vz=st_flag_vz, st_flag_vc=st_flag_vc, st_flag_s=st_flag_s,
        #    low_mach=low_mach)

        if tree_scalar is not None:
            BaseScheme.__init__(self, tree_scalar=tree_scalar,
                    diffusion=diffusion, low_mach=low_mach)

        self.scalar_mass = None
        self.scalar_inverse_mass = None
        self.adv = None
        self.lap = None
        self.grad_x = None
        self.grad_y = None
        self.diffusion = diffusion

        self.A_coefs = None
        self.B_coefs = None
        self.C_coefs = None

    def make_mass(self, tree_scalar):
        self.scalar_mass = sd.Operator(tree_scalar, 0, self.mass)

    def make_inverse_mass(self, tree_scalar):
        self.scalar_inverse_mass = sd.Operator(tree_scalar, 0, self.inverse_mass)

    def make_adv(self, tree_scalar, tree_velocity_x=None, tree_velocity_y=None, tree_velocity_z=None):
        if self.dimension == 1:
            self.adv = sd.AdvectionOperator(
            self.dimension, tree_scalar, self.divergence,
            tree_velocity_x)

        elif self.dimension == 2:
            self.adv = sd.AdvectionOperator(
            self.dimension, tree_scalar, self.divergence,
            tree_velocity_x, tree_velocity_y)

        elif self.dimension == 3:
            self.adv = sd.AdvectionOperator(
            self.dimension, tree_scalar, self.divergence,
            tree_velocity_x, tree_velocity_y, tree_velocity_z)

    def make_lap(self, tree_scalar):
        self.lap = sd.Operator(tree_scalar, 0, self.laplacian)

    def make_grad_x(self, tree_scalar):
        self.grad_x = sd.Operator(tree_scalar, 0, self.gradient)

    def make_grad_y(self, tree_scalar):
        self.grad_y = sd.Operator(tree_scalar, 1, self.gradient)

    def make_divgrad(self, scalar, density=None):

        if self.low_mach:
            temp_x = self.grad_x.apply(scalar)
            temp_y = self.grad_y.apply(scalar)

            foo_x = sd.mul_num_scalar(cfg.kappa, sd.mul_scalars(density, temp_x))
            foo_y = sd.mul_num_scalar(cfg.kappa, sd.mul_scalars(density, temp_y))

            self.grad_x.matrix.mult(foo_x.sc, temp_x.sc)
            self.grad_y.matrix.mult(foo_y.sc, temp_y.sc)

            return sd.add_scalars(temp_x, temp_y)

        else:

            return sd.mul_num_scalar(cfg.kappa, self.lap.apply(scalar))

    def make_rhs(self, scalar, velocity_x, velocity_y, source_term=None,
            density=None, one_over_density=None):

        if source_term is None:
            if self.diffusion:
                if self.low_mach:
                    return sd.add_scalars(
                        sd.mul_scalars(one_over_density, self.make_divgrad(scalar,
                            density)),
                        sd.mul_num_scalar(-1., self.adv.apply(self.dimension,
                            scalar, velocity_x, velocity_y, low_mach=self.low_mach)))

                else:
                    return sd.add_scalars(
                        self.make_divgrad(scalar),
                        sd.mul_num_scalar(-1., self.adv.apply(self.dimension,
                            scalar, velocity_x, velocity_y, low_mach=self.low_mach)))


            else:
                return sd.add_scalars(
                    sd.mul_num_scalar(-1., self.adv.apply(self.dimension,
                        scalar, velocity_x, velocity_y, low_mach=self.low_mach)))

        else:
            if self.diffusion:
                if self.low_mach:
                    return sd.add_scalars(
                        sd.mul_scalars(one_over_density, self.make_divgrad(scalar,
                            density)),
                        self.scalar_mass.apply(source_term),
                        sd.mul_num_scalar(-1., self.adv.apply(self.dimension,
                            scalar, velocity_x, velocity_y, low_mach=self.low_mach)))

                else:
                    return sd.add_scalars(
                        self.make_divgrad(scalar),
                        self.scalar_mass.apply(source_term),
                        sd.mul_num_scalar(-1., self.adv.apply(self.dimension,
                            scalar, velocity_x, velocity_y, low_mach=self.low_mach)))


            else:
                return sd.add_scalars(
                    self.scalar_mass.apply(source_term),
                    sd.mul_num_scalar(-1., self.adv.apply(self.dimension,
                        scalar, velocity_x, velocity_y, low_mach=self.low_mach)))

    #def make_rhs(self, scalar, velocity_x, velocity_y, source_term=None):

    #    if source_term is None:
    #        if self.lap is None:

    #            return sd.add_scalars(
    #                sd.mul_num_scalar(-1., self.adv.apply(self.dimension,
    #                    scalar, velocity_x, velocity_y, low_mach=self.low_mach)))

    #        else:

    #            return sd.add_scalars(
    #                sd.mul_num_scalar(cfg.kappa, self.lap.apply(scalar)),
    #                sd.mul_num_scalar(-1., self.adv.apply(self.dimension,
    #                    scalar, velocity_x, velocity_y, low_mach=self.low_mach)))

    #    else:
    #        if self.lap is None:

    #            return sd.add_scalars(
    #                self.scalar_mass.apply(source_term),
    #                sd.mul_num_scalar(-1., self.adv.apply(self.dimension,
    #                    scalar, velocity_x, velocity_y, low_mach=self.low_mach)))

    #        else:

    #            return sd.add_scalars(
    #                self.scalar_mass.apply(source_term),
    #                sd.mul_num_scalar(cfg.kappa, self.lap.apply(scalar)),
    #                sd.mul_num_scalar(-1., self.adv.apply(self.dimension,
    #                    scalar, velocity_x, velocity_y, low_mach=self.low_mach)))

    def make_operators(self, tree_scalar, tree_velocity_x, tree_velocity_y):

        print("spatial operators creation begin")
        self.make_mass(tree_scalar)
        self.make_inverse_mass(tree_scalar)
        self.make_adv(tree_scalar, tree_velocity_x, tree_velocity_y)

        if self.diffusion is True:
            if self.low_mach:
                self.make_grad_x(tree_scalar)
                self.make_grad_y(tree_scalar)
            else:
                self.make_lap(tree_scalar)
        print("spatial operators creation end")

    def compute_initial_values(self, tree_scalar, function):

        sd.finite_volume_interpolation(tree_scalar, function)

    def advance(self, s=None, v_x=None, v_y=None, t_ini=0, nsp=None,
            density=None, one_over_density=None):

        pass
