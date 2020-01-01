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


class HERKScheme(BaseScheme):
    """Base scheme for the implementation of Half-Explicit Runge-Kutta
    methods for the incompressible NS equations in 2D."""

    def __init__(self, dimension=cfg.dimension, tree_velocity_x=None,
            tree_velocity_y=None, tree_velocity_z=None, tree_pressure=None,
            tree_vorticity=None, uniform=False,
            st_flag_vx=False, st_flag_vy=False, st_flag_vz=False,
            st_flag_vc=False, st_flag_s=False, low_mach=False):

        BaseScheme.__init__(self, dimension=dimension,
            tree_velocity_x=tree_velocity_x, tree_velocity_y=tree_velocity_y,
            tree_velocity_z=tree_velocity_z, tree_pressure=tree_pressure,
            tree_vorticity=tree_vorticity,
            uniform=uniform, st_flag_vx=st_flag_vx, st_flag_vy=st_flag_vy,
            st_flag_vz=st_flag_vz, st_flag_vc=st_flag_vc, st_flag_s=st_flag_s,
            low_mach=low_mach)

    #def __init__(self, dimension=cfg.dimension, tree_velocity_x=None, tree_velocity_y=None, tree_velocity_z=None, tree_pressure=None, tree_vorticity=None):

    #    if tree_vorticity is not None:
    #        BaseScheme.__init__(self, tree_velocity_x=tree_velocity_x, tree_velocity_y=tree_velocity_y, tree_pressure=tree_pressure, tree_vorticity=tree_vorticity)
    #    else:
    #        BaseScheme.__init__(self, tree_velocity_x=tree_velocity_x, tree_velocity_y=tree_velocity_y, tree_pressure=tree_pressure)

        self.A_coefs = None
        self.B_coefs = None
        self.C_coefs = None

    def make_rhs_ode_x(self, velocity_x, velocity_y, source_term=None):

        if source_term is None:
            return sd.add_scalars(
                sd.mul_num_scalar(cfg.nu, self.velocity_lap_x.apply(velocity_x)),
                sd.mul_num_scalar(-1.,
                self.velocity_adv_x.apply(self.dimension,
                velocity_x, velocity_x, velocity_y, low_mach=self.low_mach)))

        else:
            return sd.add_scalars(
                sd.mul_num_scalar(cfg.nu, self.velocity_lap_x.apply(velocity_x)),
                self.velocity_mass.apply(source_term),
                sd.mul_num_scalar(-1., self.velocity_adv_x.apply(self.dimension,
                    velocity_x, velocity_x, velocity_y, low_mach=self.low_mach)))

    def make_rhs_ade_x(self, velocity_x, velocity_y, pressure, source_term=None):

        if source_term is None:
            return sd.add_scalars(
                sd.mul_num_scalar(cfg.nu, self.velocity_lap_x.apply(velocity_x)),
                sd.mul_num_scalar(-1., sd.add_scalars(
                    self.pressure_grad_x.apply(pressure),
                    self.velocity_adv_x.apply(self.dimension,
                    velocity_x, velocity_x, velocity_y, low_mach=self.low_mach))))

        else:
            return sd.add_scalars(
                sd.mul_num_scalar(cfg.nu, self.velocity_lap_x.apply(velocity_x)),
                self.velocity_mass.apply(source_term),
                sd.mul_num_scalar(-1., sd.add_scalars(
                    self.pressure_grad_x.apply(pressure),
                    self.velocity_adv_x.apply(self.dimension,
                    velocity_x, velocity_x, velocity_y, low_mach=self.low_mach))))

    def make_rhs_ode_y(self, velocity_x, velocity_y, source_term=None):

        if source_term is None:
            return sd.add_scalars(
                sd.mul_num_scalar(cfg.nu, self.velocity_lap_y.apply(velocity_y)),
                sd.mul_num_scalar(-1.,
                self.velocity_adv_y.apply(self.dimension,
                velocity_y, velocity_x, velocity_y, low_mach=self.low_mach)))

        else:
            return sd.add_scalars(
                sd.mul_num_scalar(cfg.nu, self.velocity_lap_y.apply(velocity_y)),
                self.velocity_mass.apply(source_term),
                sd.mul_num_scalar(-1., self.velocity_adv_y.apply(self.dimension,
                    velocity_y, velocity_x, velocity_y, low_mach=self.low_mach)))

    def make_rhs_ade_y(self, velocity_x, velocity_y, pressure, source_term=None):

        if source_term is None:
            return sd.add_scalars(
                sd.mul_num_scalar(cfg.nu, self.velocity_lap_y.apply(velocity_y)),
                sd.mul_num_scalar(-1., sd.add_scalars(
                    self.pressure_grad_y.apply(pressure),
                self.velocity_adv_y.apply(self.dimension,
                velocity_y, velocity_x, velocity_y, low_mach=self.low_mach))))

        else:
            return sd.add_scalars(
                sd.mul_num_scalar(cfg.nu, self.velocity_lap_y.apply(velocity_y)),
                self.velocity_mass.apply(source_term),
                sd.mul_num_scalar(-1., sd.add_scalars(
                    self.pressure_grad_y.apply(pressure),
                    self.velocity_adv_y.apply(self.dimension,
                    velocity_y, velocity_x, velocity_y, low_mach=self.low_mach))))

    def make_rhs_pressure_update(self, velocity_x, velocity_y, coef=1.):
        """Forms the RHS for the Poisson equation for the pressure update."""
        # needs an update to take into account a source term in the continuity
        # equation

        return sd.mul_num_scalar(1./(coef*self.dt),
                sd.add_scalars(self.velocity_div_x.apply(velocity_x),
                self.velocity_div_y.apply(velocity_y)))
        #return sd.add_scalars(self.velocity_div_x.apply(velocity_x),
        #        self.velocity_div_y.apply(velocity_y))

    def projection_velocity_x(self, velocity_x, phi, coef=1.):

        return sd.add_scalars(velocity_x, sd.mul_num_scalar(-1.*(coef*self.dt),
            self.velocity_inverse_mass.apply(self.pressure_grad_x.apply(phi))))
        #return sd.add_scalars(velocity_x, sd.mul_num_scalar(-1.,
        #    self.velocity_inverse_mass.apply(self.pressure_grad_x.apply(phi))))

    def projection_velocity_y(self, velocity_y, phi, coef=1.):

        return sd.add_scalars(velocity_y, sd.mul_num_scalar(-1.*(coef*self.dt),
            self.velocity_inverse_mass.apply(self.pressure_grad_y.apply(phi))))
        #return sd.add_scalars(velocity_y, sd.mul_num_scalar(-1.,
        #    self.velocity_inverse_mass.apply(self.pressure_grad_y.apply(phi))))

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

        self.make_velocity_adv_x(tree_velocity_x, tree_velocity_y)
        self.make_velocity_adv_y(tree_velocity_x, tree_velocity_y)

        self.make_pressure_divgrad()
