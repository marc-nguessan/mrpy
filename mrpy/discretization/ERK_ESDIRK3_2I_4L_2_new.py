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
from mrpy.discretization.ERK4_velocity_base import ERK4Scheme
import config as cfg


class Scheme(ERK4Scheme):
    """...

    """

    def __init__(self, dimension=cfg.dimension, tree_velocity_x=None,
            tree_velocity_y=None, tree_velocity_z=None, tree_pressure=None,
            tree_vorticity=None, uniform=False,
            st_flag_vx=False, st_flag_vy=False, st_flag_vz=False,
            low_mach=False):

        ERK4Scheme.__init__(self, dimension=dimension,
            tree_velocity_x=tree_velocity_x, tree_velocity_y=tree_velocity_y,
            tree_velocity_z=tree_velocity_z, tree_pressure=tree_pressure,
            tree_vorticity=tree_vorticity,
            uniform=uniform, st_flag_vx=st_flag_vx, st_flag_vy=st_flag_vy,
            st_flag_vz=st_flag_vz, low_mach=low_mach)

    #def __init__(self, dimension=cfg.dimension, tree_velocity_x=None,
    #        tree_velocity_y=None, tree_velocity_z=None, tree_pressure=None,
    #        tree_vorticity=None):

    #    if tree_vorticity is not None:
    #        ERK4Scheme.__init__(self, tree_velocity_x=tree_velocity_x, tree_velocity_y=tree_velocity_y, tree_pressure=tree_pressure, tree_vorticity=tree_vorticity)
    #    else:
    #        ERK4Scheme.__init__(self, tree_velocity_x=tree_velocity_x, tree_velocity_y=tree_velocity_y, tree_pressure=tree_pressure)

        l = 1767732205903./4055673282236
        c3 = 3./5
        a32 = c3*(c3 - 2*l)/(4*l)
        b2 = (-2 + 3*c3 + 6*l*(1 - c3))/(12*l*(c3 - 2*l))
        b3 = (1 - 6*l + 6*l*l)/(3*c3*(c3 - 2*l))

        self.B_coefs = {"b1":1. - b2 - b3 - l, "b2":b2, "b3":b3, "b4":l}
        self.C_coefs = {"c1":0, "c2":2*l, "c3":c3, "c4":1.}
        self.A_coefs = {}
        B_coefs = (self.B_coefs["b2"], self.B_coefs["b3"], self.B_coefs["b4"])
        C_coefs = (self.C_coefs["c2"], self.C_coefs["c3"], self.C_coefs["c4"])
        self.compute_A_coefs(B_coefs, C_coefs)

