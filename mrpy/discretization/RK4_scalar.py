from __future__ import print_function, division

"""This temporal-modules contain the functions needed to comute the advancement in time
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

This module implements the Radau2A scheme. It inherits from the
temporal-impl-RK2-base.py ImplicitRungeKuttaStage2Scheme.
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
from mrpy.discretization.ERK_scalar_base import ERKScheme
import config as cfg


class Scheme(ERKScheme):
    """...

    """

    def __init__(self, tree_scalar=None, diffusion=False, low_mach=False):

        ERKScheme.__init__(self, tree_scalar=tree_scalar, diffusion=diffusion,
                low_mach=low_mach)

        self.A_coefs = {"a21":1/2, "a32":1/2, "a43":1}
        self.B_coefs = {"b1":1/6, "b2":2/6, "b3":2/6, "b4":1/6}
        self.C_coefs = {"c2":1/2, "c3":1/2, "c4":1}

    def advance(self, s=None, v_x=None, v_y=None, t_ini=0, nsp=None,
            density=None, one_over_density=None):

        source_term_2 = None
        source_term_3 = None
        source_term_4 = None
        if self.uniform: #s, v_x and v_y are scalars
            scalar = sd.Scalar()
            scalar.sc = s.sc.copy()
            velocity_x = v_x
            velocity_y = v_y

            g_1 = sd.Scalar()
            g_1.sc = scalar.sc.copy()

            if self.st_flag_s:
                mesh.listing_of_leaves(self.st_tree_s)
                self.compute_source_term(self.st_tree_s, self.st_func_s,
                        t_ini + self.C_coefs["c2"]*self.dt)
                #mesh.listing_of_leaves(self.st_tree_s)
                source_term_2 = sd.Scalar(self.st_tree_s)
                self.compute_source_term(self.st_tree_s, self.st_func_s,
                        t_ini + self.C_coefs["c3"]*self.dt)
                #mesh.listing_of_leaves(self.st_tree_s)
                source_term_3 = sd.Scalar(self.st_tree_s)
                self.compute_source_term(self.st_tree_s, self.st_func_s,
                        t_ini + self.C_coefs["c4"]*self.dt)
                #mesh.listing_of_leaves(self.st_tree_s)
                source_term_4 = sd.Scalar(self.st_tree_s)
            g_2 = sd.add_scalars(
                scalar,
                self.scalar_inverse_mass.apply(
                    sd.mul_num_scalar(self.dt,
                        sd.mul_num_scalar(self.A_coefs["a21"], self.make_rhs(g_1,
                            velocity_x, velocity_y, source_term_2,
                            density=density, one_over_density=one_over_density)))))

            g_3 = sd.add_scalars(
                scalar,
                self.scalar_inverse_mass.apply(
                    sd.mul_num_scalar(self.dt,
                        sd.mul_num_scalar(self.A_coefs["a32"], self.make_rhs(g_2,
                            velocity_x, velocity_y, source_term_3,
                            density=density, one_over_density=one_over_density)))))

            g_4 = sd.add_scalars(
                scalar,
                self.scalar_inverse_mass.apply(
                    sd.mul_num_scalar(self.dt,
                        sd.mul_num_scalar(self.A_coefs["a43"], self.make_rhs(g_3,
                            velocity_x, velocity_y, source_term_4,
                            density=density, one_over_density=one_over_density)))))

            scalar = sd.add_scalars(
                scalar,
                self.scalar_inverse_mass.apply(
                    sd.mul_num_scalar(self.dt, sd.add_scalars(
                        sd.mul_num_scalar(self.B_coefs["b1"], self.make_rhs(g_1,
                            velocity_x, velocity_y,
                            density=density,
                            one_over_density=one_over_density)),
                        sd.mul_num_scalar(self.B_coefs["b2"], self.make_rhs(g_2,
                            velocity_x, velocity_y, source_term_2,
                            density=density,
                            one_over_density=one_over_density)),
                        sd.mul_num_scalar(self.B_coefs["b3"], self.make_rhs(g_3,
                            velocity_x, velocity_y, source_term_3,
                            density=density,
                            one_over_density=one_over_density)),
                        sd.mul_num_scalar(self.B_coefs["b4"], self.make_rhs(g_4,
                            velocity_x, velocity_y, source_term_4,
                            density=density,
                            one_over_density=one_over_density))))))

            s.sc = scalar.sc.copy()

        else: #s, v_x and v_y are trees
            scalar = sd.Scalar(s)
            velocity_x = sd.Scalar(v_x)
            velocity_y = sd.Scalar(v_y)

            g_1 = sd.Scalar(s)

            if self.st_flag_s: #we need to put the st_tree_s to the same grading as s
                op.set_to_same_grading(s, self.st_tree_s)
                op.run_pruning(self.st_tree_s)
                mesh.listing_of_leaves(self.st_tree_s)
                self.compute_source_term(self.st_tree_s, self.st_func_s,
                        t_ini + self.C_coefs["c2"]*self.dt)
                #mesh.listing_of_leaves(self.st_tree_s)
                source_term_2 = sd.Scalar(self.st_tree_s)
                self.compute_source_term(self.st_tree_s, self.st_func_s,
                        t_ini + self.C_coefs["c3"]*self.dt)
                #mesh.listing_of_leaves(self.st_tree_s)
                source_term_3 = sd.Scalar(self.st_tree_s)
                self.compute_source_term(self.st_tree_s, self.st_func_s,
                        t_ini + self.C_coefs["c4"]*self.dt)
                #mesh.listing_of_leaves(self.st_tree_s)
                source_term_4 = sd.Scalar(self.st_tree_s)
            g_2 = sd.add_scalars(
                scalar,
                self.scalar_inverse_mass.apply(
                    sd.mul_num_scalar(self.dt,
                        sd.mul_num_scalar(self.A_coefs["a21"], self.make_rhs(g_1,
                            velocity_x, velocity_y, source_term_2)))))

            g_3 = sd.add_scalars(
                scalar,
                self.scalar_inverse_mass.apply(
                    sd.mul_num_scalar(self.dt,
                        sd.mul_num_scalar(self.A_coefs["a32"], self.make_rhs(g_2,
                            velocity_x, velocity_y, source_term_3)))))

            g_4 = sd.add_scalars(
                scalar,
                self.scalar_inverse_mass.apply(
                    sd.mul_num_scalar(self.dt,
                        sd.mul_num_scalar(self.A_coefs["a43"], self.make_rhs(g_3,
                            velocity_x, velocity_y, source_term_4)))))

            scalar = sd.add_scalars(
                scalar,
                self.scalar_inverse_mass.apply(
                    sd.mul_num_scalar(self.dt, sd.add_scalars(
                        sd.mul_num_scalar(self.B_coefs["b1"], self.make_rhs(g_1,
                            velocity_x, velocity_y)),
                        sd.mul_num_scalar(self.B_coefs["b2"], self.make_rhs(g_2,
                            velocity_x, velocity_y, source_term_2)),
                        sd.mul_num_scalar(self.B_coefs["b3"], self.make_rhs(g_3,
                            velocity_x, velocity_y, source_term_3)),
                        sd.mul_num_scalar(self.B_coefs["b4"], self.make_rhs(g_4,
                            velocity_x, velocity_y, source_term_4))))))

            self.scalar_to_tree(scalar, s)
