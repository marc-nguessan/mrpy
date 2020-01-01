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
from mrpy.discretization.HERK_velocity_base import HERKScheme
import config as cfg


class HERK4Scheme(HERKScheme):
    """Base scheme for the implementation of 4-stage Half-Explicit Runge-Kutta
    methods for the NS equations in 2D."""

    def __init__(self, dimension=cfg.dimension, tree_velocity_x=None,
            tree_velocity_y=None, tree_velocity_z=None, tree_pressure=None,
            tree_vorticity=None, uniform=False,
            st_flag_vx=False, st_flag_vy=False, st_flag_vz=False,
            st_flag_vc=False, st_flag_s=False, low_mach=False):

        HERKScheme.__init__(self, dimension=dimension,
            tree_velocity_x=tree_velocity_x, tree_velocity_y=tree_velocity_y,
            tree_velocity_z=tree_velocity_z, tree_pressure=tree_pressure,
            tree_vorticity=tree_vorticity,
            uniform=uniform, st_flag_vx=st_flag_vx, st_flag_vy=st_flag_vy,
            st_flag_vz=st_flag_vz, st_flag_vc=st_flag_vc, st_flag_s=st_flag_s,
            low_mach=low_mach)

    #def __init__(self, dimension=cfg.dimension, tree_velocity_x=None,
    #        tree_velocity_y=None, tree_velocity_z=None, tree_pressure=None,
    #        tree_vorticity=None):

    #    if tree_vorticity is not None:
    #        HERKScheme.__init__(self, tree_velocity_x=tree_velocity_x,
    #                tree_velocity_y=tree_velocity_y,
    #                tree_pressure=tree_pressure, tree_vorticity=tree_vorticity)
    #    else:
    #        HERKScheme.__init__(self, tree_velocity_x=tree_velocity_x,
    #                tree_velocity_y=tree_velocity_y,
    #                tree_pressure=tree_pressure)

    #def compute_A_coefs(self, B_coefs, C_coefs):
    #    """Computes the A_coefs of the ERK method given the B and C coefs.

    #    We use the formulas to obtain a 4th order scheme in 4 stages. They can
    #    be found in Hairer and Wanner, Solving Ordinary Differential Equations
    #    I.
    #    """

    #    b2 = B_coefs[0]
    #    b3 = B_coefs[1]
    #    b4 = B_coefs[2]
    #    c2 = C_coefs[0]
    #    c3 = C_coefs[1]
    #    c4 = C_coefs[2]
    #    self.A_coefs["a43"] = (b3*(1 - c3))/b4
    #    self.A_coefs["a32"] = (1./(b3*b4*c2*(c4 - c3)))*(b4*c4*c2*b2*(1. - c2) + \
    #        self.A_coefs["a43"]*c3*c4*b4*b4 - 1/8.*b4)
    #    self.A_coefs["a42"] = (1./(b3*b4*c2*(c4 - c3)))*(-b3*c3*c2*b2*(1. - c2) - \
    #        self.A_coefs["a43"]*c3*c4*b3*b4 + 1/8.*b3)
    #    self.A_coefs["a21"] = c2
    #    self.A_coefs["a31"] = c3 - self.A_coefs["a32"]
    #    self.A_coefs["a41"] = c4 - self.A_coefs["a43"] - self.A_coefs["a42"]

    def advance(self, v_x=None, v_y=None, v_z=None, p=None, t_ini=0, nsp=None):

    # needs an update to take into account a source term in the continuity
    # equation

        st_rhs_12 = None
        st_rhs_13 = None
        st_rhs_14 = None
        st_rhs_22 = None
        st_rhs_23 = None
        st_rhs_24 = None
        if self.uniform: #v_x, v_y, etc are scalars, and we just advance them
            if self.st_flag_vx:
                mesh.listing_of_leaves(self.st_tree_vx)
                self.compute_source_term(self.st_tree_vx, self.st_func_vx,
                        t_ini + self.C_coefs["c2"]*self.dt)
                #mesh.listing_of_leaves(self.st_tree_vx)
                st_rhs_12 = sd.Scalar(self.st_tree_vx)
                self.compute_source_term(self.st_tree_vx, self.st_func_vx,
                        t_ini + self.C_coefs["c3"]*self.dt)
                #mesh.listing_of_leaves(self.st_tree_vx)
                st_rhs_13 = sd.Scalar(self.st_tree_vx)
                self.compute_source_term(self.st_tree_vx, self.st_func_vx,
                        t_ini + self.C_coefs["c4"]*self.dt)
                #mesh.listing_of_leaves(self.st_tree_vx)
                st_rhs_14 = sd.Scalar(self.st_tree_vx)

            if self.st_flag_vy:
                mesh.listing_of_leaves(self.st_tree_vy)
                self.compute_source_term(self.st_tree_vy, self.st_func_vy,
                    t_ini + self.C_coefs["c2"]*self.dt)
                #mesh.listing_of_leaves(self.st_tree_vy)
                st_rhs_22 = sd.Scalar(self.st_tree_vy)
                self.compute_source_term(self.st_tree_vy, self.st_func_vy,
                        t_ini + self.C_coefs["c3"]*self.dt)
                #mesh.listing_of_leaves(self.st_tree_vy)
                st_rhs_23 = sd.Scalar(self.st_tree_vy)
                self.compute_source_term(self.st_tree_vy, self.st_func_vy,
                        t_ini + self.C_coefs["c4"]*self.dt)
                #mesh.listing_of_leaves(self.st_tree_vy)
                st_rhs_24 = sd.Scalar(self.st_tree_vy)

            g_11, g_21 = sd.Scalar(), sd.Scalar()
            g_11.sc, g_21.sc = v_x.sc.copy(), v_y.sc.copy()

            print("stage 1 done")
            print("")

            g_12 = sd.add_scalars(
                v_x,
                self.velocity_inverse_mass.apply(
                    sd.mul_num_scalar(self.dt,
                        sd.mul_num_scalar(self.A_coefs["a21"],
                            self.make_rhs_ode_x(g_11, g_21, st_rhs_12)))))

            g_22 = sd.add_scalars(
                v_y,
                self.velocity_inverse_mass.apply(
                    sd.mul_num_scalar(self.dt,
                        sd.mul_num_scalar(self.A_coefs["a21"],
                            self.make_rhs_ode_y(g_11, g_21, st_rhs_22)))))

            g_31 = self.solve(self.pressure_divgrad,
                    self.make_rhs_pressure_update(g_12, g_22,
                        self.A_coefs["a21"]))

            g_12 = self.projection_velocity_x(g_12, g_31, self.A_coefs["a21"])
            g_22 = self.projection_velocity_y(g_22, g_31, self.A_coefs["a21"])

            print("stage 2 done")
            print("")

            g_13 = sd.add_scalars(
                v_x,
                self.velocity_inverse_mass.apply(
                    sd.mul_num_scalar(self.dt, sd.add_scalars(
                        sd.mul_num_scalar(self.A_coefs["a31"],
                            self.make_rhs_ade_x(g_11, g_21, g_31, st_rhs_13)),
                        sd.mul_num_scalar(self.A_coefs["a32"],
                            self.make_rhs_ode_x(g_12, g_22, st_rhs_13))))))

            g_23 = sd.add_scalars(
                v_y,
                self.velocity_inverse_mass.apply(
                    sd.mul_num_scalar(self.dt, sd.add_scalars(
                        sd.mul_num_scalar(self.A_coefs["a31"],
                            self.make_rhs_ade_y(g_11, g_21, g_31, st_rhs_23)),
                        sd.mul_num_scalar(self.A_coefs["a32"],
                            self.make_rhs_ode_y(g_12, g_22, st_rhs_23))))))

            g_32 = self.solve(self.pressure_divgrad,
                    self.make_rhs_pressure_update(g_13, g_23,
                        self.A_coefs["a32"]))

            g_13 = self.projection_velocity_x(g_13, g_32, self.A_coefs["a32"])
            g_23 = self.projection_velocity_y(g_23, g_32, self.A_coefs["a32"])

            print("stage 3 done")
            print("")

            g_14 = sd.add_scalars(
                v_x,
                self.velocity_inverse_mass.apply(
                    sd.mul_num_scalar(self.dt, sd.add_scalars(
                        sd.mul_num_scalar(self.A_coefs["a41"],
                            self.make_rhs_ade_x(g_11, g_21, g_31, st_rhs_14)),
                        sd.mul_num_scalar(self.A_coefs["a42"],
                            self.make_rhs_ade_x(g_12, g_22, g_32, st_rhs_14)),
                        sd.mul_num_scalar(self.A_coefs["a43"],
                            self.make_rhs_ode_x(g_13, g_23, st_rhs_14))))))

            g_24 = sd.add_scalars(
                v_y,
                self.velocity_inverse_mass.apply(
                    sd.mul_num_scalar(self.dt, sd.add_scalars(
                        sd.mul_num_scalar(self.A_coefs["a41"],
                            self.make_rhs_ade_y(g_11, g_21, g_31, st_rhs_24)),
                        sd.mul_num_scalar(self.A_coefs["a42"],
                            self.make_rhs_ade_y(g_12, g_22, g_32, st_rhs_24)),
                        sd.mul_num_scalar(self.A_coefs["a43"],
                            self.make_rhs_ode_y(g_13, g_23, st_rhs_24))))))

            g_33 = self.solve(self.pressure_divgrad,
                    self.make_rhs_pressure_update(g_14, g_24,
                        self.A_coefs["a43"]))

            g_14 = self.projection_velocity_x(g_14, g_33, self.A_coefs["a43"])
            g_24 = self.projection_velocity_y(g_24, g_33, self.A_coefs["a43"])

            print("stage 4 done")
            print("")

            g_1final = sd.add_scalars(
                v_x,
                self.velocity_inverse_mass.apply(
                    sd.mul_num_scalar(self.dt, sd.add_scalars(
                        sd.mul_num_scalar(self.B_coefs["b1"],
                            self.make_rhs_ade_x(g_11, g_21, g_31)),
                        sd.mul_num_scalar(self.B_coefs["b2"],
                            self.make_rhs_ade_x(g_12, g_22, g_32, st_rhs_12)),
                        sd.mul_num_scalar(self.B_coefs["b3"],
                            self.make_rhs_ade_x(g_13, g_23, g_33, st_rhs_13)),
                        sd.mul_num_scalar(self.B_coefs["b4"],
                            self.make_rhs_ode_x(g_14, g_24, st_rhs_14))))))

            g_2final = sd.add_scalars(
                v_y,
                self.velocity_inverse_mass.apply(
                    sd.mul_num_scalar(self.dt, sd.add_scalars(
                        sd.mul_num_scalar(self.B_coefs["b1"],
                            self.make_rhs_ade_y(g_11, g_21, g_31)),
                        sd.mul_num_scalar(self.B_coefs["b2"],
                            self.make_rhs_ade_y(g_12, g_22, g_32, st_rhs_22)),
                        sd.mul_num_scalar(self.B_coefs["b3"],
                            self.make_rhs_ade_y(g_13, g_23, g_33, st_rhs_23)),
                        sd.mul_num_scalar(self.B_coefs["b4"],
                            self.make_rhs_ode_y(g_14, g_24, st_rhs_24))))))

            g_34 = self.solve(self.pressure_divgrad,
                    self.make_rhs_pressure_update(g_1final, g_2final,
                    self.B_coefs["b4"]))

            v_x.sc = self.projection_velocity_x(g_1final, g_34,
            self.B_coefs["b4"]).sc.copy()
            v_y.sc = self.projection_velocity_y(g_2final, g_34,
            self.B_coefs["b4"]).sc.copy()

            # The pressure must be the right Lagrange multiplier of the
            # resulting velocity
            p.sc = self.solve(self.pressure_divgrad,
                self.make_rhs_pressure_equation(v_x, v_y,
                st_rhs_12, st_rhs_22), nsp).sc

        else: #v_x, etc are trees
            velocity_x = sd.Scalar(v_x)
            velocity_y = sd.Scalar(v_y)
            pressure = sd.Scalar(p)

            if self.st_flag_vx: #we need to put the st_tree_vx to the same grading as v_x
                op.set_to_same_grading(v_x, self.st_tree_vx)
                op.run_pruning(self.st_tree_vx)
                mesh.listing_of_leaves(self.st_tree_vx)
                self.compute_source_term(self.st_tree_vx, self.st_func_vx,
                        t_ini + self.C_coefs["c2"]*self.dt)
                #mesh.listing_of_leaves(self.st_tree_vx)
                st_rhs_12 = sd.Scalar(self.st_tree_vx)
                self.compute_source_term(self.st_tree_vx, self.st_func_vx,
                        t_ini + self.C_coefs["c3"]*self.dt)
                #mesh.listing_of_leaves(self.st_tree_vx)
                st_rhs_13 = sd.Scalar(self.st_tree_vx)
                self.compute_source_term(self.st_tree_vx, self.st_func_vx,
                        t_ini + self.C_coefs["c4"]*self.dt)
                #mesh.listing_of_leaves(self.st_tree_vx)
                st_rhs_14 = sd.Scalar(self.st_tree_vx)

            if self.st_flag_vy: #we need to put the st_tree_vy to the same grading as v_y
                op.set_to_same_grading(v_y, self.st_tree_vy)
                op.run_pruning(self.st_tree_vy)
                mesh.listing_of_leaves(self.st_tree_vy)
                self.compute_source_term(self.st_tree_vy, self.st_func_vy,
                    t_ini + self.C_coefs["c2"]*self.dt)
                #mesh.listing_of_leaves(self.st_tree_vy)
                st_rhs_22 = sd.Scalar(self.st_tree_vy)
                self.compute_source_term(self.st_tree_vy, self.st_func_vy,
                        t_ini + self.C_coefs["c3"]*self.dt)
                #mesh.listing_of_leaves(self.st_tree_vy)
                st_rhs_23 = sd.Scalar(self.st_tree_vy)
                self.compute_source_term(self.st_tree_vy, self.st_func_vy,
                        t_ini + self.C_coefs["c4"]*self.dt)
                #mesh.listing_of_leaves(self.st_tree_vy)
                st_rhs_24 = sd.Scalar(self.st_tree_vy)

            g_11, g_21 = sd.Scalar(), sd.Scalar()
            g_11.sc, g_21.sc = velocity_x.sc.copy(), velocity_y.sc.copy()

            print("stage 1 done")
            print("")

            g_12 = sd.add_scalars(
                velocity_x,
                self.velocity_inverse_mass.apply(
                    sd.mul_num_scalar(self.dt,
                        sd.mul_num_scalar(self.A_coefs["a21"],
                            self.make_rhs_ode_x(g_11, g_21, st_rhs_12)))))

            g_22 = sd.add_scalars(
                velocity_y,
                self.velocity_inverse_mass.apply(
                    sd.mul_num_scalar(self.dt,
                        sd.mul_num_scalar(self.A_coefs["a21"],
                            self.make_rhs_ode_y(g_11, g_21, st_rhs_22)))))

            g_31 = self.solve(self.pressure_divgrad,
                    self.make_rhs_pressure_update(g_12, g_22,
                        self.A_coefs["a21"]))

            g_12 = self.projection_velocity_x(g_12, g_31, self.A_coefs["a21"])
            g_22 = self.projection_velocity_y(g_22, g_31, self.A_coefs["a21"])

            print("stage 2 done")
            print("")

            g_13 = sd.add_scalars(
                velocity_x,
                self.velocity_inverse_mass.apply(
                    sd.mul_num_scalar(self.dt, sd.add_scalars(
                        sd.mul_num_scalar(self.A_coefs["a31"],
                            self.make_rhs_ade_x(g_11, g_21, g_31, st_rhs_13)),
                        sd.mul_num_scalar(self.A_coefs["a32"],
                            self.make_rhs_ode_x(g_12, g_22, st_rhs_13))))))

            g_23 = sd.add_scalars(
                velocity_y,
                self.velocity_inverse_mass.apply(
                    sd.mul_num_scalar(self.dt, sd.add_scalars(
                        sd.mul_num_scalar(self.A_coefs["a31"],
                            self.make_rhs_ade_y(g_11, g_21, g_31, st_rhs_23)),
                        sd.mul_num_scalar(self.A_coefs["a32"],
                            self.make_rhs_ode_y(g_12, g_22, st_rhs_23))))))

            g_32 = self.solve(self.pressure_divgrad,
                    self.make_rhs_pressure_update(g_13, g_23,
                        self.A_coefs["a32"]))

            g_13 = self.projection_velocity_x(g_13, g_32, self.A_coefs["a32"])
            g_23 = self.projection_velocity_y(g_23, g_32, self.A_coefs["a32"])

            print("stage 3 done")
            print("")

            g_14 = sd.add_scalars(
                velocity_x,
                self.velocity_inverse_mass.apply(
                    sd.mul_num_scalar(self.dt, sd.add_scalars(
                        sd.mul_num_scalar(self.A_coefs["a41"],
                            self.make_rhs_ade_x(g_11, g_21, g_31, st_rhs_14)),
                        sd.mul_num_scalar(self.A_coefs["a42"],
                            self.make_rhs_ade_x(g_12, g_22, g_32, st_rhs_14)),
                        sd.mul_num_scalar(self.A_coefs["a43"],
                            self.make_rhs_ode_x(g_13, g_23, st_rhs_14))))))

            g_24 = sd.add_scalars(
                velocity_y,
                self.velocity_inverse_mass.apply(
                    sd.mul_num_scalar(self.dt, sd.add_scalars(
                        sd.mul_num_scalar(self.A_coefs["a41"],
                            self.make_rhs_ade_y(g_11, g_21, g_31, st_rhs_24)),
                        sd.mul_num_scalar(self.A_coefs["a42"],
                            self.make_rhs_ade_y(g_12, g_22, g_32, st_rhs_24)),
                        sd.mul_num_scalar(self.A_coefs["a43"],
                            self.make_rhs_ode_y(g_13, g_23, st_rhs_24))))))

            g_33 = self.solve(self.pressure_divgrad,
                    self.make_rhs_pressure_update(g_14, g_24,
                        self.A_coefs["a43"]))

            g_14 = self.projection_velocity_x(g_14, g_33, self.A_coefs["a43"])
            g_24 = self.projection_velocity_y(g_24, g_33, self.A_coefs["a43"])

            print("stage 4 done")
            print("")

            g_1final = sd.add_scalars(
                velocity_x,
                self.velocity_inverse_mass.apply(
                    sd.mul_num_scalar(self.dt, sd.add_scalars(
                        sd.mul_num_scalar(self.B_coefs["b1"],
                            self.make_rhs_ade_x(g_11, g_21, g_31)),
                        sd.mul_num_scalar(self.B_coefs["b2"],
                            self.make_rhs_ade_x(g_12, g_22, g_32, st_rhs_12)),
                        sd.mul_num_scalar(self.B_coefs["b3"],
                            self.make_rhs_ade_x(g_13, g_23, g_33, st_rhs_13)),
                        sd.mul_num_scalar(self.B_coefs["b4"],
                            self.make_rhs_ode_x(g_14, g_24, st_rhs_14))))))

            g_2final = sd.add_scalars(
                velocity_y,
                self.velocity_inverse_mass.apply(
                    sd.mul_num_scalar(self.dt, sd.add_scalars(
                        sd.mul_num_scalar(self.B_coefs["b1"],
                            self.make_rhs_ade_y(g_11, g_21, g_31)),
                        sd.mul_num_scalar(self.B_coefs["b2"],
                            self.make_rhs_ade_y(g_12, g_22, g_32, st_rhs_22)),
                        sd.mul_num_scalar(self.B_coefs["b3"],
                            self.make_rhs_ade_y(g_13, g_23, g_33, st_rhs_23)),
                        sd.mul_num_scalar(self.B_coefs["b4"],
                            self.make_rhs_ode_y(g_14, g_24, st_rhs_24))))))

            g_34 = self.solve(self.pressure_divgrad,
                    self.make_rhs_pressure_update(g_1final, g_2final,
                    self.B_coefs["b4"]))

            velocity_x.sc = self.projection_velocity_x(g_1final, g_34,
            self.B_coefs["b4"]).sc.copy()
            velocity_y.sc = self.projection_velocity_y(g_2final, g_34,
            self.B_coefs["b4"]).sc.copy()

            # The pressure must be the right Lagrange multiplier of the
            # resulting velocity
            pressure.sc = self.solve(self.pressure_divgrad,
                self.make_rhs_pressure_equation(velocity_x, velocity_y,
                st_rhs_12, st_rhs_22), nsp).sc

            self.scalar_to_tree(velocity_x, v_x)
            self.scalar_to_tree(velocity_y, v_y)
            self.scalar_to_tree(pressure, p)

    #def advance(self, v_x=None, v_y=None, v_z=None, p=None, t_ini=0, nsp=None):

    ## needs an update to take into account a source term in the continuity
    ## equation

    #    st_rhs_12 = None
    #    st_rhs_13 = None
    #    st_rhs_14 = None
    #    st_rhs_22 = None
    #    st_rhs_23 = None
    #    st_rhs_24 = None
    #    if self.uniform: #v_x, v_y, etc are scalars, and we just advance them
    #        if self.st_flag_vx:
    #            self.compute_source_term(self.st_tree_vx, self.st_func_vx,
    #                    t_ini + self.C_coefs["c2"]*self.dt)
    #            mesh.listing_of_leaves(self.st_tree_vx)
    #            st_rhs_12 = sd.Scalar(self.st_tree_vx)
    #            self.compute_source_term(self.st_tree_vx, self.st_func_vx,
    #                    t_ini + self.C_coefs["c3"]*self.dt)
    #            mesh.listing_of_leaves(self.st_tree_vx)
    #            st_rhs_13 = sd.Scalar(self.st_tree_vx)
    #            self.compute_source_term(self.st_tree_vx, self.st_func_vx,
    #                    t_ini + self.C_coefs["c4"]*self.dt)
    #            mesh.listing_of_leaves(self.st_tree_vx)
    #            st_rhs_14 = sd.Scalar(self.st_tree_vx)

    #        if self.st_flag_vy:
    #            self.compute_source_term(self.st_tree_vy, self.st_func_vy,
    #                t_ini + self.C_coefs["c2"]*self.dt)
    #            mesh.listing_of_leaves(self.st_tree_vy)
    #            st_rhs_22 = sd.Scalar(self.st_tree_vy)
    #            self.compute_source_term(self.st_tree_vy, self.st_func_vy,
    #                    t_ini + self.C_coefs["c3"]*self.dt)
    #            mesh.listing_of_leaves(self.st_tree_vy)
    #            st_rhs_23 = sd.Scalar(self.st_tree_vy)
    #            self.compute_source_term(self.st_tree_vy, self.st_func_vy,
    #                    t_ini + self.C_coefs["c4"]*self.dt)
    #            mesh.listing_of_leaves(self.st_tree_vy)
    #            st_rhs_24 = sd.Scalar(self.st_tree_vy)

    #        g_11, g_21 = sd.Scalar(), sd.Scalar()
    #        g_11.sc, g_21.sc = v_x.sc.copy(), v_y.sc.copy()

    #        print("stage 1 done")
    #        print("")

    #        g_12 = sd.add_scalars(
    #            v_x,
    #            self.velocity_inverse_mass.apply(
    #                sd.mul_num_scalar(self.dt,
    #                    sd.mul_num_scalar(self.A_coefs["a21"],
    #                        self.make_rhs_ode_x(g_11, g_21, st_rhs_12)))))

    #        g_22 = sd.add_scalars(
    #            v_y,
    #            self.velocity_inverse_mass.apply(
    #                sd.mul_num_scalar(self.dt,
    #                    sd.mul_num_scalar(self.A_coefs["a21"],
    #                        self.make_rhs_ode_y(g_11, g_21, st_rhs_22)))))

    #        phi = self.solve(self.pressure_divgrad,
    #                self.make_rhs_pressure_update(g_12, g_22))

    #        g_12 = self.projection_velocity_x(g_12, phi)
    #        g_22 = self.projection_velocity_y(g_22, phi)

    #        print("stage 2 done")
    #        print("")

    #        g_13 = sd.add_scalars(
    #            v_x,
    #            self.velocity_inverse_mass.apply(
    #                sd.mul_num_scalar(self.dt, sd.add_scalars(
    #                    sd.mul_num_scalar(self.A_coefs["a31"],
    #                        self.make_rhs_ode_x(g_11, g_21, st_rhs_13)),
    #                    sd.mul_num_scalar(self.A_coefs["a32"],
    #                        self.make_rhs_ode_x(g_12, g_22, st_rhs_13))))))

    #        g_23 = sd.add_scalars(
    #            v_y,
    #            self.velocity_inverse_mass.apply(
    #                sd.mul_num_scalar(self.dt, sd.add_scalars(
    #                    sd.mul_num_scalar(self.A_coefs["a31"],
    #                        self.make_rhs_ode_y(g_11, g_21, st_rhs_23)),
    #                    sd.mul_num_scalar(self.A_coefs["a32"],
    #                        self.make_rhs_ode_y(g_12, g_22, st_rhs_23))))))

    #        phi = self.solve(self.pressure_divgrad,
    #                self.make_rhs_pressure_update(g_13, g_23))

    #        g_13 = self.projection_velocity_x(g_13, phi)
    #        g_23 = self.projection_velocity_y(g_23, phi)

    #        print("stage 3 done")
    #        print("")

    #        g_14 = sd.add_scalars(
    #            v_x,
    #            self.velocity_inverse_mass.apply(
    #                sd.mul_num_scalar(self.dt, sd.add_scalars(
    #                    sd.mul_num_scalar(self.A_coefs["a41"],
    #                        self.make_rhs_ode_x(g_11, g_21, st_rhs_14)),
    #                    sd.mul_num_scalar(self.A_coefs["a42"],
    #                        self.make_rhs_ode_x(g_12, g_22, st_rhs_14)),
    #                    sd.mul_num_scalar(self.A_coefs["a43"],
    #                        self.make_rhs_ode_x(g_13, g_23, st_rhs_14))))))

    #        g_24 = sd.add_scalars(
    #            v_y,
    #            self.velocity_inverse_mass.apply(
    #                sd.mul_num_scalar(self.dt, sd.add_scalars(
    #                    sd.mul_num_scalar(self.A_coefs["a41"],
    #                        self.make_rhs_ode_y(g_11, g_21, st_rhs_24)),
    #                    sd.mul_num_scalar(self.A_coefs["a42"],
    #                        self.make_rhs_ode_y(g_12, g_22, st_rhs_24)),
    #                    sd.mul_num_scalar(self.A_coefs["a43"],
    #                        self.make_rhs_ode_y(g_13, g_23, st_rhs_24))))))

    #        phi = self.solve(self.pressure_divgrad,
    #                self.make_rhs_pressure_update(g_14, g_24))

    #        g_14 = self.projection_velocity_x(g_14, phi)
    #        g_24 = self.projection_velocity_y(g_24, phi)

    #        print("stage 4 done")
    #        print("")

    #        g_1final = sd.add_scalars(
    #            v_x,
    #            self.velocity_inverse_mass.apply(
    #                sd.mul_num_scalar(self.dt, sd.add_scalars(
    #                    sd.mul_num_scalar(self.B_coefs["b1"],
    #                        self.make_rhs_ode_x(g_11, g_21)),
    #                    sd.mul_num_scalar(self.B_coefs["b2"],
    #                        self.make_rhs_ode_x(g_12, g_22, st_rhs_12)),
    #                    sd.mul_num_scalar(self.B_coefs["b3"],
    #                        self.make_rhs_ode_x(g_13, g_23, st_rhs_13)),
    #                    sd.mul_num_scalar(self.B_coefs["b4"],
    #                        self.make_rhs_ode_x(g_14, g_24, st_rhs_14))))))

    #        g_2final = sd.add_scalars(
    #            v_y,
    #            self.velocity_inverse_mass.apply(
    #                sd.mul_num_scalar(self.dt, sd.add_scalars(
    #                    sd.mul_num_scalar(self.B_coefs["b1"],
    #                        self.make_rhs_ode_y(g_11, g_21)),
    #                    sd.mul_num_scalar(self.B_coefs["b2"],
    #                        self.make_rhs_ode_y(g_12, g_22, st_rhs_22)),
    #                    sd.mul_num_scalar(self.B_coefs["b3"],
    #                        self.make_rhs_ode_y(g_13, g_23, st_rhs_23)),
    #                    sd.mul_num_scalar(self.B_coefs["b4"],
    #                        self.make_rhs_ode_y(g_14, g_24, st_rhs_24))))))

    #        phi = self.solve(self.pressure_divgrad,
    #                self.make_rhs_pressure_update(g_1final, g_2final))

    #        v_x.sc = self.projection_velocity_x(g_1final, phi).sc.copy()
    #        v_y.sc = self.projection_velocity_y(g_2final, phi).sc.copy()

    #        # The pressure must be the right Lagrange multiplier of the
    #        # resulting velocity
    #        p.sc = self.solve(self.pressure_divgrad,
    #            self.make_rhs_pressure_equation(v_x, v_y,
    #            st_rhs_12, st_rhs_22), nsp).sc

    #    else: #v_x, etc are trees
    #        velocity_x = sd.Scalar(v_x)
    #        velocity_y = sd.Scalar(v_y)
    #        pressure = sd.Scalar(p)

    #        if self.st_flag_vx: #we need to put the st_tree_vx to the same grading as v_x
    #            op.set_to_same_grading(v_x, self.st_tree_vx)
    #            op.run_pruning(self.st_tree_vx)
    #            self.compute_source_term(self.st_tree_vx, self.st_func_vx,
    #                    t_ini + self.C_coefs["c2"]*self.dt)
    #            mesh.listing_of_leaves(self.st_tree_vx)
    #            st_rhs_12 = sd.Scalar(self.st_tree_vx)
    #            self.compute_source_term(self.st_tree_vx, self.st_func_vx,
    #                    t_ini + self.C_coefs["c3"]*self.dt)
    #            mesh.listing_of_leaves(self.st_tree_vx)
    #            st_rhs_13 = sd.Scalar(self.st_tree_vx)
    #            self.compute_source_term(self.st_tree_vx, self.st_func_vx,
    #                    t_ini + self.C_coefs["c4"]*self.dt)
    #            mesh.listing_of_leaves(self.st_tree_vx)
    #            st_rhs_14 = sd.Scalar(self.st_tree_vx)

    #        if self.st_flag_vy: #we need to put the st_tree_vy to the same grading as v_y
    #            op.set_to_same_grading(v_y, self.st_tree_vy)
    #            op.run_pruning(self.st_tree_vy)
    #            self.compute_source_term(self.st_tree_vy, self.st_func_vy,
    #                t_ini + self.C_coefs["c2"]*self.dt)
    #            mesh.listing_of_leaves(self.st_tree_vy)
    #            st_rhs_22 = sd.Scalar(self.st_tree_vy)
    #            self.compute_source_term(self.st_tree_vy, self.st_func_vy,
    #                    t_ini + self.C_coefs["c3"]*self.dt)
    #            mesh.listing_of_leaves(self.st_tree_vy)
    #            st_rhs_23 = sd.Scalar(self.st_tree_vy)
    #            self.compute_source_term(self.st_tree_vy, self.st_func_vy,
    #                    t_ini + self.C_coefs["c4"]*self.dt)
    #            mesh.listing_of_leaves(self.st_tree_vy)
    #            st_rhs_24 = sd.Scalar(self.st_tree_vy)

    #        g_11, g_21 = sd.Scalar(), sd.Scalar()
    #        g_11.sc, g_21.sc = velocity_x.sc.copy(), velocity_y.sc.copy()

    #        print("stage 1 done")
    #        print("")

    #        g_12 = sd.add_scalars(
    #            velocity_x,
    #            self.velocity_inverse_mass.apply(
    #                sd.mul_num_scalar(self.dt,
    #                    sd.mul_num_scalar(self.A_coefs["a21"],
    #                        self.make_rhs_ode_x(g_11, g_21, st_rhs_12)))))

    #        g_22 = sd.add_scalars(
    #            velocity_y,
    #            self.velocity_inverse_mass.apply(
    #                sd.mul_num_scalar(self.dt,
    #                    sd.mul_num_scalar(self.A_coefs["a21"],
    #                        self.make_rhs_ode_y(g_11, g_21, st_rhs_22)))))

    #        phi = self.solve(self.pressure_divgrad,
    #                self.make_rhs_pressure_update(g_12, g_22))

    #        g_12 = self.projection_velocity_x(g_12, phi)
    #        g_22 = self.projection_velocity_x(g_22, phi)

    #        print("stage 2 done")
    #        print("")

    #        g_13 = sd.add_scalars(
    #            velocity_x,
    #            self.velocity_inverse_mass.apply(
    #                sd.mul_num_scalar(self.dt, sd.add_scalars(
    #                    sd.mul_num_scalar(self.A_coefs["a31"],
    #                        self.make_rhs_ode_x(g_11, g_21, st_rhs_13)),
    #                    sd.mul_num_scalar(self.A_coefs["a32"],
    #                        self.make_rhs_ode_x(g_12, g_22, st_rhs_13))))))

    #        g_23 = sd.add_scalars(
    #            velocity_y,
    #            self.velocity_inverse_mass.apply(
    #                sd.mul_num_scalar(self.dt, sd.add_scalars(
    #                    sd.mul_num_scalar(self.A_coefs["a31"],
    #                        self.make_rhs_ode_y(g_11, g_21, st_rhs_23)),
    #                    sd.mul_num_scalar(self.A_coefs["a32"],
    #                        self.make_rhs_ode_y(g_12, g_22, st_rhs_23))))))

    #        phi = self.solve(self.pressure_divgrad,
    #                self.make_rhs_pressure_update(g_13, g_23))

    #        g_13 = self.projection_velocity_x(g_13, phi)
    #        g_23 = self.projection_velocity_x(g_23, phi)

    #        print("stage 3 done")
    #        print("")

    #        g_14 = sd.add_scalars(
    #            velocity_x,
    #            self.velocity_inverse_mass.apply(
    #                sd.mul_num_scalar(self.dt, sd.add_scalars(
    #                    sd.mul_num_scalar(self.A_coefs["a41"],
    #                        self.make_rhs_ode_x(g_11, g_21, st_rhs_14)),
    #                    sd.mul_num_scalar(self.A_coefs["a42"],
    #                        self.make_rhs_ode_x(g_12, g_22, st_rhs_14)),
    #                    sd.mul_num_scalar(self.A_coefs["a43"],
    #                        self.make_rhs_ode_x(g_13, g_23, st_rhs_14))))))

    #        g_24 = sd.add_scalars(
    #            velocity_y,
    #            self.velocity_inverse_mass.apply(
    #                sd.mul_num_scalar(self.dt, sd.add_scalars(
    #                    sd.mul_num_scalar(self.A_coefs["a41"],
    #                        self.make_rhs_ode_y(g_11, g_21, st_rhs_24)),
    #                    sd.mul_num_scalar(self.A_coefs["a42"],
    #                        self.make_rhs_ode_y(g_12, g_22, st_rhs_24)),
    #                    sd.mul_num_scalar(self.A_coefs["a43"],
    #                        self.make_rhs_ode_y(g_13, g_23, st_rhs_24))))))

    #        phi = self.solve(self.pressure_divgrad,
    #                self.make_rhs_pressure_update(g_14, g_24))

    #        g_14 = self.projection_velocity_x(g_14, phi)
    #        g_24 = self.projection_velocity_x(g_24, phi)

    #        print("stage 4 done")
    #        print("")

    #        g_1final = sd.add_scalars(
    #            velocity_x,
    #            self.velocity_inverse_mass.apply(
    #                sd.mul_num_scalar(self.dt, sd.add_scalars(
    #                    sd.mul_num_scalar(self.B_coefs["b1"],
    #                        self.make_rhs_ode_x(g_11, g_21)),
    #                    sd.mul_num_scalar(self.B_coefs["b2"],
    #                        self.make_rhs_ode_x(g_12, g_22, st_rhs_12)),
    #                    sd.mul_num_scalar(self.B_coefs["b3"],
    #                        self.make_rhs_ode_x(g_13, g_23, st_rhs_13)),
    #                    sd.mul_num_scalar(self.B_coefs["b4"],
    #                        self.make_rhs_ode_x(g_14, g_24, st_rhs_14))))))

    #        g_2final = sd.add_scalars(
    #            velocity_y,
    #            self.velocity_inverse_mass.apply(
    #                sd.mul_num_scalar(self.dt, sd.add_scalars(
    #                    sd.mul_num_scalar(self.B_coefs["b1"],
    #                        self.make_rhs_ode_y(g_11, g_21)),
    #                    sd.mul_num_scalar(self.B_coefs["b2"],
    #                        self.make_rhs_ode_y(g_12, g_22, st_rhs_22)),
    #                    sd.mul_num_scalar(self.B_coefs["b3"],
    #                        self.make_rhs_ode_y(g_13, g_23, st_rhs_23)),
    #                    sd.mul_num_scalar(self.B_coefs["b4"],
    #                        self.make_rhs_ode_y(g_14, g_24, st_rhs_24))))))

    #        phi = self.solve(self.pressure_divgrad,
    #                self.make_rhs_pressure_update(g_1final, g_2final))

    #        velocity_x.sc = self.projection_velocity_x(g_1final, phi).sc.copy()
    #        velocity_y.sc = self.projection_velocity_y(g_2final, phi).sc.copy()

    #        # The pressure must be the right Lagrange multiplier of the
    #        # resulting velocity
    #        pressure.sc = self.solve(self.pressure_divgrad,
    #            self.make_rhs_pressure_equation(velocity_x, velocity_y,
    #            st_rhs_12, st_rhs_22), nsp).sc

    #        self.scalar_to_tree(velocity_x, v_x)
    #        self.scalar_to_tree(velocity_y, v_y)
    #        self.scalar_to_tree(pressure, p)
