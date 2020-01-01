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
import mrpy.discretization.ESDIRK3_2I_4L_2_so as IrkScheme
import mrpy.discretization.ERK_ESDIRK3_2I_4L_2_so as ErkScheme
import config as cfg


class Scheme(BaseScheme):

    def __init__(self, dimension=cfg.dimension, tree_velocity_x=None,
            tree_velocity_y=None, tree_velocity_z=None, tree_pressure=None,
            tree_vorticity=None):

        if tree_vorticity is not None:
            BaseScheme.__init__(self, tree_velocity_x=tree_velocity_x,
                tree_velocity_y=tree_velocity_y,
                tree_pressure=tree_pressure, tree_vorticity=tree_vorticity)
        else:
            BaseScheme.__init__(self, tree_velocity_x=tree_velocity_x,
                tree_velocity_y=tree_velocity_y,
                tree_pressure=tree_pressure)

        self.irk = IrkScheme.Scheme(tree_velocity_x=tree_velocity_x,
                tree_velocity_y=tree_velocity_y,
                tree_pressure=tree_pressure, tree_vorticity=tree_vorticity)

        self.erk = ErkScheme.Scheme(tree_velocity_x=tree_velocity_x,
                tree_velocity_y=tree_velocity_y,
                tree_pressure=tree_pressure, tree_vorticity=tree_vorticity)

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

        self.irk.make_operators(tree_velocity_x, tree_velocity_y, tree_pressure)
        self.erk.make_operators(tree_velocity_x, tree_velocity_y, tree_pressure)

    def make_ksps(self):

        self.irk.make_ksps()
        self.erk.make_ksps()

    def advance(self, v_x=None, v_y=None, v_z=None, p=None, t_ini=0, nsp=None):

        st_rhs_12 = None
        st_rhs_13 = None
        st_rhs_14 = None
        st_rhs_22 = None
        st_rhs_23 = None
        st_rhs_24 = None
        st_rhs_32 = None
        st_rhs_33 = None
        st_rhs_34 = None
        if self.uniform: #v_x, v_y, etc are scalars, and we just advance them
            if self.st_flag_vx:
                mesh.listing_of_leaves(self.st_tree_vx)
                self.compute_source_term(self.st_tree_vx, self.st_func_vx,
                        t_ini + self.irk.C_coefs["c2"]*self.dt)
                #mesh.listing_of_leaves(self.st_tree_vx)
                st_rhs_12 = sd.Scalar(self.st_tree_vx)
                self.compute_source_term(self.st_tree_vx, self.st_func_vx,
                        t_ini + self.irk.C_coefs["c3"]*self.dt)
                #mesh.listing_of_leaves(self.st_tree_vx)
                st_rhs_13 = sd.Scalar(self.st_tree_vx)
                self.compute_source_term(self.st_tree_vx, self.st_func_vx,
                        t_ini + self.irk.C_coefs["c4"]*self.dt)
                #mesh.listing_of_leaves(self.st_tree_vx)
                st_rhs_14 = sd.Scalar(self.st_tree_vx)

            if self.st_flag_vy:
                mesh.listing_of_leaves(self.st_tree_vy)
                self.compute_source_term(self.st_tree_vy, self.st_func_vy,
                    t_ini + self.irk.C_coefs["c2"]*self.dt)
                #mesh.listing_of_leaves(self.st_tree_vy)
                st_rhs_22 = sd.Scalar(self.st_tree_vy)
                self.compute_source_term(self.st_tree_vy, self.st_func_vy,
                        t_ini + self.irk.C_coefs["c3"]*self.dt)
                #mesh.listing_of_leaves(self.st_tree_vy)
                st_rhs_23 = sd.Scalar(self.st_tree_vy)
                self.compute_source_term(self.st_tree_vy, self.st_func_vy,
                        t_ini + self.irk.C_coefs["c4"]*self.dt)
                #mesh.listing_of_leaves(self.st_tree_vy)
                st_rhs_24 = sd.Scalar(self.st_tree_vy)

            if self.st_flag_vc:
                mesh.listing_of_leaves(self.st_tree_vc)
                self.compute_source_term(self.st_tree_vc, self.st_func_vc,
                        t_ini + self.irk.C_coefs["c2"]*self.dt)
                #mesh.listing_of_leaves(self.st_tree_vc)
                st_rhs_32 = sd.Scalar(self.st_tree_vc)
                self.compute_source_term(self.st_tree_vc, self.st_func_vc,
                        t_ini + self.irk.C_coefs["c3"]*self.dt)
                #mesh.listing_of_leaves(self.st_tree_vc)
                st_rhs_33 = sd.Scalar(self.st_tree_vc)
                self.compute_source_term(self.st_tree_vc, self.st_func_vc,
                        t_ini + self.irk.C_coefs["c4"]*self.dt)
                #mesh.listing_of_leaves(self.st_tree_vc)
                st_rhs_34 = sd.Scalar(self.st_tree_vc)

            g_11, g_21, g_31 = sd.Scalar(), sd.Scalar(), sd.Scalar()
            g_11.sc, g_21.sc, g_31.sc = v_x.sc.copy(), v_y.sc.copy(), p.sc.copy()

            print("stage 1 done")
            print("")

            g_12, g_22, g_32 = sd.Scalar(), sd.Scalar(), sd.Scalar()
            g_12.sc, g_22.sc, g_32.sc = g_11.sc.copy(), g_21.sc.copy(), g_31.sc.copy()
            rhs_momentum_x = sd.add_scalars(
                sd.mul_num_scalar(1/self.dt, self.irk.velocity_mass.apply(v_x)),
                sd.mul_num_scalar(self.erk.A_coefs["a21"],
                    self.erk.make_rhs_ode_x(g_11, g_21)),
                sd.mul_num_scalar(self.irk.A_coefs["a21"], self.irk.make_rhs_dae_x(g_11,
                    g_31, st_rhs_12)))
            rhs_momentum_y = sd.add_scalars(
                sd.mul_num_scalar(1/self.dt, self.irk.velocity_mass.apply(v_y)),
                sd.mul_num_scalar(self.erk.A_coefs["a21"],
                    self.erk.make_rhs_ode_y(g_11, g_21)),
                sd.mul_num_scalar(self.irk.A_coefs["a21"], self.irk.make_rhs_dae_y(g_21,
                    g_31, st_rhs_22)))

            self.irk.uzawa_solver_internal("2", velocity_x=g_12, velocity_y=g_22,
                pressure=g_32, rhs_momentum_x=rhs_momentum_x,
                rhs_momentum_y=rhs_momentum_y, rhs_continuity=st_rhs_32)

            print("stage 2 done")
            print("")

            g_13, g_23, g_33 = sd.Scalar(), sd.Scalar(), sd.Scalar()
            g_13.sc, g_23.sc, g_33.sc = g_12.sc.copy(), g_22.sc.copy(), g_32.sc.copy()
            rhs_momentum_x = sd.add_scalars(
                sd.mul_num_scalar(1/self.dt, self.irk.velocity_mass.apply(v_x)),
                sd.mul_num_scalar(self.erk.A_coefs["a31"],
                    self.erk.make_rhs_ode_x(g_11, g_21)),
                sd.mul_num_scalar(self.erk.A_coefs["a32"],
                    self.erk.make_rhs_ode_x(g_12, g_22)),
                sd.mul_num_scalar(self.irk.A_coefs["a31"], self.irk.make_rhs_dae_x(g_11,
                    g_31, st_rhs_13)),
                sd.mul_num_scalar(self.irk.A_coefs["a32"], self.irk.make_rhs_dae_x(g_12,
                    g_32, st_rhs_13)))
            rhs_momentum_y = sd.add_scalars(
                sd.mul_num_scalar(1/self.dt, self.irk.velocity_mass.apply(v_y)),
                sd.mul_num_scalar(self.erk.A_coefs["a31"],
                    self.erk.make_rhs_ode_y(g_11, g_21)),
                sd.mul_num_scalar(self.erk.A_coefs["a32"],
                    self.erk.make_rhs_ode_y(g_12, g_22)),
                sd.mul_num_scalar(self.irk.A_coefs["a31"], self.irk.make_rhs_dae_y(g_21,
                    g_31, st_rhs_23)),
                sd.mul_num_scalar(self.irk.A_coefs["a32"], self.irk.make_rhs_dae_y(g_22,
                    g_32, st_rhs_23)))

            self.irk.uzawa_solver_internal("3", velocity_x=g_13, velocity_y=g_23,
                pressure=g_33, rhs_momentum_x=rhs_momentum_x,
                rhs_momentum_y=rhs_momentum_y, rhs_continuity=st_rhs_33)

            print("stage 3 done")
            print("")
            #quit()

            g_14, g_24, g_34 = sd.Scalar(), sd.Scalar(), sd.Scalar()
            g_14.sc, g_24.sc, g_34.sc = g_13.sc.copy(), g_23.sc.copy(), g_33.sc.copy()
            rhs_momentum_x = sd.add_scalars(
                sd.mul_num_scalar(1/self.dt, self.irk.velocity_mass.apply(v_x)),
                sd.mul_num_scalar(self.erk.A_coefs["a41"],
                    self.erk.make_rhs_ode_x(g_11, g_21)),
                sd.mul_num_scalar(self.erk.A_coefs["a42"],
                    self.erk.make_rhs_ode_x(g_12, g_22)),
                sd.mul_num_scalar(self.erk.A_coefs["a43"],
                    self.erk.make_rhs_ode_x(g_13, g_23)),
                sd.mul_num_scalar(self.irk.A_coefs["a41"], self.irk.make_rhs_dae_x(g_11,
                    g_31, st_rhs_14)),
                sd.mul_num_scalar(self.irk.A_coefs["a42"], self.irk.make_rhs_dae_x(g_12,
                    g_32, st_rhs_14)),
                sd.mul_num_scalar(self.irk.A_coefs["a43"], self.irk.make_rhs_dae_x(g_13,
                    g_33, st_rhs_14)))
            rhs_momentum_y = sd.add_scalars(
                sd.mul_num_scalar(1/self.dt, self.irk.velocity_mass.apply(v_y)),
                sd.mul_num_scalar(self.erk.A_coefs["a41"],
                    self.erk.make_rhs_ode_y(g_11, g_21)),
                sd.mul_num_scalar(self.erk.A_coefs["a42"],
                    self.erk.make_rhs_ode_y(g_12, g_22)),
                sd.mul_num_scalar(self.erk.A_coefs["a43"],
                    self.erk.make_rhs_ode_y(g_13, g_23)),
                sd.mul_num_scalar(self.irk.A_coefs["a41"], self.irk.make_rhs_dae_y(g_21,
                    g_31, st_rhs_24)),
                sd.mul_num_scalar(self.irk.A_coefs["a42"], self.irk.make_rhs_dae_y(g_22,
                    g_32, st_rhs_24)),
                sd.mul_num_scalar(self.irk.A_coefs["a43"], self.irk.make_rhs_dae_y(g_23,
                    g_33, st_rhs_24)))

            self.irk.uzawa_solver_internal("4", velocity_x=g_14, velocity_y=g_24,
                pressure=g_34, rhs_momentum_x=rhs_momentum_x,
                rhs_momentum_y=rhs_momentum_y, rhs_continuity=st_rhs_34)

            print("stage 4 done")
            print("")

            #g_1f, g_2f, g_3f = sd.Scalar(), sd.Scalar(), sd.Scalar()
            #g_1f.sc, g_2f.sc, g_3f.sc = g_14.sc.copy(), g_24.sc.copy(), g_34.sc.copy()
            #rhs_momentum_x = sd.add_scalars(
            #    sd.mul_num_scalar(1/self.dt, self.irk.velocity_mass.apply(v_x)),
            #    sd.mul_num_scalar(self.erk.B_coefs["b1"],
            #        self.erk.make_rhs_ode_x(g_11, g_21)),
            #    sd.mul_num_scalar(self.erk.B_coefs["b2"],
            #        self.erk.make_rhs_ode_x(g_12, g_22)),
            #    sd.mul_num_scalar(self.erk.B_coefs["b3"],
            #        self.erk.make_rhs_ode_x(g_13, g_23)),
            #    sd.mul_num_scalar(self.erk.B_coefs["b4"],
            #        self.erk.make_rhs_ode_x(g_14, g_24)),
            #    sd.mul_num_scalar(self.irk.A_coefs["a41"], self.irk.make_rhs_dae_x(g_11,
            #        g_31, st_rhs_14)),
            #    sd.mul_num_scalar(self.irk.A_coefs["a42"], self.irk.make_rhs_dae_x(g_12,
            #        g_32, st_rhs_14)),
            #    sd.mul_num_scalar(self.irk.A_coefs["a43"], self.irk.make_rhs_dae_x(g_13,
            #        g_33, st_rhs_14)))
            #rhs_momentum_y = sd.add_scalars(
            #    sd.mul_num_scalar(1/self.dt, self.irk.velocity_mass.apply(v_y)),
            #    sd.mul_num_scalar(self.erk.B_coefs["b1"],
            #        self.erk.make_rhs_ode_y(g_11, g_21)),
            #    sd.mul_num_scalar(self.erk.B_coefs["b2"],
            #        self.erk.make_rhs_ode_y(g_12, g_22)),
            #    sd.mul_num_scalar(self.erk.B_coefs["b3"],
            #        self.erk.make_rhs_ode_y(g_13, g_23)),
            #    sd.mul_num_scalar(self.erk.B_coefs["b4"],
            #        self.erk.make_rhs_ode_y(g_14, g_24)),
            #    sd.mul_num_scalar(self.irk.A_coefs["a41"], self.irk.make_rhs_dae_y(g_21,
            #        g_31, st_rhs_24)),
            #    sd.mul_num_scalar(self.irk.A_coefs["a42"], self.irk.make_rhs_dae_y(g_22,
            #        g_32, st_rhs_24)),
            #    sd.mul_num_scalar(self.irk.A_coefs["a43"], self.irk.make_rhs_dae_y(g_23,
            #        g_33, st_rhs_24)))

            #self.irk.uzawa_solver_internal("4", velocity_x=g_1f, velocity_y=g_2f,
            #    pressure=g_3f, rhs_momentum_x=rhs_momentum_x,
            #    rhs_momentum_y=rhs_momentum_y, rhs_continuity=st_rhs_34)

            #print("final stage done")
            #print("")

            g_1f, g_2f, g_3f = sd.Scalar(), sd.Scalar(), sd.Scalar()
            #g_1f.sc, g_2f.sc, g_3f.sc = g_14.sc.copy(), g_24.sc.copy(), g_34.sc.copy()
            g_1f = sd.add_scalars(
                v_x,
                self.velocity_inverse_mass.apply(
                sd.mul_num_scalar(self.dt, sd.add_scalars(
                    sd.mul_num_scalar(self.erk.B_coefs["b1"],
                        self.erk.make_rhs_ode_x(g_11, g_21)),
                    sd.mul_num_scalar(self.erk.B_coefs["b2"],
                        self.erk.make_rhs_ode_x(g_12, g_22)),
                    sd.mul_num_scalar(self.erk.B_coefs["b3"],
                        self.erk.make_rhs_ode_x(g_13, g_23)),
                    sd.mul_num_scalar(self.erk.B_coefs["b4"],
                        self.erk.make_rhs_ode_x(g_14, g_24)),
                    sd.mul_num_scalar(self.irk.A_coefs["a41"],
                        self.irk.make_rhs_dae_x(g_11,
                        g_31, st_rhs_14)),
                    sd.mul_num_scalar(self.irk.A_coefs["a42"],
                        self.irk.make_rhs_dae_x(g_12,
                        g_32, st_rhs_14)),
                    sd.mul_num_scalar(self.irk.A_coefs["a43"],
                        self.irk.make_rhs_dae_x(g_13,
                        g_33, st_rhs_14)),
                    sd.mul_num_scalar(self.irk.A_coefs["a44"],
                        self.irk.make_rhs_dae_x(g_14,
                        g_34, st_rhs_14))))))
            g_2f = sd.add_scalars(
                v_y,
                self.velocity_inverse_mass.apply(
                sd.mul_num_scalar(self.dt, sd.add_scalars(
                    sd.mul_num_scalar(self.erk.B_coefs["b1"],
                        self.erk.make_rhs_ode_y(g_11, g_21)),
                    sd.mul_num_scalar(self.erk.B_coefs["b2"],
                        self.erk.make_rhs_ode_y(g_12, g_22)),
                    sd.mul_num_scalar(self.erk.B_coefs["b3"],
                        self.erk.make_rhs_ode_y(g_13, g_23)),
                    sd.mul_num_scalar(self.erk.B_coefs["b4"],
                        self.erk.make_rhs_ode_y(g_14, g_24)),
                    sd.mul_num_scalar(self.irk.A_coefs["a41"],
                        self.irk.make_rhs_dae_y(g_21,
                        g_31, st_rhs_24)),
                    sd.mul_num_scalar(self.irk.A_coefs["a42"],
                        self.irk.make_rhs_dae_y(g_22,
                        g_32, st_rhs_24)),
                    sd.mul_num_scalar(self.irk.A_coefs["a43"],
                        self.irk.make_rhs_dae_y(g_23,
                        g_33, st_rhs_24)),
                    sd.mul_num_scalar(self.irk.A_coefs["a44"],
                        self.irk.make_rhs_dae_y(g_24,
                        g_34, st_rhs_24))))))

            print("final stage done")
            print("")

            #v_x.sc, v_y.sc, p.sc = g_14.sc.copy(), g_24.sc.copy(), g_34.sc.copy()
            #v_x.sc, v_y.sc, p.sc = g_1f.sc.copy(), g_2f.sc.copy(), g_3f.sc.copy()
            v_x.sc, v_y.sc, p.sc = g_1f.sc.copy(), g_2f.sc.copy(), g_34.sc.copy()
            # Do we need to recompute the right pressure?

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
                        t_ini + self.irk.C_coefs["c2"]*self.dt)
                #mesh.listing_of_leaves(self.st_tree_vx)
                st_rhs_12 = sd.Scalar(self.st_tree_vx)
                self.compute_source_term(self.st_tree_vx, self.st_func_vx,
                        t_ini + self.irk.C_coefs["c3"]*self.dt)
                #mesh.listing_of_leaves(self.st_tree_vx)
                st_rhs_13 = sd.Scalar(self.st_tree_vx)
                self.compute_source_term(self.st_tree_vx, self.st_func_vx,
                        t_ini + self.irk.C_coefs["c4"]*self.dt)
                #mesh.listing_of_leaves(self.st_tree_vx)
                st_rhs_14 = sd.Scalar(self.st_tree_vx)

            if self.st_flag_vy: #we need to put the st_tree_vy to the same grading as v_y
                op.set_to_same_grading(v_y, self.st_tree_vy)
                op.run_pruning(self.st_tree_vy)
                mesh.listing_of_leaves(self.st_tree_vy)
                self.compute_source_term(self.st_tree_vy, self.st_func_vy,
                    t_ini + self.irk.C_coefs["c2"]*self.dt)
                #mesh.listing_of_leaves(self.st_tree_vy)
                st_rhs_22 = sd.Scalar(self.st_tree_vy)
                self.compute_source_term(self.st_tree_vy, self.st_func_vy,
                        t_ini + self.irk.C_coefs["c3"]*self.dt)
                #mesh.listing_of_leaves(self.st_tree_vy)
                st_rhs_23 = sd.Scalar(self.st_tree_vy)
                self.compute_source_term(self.st_tree_vy, self.st_func_vy,
                        t_ini + self.irk.C_coefs["c4"]*self.dt)
                #mesh.listing_of_leaves(self.st_tree_vy)
                st_rhs_24 = sd.Scalar(self.st_tree_vy)

            if self.st_flag_vc: #we need to put the st_tree_vy to the same grading as v_y
                op.set_to_same_grading(v_x, self.st_tree_vc)
                op.run_pruning(self.st_tree_vc)
                mesh.listing_of_leaves(self.st_tree_vc)
                self.compute_source_term(self.st_tree_vc, self.st_func_vc,
                        t_ini + self.irk.C_coefs["c2"]*self.dt)
                #mesh.listing_of_leaves(self.st_tree_vc)
                st_rhs_32 = sd.Scalar(self.st_tree_vc)
                self.compute_source_term(self.st_tree_vc, self.st_func_vc,
                        t_ini + self.irk.C_coefs["c3"]*self.dt)
                #mesh.listing_of_leaves(self.st_tree_vc)
                st_rhs_33 = sd.Scalar(self.st_tree_vc)
                self.compute_source_term(self.st_tree_vc, self.st_func_vc,
                        t_ini + self.irk.C_coefs["c4"]*self.dt)
                #mesh.listing_of_leaves(self.st_tree_vc)
                st_rhs_34 = sd.Scalar(self.st_tree_vc)

            g_11, g_21, g_31 = sd.Scalar(), sd.Scalar(), sd.Scalar()
            g_11.sc, g_21.sc, g_31.sc = velocity_x.sc.copy(), \
            velocity_y.sc.copy(), pressure.sc.copy()

            print("stage 1 done")
            print("")

            g_12, g_22, g_32 = sd.Scalar(), sd.Scalar(), sd.Scalar()
            g_12.sc, g_22.sc, g_32.sc = g_11.sc.copy(), g_21.sc.copy(), g_31.sc.copy()
            rhs_momentum_x = sd.add_scalars(
                sd.mul_num_scalar(1/self.dt, self.irk.velocity_mass.apply(velocity_x)),
                sd.mul_num_scalar(self.erk.A_coefs["a21"],
                    self.erk.make_rhs_ode_x(g_11, g_21)),
                sd.mul_num_scalar(self.irk.A_coefs["a21"], self.irk.make_rhs_dae_x(g_11,
                    g_31, st_rhs_12)))
            rhs_momentum_y = sd.add_scalars(
                sd.mul_num_scalar(1/self.dt, self.irk.velocity_mass.apply(velocity_y)),
                sd.mul_num_scalar(self.erk.A_coefs["a21"],
                    self.erk.make_rhs_ode_y(g_11, g_21)),
                sd.mul_num_scalar(self.irk.A_coefs["a21"], self.irk.make_rhs_dae_y(g_21,
                    g_31, st_rhs_22)))

            self.irk.uzawa_solver_internal("2", velocity_x=g_12, velocity_y=g_22,
                pressure=g_32, rhs_momentum_x=rhs_momentum_x,
                rhs_momentum_y=rhs_momentum_y, rhs_continuity=st_rhs_32)

            print("stage 2 done")
            print("")

            g_13, g_23, g_33 = sd.Scalar(), sd.Scalar(), sd.Scalar()
            g_13.sc, g_23.sc, g_33.sc = g_12.sc.copy(), g_22.sc.copy(), g_32.sc.copy()
            rhs_momentum_x = sd.add_scalars(
                sd.mul_num_scalar(1/self.dt, self.irk.velocity_mass.apply(velocity_x)),
                sd.mul_num_scalar(self.erk.A_coefs["a31"],
                    self.erk.make_rhs_ode_x(g_11, g_21)),
                sd.mul_num_scalar(self.erk.A_coefs["a32"],
                    self.erk.make_rhs_ode_x(g_12, g_22)),
                sd.mul_num_scalar(self.irk.A_coefs["a31"], self.irk.make_rhs_dae_x(g_11,
                    g_31, st_rhs_13)),
                sd.mul_num_scalar(self.irk.A_coefs["a32"], self.irk.make_rhs_dae_x(g_12,
                    g_32, st_rhs_13)))
            rhs_momentum_y = sd.add_scalars(
                sd.mul_num_scalar(1/self.dt, self.irk.velocity_mass.apply(velocity_y)),
                sd.mul_num_scalar(self.erk.A_coefs["a31"],
                    self.erk.make_rhs_ode_y(g_11, g_21)),
                sd.mul_num_scalar(self.erk.A_coefs["a32"],
                    self.erk.make_rhs_ode_y(g_12, g_22)),
                sd.mul_num_scalar(self.irk.A_coefs["a31"], self.irk.make_rhs_dae_y(g_21,
                    g_31, st_rhs_23)),
                sd.mul_num_scalar(self.irk.A_coefs["a32"], self.irk.make_rhs_dae_y(g_22,
                    g_32, st_rhs_23)))

            self.irk.uzawa_solver_internal("3", velocity_x=g_13, velocity_y=g_23,
                pressure=g_33, rhs_momentum_x=rhs_momentum_x,
                rhs_momentum_y=rhs_momentum_y, rhs_continuity=st_rhs_33)

            print("stage 3 done")
            print("")

            g_14, g_24, g_34 = sd.Scalar(), sd.Scalar(), sd.Scalar()
            g_14.sc, g_24.sc, g_34.sc = g_13.sc.copy(), g_23.sc.copy(), g_33.sc.copy()
            rhs_momentum_x = sd.add_scalars(
                sd.mul_num_scalar(1/self.dt, self.irk.velocity_mass.apply(velocity_x)),
                sd.mul_num_scalar(self.erk.A_coefs["a41"],
                    self.erk.make_rhs_ode_x(g_11, g_21, st_rhs_14)),
                sd.mul_num_scalar(self.erk.A_coefs["a42"],
                    self.erk.make_rhs_ode_x(g_12, g_22, st_rhs_14)),
                sd.mul_num_scalar(self.erk.A_coefs["a43"],
                    self.erk.make_rhs_ode_x(g_13, g_23, st_rhs_14)),
                sd.mul_num_scalar(self.irk.A_coefs["a41"], self.irk.make_rhs_dae_x(g_11,
                    g_31, st_rhs_14)),
                sd.mul_num_scalar(self.irk.A_coefs["a42"], self.irk.make_rhs_dae_x(g_12,
                    g_32, st_rhs_14)),
                sd.mul_num_scalar(self.irk.A_coefs["a43"], self.irk.make_rhs_dae_x(g_13,
                    g_33, st_rhs_14)))
            rhs_momentum_y = sd.add_scalars(
                sd.mul_num_scalar(1/self.dt, self.irk.velocity_mass.apply(velocity_y)),
                sd.mul_num_scalar(self.erk.A_coefs["a41"],
                    self.erk.make_rhs_ode_y(g_11, g_21, st_rhs_24)),
                sd.mul_num_scalar(self.erk.A_coefs["a42"],
                    self.erk.make_rhs_ode_y(g_12, g_22, st_rhs_24)),
                sd.mul_num_scalar(self.erk.A_coefs["a43"],
                    self.erk.make_rhs_ode_y(g_13, g_23, st_rhs_24)),
                sd.mul_num_scalar(self.irk.A_coefs["a41"], self.irk.make_rhs_dae_y(g_21,
                    g_31, st_rhs_24)),
                sd.mul_num_scalar(self.irk.A_coefs["a42"], self.irk.make_rhs_dae_y(g_22,
                    g_32, st_rhs_24)),
                sd.mul_num_scalar(self.irk.A_coefs["a43"], self.irk.make_rhs_dae_y(g_23,
                    g_33, st_rhs_24)))

            self.irk.uzawa_solver_internal("4", velocity_x=g_14, velocity_y=g_24,
                pressure=g_34, rhs_momentum_x=rhs_momentum_x,
                rhs_momentum_y=rhs_momentum_y, rhs_continuity=st_rhs_34)

            print("stage 4 done")
            print("")

            velocity_x.sc, velocity_y.sc, pressure.sc = g_14.sc.copy(), g_24.sc.copy(), g_34.sc.copy()
            # Do we need to recompute the right pressure?

            self.scalar_to_tree(velocity_x, v_x)
            self.scalar_to_tree(velocity_y, v_y)
            self.scalar_to_tree(pressure, p)
