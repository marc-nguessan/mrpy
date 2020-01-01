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
from mrpy.discretization.temporal_impl_RK2_base import ImplicitRungeKuttaStage2Scheme
import config as cfg


class Scheme(ImplicitRungeKuttaStage2Scheme):
    """...

    """

    def __init__(self, dimension=cfg.dimension, tree_velocity_x=None,
            tree_velocity_y=None, tree_velocity_z=None, tree_pressure=None,
            tree_vorticity=None, tree_scalar=None, uniform=False,
            st_flag_vx=False, st_flag_vy=False, st_flag_vz=False,
            st_flag_vc=False, st_flag_s=False, low_mach=False,
            nonlin_solver="pic_fullexp"):

        ImplicitRungeKuttaStage2Scheme.__init__(self, dimension=dimension,
            tree_velocity_x=tree_velocity_x, tree_velocity_y=tree_velocity_y,
            tree_velocity_z=tree_velocity_z, tree_pressure=tree_pressure,
            tree_vorticity=tree_vorticity, tree_scalar=tree_scalar,
            uniform=uniform, st_flag_vx=st_flag_vx, st_flag_vy=st_flag_vy,
            st_flag_vz=st_flag_vz, st_flag_vc=st_flag_vc, st_flag_s=st_flag_s,
            low_mach=low_mach, nonlin_solver=nonlin_solver)

    #def __init__(self, dimension=cfg.dimension, tree_velocity_x=None,
    #    tree_velocity_y=None, tree_velocity_z=None, tree_pressure=None, tree_vorticity=None):

    #    if tree_vorticity is not None:
    #        ImplicitRungeKuttaStage2Scheme.__init__(self,
    #            tree_velocity_x=tree_velocity_x, tree_velocity_y=tree_velocity_y, tree_pressure=tree_pressure, tree_vorticity=tree_vorticity)
    #    else:
    #        ImplicitRungeKuttaStage2Scheme.__init__(self,
    #            tree_velocity_x=tree_velocity_x, tree_velocity_y=tree_velocity_y, tree_pressure=tree_pressure)

        self.A_coefs = {"a11":5/12, "a12":-1/12, "a21":3/4, "a22":1/4}
        self.C_coefs = {"c1":1/3, "c2":1}

        #self.approx_solver_1.dt = self.C_coefs["c1"]*self.dt
        #self.approx_solver_2.dt = self.C_coefs["c2"]*self.dt

    def advance(self, v_x=None, v_y=None, v_z=None, p=None, t_ini=0, nsp=None):

        st_rhs_11 = None
        st_rhs_12 = None
        st_rhs_21 = None
        st_rhs_22 = None
        st_rhs_31 = None
        st_rhs_32 = None
        if self.uniform: #v_x, v_y, etc are scalars, and we just advance them

            #self.make_jacobian(tree_velocity_x, tree_velocity_y)
            #self.make_ksp_jacobian_A11()

            if self.st_flag_vx:
                mesh.listing_of_leaves(self.st_tree_vx)
                self.compute_source_term(self.st_tree_vx, self.st_func_vx,
                    t_ini + self.C_coefs["c1"]*self.dt)
                #mesh.listing_of_leaves(self.st_tree_vx)
                st_rhs_11 = sd.Scalar(self.st_tree_vx)
                self.compute_source_term(self.st_tree_vx, self.st_func_vx,
                    t_ini + self.C_coefs["c2"]*self.dt)
                mesh.listing_of_leaves(self.st_tree_vx)
                st_rhs_12 = sd.Scalar(self.st_tree_vx)

            if self.st_flag_vy:
                mesh.listing_of_leaves(self.st_tree_vy)
                self.compute_source_term(self.st_tree_vy, self.st_func_vy,
                    t_ini + self.C_coefs["c1"]*self.dt)
                #mesh.listing_of_leaves(self.st_tree_vy)
                st_rhs_21 = sd.Scalar(self.st_tree_vy)
                mesh.listing_of_leaves(self.st_tree_vy)
                self.compute_source_term(self.st_tree_vy, self.st_func_vy,
                    t_ini + self.C_coefs["c2"]*self.dt)
                #mesh.listing_of_leaves(self.st_tree_vy)
                st_rhs_22 = sd.Scalar(self.st_tree_vy)

            if self.st_flag_vc:
                mesh.listing_of_leaves(self.st_tree_vc)
                self.compute_source_term(self.st_tree_vc, self.st_func_vc,
                    t_ini + self.C_coefs["c1"]*self.dt)
                #mesh.listing_of_leaves(self.st_tree_vc)
                st_rhs_31 = sd.Scalar(self.st_tree_vc)
                mesh.listing_of_leaves(self.st_tree_vc)
                self.compute_source_term(self.st_tree_vc, self.st_func_vc,
                    t_ini + self.C_coefs["c2"]*self.dt)
                #mesh.listing_of_leaves(self.st_tree_vc)
                st_rhs_32 = sd.Scalar(self.st_tree_vc)

            if self.nonlin_solver == "newton":
                self.Newton_init_val(v_x, v_y, p)
                    #source_term_vx, source_term_vy)
                self.Newton_solver(v_x, v_y, p,
                    st_rhs_11, st_rhs_12, st_rhs_21, st_rhs_22, st_rhs_31,
                    st_rhs_32, t_ini)
            elif self.nonlin_solver == "pic_fullexp":
                self.Nonlin_init_val(v_x, v_y, p)
                    #source_term_vx, source_term_vy)
                self.Nonlin_solver(v_x, v_y, p,
                    st_rhs_11, st_rhs_12, st_rhs_21, st_rhs_22, st_rhs_31,
                    st_rhs_32, t_ini)

            if self.nonlin_solver == "newton":
                v_x.sc[:] = self.z_1.sc[self.n_mesh_vx:2*self.n_mesh_vx]
                v.sc[:] = self.z_1.sc[2*self.n_mesh_vx + self.n_mesh_vy:]
            elif self.nonlin_solver == "pic_fullexp":
                v_x.sc[:] = self.g_1.sc[self.n_mesh_vx:]
                v_y.sc[:] = self.g_2.sc[self.n_mesh_vy:]

            # The pressure must be the right Lagrange multiplier of the
            # resulting velocity
            p.sc = self.solve(self.pressure_divgrad,
                self.make_rhs_pressure_equation(v_x, v_y,
                st_rhs_12, st_rhs_22), nsp).sc

            #pressure.sc[:] = self.z_2.sc[self.n_mesh_p:]

        else: #v_x, etc are trees
            velocity_x = sd.Scalar(v_x)
            velocity_y = sd.Scalar(v_y)
            pressure = sd.Scalar(p)

            if self.st_flag_vx: #we need to put the st_tree_vx to the same grading as v_x
                op.set_to_same_grading(v_x, self.st_tree_vx)
                op.run_pruning(self.st_tree_vx)
                mesh.listing_of_leaves(self.st_tree_vx)
                self.compute_source_term(self.st_tree_vx, self.st_func_vx,
                    t_ini + self.C_coefs["c1"]*self.dt)
                #mesh.listing_of_leaves(self.st_tree_vx)
                st_rhs_11 = sd.Scalar(self.st_tree_vx)
                mesh.listing_of_leaves(self.st_tree_vx)
                self.compute_source_term(self.st_tree_vx, self.st_func_vx,
                    t_ini + self.C_coefs["c2"]*self.dt)
                #mesh.listing_of_leaves(self.st_tree_vx)
                st_rhs_12 = sd.Scalar(self.st_tree_vx)

            if self.st_flag_vy: #we need to put the st_tree_vy to the same grading as v_y
                op.set_to_same_grading(v_y, self.st_tree_vy)
                op.run_pruning(self.st_tree_vy)
                mesh.listing_of_leaves(self.st_tree_vy)
                self.compute_source_term(self.st_tree_vy, self.st_func_vy,
                        t_ini + self.C_coefs["c1"]*self.dt)
                #mesh.listing_of_leaves(self.st_tree_vy)
                st_rhs_21 = sd.Scalar(self.st_tree_vy)
                mesh.listing_of_leaves(self.st_tree_vy)
                self.compute_source_term(self.st_tree_vy, self.st_func_vy,
                        t_ini + self.C_coefs["c2"]*self.dt)
                #mesh.listing_of_leaves(self.st_tree_vy)
                st_rhs_22 = sd.Scalar(self.st_tree_vy)

            if self.st_flag_vc: #we need to put the st_tree_vc to the same grading as v_x
                op.set_to_same_grading(v_x, self.st_tree_vc)
                op.run_pruning(self.st_tree_vc)
                mesh.listing_of_leaves(self.st_tree_vc)
                self.compute_source_term(self.st_tree_vc, self.st_func_vc,
                        t_ini + self.C_coefs["c1"]*self.dt)
                #mesh.listing_of_leaves(self.st_tree_vc)
                st_rhs_31 = sd.Scalar(self.st_tree_vc)
                mesh.listing_of_leaves(self.st_tree_vc)
                self.compute_source_term(self.st_tree_vc, self.st_func_vc,
                        t_ini + self.C_coefs["c2"]*self.dt)
                #mesh.listing_of_leaves(self.st_tree_vc)
                st_rhs_32 = sd.Scalar(self.st_tree_vc)

            #self.make_jacobian(tree_velocity_x, tree_velocity_y)
            #self.make_ksp_jacobian_A11()

            if self.nonlin_solver == "newton":
                self.Newton_init_val(v_x, v_y, p)
                    #source_term_vx, source_term_vy)
                self.Newton_solver(velocity_x, velocity_y, pressure,
                    st_rhs_11, st_rhs_12, st_rhs_21, st_rhs_22, st_rhs_31,
                    st_rhs_32, t_ini)
            elif self.nonlin_solver == "pic_fullexp":
                self.Nonlin_init_val(v_x, v_y, p)
                    #source_term_vx, source_term_vy)
                self.Nonlin_solver(velocity_x, velocity_y, pressure,
                    st_rhs_11, st_rhs_12, st_rhs_21, st_rhs_22, st_rhs_31,
                    st_rhs_32, t_ini)

            if self.nonlin_solver == "newton":
                velocity_x.sc[:] = self.z_1.sc[self.n_mesh_vx:2*self.n_mesh_vx]
                velocity_y.sc[:] = self.z_1.sc[2*self.n_mesh_vx + self.n_mesh_vy:]
            elif self.nonlin_solver == "pic_fullexp":
                velocity_x.sc[:] = self.g_1.sc[self.n_mesh_vx:]
                velocity_y.sc[:] = self.g_2.sc[self.n_mesh_vy:]

            # The pressure must be the right Lagrange multiplier of the
            # resulting velocity
            pressure.sc = self.solve(self.pressure_divgrad,
                self.make_rhs_pressure_equation(velocity_x, velocity_y,
                st_rhs_12, st_rhs_22), nsp).sc

            #pressure.sc[:] = self.z_2.sc[self.n_mesh_p:]

            for index in range(v_x.number_of_leaves):
                v_x.nvalue[v_x.tree_leaves[index]] = velocity_x.sc[index]

            for index in range(v_y.number_of_leaves):
                v_y.nvalue[v_y.tree_leaves[index]] = velocity_y.sc[index]

            for index in range(p.number_of_leaves):
                p.nvalue[p.tree_leaves[index]] = pressure.sc[index]
