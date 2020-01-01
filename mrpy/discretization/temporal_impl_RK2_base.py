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

This module contains a base class for the implementation of Implicit 2-stage
Runge-Kutta schemes.
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
import mrpy.discretization.temporal_impl_expl_euler as approx_solver
import config as cfg


class ImplicitRungeKuttaStage2Scheme(BaseScheme):
    "Base class for the implementation of stage-2 IRK methods."

    def __init__(self, dimension=cfg.dimension, tree_velocity_x=None,
            tree_velocity_y=None, tree_velocity_z=None, tree_pressure=None,
            tree_vorticity=None, tree_scalar=None, uniform=False,
            st_flag_vx=False, st_flag_vy=False, st_flag_vz=False,
            st_flag_vc=False, st_flag_s=False, low_mach=False,
            nonlin_solver="pic_fullexp"):

        BaseScheme.__init__(self, dimension=dimension,
            tree_velocity_x=tree_velocity_x, tree_velocity_y=tree_velocity_y,
            tree_velocity_z=tree_velocity_z, tree_pressure=tree_pressure,
            tree_vorticity=tree_vorticity, tree_scalar=tree_scalar,
            uniform=uniform, st_flag_vx=st_flag_vx, st_flag_vy=st_flag_vy,
            st_flag_vz=st_flag_vz, st_flag_vc=st_flag_vc, st_flag_s=st_flag_s,
            low_mach=low_mach)

    #def __init__(self, dimension=cfg.dimension, tree_velocity_x=None,
    #        tree_velocity_y=None, tree_velocity_z=None, tree_pressure=None,
    #        tree_vorticity=None, nonlin_solver="pic_fullexp"):

#        if tree_vorticity is not None:
#            BaseScheme.__init__(self, tree_velocity_x=tree_velocity_x, tree_velocity_y=tree_velocity_y, tree_pressure=tree_pressure, tree_vorticity=tree_vorticity)
#        else:
#            BaseScheme.__init__(self, tree_velocity_x=tree_velocity_x, tree_velocity_y=tree_velocity_y, tree_pressure=tree_pressure)

        self.stfv_x = cfg.source_term_function_velocity_x
        self.stfv_y = cfg.source_term_function_velocity_y

        self.A_coefs = None
        self.C_coefs = None

        # The type of nonlinear solver used for the computation
        self.nonlin_solver= nonlin_solver

        self.jacobian_base = None
        self.jacobian_variable = None
        self.jacobian = None

        self.mat_iter_base = None
        self.mat_iter_variable = None
        self.mat_iter = None

        self.pressure_divgrad = None

        self.pc_schur_jacobian = None

        # KSP object for solving the jacobian_A11
        self.ksp_jacobian_A11 = None

        # KSP object for solving the mat_iter_A11
        self.ksp_mat_iter_A11 = None

        # KSP object for solving the mat_iter_A22
        self.ksp_mat_iter_A22 = None

        # KSP objects for solving the preconditioner of the the schur complement
        # of jacobian
        self.ksp_pc_schur_jacobian = None
        self.ksp_pc_schur_jacobian1 = None
        self.ksp_pc_schur_jacobian2 = None

        # KSP objects for solving the preconditioner of the the schur complement
        # of the iteration matrix of the nonlinear solver
        self.ksp_pc_schur_mat_iter1 = None
        self.ksp_pc_schur_mat_iter2 = None

        # Approximate solver for the initial values of the Newton solver
        #self.approx_solver_1 = approx_solver.Scheme(tree_velocity_x=tree_velocity_x, tree_velocity_y=tree_velocity_y, tree_pressure=tree_pressure)
        #self.approx_solver_2 = approx_solver.Scheme(tree_velocity_x=tree_velocity_x, tree_velocity_y=tree_velocity_y, tree_pressure=tree_pressure)

        # Approximate solutions for the initial values of the Newton solver
        self.approx_vx_1 = None
        self.approx_vy_1 = None
        self.approx_p_1 = None
        self.approx_vx_2 = None
        self.approx_vy_2 = None
        self.approx_p_2 = None

        # Solution of the nonlinear system at each timestep; it is initialized
        # with the initial velocity and pressure fields

        self.z_1 = None

        self.z_2 = None

        # Increment in the jacobian solver

        self.delta_z_1 = None

        self.delta_z_2 = None

        # Criteria used in the stopping criterion of the nonlinear solver
        self.eta_k = 1.

        self.n_mesh_vx = None
        self.n_mesh_vy = None
        self.n_mesh_p = None

    def compute_n_mesh_vx(self, tree_velocity_x):

        self.n_mesh_vx = len(tree_velocity_x.tree_leaves)

    def compute_n_mesh_vy(self, tree_velocity_y):

        self.n_mesh_vy = len(tree_velocity_y.tree_leaves)

    def compute_n_mesh_p(self, tree_pressure):

        self.n_mesh_p = len(tree_pressure.tree_leaves)

    def setup_internal_variables(self, tree_velocity_x, tree_velocity_y,
            tree_pressure):

        self.compute_n_mesh_vx(tree_velocity_x)
        self.compute_n_mesh_vy(tree_velocity_y)
        self.compute_n_mesh_p(tree_pressure)

        if self.nonlin_solver == "newton":
            # Variable for the Newton solver
            self.z_1 = sd.Scalar()
            self.z_1.sc = petsc.Vec().create()
            self.z_1.sc.setSizes(2*self.n_mesh_vx + 2*self.n_mesh_vy)
            self.z_1.sc.setUp()

            self.z_2 = sd.Scalar()
            self.z_2.sc = petsc.Vec().create()
            self.z_2.sc.setSizes(2*self.n_mesh_p)
            self.z_2.sc.setUp()

            self.delta_z_1 = sd.Scalar()
            self.delta_z_1.sc = petsc.Vec().create()
            self.delta_z_1.sc.setSizes(2*self.n_mesh_vx + 2*self.n_mesh_vy)
            self.delta_z_1.sc.setUp()

            self.delta_z_2 = sd.Scalar()
            self.delta_z_2.sc = petsc.Vec().create()
            self.delta_z_2.sc.setSizes(2*self.n_mesh_p)
            self.delta_z_2.sc.setUp()
        elif self.nonlin_solver == "pic_fullexp":
            # Variable for the Picard solver
            self.g_1 = sd.Scalar()
            self.g_1.sc = petsc.Vec().create()
            self.g_1.sc.setSizes(2*self.n_mesh_vx)
            self.g_1.sc.setUp()

            self.g_2 = sd.Scalar()
            self.g_2.sc = petsc.Vec().create()
            self.g_2.sc.setSizes(2*self.n_mesh_vy)
            self.g_2.sc.setUp()

            self.g_3 = sd.Scalar()
            self.g_3.sc = petsc.Vec().create()
            self.g_3.sc.setSizes(2*self.n_mesh_p)
            self.g_3.sc.setUp()

    def nonlin_op_apply(self, x1, y1, z1, x2, y2, z2):
        """Computes a petsc vector that is the result of opplying the nonlinear
        operator of the scheme to internal stage variables of the RK method.
        """

        result = petsc.Vec().create()
        result.setSizes(2*self.n_mesh_vx + 2*self.n_mesh_vy +
                2*self.n_mesh_p)
        result.setUp()

        temp, foo_x, bar = self.g_1.sc.duplicate(), self.g_1.sc.duplicate(), \
        self.g_1.sc.duplicate()
        baz = self.g_3.sc.duplicate()
        foo_x[:self.n_mesh_vx] = x1.sc
        foo_x[self.n_mesh_vx:] = x2.sc
        baz[:self.n_mesh_p] = z1.sc
        baz[self.n_mesh_p:] = z2.sc
        self.mat_iter["A11"].mult(foo_x, temp)
        self.mat_iter["A13"].mult(baz, bar)
        temp += bar
        result[:2*self.n_mesh_vx] = temp

        temp, foo_y, bar = self.g_2.sc.duplicate(), self.g_2.sc.duplicate(), \
        self.g_2.sc.duplicate()
        foo_y[:self.n_mesh_vy] = y1.sc
        foo_y[self.n_mesh_vy:] = y2.sc
        self.mat_iter["A22"].mult(foo_y, temp)
        self.mat_iter["A23"].mult(baz, bar)
        temp += bar
        result[2*self.n_mesh_vx:2*self.n_mesh_vx + 2*self.n_mesh_vy] = temp

        temp_div_x = self.g_3.sc.duplicate()
        self.mat_iter["A31"].mult(foo_x, temp_div_x)

        temp_div_y = self.g_3.sc.duplicate()
        self.mat_iter["A32"].mult(foo_y, temp_div_y)
        result[2*self.n_mesh_vx + 2*self.n_mesh_vy:] = temp_div_x + temp_div_y

        temp_adv_x_1 = self.velocity_adv_x.apply(self.dimension, x1, x1, y1, low_mach=self.low_mach).sc
        temp_adv_x_1 -= self.velocity_adv_x.bc_x
        temp_adv_x_1 -= self.velocity_adv_x.bc_y

        temp_adv_x_2 = self.velocity_adv_x.apply(self.dimension, x2, x2, y2, low_mach=self.low_mach).sc
        temp_adv_x_2 -= self.velocity_adv_x.bc_x
        temp_adv_x_2 -= self.velocity_adv_x.bc_y

        temp_adv_y_1 = self.velocity_adv_y.apply(self.dimension, y1, x1, y1, low_mach=self.low_mach).sc
        temp_adv_y_1 -= self.velocity_adv_y.bc_x
        temp_adv_y_1 -= self.velocity_adv_y.bc_y

        temp_adv_y_2 = self.velocity_adv_y.apply(self.dimension, y2, x2, y2, low_mach=self.low_mach).sc
        temp_adv_y_2 -= self.velocity_adv_y.bc_x
        temp_adv_y_2 -= self.velocity_adv_y.bc_y

        result[:self.n_mesh_vx] += self.A_coefs["a11"]*temp_adv_x_1 + \
            self.A_coefs["a12"]*temp_adv_x_2

        result[self.n_mesh_vx:2*self.n_mesh_vx] += \
            self.A_coefs["a21"]*temp_adv_x_1 + self.A_coefs["a22"]*temp_adv_x_2

        result[2*self.n_mesh_vx:2*self.n_mesh_vx + self.n_mesh_vy] += \
            self.A_coefs["a11"]*temp_adv_y_1 + self.A_coefs["a12"]*temp_adv_y_2

        result[2*self.n_mesh_vx + self.n_mesh_vy:2*self.n_mesh_vx + 2*self.n_mesh_vy] += \
            self.A_coefs["a21"]*temp_adv_y_1 + self.A_coefs["a22"]*temp_adv_y_2

        return result

    def make_rhs_pressure_equation(self, velocity_x, velocity_y,
                                   source_term_vx=None, source_term_vy=None):

        """Forms the RHS for the pressure Poisson equation."""

        if (source_term_vx is None) and (source_term_vy is None):
            temp_1 = sd.add_scalars(
                #sd.mul_num_scalar(cfg.nu, self.velocity_lap_x.apply(velocity_x)))
                sd.mul_num_scalar(cfg.nu, self.velocity_lap_x.apply(velocity_x)),
                sd.mul_num_scalar(-1, self.velocity_adv_x.apply(self.dimension, velocity_x,
                    velocity_x, velocity_y, low_mach=self.low_mach)))

            temp_2 = sd.add_scalars(
                #sd.mul_num_scalar(cfg.nu, self.velocity_lap_y.apply(velocity_y)))
                sd.mul_num_scalar(cfg.nu, self.velocity_lap_y.apply(velocity_y)),
                sd.mul_num_scalar(-1, self.velocity_adv_y.apply(self.dimension, velocity_y,
                    velocity_x, velocity_y, low_mach=self.low_mach)))

        if (source_term_vx is not None):
            temp_1 = sd.add_scalars(
                self.velocity_mass.apply(source_term_vx),
                #sd.mul_num_scalar(cfg.nu, self.velocity_lap_x.apply(velocity_x)))
                sd.mul_num_scalar(cfg.nu, self.velocity_lap_x.apply(velocity_x)),
                sd.mul_num_scalar(-1, self.velocity_adv_x.apply(self.dimension, velocity_x,
                    velocity_x, velocity_y, low_mach=self.low_mach)))

        if (source_term_vy is not None):
            temp_2 = sd.add_scalars(
                self.velocity_mass.apply(source_term_vy),
                #sd.mul_num_scalar(cfg.nu, self.velocity_lap_y.apply(velocity_y)))
                sd.mul_num_scalar(cfg.nu, self.velocity_lap_y.apply(velocity_y)),
                sd.mul_num_scalar(-1, self.velocity_adv_y.apply(self.dimension, velocity_y,
                    velocity_x, velocity_y, low_mach=self.low_mach)))

        foo_1, foo_2 = sd.Scalar(), sd.Scalar()
        foo_1.sc, foo_2.sc = temp_1.sc.duplicate(), temp_2.sc.duplicate()
        self.velocity_div_x.matrix.mult(temp_1.sc, foo_1.sc)
        self.velocity_div_y.matrix.mult(temp_2.sc, foo_2.sc)

        return sd.add_scalars(foo_1, foo_2)

    #===========================================================================
    #+++++++++++++++++++++++++++ Newton solver +++++++++++++++++++++++++++++++++
    #===========================================================================

    def rhs_1(self, x, y, z, source_term_vx=None):
        """...

        """

        result = sd.Scalar()
        temp_1 = sd.Scalar()
        temp_1.sc = x.sc.duplicate()

        #self.velocity_lap_x.matrix.mult(x.sc, temp_1.sc)
        temp_1 = self.velocity_lap_x.apply(x)
        temp_1 = sd.mul_num_scalar(cfg.nu, temp_1)
        result.sc = temp_1.sc

        #self.pressure_grad_x.matrix.mult(z.sc, temp_1.sc)
        temp_1 = self.pressure_grad_x.apply(z)
        temp_1 = sd.mul_num_scalar(-1, temp_1)
        result.sc += temp_1.sc

        #temp_1 = self.velocity_adv_x.apply(x, x, y)
        #temp_1 = sd.mul_num_scalar(-1, temp_1)
        #result.sc += temp_1.sc

        if source_term_vx is not None:
            #if not self.uniform: #source_term_vx is a tree
            #    self.compute_source_term(source_term_vx, self.stfv_x, t)
            #    temp_1 = sd.Scalar(source_term_vx)
            #else: #source_term_vx is a scalar
            #    temp_1 = sd.Scalar()
            #    temp_1.sc = source_term_vx.sc.copy()
            #temp_1 = self.velocity_mass.apply(temp_1)
            temp_1 = self.velocity_mass.apply(source_term_vx)
            result.sc += temp_1.sc

        return result

    def rhs_2(self, x, y, z, source_term_vy=None):
        """...

        """

        result = sd.Scalar()
        temp_1 = sd.Scalar()
        temp_1.sc = x.sc.duplicate()

        #self.velocity_lap_y.matrix.mult(y.sc, temp_1.sc)
        temp_1 = self.velocity_lap_y.apply(y)
        temp_1 = sd.mul_num_scalar(cfg.nu, temp_1)
        result.sc = temp_1.sc

        #self.pressure_grad_y.matrix.mult(z.sc, temp_1.sc)
        temp_1 = self.pressure_grad_y.apply(z)
        temp_1 = sd.mul_num_scalar(-1, temp_1)
        result.sc += temp_1.sc

        #temp_1 = self.velocity_adv_y.apply(y, x, y)
        #temp_1 = sd.mul_num_scalar(-1, temp_1)
        #result.sc += temp_1.sc

        if source_term_vy is not None:
        #    if not self.uniform: #source_term_vy is a tree
        #        self.compute_source_term(source_term_vy, self.stfv_y, t)
        #        temp_1 = sd.Scalar(source_term_vy)
        #    else: #source_term_vy is a scalar
        #        temp_1 = sd.Scalar()
        #        temp_1.sc = source_term_vy.sc.copy()
        #    temp_1 = self.velocity_mass.apply(temp_1)
            temp_1 = self.velocity_mass.apply(source_term_vy)
            result.sc += temp_1.sc

        return result

    def rhs_3(self, source_term=None):
        """...

        """

        result = sd.Scalar()
        result.sc = self.velocity_div_x.bc
        result.sc += self.velocity_div_y.bc
        result = sd.mul_num_scalar(-1, result)
        if source_term is not None:
            result.sc += source_term.sc

        return result

    #def rhs_3(self, x, y, z, t=0.):
    #    """...

    #    """

    #    return sd.Scalar()

    def rhs_newton_iteration(self, velocity_x, velocity_y,
            x1, y1, z1, x2, y2, z2,  st_rhs_11=None, st_rhs_12=None,
            st_rhs_21=None, st_rhs_22=None, st_rhs_31=None, st_rhs_32=None):
        """...

        """

        global_rhs_1 =  self.z_1.sc.duplicate()
        #global_rhs_1 =  petsc.Vec().create()
        #global_rhs_1.setSizes(4*self.n_mesh)
        #global_rhs_1.setUp()
        global_rhs_2 =  self.z_2.sc.duplicate()
        #global_rhs_2 =  petsc.Vec().create()
        #global_rhs_2.setSizes(2*self.n_mesh)
        #global_rhs_2.setUp()

        #bc_1 = sd.Scalar()
        #bc_1.sc = velocity_x.sc.duplicate()
        #bc_1.sc += self.velocity_lap_x.bc
        #bc_1 = sd.mul_num_scalar(cfg.nu, bc_1)
        temp = sd.add_scalars(
            sd.mul_num_scalar(1/self.dt, self.velocity_mass.apply(velocity_x)),
            sd.mul_num_scalar(-1/self.dt, self.velocity_mass.apply(x1)),
            sd.mul_num_scalar(self.A_coefs["a11"], self.rhs_1(x1, y1, z1,
                st_rhs_11)),
                #source_term_vx, t_ini + self.C_coefs["c1"]*self.dt)),
            #sd.mul_num_scalar(self.A_coefs["a11"], bc_1),
            #sd.mul_num_scalar(self.A_coefs["a12"], bc_1),
            sd.mul_num_scalar(self.A_coefs["a12"], self.rhs_1(x2, y2, z2,
                st_rhs_12)))
                #source_term_vx, t_ini + self.C_coefs["c2"]*self.dt)))

        global_rhs_1[:self.n_mesh_vx] = temp.sc

        temp = sd.add_scalars(
            sd.mul_num_scalar(1/self.dt, self.velocity_mass.apply(velocity_x)),
            sd.mul_num_scalar(-1/self.dt, self.velocity_mass.apply(x2)),
            sd.mul_num_scalar(self.A_coefs["a21"], self.rhs_1(x1, y1, z1,
                st_rhs_11)),
                #source_term_vx, t_ini + self.C_coefs["c1"]*self.dt)),
            #sd.mul_num_scalar(self.A_coefs["a21"], bc_1),
            #sd.mul_num_scalar(self.A_coefs["a22"], bc_1),
            sd.mul_num_scalar(self.A_coefs["a22"], self.rhs_1(x2, y2, z2,
                st_rhs_12)))
                #source_term_vx, t_ini + self.C_coefs["c2"]*self.dt)))

        global_rhs_1[self.n_mesh_vx:2*self.n_mesh_vx] = temp.sc

        #bc_2 = sd.Scalar()
        #bc_2.sc = velocity_y.sc.duplicate()
        #bc_2.sc += self.velocity_lap_y.bc
        #bc_2 = sd.mul_num_scalar(cfg.nu, bc_2)
        temp = sd.add_scalars(
            sd.mul_num_scalar(1/self.dt, self.velocity_mass.apply(velocity_y)),
            sd.mul_num_scalar(-1/self.dt, self.velocity_mass.apply(y1)),
            sd.mul_num_scalar(self.A_coefs["a11"], self.rhs_2(x1, y1, z1,
                st_rhs_21)),
                #source_term_vy, t_ini + self.C_coefs["c1"]*self.dt)),
            #sd.mul_num_scalar(self.A_coefs["a11"], bc_2),
            #sd.mul_num_scalar(self.A_coefs["a12"], bc_2),
            sd.mul_num_scalar(self.A_coefs["a12"], self.rhs_2(x2, y2, z2,
                source_term_vy, t_ini
                + self.C_coefs["c2"]*self.dt)))

        global_rhs_1[2*self.n_mesh_vx:2*self.n_mesh_vx + self.n_mesh_vy] = temp.sc

        temp = sd.add_scalars(
            sd.mul_num_scalar(1/self.dt, self.velocity_mass.apply(velocity_y)),
            sd.mul_num_scalar(-1/self.dt, self.velocity_mass.apply(y2)),
            sd.mul_num_scalar(self.A_coefs["a21"], self.rhs_2(x1, y1, z1,
                st_rhs_21)),
                #source_term_vy, t_ini + self.C_coefs["c1"]*self.dt)),
            #sd.mul_num_scalar(self.A_coefs["a21"], bc_2),
            #sd.mul_num_scalar(self.A_coefs["a22"], bc_2),
            sd.mul_num_scalar(self.A_coefs["a22"], self.rhs_2(x2, y2, z2,
                st_rhs_21)))
                #source_term_vy, t_ini + self.C_coefs["c2"]*self.dt)))

        global_rhs_1[2*self.n_mesh_vx + self.n_mesh_vy:] = temp.sc

        #bc_3 = sd.Scalar()
        #bc_3.sc = self.velocity_div_x.bc
        #bc_3.sc += self.velocity_div_y.bc
        #bc_3 = sd.mul_num_scalar(-1, bc_3)
        #temp = sd.mul_num_scalar(-1, self.rhs_3(x1, y1, z1))
        #temp = bc_3

        global_rhs_2[:self.n_mesh_p] = self.rhs_3(st_rhs_31).sc

        #temp = sd.mul_num_scalar(-1, self.rhs_3(x2, y2, z2))
        #temp = bc_3

        global_rhs_2[self.n_mesh_p:] = self.rhs_3(st_rhs_32).sc

        temp_1 =  global_rhs_1.duplicate()
        #temp_1 =  petsc.Vec().create()
        #temp_1.setSizes(4*self.n_mesh)
        #temp_1.setUp()
        temp_2 =  global_rhs_2.duplicate()
        #temp_2 =  petsc.Vec().create()
        #temp_2.setSizes(2*self.n_mesh)
        #temp_2.setUp()

        temp_1[:self.n_mesh_vx] = x1.sc
        temp_1[self.n_mesh_vx:2*self.n_mesh_vx] = x2.sc
        temp_1[2*self.n_mesh_vx:2*self.n_mesh_vx + self.n_mesh_vy] = y1.sc
        temp_1[2*self.n_mesh_vx + self.n_mesh_vy:] = y2.sc

        temp_2[:self.n_mesh_p] = z1.sc
        temp_2[self.n_mesh_p:] = z2.sc

        temp_z_1, temp = temp_1.duplicate(), temp_1.duplicate()
        self.jacobian["A11"].mult(temp_1, temp_z_1)
        self.jacobian["A12"].mult(temp_2, temp)

        temp_1 = temp_z_1 + temp

        global_rhs_1 += temp_1

        return global_rhs_1, global_rhs_2

    def make_jacobian_base(self):
        """...

        """

        #++++++++++++ Creation of the A11 block ++++++++++++++++++++++++++++++++
        jacobian_A11 = petsc.Mat().create()
        number_of_rows = 2*self.n_mesh_vx + 2*self.n_mesh_vy
        size_row = (number_of_rows, number_of_rows)
        size_col = (number_of_rows, number_of_rows)
        jacobian_A11.setSizes((size_row, size_col))
        jacobian_A11.setUp()

        # The A11 block can be divided in 4x4 blocks, and we build each of this
        # block

        # Creation of the a11 block of A11
        temp_a11 = sd.add_operators(
            sd.mul_num_operator(1/self.dt, self.velocity_mass),
            sd.mul_num_operator(-cfg.nu*self.A_coefs["a11"], self.velocity_lap_x))

        # Creation of the a12 block of A11
        temp_a12 = sd.mul_num_operator(-cfg.nu*self.A_coefs["a12"], self.velocity_lap_x)

        # Creation of the a21 block of A11
        temp_a21 = sd.mul_num_operator(-cfg.nu*self.A_coefs["a21"], self.velocity_lap_x)

        # Creation of the a22 block of A11
        temp_a22 = sd.add_operators(
            sd.mul_num_operator(1/self.dt, self.velocity_mass),
            sd.mul_num_operator(-cfg.nu*self.A_coefs["a22"], self.velocity_lap_x))

        # Creation of the a33 block of A11
        temp_a33 = sd.add_operators(
            sd.mul_num_operator(1/self.dt, self.velocity_mass),
            sd.mul_num_operator(-cfg.nu*self.A_coefs["a11"], self.velocity_lap_y))

        # Creation of the a34 block of A11
        temp_a34 = sd.mul_num_operator(-cfg.nu*self.A_coefs["a12"], self.velocity_lap_y)

        # Creation of the a43 block of A11
        temp_a43 = sd.mul_num_operator(-cfg.nu*self.A_coefs["a21"], self.velocity_lap_y)

        # Creation of the a44 block of A11
        temp_a44 = sd.add_operators(
            sd.mul_num_operator(1/self.dt, self.velocity_mass),
            sd.mul_num_operator(-cfg.nu*self.A_coefs["a22"], self.velocity_lap_y))

        # Block a11
        for row in range(self.n_mesh_vx):
            cols, vals = temp_a11.matrix.getRow(row)
            for index_col in range(len(cols)):
                jacobian_A11.setValue(row, cols[index_col], vals[index_col], True)

        # Block a12
        for row in range(self.n_mesh_vx):
            cols, vals = temp_a12.matrix.getRow(row)
            for index_col in range(len(cols)):
                jacobian_A11.setValue(row, cols[index_col] + self.n_mesh_vx,
                    vals[index_col], True)

        # Block a21
        for row in range(self.n_mesh_vx):
            cols, vals = temp_a21.matrix.getRow(row)
            for index_col in range(len(cols)):
                jacobian_A11.setValue(row + self.n_mesh_vx, cols[index_col],
                    vals[index_col], True)

        # Block a22
        for row in range(self.n_mesh_vx):
            cols, vals = temp_a22.matrix.getRow(row)
            for index_col in range(len(cols)):
                jacobian_A11.setValue(row + self.n_mesh_vx, cols[index_col] +
                    self.n_mesh_vx, vals[index_col], True)

        # Block a33
        for row in range(self.n_mesh_vy):
            cols, vals = temp_a33.matrix.getRow(row)
            for index_col in range(len(cols)):
                jacobian_A11.setValue(row + 2*self.n_mesh_vx, cols[index_col] +
                    2*self.n_mesh_vx, vals[index_col], True)

        # Block a34
        for row in range(self.n_mesh_vy):
            cols, vals = temp_a34.matrix.getRow(row)
            for index_col in range(len(cols)):
                jacobian_A11.setValue(row + 2*self.n_mesh_vx, cols[index_col] +
                    3*self.n_mesh_vx, vals[index_col], True)

        # Block a43
        for row in range(self.n_mesh_vy):
            cols, vals = temp_a43.matrix.getRow(row)
            for index_col in range(len(cols)):
                jacobian_A11.setValue(row + 3*self.n_mesh_vx, cols[index_col] +
                    2*self.n_mesh_vx, vals[index_col], True)

        # Block a44
        for row in range(self.n_mesh_vy):
            cols, vals = temp_a44.matrix.getRow(row)
            for index_col in range(len(cols)):
                jacobian_A11.setValue(row + 3*self.n_mesh_vx, cols[index_col] +
                    3*self.n_mesh_vx, vals[index_col], True)

        jacobian_A11.assemble()
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #++++++++++++ Creation of the A12 block ++++++++++++++++++++++++++++++++
        jacobian_A12 = petsc.Mat().create()
        number_of_rows = 2*self.n_mesh_vx + 2*self.n_mesh_vy
        number_of_cols = 2*self.n_mesh_p
        size_row = (number_of_rows, number_of_rows)
        size_col = (number_of_cols, number_of_cols)
        jacobian_A12.setSizes((size_row, size_col))
        jacobian_A12.setUp()

        # The A12 block can be divided in 4x2 blocks, and we build each of this
        # block

        # Creation of the a11 block of A12
        temp_a11 = sd.mul_num_operator(self.A_coefs["a11"], self.pressure_grad_x)

        # Creation of the a12 block of A12
        temp_a12 = sd.mul_num_operator(self.A_coefs["a12"], self.pressure_grad_x)

        # Creation of the a21 block of A12
        temp_a21 = sd.mul_num_operator(self.A_coefs["a21"], self.pressure_grad_x)

        # Creation of the a22 block of A12
        temp_a22 = sd.mul_num_operator(self.A_coefs["a22"], self.pressure_grad_x)

        # Creation of the a31 block of A12
        temp_a31 = sd.mul_num_operator(self.A_coefs["a11"], self.pressure_grad_y)

        # Creation of the a32 block of A12
        temp_a32 = sd.mul_num_operator(self.A_coefs["a12"], self.pressure_grad_y)

        # Creation of the a41 block of A12
        temp_a41 = sd.mul_num_operator(self.A_coefs["a21"], self.pressure_grad_y)

        # Creation of the a42 block of A12
        temp_a42 = sd.mul_num_operator(self.A_coefs["a22"], self.pressure_grad_y)

        # Block a11
        for row in range(self.n_mesh_vx):
            cols, vals = temp_a11.matrix.getRow(row)
            for index_col in range(len(cols)):
                jacobian_A12.setValue(row, cols[index_col], vals[index_col], True)

        # Block a12
        for row in range(self.n_mesh_vx):
            cols, vals = temp_a12.matrix.getRow(row)
            for index_col in range(len(cols)):
                jacobian_A12.setValue(row, cols[index_col] + self.n_mesh_p,
                    vals[index_col], True)

        # Block a21
        for row in range(self.n_mesh_vx):
            cols, vals = temp_a21.matrix.getRow(row)
            for index_col in range(len(cols)):
                jacobian_A12.setValue(row + self.n_mesh_p, cols[index_col],
                    vals[index_col], True)

        # Block a22
        for row in range(self.n_mesh_vx):
            cols, vals = temp_a22.matrix.getRow(row)
            for index_col in range(len(cols)):
                jacobian_A12.setValue(row + self.n_mesh_p, cols[index_col] +
                    self.n_mesh_p, vals[index_col], True)

        # Block a31
        for row in range(self.n_mesh_vy):
            cols, vals = temp_a31.matrix.getRow(row)
            for index_col in range(len(cols)):
                jacobian_A12.setValue(row + 2*self.n_mesh_p, cols[index_col],
                        vals[index_col], True)

        # Block a32
        for row in range(self.n_mesh_vy):
            cols, vals = temp_a32.matrix.getRow(row)
            for index_col in range(len(cols)):
                jacobian_A12.setValue(row + 2*self.n_mesh_p, cols[index_col] +
                    self.n_mesh_p, vals[index_col], True)

        # Block a41
        for row in range(self.n_mesh_vy):
            cols, vals = temp_a41.matrix.getRow(row)
            for index_col in range(len(cols)):
                jacobian_A12.setValue(row + 3*self.n_mesh_p, cols[index_col],
                        vals[index_col], True)

        # Block a42
        for row in range(self.n_mesh_vy):
            cols, vals = temp_a42.matrix.getRow(row)
            for index_col in range(len(cols)):
                jacobian_A12.setValue(row + 3*self.n_mesh_p, cols[index_col] +
                    self.n_mesh_p, vals[index_col], True)

        jacobian_A12.assemble()
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #++++++++++++ Creation of the A21 block ++++++++++++++++++++++++++++++++
        jacobian_A21 = petsc.Mat().create()
        number_of_rows = 2*self.n_mesh_p
        number_of_cols = 2*self.n_mesh_vx + 2*self.n_mesh_vy
        size_row = (number_of_rows, number_of_rows)
        size_col = (number_of_cols, number_of_cols)
        jacobian_A21.setSizes((size_row, size_col))
        jacobian_A21.setUp()

        # The A21 block can be divided in 2x4 blocks, and we build each of this
        # block

        # Creation of the a11 block of A21
        temp_a11 = self.velocity_div_x

        # Creation of the a22 block of A21
        temp_a22 = self.velocity_div_x

        # Creation of the a13 block of A21
        temp_a13 = self.velocity_div_y

        # Creation of the a24 block of A21
        temp_a24 = self.velocity_div_y

        # Block a11
        for row in range(self.n_mesh_p):
            cols, vals = temp_a11.matrix.getRow(row)
            for index_col in range(len(cols)):
                jacobian_A21.setValue(row, cols[index_col], vals[index_col], True)

        # Block a22
        for row in range(self.n_mesh_p):
            cols, vals = temp_a22.matrix.getRow(row)
            for index_col in range(len(cols)):
                jacobian_A21.setValue(row + self.n_mesh_vx, cols[index_col] +
                    self.n_mesh_vx, vals[index_col], True)

        # Block a13
        for row in range(self.n_mesh_p):
            cols, vals = temp_a13.matrix.getRow(row)
            for index_col in range(len(cols)):
                jacobian_A21.setValue(row, cols[index_col] + 2*self.n_mesh_vx,
                    vals[index_col], True)

        # Block a24
        for row in range(self.n_mesh_p):
            cols, vals = temp_a24.matrix.getRow(row)
            for index_col in range(len(cols)):
                jacobian_A21.setValue(row + self.n_mesh_vx, cols[index_col] + 3*self.n_mesh_vx,
                    vals[index_col], True)

        jacobian_A21.assemble()
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


        self.jacobian_base = {"A11":jacobian_A11, "A12":jacobian_A12,
                "A21":jacobian_A21}

    def make_jacobian_variable(self, tree_velocity_x, tree_velocity_y):
        """...

        """

        velocity_x = sd.Scalar(tree_velocity_x)
        velocity_y = sd.Scalar(tree_velocity_y)

        aux_1 = sd.Operator(tree_velocity_x, 0, self.divergence, velocity_x.sc)
        aux_2 = sd.Operator(tree_velocity_x, 0, self.divergence, velocity_y.sc)
        aux_3 = sd.Operator(tree_velocity_y, 1, self.divergence, velocity_x.sc)
        aux_4 = sd.Operator(tree_velocity_y, 1, self.divergence, velocity_y.sc)

        #++++++++++++ Creation of the A11 block ++++++++++++++++++++++++++++++++
        jacobian_A11 = petsc.Mat().create()
        number_of_rows = 2*self.n_mesh_vx + 2*self.n_mesh_vy
        size_row = (number_of_rows, number_of_rows)
        size_col = (number_of_rows, number_of_rows)
        jacobian_A11.setSizes((size_row, size_col))
        jacobian_A11.setUp()

        # The A11 block can be divided in 4x4 blocks, and we build each of this
        # block

        # Creation of the a11 block of A11
        temp_a11 = sd.add_operators(
            sd.mul_num_operator(2*self.A_coefs["a11"], aux_1),
            sd.mul_num_operator(self.A_coefs["a11"], aux_4))

        # Creation of the a12 block of A11
        temp_a12 = sd.add_operators(
            sd.mul_num_operator(2*self.A_coefs["a12"], aux_1),
            sd.mul_num_operator(self.A_coefs["a12"], aux_4))

        # Creation of the a21 block of A11
        temp_a21 = sd.add_operators(
            sd.mul_num_operator(2*self.A_coefs["a21"], aux_1),
            sd.mul_num_operator(self.A_coefs["a21"], aux_4))

        # Creation of the a22 block of A11
        temp_a22 = sd.add_operators(
            sd.mul_num_operator(2*self.A_coefs["a22"], aux_1),
            sd.mul_num_operator(self.A_coefs["a22"], aux_4))

        # Creation of the a33 block of A11
        temp_a33 = sd.add_operators(
            sd.mul_num_operator(self.A_coefs["a11"], aux_1),
            sd.mul_num_operator(2*self.A_coefs["a11"], aux_4))

        # Creation of the a34 block of A11
        temp_a34 = sd.add_operators(
            sd.mul_num_operator(self.A_coefs["a12"], aux_1),
            sd.mul_num_operator(2*self.A_coefs["a12"], aux_4))

        # Creation of the a43 block of A11
        temp_a43 = sd.add_operators(
            sd.mul_num_operator(self.A_coefs["a21"], aux_1),
            sd.mul_num_operator(2*self.A_coefs["a21"], aux_4))

        # Creation of the a44 block of A11
        temp_a44 = sd.add_operators(
            sd.mul_num_operator(self.A_coefs["a22"], aux_1),
            sd.mul_num_operator(2*self.A_coefs["a22"], aux_4))

        # Creation of the a13 block of A11
        temp_a13 = sd.mul_num_operator(self.A_coefs["a11"], aux_3)

        # Creation of the a14 block of A11
        temp_a14 = sd.mul_num_operator(self.A_coefs["a12"], aux_3)

        # Creation of the a23 block of A11
        temp_a23 = sd.mul_num_operator(self.A_coefs["a21"], aux_3)

        # Creation of the a24 block of A11
        temp_a24 = sd.mul_num_operator(self.A_coefs["a22"], aux_3)

        # Creation of the a31 block of A11
        temp_a31 = sd.mul_num_operator(self.A_coefs["a11"], aux_2)

        # Creation of the a32 block of A11
        temp_a32 = sd.mul_num_operator(self.A_coefs["a12"], aux_2)

        # Creation of the a41 block of A11
        temp_a41 = sd.mul_num_operator(self.A_coefs["a21"], aux_2)

        # Creation of the a42 block of A11
        temp_a42 = sd.mul_num_operator(self.A_coefs["a22"], aux_2)

        # Block a11
        for row in range(self.n_mesh_vx):
            cols, vals = temp_a11.matrix.getRow(row)
            for index_col in range(len(cols)):
                jacobian_A11.setValue(row, cols[index_col], vals[index_col], True)

        # Block a12
        for row in range(self.n_mesh_vx):
            cols, vals = temp_a12.matrix.getRow(row)
            for index_col in range(len(cols)):
                jacobian_A11.setValue(row, cols[index_col] + self.n_mesh_vx,
                    vals[index_col], True)

        # Block a21
        for row in range(self.n_mesh_vx):
            cols, vals = temp_a21.matrix.getRow(row)
            for index_col in range(len(cols)):
                jacobian_A11.setValue(row + self.n_mesh_vx, cols[index_col],
                    vals[index_col], True)

        # Block a22
        for row in range(self.n_mesh_vx):
            cols, vals = temp_a22.matrix.getRow(row)
            for index_col in range(len(cols)):
                jacobian_A11.setValue(row + self.n_mesh_vx, cols[index_col] +
                    self.n_mesh_vx, vals[index_col], True)

        # Block a33
        for row in range(self.n_mesh_vy):
            cols, vals = temp_a33.matrix.getRow(row)
            for index_col in range(len(cols)):
                jacobian_A11.setValue(row + 2*self.n_mesh_vx, cols[index_col] +
                    2*self.n_mesh_vx, vals[index_col], True)

        # Block a34
        for row in range(self.n_mesh_vy):
            cols, vals = temp_a34.matrix.getRow(row)
            for index_col in range(len(cols)):
                jacobian_A11.setValue(row + 2*self.n_mesh_vx, cols[index_col] +
                    3*self.n_mesh_vx, vals[index_col], True)

        # Block a43
        for row in range(self.n_mesh_vy):
            cols, vals = temp_a43.matrix.getRow(row)
            for index_col in range(len(cols)):
                jacobian_A11.setValue(row + 3*self.n_mesh_vx, cols[index_col] +
                    2*self.n_mesh_vx, vals[index_col], True)

        # Block a44
        for row in range(self.n_mesh_vy):
            cols, vals = temp_a44.matrix.getRow(row)
            for index_col in range(len(cols)):
                jacobian_A11.setValue(row + 3*self.n_mesh_vx, cols[index_col] +
                    3*self.n_mesh_vx, vals[index_col], True)

        # Block a13
        for row in range(self.n_mesh_vx):
            cols, vals = temp_a13.matrix.getRow(row)
            for index_col in range(len(cols)):
                jacobian_A11.setValue(row, cols[index_col] + 2*self.n_mesh_vx,
                    vals[index_col], True)

        # Block a14
        for row in range(self.n_mesh_vx):
            cols, vals = temp_a14.matrix.getRow(row)
            for index_col in range(len(cols)):
                jacobian_A11.setValue(row, cols[index_col] + 3*self.n_mesh_vx,
                    vals[index_col], True)

        # Block a23
        for row in range(self.n_mesh_vx):
            cols, vals = temp_a23.matrix.getRow(row)
            for index_col in range(len(cols)):
                jacobian_A11.setValue(row + self.n_mesh_vx, cols[index_col] +
                    2*self.n_mesh_vx, vals[index_col], True)

        # Block a24
        for row in range(self.n_mesh_vx):
            cols, vals = temp_a24.matrix.getRow(row)
            for index_col in range(len(cols)):
                jacobian_A11.setValue(row + n_mesh_vx, cols[index_col] +
                3*self.n_mesh_vx, vals[index_col], True)

        # Block a31
        for row in range(self.n_mesh_vy):
            cols, vals = temp_a31.matrix.getRow(row)
            for index_col in range(len(cols)):
                jacobian_A11.setValue(row + 2*self.n_mesh_vx, cols[index_col],
                    vals[index_col], True)

        # Block a32
        for row in range(self.n_mesh_vy):
            cols, vals = temp_a32.matrix.getRow(row)
            for index_col in range(len(cols)):
                jacobian_A11.setValue(row + 2*self.n_mesh_vx, cols[index_col] +
                    self.n_mesh_vx, vals[index_col], True)

        # Block a41
        for row in range(self.n_mesh_vy):
            cols, vals = temp_a41.matrix.getRow(row)
            for index_col in range(len(cols)):
                jacobian_A11.setValue(row + 3*self.n_mesh_vx, cols[index_col],
                    vals[index_col], True)

        # Block a42
        for row in range(self.n_mesh_vy):
            cols, vals = temp_a42.matrix.getRow(row)
            for index_col in range(len(cols)):
                jacobian_A11.setValue(row + 3*self.n_mesh_vx, cols[index_col] +
                    self.n_mesh_vx, vals[index_col], True)

        jacobian_A11.assemble()

        self.jacobian_variable = jacobian_A11

    def make_jacobian(self, velocity_x, velocity_y):
        """...

        """

        self.jacobian = self.jacobian_base
        #self.make_jacobian_variable(velocity_x, velocity_y)
        #self.jacobian["A11"] += self.jacobian_variable

    #===========================================================================
    #+++++++++++++ Picard solver - full explicit convection ++++++++++++++++++++
    #===========================================================================

    def rhs_1_pic_fullexp(self, source_term_vx=None):
        """...

        """

        result = sd.Scalar()
        temp_1 = sd.Scalar()

        temp_1.sc = self.velocity_lap_x.bc
        temp_1 = sd.mul_num_scalar(cfg.nu, temp_1)
        result.sc = temp_1.sc

        temp_1.sc = self.velocity_adv_x.bc_x
        temp_1.sc += self.velocity_adv_x.bc_y
        temp_1 = sd.mul_num_scalar(-1, temp_1)
        result.sc += temp_1.sc

        if source_term_vx is not None:
            #if not self.uniform: #source_term_vx is a tree
            #    self.compute_source_term(source_term_vx, self.stfv_x, t)
            #    temp_1 = sd.Scalar(source_term_vx)
            #else: #source_term_vx is a scalar
            #    temp_1.sc = source_term_vx.sc.copy()
            temp_1 = self.velocity_mass.apply(temp_1)
            result.sc += temp_1.sc

        return result

    def rhs_2_pic_fullexp(self, source_term_vy=None):
        """...

        """

        result = sd.Scalar()
        temp_1 = sd.Scalar()

        temp_1.sc = self.velocity_lap_y.bc
        temp_1 = sd.mul_num_scalar(cfg.nu, temp_1)
        result.sc = temp_1.sc

        temp_1.sc = self.velocity_adv_y.bc_x
        temp_1.sc += self.velocity_adv_y.bc_y
        temp_1 = sd.mul_num_scalar(-1, temp_1)
        result.sc += temp_1.sc

        if source_term_vy is not None:
            #if not self.uniform: #source_term_vy is a tree
            #    self.compute_source_term(source_term_vy, self.stfv_y, t)
            #    temp_1 = sd.Scalar(source_term_vy)
            #else: #source_term_vy is a scalar
            #    temp_1.sc = source_term_vy.sc.copy()
            temp_1 = self.velocity_mass.apply(temp_1)
            result.sc += temp_1.sc

        return result

    def rhs_3_pic_fullexp(self, source_term=None):
        """...

        """

        result = sd.Scalar()
        result.sc = self.velocity_div_x.bc
        result.sc += self.velocity_div_y.bc
        result = sd.mul_num_scalar(-1, result)
        if source_term is not None:
            result.sc += source_term.sc

        return result

    def nonlin_rhs_pic_fullexp(self, velocity_x, velocity_y,
            st_rhs_11=None, st_rhs_12=None, st_rhs_21=None,
            st_rhs_22=None, st_rhs_31=None, st_rhs_32=None):
        """Returns a petsc vector that is the rhs of the full nonlinear system
        from the RK method.
        """

        result = petsc.Vec().create()
        result.setSizes(2*self.n_mesh_vx + 2*self.n_mesh_vy +
                2*self.n_mesh_p)
        result.setUp()

        temp = sd.add_scalars(
            sd.mul_num_scalar(1/self.dt, self.velocity_mass.apply(velocity_x)),
            sd.mul_num_scalar(self.A_coefs["a11"], self.rhs_1_pic_fullexp(
                st_rhs_11)),
                #source_term_vx, t_ini + self.C_coefs["c1"]*self.dt)),
            sd.mul_num_scalar(self.A_coefs["a12"], self.rhs_1_pic_fullexp(
                st_rhs_12)))
                #source_term_vx, t_ini + self.C_coefs["c2"]*self.dt)))

        result[:self.n_mesh_vx] = temp.sc

        temp = sd.add_scalars(
            sd.mul_num_scalar(1/self.dt, self.velocity_mass.apply(velocity_x)),
            sd.mul_num_scalar(self.A_coefs["a21"], self.rhs_1_pic_fullexp(
                st_rhs_11)),
                #source_term_vx, t_ini + self.C_coefs["c1"]*self.dt)),
            sd.mul_num_scalar(self.A_coefs["a22"], self.rhs_1_pic_fullexp(
                st_rhs_12)))
                #source_term_vx, t_ini + self.C_coefs["c2"]*self.dt)))

        result[self.n_mesh_vx:2*self.n_mesh_vx] = temp.sc

        temp = sd.add_scalars(
            sd.mul_num_scalar(1/self.dt, self.velocity_mass.apply(velocity_y)),
            sd.mul_num_scalar(self.A_coefs["a11"], self.rhs_2_pic_fullexp(
                st_rhs_21)),
                #source_term_vy, t_ini + self.C_coefs["c1"]*self.dt)),
            sd.mul_num_scalar(self.A_coefs["a12"], self.rhs_2_pic_fullexp(
                st_rhs_22)))
                #source_term_vy, t_ini + self.C_coefs["c2"]*self.dt)))

        result[2*self.n_mesh_vx:2*self.n_mesh_vx + self.n_mesh_vy] = temp.sc

        temp = sd.add_scalars(
            sd.mul_num_scalar(1/self.dt, self.velocity_mass.apply(velocity_y)),
            sd.mul_num_scalar(self.A_coefs["a21"], self.rhs_2_pic_fullexp(
                st_rhs_21)),
                #source_term_vy, t_ini + self.C_coefs["c1"]*self.dt)),
            sd.mul_num_scalar(self.A_coefs["a22"], self.rhs_2_pic_fullexp(
                st_rhs_22)))
                #source_term_vy, t_ini + self.C_coefs["c2"]*self.dt)))

        result[2*self.n_mesh_vx + self.n_mesh_vy:2*self.n_mesh_vx + 2*self.n_mesh_vy] = temp.sc

        result[2*self.n_mesh_vx + 2*self.n_mesh_vy:2*self.n_mesh_vx +
                2*self.n_mesh_vy + self.n_mesh_p] = \
        self.rhs_3_pic_fullexp(st_rhs_31).sc
        result[2*self.n_mesh_vx + 2*self.n_mesh_vy + self.n_mesh_p:] = \
            self.rhs_3_pic_fullexp(st_rhs_32).sc

        return result

    def rhs_pic_fullexp_iteration(self, velocity_x, velocity_y, x1,
            y1, x2, y2, st_rhs_11=None, st_rhs_12=None, st_rhs_21=None,
            st_rhs_22=None, st_rhs_31=None, st_rhs_32=None):
        """...

        """

        iter_rhs_1 =  self.g_1.sc.duplicate()
        iter_rhs_2 =  self.g_2.sc.duplicate()
        iter_rhs_3 =  self.g_3.sc.duplicate()

        temp = sd.add_scalars(
            sd.mul_num_scalar(1/self.dt, self.velocity_mass.apply(velocity_x)),
            sd.mul_num_scalar(self.A_coefs["a11"], self.rhs_1_pic_fullexp(
                st_rhs_11)),
                #source_term_vx, t_ini + self.C_coefs["c1"]*self.dt)),
            sd.mul_num_scalar(self.A_coefs["a12"], self.rhs_1_pic_fullexp(
                st_rhs_12)))
                #source_term_vx, t_ini + self.C_coefs["c2"]*self.dt)))

        iter_rhs_1[:self.n_mesh_vx] = temp.sc

        temp = sd.add_scalars(
            sd.mul_num_scalar(1/self.dt, self.velocity_mass.apply(velocity_x)),
            sd.mul_num_scalar(self.A_coefs["a21"], self.rhs_1_pic_fullexp(
                st_rhs_11)),
                #source_term_vx, t_ini + self.C_coefs["c1"]*self.dt)),
            sd.mul_num_scalar(self.A_coefs["a22"], self.rhs_1_pic_fullexp(
                st_rhs_12)))
                #source_term_vx, t_ini + self.C_coefs["c2"]*self.dt)))

        iter_rhs_1[self.n_mesh_vx:] = temp.sc

        temp = sd.add_scalars(
            sd.mul_num_scalar(1/self.dt, self.velocity_mass.apply(velocity_y)),
            sd.mul_num_scalar(self.A_coefs["a11"], self.rhs_2_pic_fullexp(
                st_rhs_21)),
                #source_term_vy, t_ini + self.C_coefs["c1"]*self.dt)),
            sd.mul_num_scalar(self.A_coefs["a12"], self.rhs_2_pic_fullexp(
                st_rhs_22)))
                #source_term_vx, t_ini + self.C_coefs["c2"]*self.dt)))

        iter_rhs_2[:self.n_mesh_vy] = temp.sc

        temp = sd.add_scalars(
            sd.mul_num_scalar(1/self.dt, self.velocity_mass.apply(velocity_y)),
            sd.mul_num_scalar(self.A_coefs["a21"], self.rhs_2_pic_fullexp(
                st_rhs_21)),
                #source_term_vy, t_ini + self.C_coefs["c1"]*self.dt)),
            sd.mul_num_scalar(self.A_coefs["a22"], self.rhs_2_pic_fullexp(
                st_rhs_22)))
                #source_term_vy, t_ini + self.C_coefs["c2"]*self.dt)))

        iter_rhs_2[self.n_mesh_vy:] = temp.sc

        iter_rhs_3[:self.n_mesh_p] = self.rhs_3_pic_fullexp(st_rhs_31).sc
        iter_rhs_3[self.n_mesh_p:] = self.rhs_3_pic_fullexp(st_rhs_32).sc

        temp_adv_x_1 = self.velocity_adv_x.apply(self.dimension, x1, x1, y1, low_mach=self.low_mach).sc
        temp_adv_x_1 -= self.velocity_adv_x.bc_x
        temp_adv_x_1 -= self.velocity_adv_x.bc_y

        temp_adv_x_2 = self.velocity_adv_x.apply(self.dimension, x2, x2, y2, low_mach=self.low_mach).sc
        temp_adv_x_2 -= self.velocity_adv_x.bc_x
        temp_adv_x_2 -= self.velocity_adv_x.bc_y

        temp_adv_y_1 = self.velocity_adv_y.apply(self.dimension, y1, x1, y1, low_mach=self.low_mach).sc
        temp_adv_y_1 -= self.velocity_adv_y.bc_x
        temp_adv_y_1 -= self.velocity_adv_y.bc_y

        temp_adv_y_2 = self.velocity_adv_y.apply(self.dimension, y2, x2, y2, low_mach=self.low_mach).sc
        temp_adv_y_2 -= self.velocity_adv_y.bc_x
        temp_adv_y_2 -= self.velocity_adv_y.bc_y

        iter_rhs_1[:self.n_mesh_vx] -= self.A_coefs["a11"]*temp_adv_x_1
        iter_rhs_1[:self.n_mesh_vx] -= self.A_coefs["a12"]*temp_adv_x_2

        iter_rhs_1[self.n_mesh_vx:] -= self.A_coefs["a21"]*temp_adv_x_1
        iter_rhs_1[self.n_mesh_vx:] -= self.A_coefs["a22"]*temp_adv_x_2

        iter_rhs_2[:self.n_mesh_vy] -= self.A_coefs["a11"]*temp_adv_y_1
        iter_rhs_2[:self.n_mesh_vy] -= self.A_coefs["a12"]*temp_adv_y_2

        iter_rhs_2[self.n_mesh_vy:] -= self.A_coefs["a21"]*temp_adv_y_1
        iter_rhs_2[self.n_mesh_vy:] -= self.A_coefs["a22"]*temp_adv_y_2

        return iter_rhs_1, iter_rhs_2, iter_rhs_3

    def make_mat_iter_base(self):
        """...

        """

        #++++++++++++ Creation of the A11 block ++++++++++++++++++++++++++++++++
        mat_iter_A11 = petsc.Mat().create()
        number_of_rows = 2*self.n_mesh_vx
        size_row = (number_of_rows, number_of_rows)
        size_col = (number_of_rows, number_of_rows)
        mat_iter_A11.setSizes((size_row, size_col))
        mat_iter_A11.setUp()

        # The A11 block can be divided in 2x2 blocks, and we build each of this
        # block

        # Creation of the a11 block of A11
        temp_a11 = sd.add_operators(
            sd.mul_num_operator(1/self.dt, self.velocity_mass),
            sd.mul_num_operator(-cfg.nu*self.A_coefs["a11"], self.velocity_lap_x))

        # Creation of the a12 block of A11
        temp_a12 = sd.mul_num_operator(-cfg.nu*self.A_coefs["a12"], self.velocity_lap_x)

        # Creation of the a21 block of A11
        temp_a21 = sd.mul_num_operator(-cfg.nu*self.A_coefs["a21"], self.velocity_lap_x)

        # Creation of the a22 block of A11
        temp_a22 = sd.add_operators(
            sd.mul_num_operator(1/self.dt, self.velocity_mass),
            sd.mul_num_operator(-cfg.nu*self.A_coefs["a22"], self.velocity_lap_x))

        # Block a11
        for row in range(self.n_mesh_vx):
            cols, vals = temp_a11.matrix.getRow(row)
            for index_col in range(len(cols)):
                mat_iter_A11.setValue(row, cols[index_col], vals[index_col], True)

        # Block a12
        for row in range(self.n_mesh_vx):
            cols, vals = temp_a12.matrix.getRow(row)
            for index_col in range(len(cols)):
                mat_iter_A11.setValue(row, cols[index_col] + self.n_mesh_vx,
                    vals[index_col], True)

        # Block a21
        for row in range(self.n_mesh_vx):
            cols, vals = temp_a21.matrix.getRow(row)
            for index_col in range(len(cols)):
                mat_iter_A11.setValue(row + self.n_mesh_vx, cols[index_col],
                    vals[index_col], True)

        # Block a22
        for row in range(self.n_mesh_vx):
            cols, vals = temp_a22.matrix.getRow(row)
            for index_col in range(len(cols)):
                mat_iter_A11.setValue(row + self.n_mesh_vx, cols[index_col] +
                    self.n_mesh_vx, vals[index_col], True)

        mat_iter_A11.assemble()
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #++++++++++++ Creation of the A22 block ++++++++++++++++++++++++++++++++
        mat_iter_A22 = petsc.Mat().create()
        number_of_rows = 2*self.n_mesh_vy
        size_row = (number_of_rows, number_of_rows)
        size_col = (number_of_rows, number_of_rows)
        mat_iter_A22.setSizes((size_row, size_col))
        mat_iter_A22.setUp()

        # The A22 block can be divided in 2x2 blocks, and we build each of this
        # block

        # Creation of the a11 block of A22
        temp_a11 = sd.add_operators(
            sd.mul_num_operator(1/self.dt, self.velocity_mass),
            sd.mul_num_operator(-cfg.nu*self.A_coefs["a11"], self.velocity_lap_y))

        # Creation of the a12 block of A22
        temp_a12 = sd.mul_num_operator(-cfg.nu*self.A_coefs["a12"], self.velocity_lap_y)

        # Creation of the a21 block of A22
        temp_a21 = sd.mul_num_operator(-cfg.nu*self.A_coefs["a21"], self.velocity_lap_y)

        # Creation of the a22 block of A22
        temp_a22 = sd.add_operators(
            sd.mul_num_operator(1/self.dt, self.velocity_mass),
            sd.mul_num_operator(-cfg.nu*self.A_coefs["a22"], self.velocity_lap_y))

        # Block a11
        for row in range(self.n_mesh_vy):
            cols, vals = temp_a11.matrix.getRow(row)
            for index_col in range(len(cols)):
                mat_iter_A22.setValue(row, cols[index_col], vals[index_col], True)

        # Block a12
        for row in range(self.n_mesh_vy):
            cols, vals = temp_a12.matrix.getRow(row)
            for index_col in range(len(cols)):
                mat_iter_A22.setValue(row, cols[index_col] + self.n_mesh_vy,
                    vals[index_col], True)

        # Block a21
        for row in range(self.n_mesh_vy):
            cols, vals = temp_a21.matrix.getRow(row)
            for index_col in range(len(cols)):
                mat_iter_A22.setValue(row + self.n_mesh_vy, cols[index_col],
                    vals[index_col], True)

        # Block a22
        for row in range(self.n_mesh_vy):
            cols, vals = temp_a22.matrix.getRow(row)
            for index_col in range(len(cols)):
                mat_iter_A22.setValue(row + self.n_mesh_vy, cols[index_col] +
                    self.n_mesh_vy, vals[index_col], True)

        mat_iter_A22.assemble()
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #++++++++++++ Creation of the A13 block ++++++++++++++++++++++++++++++++
        mat_iter_A13 = petsc.Mat().create()
        number_of_rows = 2*self.n_mesh_vx
        number_of_cols = 2*self.n_mesh_p
        size_row = (number_of_rows, number_of_rows)
        size_col = (number_of_cols, number_of_cols)
        mat_iter_A13.setSizes((size_row, size_col))
        mat_iter_A13.setUp()

        # The A13 block can be divided in 2x2 blocks, and we build each of this
        # block

        # Creation of the a11 block of A12
        temp_a11 = sd.mul_num_operator(self.A_coefs["a11"], self.pressure_grad_x)

        # Creation of the a12 block of A12
        temp_a12 = sd.mul_num_operator(self.A_coefs["a12"], self.pressure_grad_x)

        # Creation of the a21 block of A12
        temp_a21 = sd.mul_num_operator(self.A_coefs["a21"], self.pressure_grad_x)

        # Creation of the a22 block of A12
        temp_a22 = sd.mul_num_operator(self.A_coefs["a22"], self.pressure_grad_x)

        # Block a11
        for row in range(self.n_mesh_vx):
            cols, vals = temp_a11.matrix.getRow(row)
            for index_col in range(len(cols)):
                mat_iter_A13.setValue(row, cols[index_col], vals[index_col], True)

        # Block a12
        for row in range(self.n_mesh_vx):
            cols, vals = temp_a12.matrix.getRow(row)
            for index_col in range(len(cols)):
                mat_iter_A13.setValue(row, cols[index_col] + self.n_mesh_p,
                    vals[index_col], True)

        # Block a21
        for row in range(self.n_mesh_vx):
            cols, vals = temp_a21.matrix.getRow(row)
            for index_col in range(len(cols)):
                mat_iter_A13.setValue(row + self.n_mesh_p, cols[index_col],
                    vals[index_col], True)

        # Block a22
        for row in range(self.n_mesh_vx):
            cols, vals = temp_a22.matrix.getRow(row)
            for index_col in range(len(cols)):
                mat_iter_A13.setValue(row + self.n_mesh_p, cols[index_col] +
                    self.n_mesh_p, vals[index_col], True)

        mat_iter_A13.assemble()
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #++++++++++++ Creation of the A23 block ++++++++++++++++++++++++++++++++
        mat_iter_A23 = petsc.Mat().create()
        number_of_rows = 2*self.n_mesh_vy
        number_of_cols = 2*self.n_mesh_p
        size_row = (number_of_rows, number_of_rows)
        size_col = (number_of_cols, number_of_cols)
        mat_iter_A23.setSizes((size_row, size_col))
        mat_iter_A23.setUp()

        # The A23 block can be divided in 2x2 blocks, and we build each of this
        # block

        # Creation of the a11 block of A23
        temp_a11 = sd.mul_num_operator(self.A_coefs["a11"], self.pressure_grad_y)

        # Creation of the a12 block of A23
        temp_a12 = sd.mul_num_operator(self.A_coefs["a12"], self.pressure_grad_y)

        # Creation of the a21 block of A23
        temp_a21 = sd.mul_num_operator(self.A_coefs["a21"], self.pressure_grad_y)

        # Creation of the a22 block of A23
        temp_a22 = sd.mul_num_operator(self.A_coefs["a22"], self.pressure_grad_y)

        # Block a11
        for row in range(self.n_mesh_vy):
            cols, vals = temp_a11.matrix.getRow(row)
            for index_col in range(len(cols)):
                mat_iter_A23.setValue(row, cols[index_col], vals[index_col], True)

        # Block a12
        for row in range(self.n_mesh_vy):
            cols, vals = temp_a12.matrix.getRow(row)
            for index_col in range(len(cols)):
                mat_iter_A23.setValue(row, cols[index_col] +
                    self.n_mesh_p, vals[index_col], True)

        # Block a21
        for row in range(self.n_mesh_vy):
            cols, vals = temp_a21.matrix.getRow(row)
            for index_col in range(len(cols)):
                mat_iter_A23.setValue(row + self.n_mesh_p, cols[index_col],
                    vals[index_col], True)

        # Block a22
        for row in range(self.n_mesh_vy):
            cols, vals = temp_a22.matrix.getRow(row)
            for index_col in range(len(cols)):
                mat_iter_A23.setValue(row + self.n_mesh_p, cols[index_col] +
                    self.n_mesh_p, vals[index_col], True)

        mat_iter_A23.assemble()
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #++++++++++++ Creation of the A31 block ++++++++++++++++++++++++++++++++
        mat_iter_A31 = petsc.Mat().create()
        number_of_rows = 2*self.n_mesh_p
        number_of_cols = 2*self.n_mesh_vx
        size_row = (number_of_rows, number_of_rows)
        size_col = (number_of_cols, number_of_cols)
        mat_iter_A31.setSizes((size_row, size_col))
        mat_iter_A31.setUp()

        # The A31 block can be divided in 2x2 blocks, and we build each of this
        # block

        # Creation of the a11 block of A31
        temp_a11 = self.velocity_div_x

        # Creation of the a22 block of A21
        temp_a22 = self.velocity_div_x

        # Block a11
        for row in range(self.n_mesh_p):
            cols, vals = temp_a11.matrix.getRow(row)
            for index_col in range(len(cols)):
                mat_iter_A31.setValue(row, cols[index_col], vals[index_col], True)

        # Block a22
        for row in range(self.n_mesh_p):
            cols, vals = temp_a22.matrix.getRow(row)
            for index_col in range(len(cols)):
                mat_iter_A31.setValue(row + self.n_mesh_vx, cols[index_col] +
                    self.n_mesh_vx, vals[index_col], True)

        mat_iter_A31.assemble()
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #++++++++++++ Creation of the A32 block ++++++++++++++++++++++++++++++++
        mat_iter_A32 = petsc.Mat().create()
        number_of_rows = 2*self.n_mesh_p
        number_of_cols = 2*self.n_mesh_vy
        size_row = (number_of_rows, number_of_rows)
        size_col = (number_of_cols, number_of_cols)
        mat_iter_A32.setSizes((size_row, size_col))
        mat_iter_A32.setUp()

        # The A32 block can be divided in 2x2 blocks, and we build each of this
        # block

        # Creation of the a13 block of A32
        temp_a11 = self.velocity_div_y

        # Creation of the a24 block of A21
        temp_a22 = self.velocity_div_y

        # Block a11
        for row in range(self.n_mesh_p):
            cols, vals = temp_a11.matrix.getRow(row)
            for index_col in range(len(cols)):
                mat_iter_A32.setValue(row, cols[index_col], vals[index_col], True)

        # Block a22
        for row in range(self.n_mesh_p):
            cols, vals = temp_a22.matrix.getRow(row)
            for index_col in range(len(cols)):
                mat_iter_A32.setValue(row + self.n_mesh_vy, cols[index_col] +
                    self.n_mesh_vy, vals[index_col], True)

        mat_iter_A32.assemble()
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


        self.mat_iter_base = {"A11":mat_iter_A11, "A22":mat_iter_A22,
                "A13":mat_iter_A13, "A23":mat_iter_A23, "A31":mat_iter_A31,
                "A32":mat_iter_A32}

    def make_mat_iter_variable(self, tree_velocity_x, tree_velocity_y):
        """...

        """

        velocity_x = sd.Scalar(tree_velocity_x)
        velocity_y = sd.Scalar(tree_velocity_y)

        aux_1 = sd.Operator(tree_velocity_x, 0, self.divergence, velocity_x.sc)
        aux_2 = sd.Operator(tree_velocity_x, 0, self.divergence, velocity_y.sc)
        aux_3 = sd.Operator(tree_velocity_y, 1, self.divergence, velocity_x.sc)
        aux_4 = sd.Operator(tree_velocity_y, 1, self.divergence, velocity_y.sc)

        #++++++++++++ Creation of the A11 block ++++++++++++++++++++++++++++++++
        mat_iter_A11 = petsc.Mat().create()
        number_of_rows = 2*self.n_mesh_vx + 2*self.n_mesh_vy
        size_row = (number_of_rows, number_of_rows)
        size_col = (number_of_rows, number_of_rows)
        mat_iter_A11.setSizes((size_row, size_col))
        mat_iter_A11.setUp()

        # The A11 block can be divided in 4x4 blocks, and we build each of this
        # block

        # Creation of the a11 block of A11
        temp_a11 = sd.add_operators(
            sd.mul_num_operator(2*self.A_coefs["a11"], aux_1),
            sd.mul_num_operator(self.A_coefs["a11"], aux_4))

        # Creation of the a12 block of A11
        temp_a12 = sd.add_operators(
            sd.mul_num_operator(2*self.A_coefs["a12"], aux_1),
            sd.mul_num_operator(self.A_coefs["a12"], aux_4))

        # Creation of the a21 block of A11
        temp_a21 = sd.add_operators(
            sd.mul_num_operator(2*self.A_coefs["a21"], aux_1),
            sd.mul_num_operator(self.A_coefs["a21"], aux_4))

        # Creation of the a22 block of A11
        temp_a22 = sd.add_operators(
            sd.mul_num_operator(2*self.A_coefs["a22"], aux_1),
            sd.mul_num_operator(self.A_coefs["a22"], aux_4))

        # Creation of the a33 block of A11
        temp_a33 = sd.add_operators(
            sd.mul_num_operator(self.A_coefs["a11"], aux_1),
            sd.mul_num_operator(2*self.A_coefs["a11"], aux_4))

        # Creation of the a34 block of A11
        temp_a34 = sd.add_operators(
            sd.mul_num_operator(self.A_coefs["a12"], aux_1),
            sd.mul_num_operator(2*self.A_coefs["a12"], aux_4))

        # Creation of the a43 block of A11
        temp_a43 = sd.add_operators(
            sd.mul_num_operator(self.A_coefs["a21"], aux_1),
            sd.mul_num_operator(2*self.A_coefs["a21"], aux_4))

        # Creation of the a44 block of A11
        temp_a44 = sd.add_operators(
            sd.mul_num_operator(self.A_coefs["a22"], aux_1),
            sd.mul_num_operator(2*self.A_coefs["a22"], aux_4))

        # Creation of the a13 block of A11
        temp_a13 = sd.mul_num_operator(self.A_coefs["a11"], aux_3)

        # Creation of the a14 block of A11
        temp_a14 = sd.mul_num_operator(self.A_coefs["a12"], aux_3)

        # Creation of the a23 block of A11
        temp_a23 = sd.mul_num_operator(self.A_coefs["a21"], aux_3)

        # Creation of the a24 block of A11
        temp_a24 = sd.mul_num_operator(self.A_coefs["a22"], aux_3)

        # Creation of the a31 block of A11
        temp_a31 = sd.mul_num_operator(self.A_coefs["a11"], aux_2)

        # Creation of the a32 block of A11
        temp_a32 = sd.mul_num_operator(self.A_coefs["a12"], aux_2)

        # Creation of the a41 block of A11
        temp_a41 = sd.mul_num_operator(self.A_coefs["a21"], aux_2)

        # Creation of the a42 block of A11
        temp_a42 = sd.mul_num_operator(self.A_coefs["a22"], aux_2)

        # Block a11
        for row in range(self.n_mesh_vx):
            cols, vals = temp_a11.matrix.getRow(row)
            for index_col in range(len(cols)):
                mat_iter_A11.setValue(row, cols[index_col], vals[index_col], True)

        # Block a12
        for row in range(self.n_mesh_vx):
            cols, vals = temp_a12.matrix.getRow(row)
            for index_col in range(len(cols)):
                mat_iter_A11.setValue(row, cols[index_col] + self.n_mesh_vx,
                    vals[index_col], True)

        # Block a21
        for row in range(self.n_mesh_vx):
            cols, vals = temp_a21.matrix.getRow(row)
            for index_col in range(len(cols)):
                mat_iter_A11.setValue(row + self.n_mesh_vx, cols[index_col],
                    vals[index_col], True)

        # Block a22
        for row in range(self.n_mesh_vx):
            cols, vals = temp_a22.matrix.getRow(row)
            for index_col in range(len(cols)):
                mat_iter_A11.setValue(row + self.n_mesh_vx, cols[index_col] +
                    self.n_mesh_vx, vals[index_col], True)

        # Block a33
        for row in range(self.n_mesh_vy):
            cols, vals = temp_a33.matrix.getRow(row)
            for index_col in range(len(cols)):
                mat_iter_A11.setValue(row + 2*self.n_mesh_vx, cols[index_col] +
                    2*self.n_mesh_vx, vals[index_col], True)

        # Block a34
        for row in range(self.n_mesh_vy):
            cols, vals = temp_a34.matrix.getRow(row)
            for index_col in range(len(cols)):
                mat_iter_A11.setValue(row + 2*self.n_mesh_vx, cols[index_col] +
                    3*self.n_mesh_vx, vals[index_col], True)

        # Block a43
        for row in range(self.n_mesh_vy):
            cols, vals = temp_a43.matrix.getRow(row)
            for index_col in range(len(cols)):
                mat_iter_A11.setValue(row + 3*self.n_mesh_vx, cols[index_col] +
                    2*self.n_mesh_vx, vals[index_col], True)

        # Block a44
        for row in range(self.n_mesh_vy):
            cols, vals = temp_a44.matrix.getRow(row)
            for index_col in range(len(cols)):
                mat_iter_A11.setValue(row + 3*self.n_mesh_vx, cols[index_col] +
                    3*self.n_mesh_vx, vals[index_col], True)

        # Block a13
        for row in range(self.n_mesh_vx):
            cols, vals = temp_a13.matrix.getRow(row)
            for index_col in range(len(cols)):
                mat_iter_A11.setValue(row, cols[index_col] + 2*self.n_mesh_vx,
                    vals[index_col], True)

        # Block a14
        for row in range(self.n_mesh_vx):
            cols, vals = temp_a14.matrix.getRow(row)
            for index_col in range(len(cols)):
                mat_iter_A11.setValue(row, cols[index_col] + 3*self.n_mesh_vx,
                    vals[index_col], True)

        # Block a23
        for row in range(self.n_mesh_vx):
            cols, vals = temp_a23.matrix.getRow(row)
            for index_col in range(len(cols)):
                mat_iter_A11.setValue(row + self.n_mesh_vx, cols[index_col] +
                    2*self.n_mesh_vx, vals[index_col], True)

        # Block a24
        for row in range(self.n_mesh_vx):
            cols, vals = temp_a24.matrix.getRow(row)
            for index_col in range(len(cols)):
                mat_iter_A11.setValue(row + n_mesh_vx, cols[index_col] +
                3*self.n_mesh_vx, vals[index_col], True)

        # Block a31
        for row in range(self.n_mesh_vy):
            cols, vals = temp_a31.matrix.getRow(row)
            for index_col in range(len(cols)):
                mat_iter_A11.setValue(row + 2*self.n_mesh_vx, cols[index_col],
                    vals[index_col], True)

        # Block a32
        for row in range(self.n_mesh_vy):
            cols, vals = temp_a32.matrix.getRow(row)
            for index_col in range(len(cols)):
                mat_iter_A11.setValue(row + 2*self.n_mesh_vx, cols[index_col] +
                    self.n_mesh_vx, vals[index_col], True)

        # Block a41
        for row in range(self.n_mesh_vy):
            cols, vals = temp_a41.matrix.getRow(row)
            for index_col in range(len(cols)):
                mat_iter_A11.setValue(row + 3*self.n_mesh_vx, cols[index_col],
                    vals[index_col], True)

        # Block a42
        for row in range(self.n_mesh_vy):
            cols, vals = temp_a42.matrix.getRow(row)
            for index_col in range(len(cols)):
                mat_iter_A11.setValue(row + 3*self.n_mesh_vx, cols[index_col] +
                    self.n_mesh_vx, vals[index_col], True)

        mat_iter_A11.assemble()

        self.mat_iter_variable = mat_iter_A11

    def make_mat_iter(self, velocity_x, velocity_y):
        """...

        """

        self.mat_iter = self.mat_iter_base
        #self.make_mat_iter_variable(velocity_x, velocity_y)
        #self.mat_iter["A11"] += self.mat_iter_variable

    def make_pressure_divgrad(self):

        self.pressure_divgrad = sd.add_operators(
            sd.mul_operators(self.pressure_grad_x, self.velocity_div_x),
            sd.mul_operators(self.pressure_grad_y, self.velocity_div_y))

        self.pressure_divgrad.bc.set(0) # we need a special right-hand side for this pressure Poisson equation, and for now we set it to zero

    def make_operators(self, tree_velocity_x, tree_velocity_y, tree_pressure,
            tree_density=None):

        print("spatial operators creation begin")
        self.make_velocity_mass(tree_velocity_x)
        self.make_velocity_inverse_mass(tree_velocity_x)

        if self.low_mach:
            self.make_one_over_density(tree_density)

        self.make_velocity_div_x(tree_velocity_x)
        self.make_velocity_div_y(tree_velocity_y)
        #print("div x")
        #self.velocity_div_x.matrix.view()
        #print("")

        #print("div y")
        #self.velocity_div_y.matrix.view()
        #print("")

        self.make_velocity_adv_x(tree_velocity_x, tree_velocity_y)
        self.make_velocity_adv_y(tree_velocity_x, tree_velocity_y)

        self.make_velocity_lap_x(tree_velocity_x)
        self.make_velocity_lap_y(tree_velocity_y)

        #print("lap x")
        #self.velocity_lap_x.matrix.view()
        #print("")

        #print("lap y")
        #self.velocity_lap_y.matrix.view()
        #print("")

        self.make_pressure_grad_x(tree_pressure)
        self.make_pressure_grad_y(tree_pressure)

        #print("grad x")
        #self.pressure_grad_x.matrix.view()
        #print("")

        #print("grad y")
        #self.pressure_grad_y.matrix.view()
        #print("")

        if self.nonlin_solver == "newton":
            self.make_jacobian_base()
            self.jacobian = self.jacobian_base
            #print("jacobian")
            #self.jacobian["A11"].view()
            #self.jacobian["A12"].view()
            #self.jacobian["A21"].view()
            #quit()
        elif self.nonlin_solver == "pic_fullexp":
            self.make_mat_iter_base()
            self.mat_iter = self.mat_iter_base
            #print("mat_iter")
            #self.mat_iter["A11"].view()
            #self.mat_iter["A12"].view()
            #self.mat_iter["A21"].view()
            #quit()

        self.make_pressure_divgrad()

        #self.approx_solver_1.make_operators(tree_velocity_x, tree_velocity_y, tree_pressure)
        #self.approx_solver_2.make_operators(tree_velocity_x, tree_velocity_y, tree_pressure)

    def make_ksp_jacobian_A11(self):
        # This structure allows to use the same ksp context to solve successive
        # linear systems with the same matrix over the successive iterations
        self.ksp_jacobian_A11 = petsc.KSP().create()
        self.ksp_jacobian_A11.setOperators(self.jacobian["A11"])
        #self.ksp_jacobian_A11.setType("preonly")
        self.ksp_jacobian_A11.getPC().setType("lu")
        self.ksp_jacobian_A11.setTolerances(rtol=1.e-12)
        #self.ksp_jacobian_A11.setFromOptions()

    def make_ksp_pc_schur_jacobian(self, preconditioner="Chorin-Temam"):
        # This structure allows to use the same ksp context to solve successive
        # linear systems with the same matrix over the successive iterations
        #self.ksp_pc_schur_jacobian = None
        self.ksp_pc_schur_jacobian1 = None
        self.ksp_pc_schur_jacobian2 = None

        # We compute a pc operator and use it as a preconditoner for the
        # iterative solver on the pressure
        if preconditioner == "Chorin-Temam":
            #matrix = self.jacobian["A21"].matMult(self.jacobian["A12"])
            #matrix.scale(self.dt)

            pc1 = sd.mul_num_operator(self.dt*self.A_coefs["a11"], sd.add_operators(
            #pc1 = sd.mul_num_operator(self.dt, sd.add_operators(
                sd.mul_operators(self.pressure_grad_x,
                    self.velocity_inverse_mass, self.velocity_div_x),
                sd.mul_operators(self.pressure_grad_y, self.velocity_inverse_mass, self.velocity_div_y)))

            pc2 = sd.mul_num_operator(self.dt*self.A_coefs["a22"], sd.add_operators(
            #pc2 = sd.mul_num_operator(self.dt, sd.add_operators(
                sd.mul_operators(self.pressure_grad_x, self.velocity_inverse_mass, self.velocity_div_x),
                sd.mul_operators(self.pressure_grad_y, self.velocity_inverse_mass, self.velocity_div_y)))

            #self.ksp_pc_schur_jacobian = petsc.KSP().create()
            #self.ksp_pc_schur_jacobian.setOperators(matrix)

            #nsp = petsc.NullSpace().create(constant=True)
            #matrix.setNullSpace(nsp)

            self.ksp_pc_schur_jacobian1 = petsc.KSP().create()
            self.ksp_pc_schur_jacobian1.setOperators(pc1.matrix)

            nsp = petsc.NullSpace().create(constant=True)
            pc1.matrix.setNullSpace(nsp)

            self.ksp_pc_schur_jacobian2 = petsc.KSP().create()
            self.ksp_pc_schur_jacobian2.setOperators(pc2.matrix)

            nsp = petsc.NullSpace().create(constant=True)
            pc2.matrix.setNullSpace(nsp)

            #self.ksp_pc_schur_jacobian.getPC().setType("lu")
            #self.ksp_pc_schur_jacobian.setTolerances(rtol=1.e-12)
            self.ksp_pc_schur_jacobian1.setTolerances(rtol=1.e-12)
            self.ksp_pc_schur_jacobian2.setTolerances(rtol=1.e-12)

            # Run the program with the option -help to see all the possible
            # linear solver options.
            self.ksp_pc_schur_jacobian1.setFromOptions()
            #self.ksp_pc_schur_jacobian2.setFromOptions()

    def make_ksp_mat_iter_A11(self):
        # This structure allows to use the same ksp context to solve successive
        # linear systems with the same matrix over the successive iterations
        self.ksp_mat_iter_A11 = petsc.KSP().create()
        self.ksp_mat_iter_A11.setOperators(self.mat_iter["A11"])
        #self.ksp_mat_iter_A11.setType("preonly")
        self.ksp_mat_iter_A11.getPC().setType("lu")
        self.ksp_mat_iter_A11.setTolerances(rtol=1.e-12)
        #self.ksp_mat_iter_A11.setFromOptions()

    def make_ksp_mat_iter_A22(self):
        # This structure allows to use the same ksp context to solve successive
        # linear systems with the same matrix over the successive iterations
        self.ksp_mat_iter_A22 = petsc.KSP().create()
        self.ksp_mat_iter_A22.setOperators(self.mat_iter["A22"])
        #self.ksp_mat_iter_A22.setType("preonly")
        self.ksp_mat_iter_A22.getPC().setType("lu")
        self.ksp_mat_iter_A22.setTolerances(rtol=1.e-12)
        #self.ksp_mat_iter_A22.setFromOptions()

    def make_ksp_pc_schur_mat_iter(self, preconditioner="Chorin-Temam"):
        # This structure allows to use the same ksp context to solve successive
        # linear systems with the same matrix over the successive iterations
        #self.ksp_pc_schur_mat_iter = None
        self.ksp_pc_schur_mat_iter1 = None
        self.ksp_pc_schur_mat_iter2 = None

        # We compute a pc operator and use it as a preconditoner for the
        # iterative solver on the pressure
        if preconditioner == "Chorin-Temam":
            #matrix = self.mat_iter["A21"].matMult(self.mat_iter["A12"])
            #matrix.scale(self.dt)

            pc1 = sd.mul_num_operator(self.dt*self.A_coefs["a11"], sd.add_operators(
            #pc1 = sd.mul_num_operator(self.dt, sd.add_operators(
                sd.mul_operators(self.pressure_grad_x,
                    self.velocity_inverse_mass, self.velocity_div_x),
                sd.mul_operators(self.pressure_grad_y, self.velocity_inverse_mass, self.velocity_div_y)))

            pc2 = sd.mul_num_operator(self.dt*self.A_coefs["a22"], sd.add_operators(
            #pc2 = sd.mul_num_operator(self.dt, sd.add_operators(
                sd.mul_operators(self.pressure_grad_x, self.velocity_inverse_mass, self.velocity_div_x),
                sd.mul_operators(self.pressure_grad_y, self.velocity_inverse_mass, self.velocity_div_y)))

            #self.ksp_pc_schur_mat_iter = petsc.KSP().create()
            #self.ksp_pc_schur_mat_iter.setOperators(matrix)

            #nsp = petsc.NullSpace().create(constant=True)
            #matrix.setNullSpace(nsp)

            self.ksp_pc_schur_mat_iter1 = petsc.KSP().create()
            self.ksp_pc_schur_mat_iter1.setOperators(pc1.matrix)

            nsp = petsc.NullSpace().create(constant=True)
            pc1.matrix.setNullSpace(nsp)

            self.ksp_pc_schur_mat_iter2 = petsc.KSP().create()
            self.ksp_pc_schur_mat_iter2.setOperators(pc2.matrix)

            nsp = petsc.NullSpace().create(constant=True)
            pc2.matrix.setNullSpace(nsp)

            #self.ksp_pc_schur_mat_iter.getPC().setType("lu")
            #self.ksp_pc_schur_mat_iter.setTolerances(rtol=1.e-12)
            self.ksp_pc_schur_mat_iter1.setTolerances(rtol=1.e-12)
            self.ksp_pc_schur_mat_iter2.setTolerances(rtol=1.e-12)

            # Run the program with the option -help to see all the possible
            # linear solver options.
            self.ksp_pc_schur_mat_iter1.setFromOptions()
            #self.ksp_pc_schur_mat_iter2.setFromOptions()

    def make_ksps(self):
        if self.nonlin_solver == "newton":
            self.make_ksp_pc_schur_jacobian()
            self.make_ksp_jacobian_A11()
            self.ksp = None # We force the default solve method to renew its own ksp
            #self.make_ksp_pc_schur_jacobian1()
            #self.make_ksp_pc_schur_jacobian2()
            #self.approx_solver_1.make_ksps()
            #self.approx_solver_2.make_ksps()
        elif self.nonlin_solver == "pic_fullexp":
            self.make_ksp_pc_schur_mat_iter()
            self.make_ksp_mat_iter_A11()
            self.make_ksp_mat_iter_A22()
            self.ksp = None # We force the default solve method to renew its own ksp
            #self.make_ksp_pc_schur_mat_iter1()
            #self.make_ksp_pc_schur_mat_iter2()

    #def jacobian_solver(self, delta_z_1, delta_z_2,
    def jacobian_solver(self, z_1, z_2,
                        global_rhs_1, global_rhs_2,
                        relax_param=5e-1, max_it=1000,
                        rtol=1e-06, atol=1e-10, preconditioner="Chorin-Temam"):
                        #rtol=1e-08, atol=1e-10, preconditioner=None):
        """implementation of the uzawa alogrithm to solve the saddle-point
        problem arising from the Newton iteration solver.

        """

        #dz_1 = delta_z_1.sc.copy()
        #dz_2 = delta_z_2.sc.copy()
        #dz_1 = delta_z_1.sc.duplicate()
        #dz_2 = delta_z_2.sc.duplicate()
        dz_1 = z_1.sc.copy()
        dz_2 = z_2.sc.copy()

        # This global rhs is used to compute the residual norm

        global_rhs = petsc.Vec().create()
        global_rhs.setSizes(2*self.n_mesh_vx + 2*self.n_mesh_vy +
                2*self.n_mesh_p)
        global_rhs.setUp()

        global_rhs[:2*self.n_mesh_vx + 2*self.n_mesh_vy] = global_rhs_1
        global_rhs[2*self.n_mesh_vx + 2*self.n_mesh_vy:] = global_rhs_2

        # Procedure to compute the residual norm
        def residual_norm_test(dz_1, dz_2):

            temp_dz_1, temp = dz_1.duplicate(), dz_1.duplicate()
            self.jacobian["A11"].mult(dz_1, temp_dz_1)
            self.jacobian["A12"].mult(dz_2, temp)
            temp_dz_1 += temp

            temp_div = dz_2.duplicate()
            self.jacobian["A21"].mult(dz_1, temp_div)

            temp = global_rhs.duplicate()

            temp[:2*self.n_mesh_vx + 2*self.n_mesh_vy] = temp_dz_1

            temp[2*self.n_mesh_vx + 2*self.n_mesh_vy:] = temp_div

            residual_vector = temp - global_rhs
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

            # computation of dz_1_next
            temp = global_rhs_1.duplicate()
            self.jacobian["A12"].mult(dz_2, temp)
            temp = global_rhs_1 - temp
            dz_1_next = dz_1.duplicate()
            self.ksp_jacobian_A11.solve(temp, dz_1_next)

            # computation of dz_2_next
            temp_div = global_rhs_2.duplicate()
            self.jacobian["A21"].mult(dz_1_next, temp_div)
            #if preconditioner is not None:
            if preconditioner == "Chorin-Temam":
                delta_dp_1, delta_dp_2, temp1, temp2 = sd.Scalar(), sd.Scalar(), sd.Scalar(), sd.Scalar()
                delta_dp_1.sc = petsc.Vec().create()
                delta_dp_1.sc.setSizes(self.n_mesh_p)
                delta_dp_1.sc.setUp()
                delta_dp_2.sc = petsc.Vec().create()
                delta_dp_2.sc.setSizes(self.n_mesh_p)
                delta_dp_2.sc.setUp()
                temp1.sc = petsc.Vec().create()
                temp1.sc.setSizes(self.n_mesh_p)
                temp1.sc.setUp()
                temp2.sc = petsc.Vec().create()
                temp2.sc.setSizes(self.n_mesh_p)
                temp2.sc.setUp()

                temp1.sc[:], temp2.sc[:] = temp_div[:self.n_mesh_p], \
                temp_div[self.n_mesh_p:]
                temp1.sc[:] = temp1.sc[:] - global_rhs_2[:self.n_mesh_p]
                temp2.sc[:] = temp2.sc[:] - global_rhs_2[self.n_mesh_p:]
                #nsp.remove(temp_x + temp_y - new_rhs_continuity)
                #self.ksp_pc_schur_jacobian.solve(temp_div - global_rhs_2, delta_dz_2)
                self.ksp_pc_schur_jacobian1.solve(temp1.sc, delta_dp_1.sc)
                self.ksp_pc_schur_jacobian2.solve(temp2.sc, delta_dp_2.sc)

                #dz_2_next = dz_2 + relax_param * delta_dz_2
                dz_2_next = dz_2.duplicate()
                dz_2_next[:self.n_mesh_p] = dz_2[:self.n_mesh_p] + relax_param * delta_dp_1.sc[:]
                dz_2_next[self.n_mesh_p:] = dz_2[self.n_mesh_p:] + relax_param * delta_dp_2.sc[:]
            else:
                dz_2_next = dz_2 - relax_param * (temp_div - global_rhs_2)

            if residual_norm_test(dz_1_next, dz_2_next): converged = True
            count += 1

            dz_1 = dz_1_next.copy()
            dz_2 = dz_2_next.copy()

        #delta_z_1.sc = dz_1_next.copy()
        #delta_z_2.sc = dz_2_next.copy()
        self.delta_z_1.sc = z_1.sc - dz_1_next
        self.delta_z_2.sc = z_2.sc - dz_2_next
        z_1.sc = dz_1_next.copy()
        z_2.sc = dz_2_next.copy()
        #quit()

    def test_jacobian_solver(self, test_func_1, test_func_2, test_func_3):

        target = petsc.Vec().create()
        target.setSizes(2*self.n_mesh_vx + 2*self.n_mesh_vy +
                2*self.n_mesh_p)
        target.setUp()

        temp = self.compute_source_term(test_func_1, 0.0)
        target[:self.n_mesh_vx] = temp.sc
        target[self.n_mesh_vx:2*self.n_mesh_vx] = temp.sc
        temp = self.compute_source_term(test_func_2, 0.0)
        target[2*self.n_mesh_vx:2*self.n_mesh_vx + self.n_mesh_vy] = temp.sc
        target[2*self.n_mesh_vx + self.n_mesh_vy:2*self.n_mesh_vx + 2*self.n_mesh_vy] = temp.sc
        temp = self.compute_source_term(test_func_3, 0.0)
        target[2*self.n_mesh_vx + 2*self.n_mesh_vy:2*self.n_mesh_vx + 2*self.n_mesh_vy + self.n_mesh_p] = temp.sc
        target[2*self.n_mesh_vx + 2*self.n_mesh_vy + self.n_mesh_p:] = temp.sc
        #target.view()

        targetdt = petsc.Vec().create()
        targetdt.setSizes(6*self.n_mesh)
        targetdt.setUp()

        temp = self.compute_source_term(test_func_1, 0.1)
        targetdt[:self.n_mesh_vx] = temp.sc
        targetdt[self.n_mesh_vx:2*self.n_mesh_vx] = temp.sc
        temp = self.compute_source_term(test_func_2, 0.1)
        targetdt[2*self.n_mesh_vx:2*self.n_mesh_vx + self.n_mesh_vy] = temp.sc
        targetdt[2*self.n_mesh_vx + self.n_mesh_vy:2*self.n_mesh_vx + 2*self.n_mesh_vy] = temp.sc
        temp = self.compute_source_term(test_func_3, 0.1)
        targetdt[2*self.n_mesh_vx + 2*self.n_mesh_vy:2*self.n_mesh_vx + 2*self.n_mesh_vy + self.n_mesh_p] = temp.sc
        targetdt[2*self.n_mesh_vx + 2*self.n_mesh_vy + self.n_mesh_p:] = temp.sc
        targetdt.view()

        target_1 = petsc.Vec().create()
        target_1.setSizes(2*self.n_mesh_vx + 2*self.n_mesh_vy)
        target_1.setUp()
        target_1[:] = targetdt[:2*self.n_mesh_vx + 2*self.n_mesh_vy] - \
        target[:2*self.n_mesh_vx + 2*self.n_mesh_vy]

        target_2 = petsc.Vec().create()
        target_2.setSizes(2*self.n_mesh_p)
        target_2.setUp()
        target_2[:] = targetdt[2*self.n_mesh_vx + 2*self.n_mesh_vy:] - \
        target[2*self.n_mesh_vx + 2*self.n_mesh_vy:]

        rhs_1, temp = target_1.duplicate(), target_1.duplicate()
        self.jacobian["A11"].mult(target_1, rhs_1)
        self.jacobian["A12"].mult(target_2, temp)
        rhs_1 += temp

        rhs_2 = target_2.duplicate()
        self.jacobian["A21"].mult(target_1, rhs_2)

        sol_1 = sd.Scalar()
        sol_1.sc = petsc.Vec().create()
        sol_1.sc.setSizes(2*self.n_mesh_vx + 2*self.n_mesh_vy)
        sol_1.sc.setUp()

        sol_2 = sd.Scalar()
        sol_2.sc = petsc.Vec().create()
        sol_2.sc.setSizes(2*self.n_mesh_p)
        sol_2.sc.setUp()

        self.make_ksp_jacobian_A11()
        self.jacobian_solver(sol_1, sol_2, rhs_1, rhs_2)

        sol = target.duplicate()
        sol[:2*self.n_mesh_vx + 2*self.n_mesh_vy] = sol_1.sc
        sol[2*self.n_mesh_vx + 2*self.n_mesh_vy:] = sol_2.sc
        sol += target
        sol.view()

        residual_vector = targetdt - sol
        residual_norm = residual_vector.norm()
        print("residual_norm: " + repr(residual_norm))

    def Newton_init_val(self, v_x, v_y, p):
            #source_term_vx=None, source_term_vy=None):
        """...

        """

        if self.uniform:
            self.approx_vx_1 = sd.Scalar()
            self.approx_vx_1.sc = v_x.sc.copy()
            self.approx_vx_1.tag = v_x.tag
            self.approx_vx_2 = sd.Scalar()
            self.approx_vx_2.sc = v_x.sc.copy()
            self.approx_vx_2.tag = v_x.tag
            self.approx_vy_1 = sd.Scalar()
            self.approx_vy_1.sc = v_y.sc.copy()
            self.approx_vy_1.tag = v_y.tag
            self.approx_vy_2 = sd.Scalar()
            self.approx_vy_2.sc = v_y.sc.copy()
            self.approx_vy_2.tag = v_y.tag
            self.approx_p_1 = sd.Scalar()
            self.approx_p_1.sc = p.sc.copy()
            self.approx_p_1.tag = p.tag
            self.approx_p_2 = sd.Scalar()
            self.approx_p_2.sc = p.sc.copy()
            self.approx_p_2.tag = p.tag

        else:
            self.approx_vx_1 = sd.Scalar(v_x)
            self.approx_vx_2 = sd.Scalar(v_x)
            self.approx_vy_1 = sd.Scalar(v_y)
            self.approx_vy_2 = sd.Scalar(v_y)
            self.approx_p_1 = sd.Scalar(p)
            self.approx_p_2 = sd.Scalar(p)

        #self.approx_vx_1.sc = velocity_x.sc.duplicate()
        #self.approx_vx_2.sc = velocity_x.sc.duplicate()
        #self.approx_vy_1.sc = velocity_y.sc.duplicate()
        #self.approx_vy_2.sc = velocity_y.sc.duplicate()
        #self.approx_p_1.sc = pressure.sc.duplicate()
        #self.approx_p_2.sc = pressure.sc.duplicate()
        #self.approx_solver_1.advance_given_scalar(self.approx_vx_1, self.approx_vy_1,
        #        self.approx_p_1, source_term_vx, source_term_vy)
        #self.approx_solver_2.advance_given_scalar(self.approx_vx_2, self.approx_vy_2,
        #        self.approx_p_2, source_term_vx, source_term_vy)

        self.z_1.sc[:self.n_mesh_vx] = self.approx_vx_1.sc
        self.z_1.sc[self.n_mesh_vx:2*self.n_mesh_vx] = self.approx_vx_2.sc
        self.z_1.sc[2*self.n_mesh_vx:2*self.n_mesh_vx + self.n_mesh_vy] = self.approx_vy_1.sc
        self.z_1.sc[2*self.n_mesh_vx + self.n_mesh_vy:] = self.approx_vy_2.sc

        self.z_2.sc[:self.n_mesh_p] = self.approx_p_1.sc
        self.z_2.sc[self.n_mesh_p:] = self.approx_p_2.sc

    def Newton_solver(self, velocity_x, velocity_y, pressure,
            st_rhs_11=None, st_rhs_12=None, st_rhs_21=None, st_rhs_22=None,
            st_rhs_31=None, st_rhs_32=None,
            t_ini=0., max_it=1, rtol=1e-06):
        """Inspired by the implementation of Implicit Runge-Kutta methods of
        Hairer & Wanner (1996), specially the stopping criterion.

        """

        # This global dz is used to compute the residual norm

        global_dz = petsc.Vec().create()
        global_dz.setSizes(2*self.n_mesh_vx + 2*self.n_mesh_vy +
                2*self.n_mesh_p)
        global_dz.setUp()

        norm_dz = 0.

        # Procedure to test the convergence
        def residual_norm_test(norm_dz):

            print("eta_k*norm_dz: " + repr(norm_dz))
            #print("eta_k*norm_dz: " + repr(self.eta_k*norm_dz))
            print("rtol: " + repr(rtol))

            if abs(self.eta_k*norm_dz) <= rtol:
                return True
            else:
                return False

        x1, x2 = sd.Scalar(), sd.Scalar()
        y1, y2 = sd.Scalar(), sd.Scalar()
        z1, z2 = sd.Scalar(), sd.Scalar()
        x1.sc, x2.sc = velocity_x.sc.duplicate(), velocity_x.sc.duplicate()
        y1.sc, y2.sc = velocity_y.sc.duplicate(), velocity_y.sc.duplicate()
        z1.sc, z2.sc = pressure.sc.duplicate(), pressure.sc.duplicate()
        x1.sc[:], x2.sc[:] = self.z_1.sc[:self.n_mesh_vx], \
            self.z_1.sc[self.n_mesh_vx:2*self.n_mesh_vx]
        y1.sc[:], y2.sc[:] = self.z_1.sc[2*self.n_mesh_vx:2*self.n_mesh_vx +
            self.n_mesh_vy], self.z_1.sc[2*self.n_mesh_vx + self.n_mesh_vy:]
        z1.sc[:], z2.sc[:] = self.z_2.sc[:self.n_mesh_p], \
            self.z_2.sc[self.n_mesh_p:]

        converged = False
        count = 0

        while ((not converged) and (count < max_it)):
        #while (count < max_it):

            # First iteration
            if count == 0:
                global_rhs_1, global_rhs_2 = self.rhs_newton_iteration(velocity_x,
                        velocity_y, x1, y1, z1, x2, y2, z2, st_rhs_11,
                        st_rhs_12, st_rhs_21, st_rhs_22, st_rhs_31, st_rhs_32)

                #self.jacobian_solver(self.delta_z_1, self.delta_z_2, global_rhs_1,
                #        global_rhs_2)
                self.jacobian_solver(self.z_1, self.z_2, global_rhs_1,
                        global_rhs_2)

                global_dz[:2*self.n_mesh_vx + 2*self.n_mesh_vy] = self.delta_z_1.sc
                global_dz[2*self.n_mesh_vx + 2*self.n_mesh_vy:] = self.delta_z_2.sc

                #norm_dz = global_dz.norm()
                norm_dz = self.delta_z_1.sc.norm()

                if residual_norm_test(norm_dz): converged = True
                count += 1

                #self.delta_z_1.sc.set(1.e-40)
                #self.delta_z_2.sc.set(1.e-40)

                #w = 1.0
                #self.z_1 = sd.add_scalars(self.z_1,
                #        sd.mul_num_scalar(w, self.delta_z_1))
                #self.z_2 = sd.add_scalars(self.z_2,
                #        sd.mul_num_scalar(w, self.delta_z_2))

            # Others iterations
            else:
                x1.sc[:], x2.sc[:] = self.z_1.sc[:self.n_mesh_vx], \
                    self.z_1.sc[self.n_mesh_vx:2*self.n_mesh_vx]
                y1.sc[:], y2.sc[:] = self.z_1.sc[2*self.n_mesh_vx:2*self.n_mesh_vx +
                    self.n_mesh_vy], self.z_1.sc[2*self.n_mesh_vx + self.n_mesh_vy:]
                z1.sc[:], z2.sc[:] = self.z_2.sc[:self.n_mesh_p], \
                    self.z_2.sc[self.n_mesh_p:]

                global_rhs_1, global_rhs_2 = self.rhs_newton_iteration(velocity_x,
                        velocity_y, x1, y1, z1, x2, y2, z2,  st_rhs_11,
                        st_rhs_12, st_rhs_21, st_rhs_22, st_rhs_31, st_rhs_32)

                #self.make_jacobian(velocity_x, velocity_y)
                #self.make_ksp_jacobian_A11()

                #self.jacobian_solver(self.delta_z_1, self.delta_z_2, global_rhs_1,
                #        global_rhs_2)
                self.jacobian_solver(self.z_1, self.z_2, global_rhs_1,
                        global_rhs_2)

                global_dz[:2*self.n_mesh_vx + 2*self.n_mesh_vy] = self.delta_z_1.sc
                global_dz[2*self.n_mesh_vx + 2*self.n_mesh_vy:] = self.delta_z_2.sc

                #theta_k = global_dz.norm() / norm_dz
                theta_k = self.delta_z_1.sc.norm() / norm_dz
                if theta_k > 1:
                    print("the iteration 'diverges'")
                    #quit()
                self.eta_k = theta_k / (1 - theta_k)

                #norm_dz = global_dz.norm()
                norm_dz = self.delta_z_1.sc.norm()

                if residual_norm_test(norm_dz): converged = True
                count += 1

                #self.z_1 = sd.add_scalars(self.z_1, self.delta_z_1)
                #self.z_2 = sd.add_scalars(self.z_2, self.delta_z_2)

    def mat_iter_solver(self, g_1, g_2, g_3,
                        global_rhs_1, global_rhs_2, global_rhs_3,
                        relax_param=1e-0, max_it=500,
                        rtol=1e-07, atol=1e-10, preconditioner="Chorin-Temam"):
                        #rtol=1e-08, atol=1e-10, preconditioner=None):
        """implementation of the uzawa alogrithm to solve the saddle-point
        problem arising from the Picard iteration solver.
        """

        dg_1 = g_1.sc.copy()
        dg_2 = g_2.sc.copy()
        dg_3 = g_3.sc.copy()

        # This global rhs is used to compute the residual norm

        global_rhs = petsc.Vec().create()
        global_rhs.setSizes(2*self.n_mesh_vx + 2*self.n_mesh_vy +
                2*self.n_mesh_p)
        global_rhs.setUp()

        global_rhs[:2*self.n_mesh_vx] = global_rhs_1
        global_rhs[2*self.n_mesh_vx:2*self.n_mesh_vx + 2*self.n_mesh_vy] = global_rhs_2
        global_rhs[2*self.n_mesh_vx + 2*self.n_mesh_vy:] = global_rhs_3

        # Procedure to compute the residual norm
        def residual_norm_test(dg_1, dg_2, dg_3):

            temp_dg_1, temp = dg_1.duplicate(), dg_1.duplicate()
            self.mat_iter["A11"].mult(dg_1, temp_dg_1)
            self.mat_iter["A13"].mult(dg_3, temp)
            temp_dg_1 += temp
            #temp_dg_1.view()

            temp_dg_2, temp = dg_2.duplicate(), dg_2.duplicate()
            self.mat_iter["A22"].mult(dg_2, temp_dg_2)
            self.mat_iter["A23"].mult(dg_3, temp)
            temp_dg_2 += temp

            temp_div_x = dg_3.duplicate()
            self.mat_iter["A31"].mult(dg_1, temp_div_x)

            temp_div_y = dg_3.duplicate()
            self.mat_iter["A32"].mult(dg_2, temp_div_y)

            temp = global_rhs.duplicate()

            temp[:2*self.n_mesh_vx] = temp_dg_1
            temp[2*self.n_mesh_vx:2*self.n_mesh_vx + 2*self.n_mesh_vy] = temp_dg_2
            temp[2*self.n_mesh_vx + 2*self.n_mesh_vy:] = temp_div_x + temp_div_y

            residual_vector = temp - global_rhs
            #temp.view()
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

            # computation of dg_1_next
            temp = global_rhs_1.duplicate()
            self.mat_iter["A13"].mult(dg_3, temp)
            temp = global_rhs_1 - temp
            dg_1_next = dg_1.duplicate()
            self.ksp_mat_iter_A11.solve(temp, dg_1_next)

            # computation of dg_2_next
            temp = global_rhs_2.duplicate()
            self.mat_iter["A23"].mult(dg_3, temp)
            temp = global_rhs_2 - temp
            dg_2_next = dg_2.duplicate()
            self.ksp_mat_iter_A22.solve(temp, dg_2_next)

            # computation of dg_3_next
            temp_div_x = global_rhs_3.duplicate()
            self.mat_iter["A31"].mult(dg_1_next, temp_div_x)
            temp_div_y = global_rhs_3.duplicate()
            self.mat_iter["A32"].mult(dg_2_next, temp_div_y)
            temp_div = temp_div_x + temp_div_y
            #if preconditioner is not None:
            if preconditioner == "Chorin-Temam":
                delta_dp_1, delta_dp_2, temp1, temp2 = sd.Scalar(), sd.Scalar(), sd.Scalar(), sd.Scalar()
                delta_dp_1.sc = petsc.Vec().create()
                delta_dp_1.sc.setSizes(self.n_mesh_p)
                delta_dp_1.sc.setUp()
                delta_dp_2.sc = petsc.Vec().create()
                delta_dp_2.sc.setSizes(self.n_mesh_p)
                delta_dp_2.sc.setUp()
                temp1.sc = petsc.Vec().create()
                temp1.sc.setSizes(self.n_mesh_p)
                temp1.sc.setUp()
                temp2.sc = petsc.Vec().create()
                temp2.sc.setSizes(self.n_mesh_p)
                temp2.sc.setUp()

                temp1.sc[:], temp2.sc[:] = temp_div[:self.n_mesh_p], \
                temp_div[self.n_mesh_p:]
                temp1.sc[:] = temp1.sc[:] - global_rhs_3[:self.n_mesh_p]
                temp2.sc[:] = temp2.sc[:] - global_rhs_3[self.n_mesh_p:]
                #nsp.remove(temp_x + temp_y - new_rhs_continuity)
                #self.ksp_pc_schur_mat_iter.solve(temp_div - global_rhs_2, delta_dz_2)
                self.ksp_pc_schur_mat_iter1.solve(temp1.sc, delta_dp_1.sc)
                self.ksp_pc_schur_mat_iter2.solve(temp2.sc, delta_dp_2.sc)

                #dz_2_next = dz_2 + relax_param * delta_dz_2
                dg_3_next = dg_3.duplicate()
                dg_3_next[:self.n_mesh_p] = dg_3[:self.n_mesh_p] + relax_param * delta_dp_1.sc[:]
                dg_3_next[self.n_mesh_p:] = dg_3[self.n_mesh_p:] + relax_param * delta_dp_2.sc[:]
            else:
                dg_3_next = dg_3 - relax_param * (temp_div - global_rhs_3)

            if residual_norm_test(dg_1_next, dg_2_next, dg_3_next): converged = True
            count += 1

            dg_1 = dg_1_next.copy()
            dg_2 = dg_2_next.copy()
            dg_3 = dg_3_next.copy()

        g_1.sc = dg_1_next.copy()
        g_2.sc = dg_2_next.copy()
        g_3.sc = dg_3_next.copy()

        #g_1.sc.view()
        #g_2.sc.view()
        #g_3.sc.view()
        #quit()

    def Nonlin_init_val(self, v_x, v_y, p):
            #source_term_vx=None, source_term_vy=None):
        """...

        """

        if self.uniform:
            self.approx_vx_1 = sd.Scalar()
            self.approx_vx_1.sc = v_x.sc.copy()
            self.approx_vx_1.tag = v_x.tag
            self.approx_vx_2 = sd.Scalar()
            self.approx_vx_2.sc = v_x.sc.copy()
            self.approx_vx_2.tag = v_x.tag
            self.approx_vy_1 = sd.Scalar()
            self.approx_vy_1.sc = v_y.sc.copy()
            self.approx_vy_1.tag = v_y.tag
            self.approx_vy_2 = sd.Scalar()
            self.approx_vy_2.sc = v_y.sc.copy()
            self.approx_vy_2.tag = v_y.tag
            self.approx_p_1 = sd.Scalar()
            self.approx_p_1.sc = p.sc.copy()
            self.approx_p_1.tag = p.tag
            self.approx_p_2 = sd.Scalar()
            self.approx_p_2.sc = p.sc.copy()
            self.approx_p_2.tag = p.tag

        else:
            self.approx_vx_1 = sd.Scalar(v_x)
            self.approx_vx_2 = sd.Scalar(v_x)
            self.approx_vy_1 = sd.Scalar(v_y)
            self.approx_vy_2 = sd.Scalar(v_y)
            self.approx_p_1 = sd.Scalar(p)
            self.approx_p_2 = sd.Scalar(p)

        #self.approx_vx_1.sc = velocity_x.sc.duplicate()
        #self.approx_vx_2.sc = velocity_x.sc.duplicate()
        #self.approx_vy_1.sc = velocity_y.sc.duplicate()
        #self.approx_vy_2.sc = velocity_y.sc.duplicate()
        #self.approx_p_1.sc = pressure.sc.duplicate()
        #self.approx_p_2.sc = pressure.sc.duplicate()
        #self.approx_solver_1.advance_given_scalar(self.approx_vx_1, self.approx_vy_1,
        #        self.approx_p_1, source_term_vx, source_term_vy)
        #self.approx_solver_2.advance_given_scalar(self.approx_vx_2, self.approx_vy_2,
        #        self.approx_p_2, source_term_vx, source_term_vy)

        self.g_1.sc[:self.n_mesh_vx] = self.approx_vx_1.sc
        self.g_1.sc[self.n_mesh_vx:] = self.approx_vx_2.sc
        self.g_2.sc[:self.n_mesh_vy] = self.approx_vy_1.sc
        self.g_2.sc[self.n_mesh_vy:] = self.approx_vy_2.sc

        self.g_3.sc[:self.n_mesh_p] = self.approx_p_1.sc
        self.g_3.sc[self.n_mesh_p:] = self.approx_p_2.sc

    def Nonlin_solver(self, velocity_x, velocity_y, pressure,
            st_rhs_11=None, st_rhs_12=None, st_rhs_21=None, st_rhs_22=None,
            st_rhs_31=None, st_rhs_32=None,
            t_ini=0., max_it=10, rtol=1e-06):
        """Picard iteration to solve the nonlinear system from the RK method.
        """

        # This rhs_nonlin is used to compute the residual norm
        rhs_nonlin = self.nonlin_rhs_pic_fullexp(velocity_x, velocity_y,
                st_rhs_11, st_rhs_12, st_rhs_21, st_rhs_22, st_rhs_31, st_rhs_32)

        # Procedure to test the convergence
        def residual_norm_test(x1, y1, z1, x2, y2, z2):

            residual_vector = self.nonlin_op_apply(x1, y1, z1, x2, y2, z2) - \
                rhs_nonlin
            #residual_vector.view()
            temp = petsc.Vec().create()
            temp.setSizes(2*self.n_mesh_vx + 2*self.n_mesh_vy)
            temp.setUp()

            #temp[:] = residual_vector[:2*self.n_mesh_vx + 2*self.n_mesh_vy]
            residual_norm = residual_vector.norm()
            #residual_norm = temp.norm()
            print("residual norm nonlinear solver: " + repr(residual_norm))
            print("rtol*rhs_nonlinear_norm: " + repr(rtol * rhs_nonlin.norm()))

            if residual_norm <= rtol * rhs_nonlin.norm():
                return True
            else:
                return False

        x1, x2 = sd.Scalar(), sd.Scalar()
        y1, y2 = sd.Scalar(), sd.Scalar()
        z1, z2 = sd.Scalar(), sd.Scalar()
        x1.sc, x2.sc = velocity_x.sc.duplicate(), velocity_x.sc.duplicate()
        y1.sc, y2.sc = velocity_y.sc.duplicate(), velocity_y.sc.duplicate()
        z1.sc, z2.sc = pressure.sc.duplicate(), pressure.sc.duplicate()
        x1.sc[:], x2.sc[:] = self.g_1.sc[:self.n_mesh_vx], \
            self.g_1.sc[self.n_mesh_vx:2*self.n_mesh_vx]
        y1.sc[:], y2.sc[:] = self.g_2.sc[:self.n_mesh_vy], \
            self.g_1.sc[self.n_mesh_vy:2*self.n_mesh_vy]
        z1.sc[:], z2.sc[:] = self.g_3.sc[:self.n_mesh_p], \
            self.g_3.sc[self.n_mesh_p:]

        converged = False
        count = 0

        while ((not converged) and (count < max_it)):
        #while (count < max_it):

            iter_rhs_1, iter_rhs_2, iter_rhs_3 = \
                self.rhs_pic_fullexp_iteration(velocity_x,
                    velocity_y, x1, y1, x2, y2, st_rhs_11, st_rhs_12,
                    st_rhs_21, st_rhs_22, st_rhs_31, st_rhs_32)

            #iter_rhs_3.view()
            self.mat_iter_solver(self.g_1, self.g_2, self.g_3, iter_rhs_1,
                    iter_rhs_2, iter_rhs_3)

            x1.sc[:], x2.sc[:] = self.g_1.sc[:self.n_mesh_vx], \
                self.g_1.sc[self.n_mesh_vx:]
            y1.sc[:], y2.sc[:] = self.g_2.sc[:self.n_mesh_vy], \
                self.g_2.sc[self.n_mesh_vy:]
            z1.sc[:], z2.sc[:] = self.g_3.sc[:self.n_mesh_p], \
                self.g_3.sc[self.n_mesh_p:]

            if residual_norm_test(x1, y1, z1, x2, y2, z2): converged = True
            count += 1

        #self.nonlin_op_apply(x1, y1, z1, x2, y2, z2).view()
        #quit()
