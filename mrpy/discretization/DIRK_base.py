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


class DIRKScheme(BaseScheme):
    """Base scheme for the implementation of Diagonally Implicit Runge-Kutta
    methods for the Stokes equation in 2D."""

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

    #def __init__(self, dimension=cfg.dimension, tree_velocity_x=None, tree_velocity_y=None, tree_velocity_z=None, tree_pressure=None, tree_vorticity=None):

    #    if tree_vorticity is not None:
    #        BaseScheme.__init__(self, tree_velocity_x=tree_velocity_x, tree_velocity_y=tree_velocity_y, tree_pressure=tree_pressure, tree_vorticity=tree_vorticity)
    #    else:
    #        BaseScheme.__init__(self, tree_velocity_x=tree_velocity_x, tree_velocity_y=tree_velocity_y, tree_pressure=tree_pressure)

        self.operators_dict = {}
        self.ksps_dict = {}

    def make_hlm_x(self, stage):
        """returns the helmholtz_x operator needed at the stage 'stage' of the RK
        method. stage must be a string"""

        key_stage = "a" + stage + stage

        return sd.add_operators(
        sd.mul_num_operator(1/self.dt, self.velocity_mass),
        sd.mul_num_operator(-cfg.nu*self.A_coefs[key_stage], self.velocity_lap_x))

    def make_hlm_y(self, stage):
        """returns the helmholtz_y operator needed at the stage 'stage' of the RK
        method. stage must be a string"""

        key_stage = "a" + stage + stage

        return sd.add_operators(
        sd.mul_num_operator(1/self.dt, self.velocity_mass),
        sd.mul_num_operator(-cfg.nu*self.A_coefs[key_stage], self.velocity_lap_y))

    def make_grad_x(self, stage):
        """returns the gradient_x operator needed at the stage 'stage' of the RK
        method. stage must be a string"""

        key_stage = "a" + stage + stage

        return sd.mul_num_operator(self.A_coefs[key_stage], self.pressure_grad_x)

    def make_grad_y(self, stage):
        """returns the gradient_y operator needed at the stage 'stage' of the RK
        method. stage must be a string"""

        key_stage = "a" + stage + stage

        return sd.mul_num_operator(self.A_coefs[key_stage], self.pressure_grad_y)

    def make_ksp_hlm_x(self, stage):
        """returns the ksp helmholtz_x operator needed at the stage 'stage' of the RK
        method. stage must be a string"""

        temp = petsc.KSP().create()
        temp.setOperators(self.operators_dict[stage]["hlm_x"].matrix)
        temp.getPC().setType("lu")
        temp.setTolerances(rtol=1.e-12)

        return temp

    def make_ksp_hlm_y(self, stage):
        """returns the ksp helmholtz_y operator needed at the stage 'stage' of the RK
        method. stage must be a string"""

        temp = petsc.KSP().create()
        temp.setOperators(self.operators_dict[stage]["hlm_y"].matrix)
        temp.getPC().setType("lu")
        temp.setTolerances(rtol=1.e-12)

        #temp.setFromOptions()

        return temp

    def make_ksp_pc(self, stage, preconditioner="Chorin-Temam"):
        """returns the ksp object needed for the preconditioner used for the
        pressure in the Uzawa alogrithm at the stage 'stage' of the RK
        method. stage must be a string"""

        if preconditioner == "Chorin-Temam":
            temp = sd.mul_num_operator(self.dt, sd.add_operators(
                sd.mul_operators(self.operators_dict[stage]["grad_x"],
                    self.velocity_inverse_mass, self.velocity_div_x),
                sd.mul_operators(self.operators_dict[stage]["grad_y"],
                    self.velocity_inverse_mass, self.velocity_div_y)))

            foo = petsc.KSP().create()
            foo.setOperators(temp.matrix)
            foo.setTolerances(max_it=500)

            #foo.setFromOptions()

            return foo


    #def make_velocity_helmholtz_y(self, tree_velocity_y):
    #    self.velocity_helmholtz_y = sd.add_operators(
    #    sd.mul_num_operator(1/self.dt, self.velocity_mass),
    #    sd.mul_num_operator(-cfg.nu, self.velocity_lap_y))

    #def make_velocity_helmholtz_z(self, tree_velocity_z):
    #    self.velocity_helmholtz_z = sd.add_operators(
    #    sd.mul_num_operator(1/self.dt, self.velocity_mass),
    #    sd.mul_num_operator(-cfg.nu, self.velocity_lap_z))

    def make_rhs_dae_x(self, velocity_x, pressure, source_term=None,
            low_mach_visc=None, one_over_density=None):

        if source_term is None:
            if self.low_mach:
                return sd.add_scalars(
                    sd.mul_scalars(one_over_density, low_mach_visc),
                    sd.mul_num_scalar(cfg.nu, self.velocity_lap_x.apply(velocity_x)),
                    sd.mul_num_scalar(-1., self.pressure_grad_x.apply(pressure)))
            else:
                return sd.add_scalars(
                    sd.mul_num_scalar(cfg.nu, self.velocity_lap_x.apply(velocity_x)),
                    sd.mul_num_scalar(-1., self.pressure_grad_x.apply(pressure)))

        else:
            if self.low_mach:
                return sd.add_scalars(
                    sd.mul_scalars(one_over_density, low_mach_visc),
                    self.velocity_mass.apply(source_term),
                    sd.mul_num_scalar(cfg.nu, self.velocity_lap_x.apply(velocity_x)),
                    sd.mul_num_scalar(-1., self.pressure_grad_x.apply(pressure)))
            else:
                return sd.add_scalars(
                    self.velocity_mass.apply(source_term),
                    sd.mul_num_scalar(cfg.nu, self.velocity_lap_x.apply(velocity_x)),
                    sd.mul_num_scalar(-1., self.pressure_grad_x.apply(pressure)))

    def make_rhs_dae_y(self, velocity_y, pressure, source_term=None,
            low_mach_visc=None, one_over_density=None):

        if source_term is None:
            if self.low_mach:
                return sd.add_scalars(
                    sd.mul_scalars(one_over_density, low_mach_visc),
                    sd.mul_num_scalar(cfg.nu, self.velocity_lap_y.apply(velocity_y)),
                    sd.mul_num_scalar(-1., self.pressure_grad_y.apply(pressure)))
            else:
                return sd.add_scalars(
                    sd.mul_num_scalar(cfg.nu, self.velocity_lap_y.apply(velocity_y)),
                    sd.mul_num_scalar(-1., self.pressure_grad_y.apply(pressure)))

        else:
            if self.low_mach:
                return sd.add_scalars(
                    sd.mul_scalars(one_over_density, low_mach_visc),
                    sd.mul_num_scalar(cfg.nu, self.velocity_lap_y.apply(velocity_y)),
                    self.velocity_mass.apply(source_term),
                    sd.mul_num_scalar(-1., self.pressure_grad_y.apply(pressure)))
            else:
                return sd.add_scalars(
                    sd.mul_num_scalar(cfg.nu, self.velocity_lap_y.apply(velocity_y)),
                    self.velocity_mass.apply(source_term),
                    sd.mul_num_scalar(-1., self.pressure_grad_y.apply(pressure)))

    def make_rhs_dae_alg(self, source_term): #probably useless, but we put it here for now

        return source_term

    def uzawa_solver_internal(self, stage, velocity_x=None, velocity_y=None,
            pressure=None, rhs_momentum_x=None, rhs_momentum_y=None,
            rhs_continuity=None, relax_param=1e-0, max_it=500, rtol=1e-07,
            atol=1e-10, preconditioner="Chorin-Temam", right_rhs=False):
        """implementation of the uzawa alogrithm to solve the saddle-point
        problem arising from the spatial discretization of the NS equation.

        It is used for to solve the saddle-poijnt problem arising at each
        internal stage of a DIRK method applied to the incompressible NS
        equations. The divergence, helmhotz and gradient operators depend on
        each stage, so the variable "stage" is used to determine them among the
        dictionary of operators of the solver used.
        """

        hlm_x = self.operators_dict[stage]["hlm_x"]
        hlm_y = self.operators_dict[stage]["hlm_y"]
        grad_x = self.operators_dict[stage]["grad_x"]
        grad_y = self.operators_dict[stage]["grad_y"]
        div_x = self.operators_dict[stage]["div_x"]
        div_y = self.operators_dict[stage]["div_y"]
        #pc = self.operators_dict[stage]["pc"]
        ksp_hlm_x = self.ksps_dict[stage]["ksp_hlm_x"]
        ksp_hlm_y = self.ksps_dict[stage]["ksp_hlm_y"]
        ksp_pc = self.ksps_dict[stage]["ksp_pc"]
        vx = velocity_x.sc.copy()
        vy = velocity_y.sc.copy()
        p = pressure.sc.copy()

        if right_rhs is False:
            new_rhs_momentum_x = rhs_momentum_x.sc - hlm_x.bc
            new_rhs_momentum_y = rhs_momentum_y.sc - hlm_y.bc
            if rhs_continuity is None:
                new_rhs_continuity = -div_x.bc - div_y.bc
            else:
                new_rhs_continuity = rhs_continuity.sc - div_x.bc - div_y.bc
        elif right_rhs is True:
            new_rhs_momentum_x = rhs_momentum_x.sc
            new_rhs_momentum_y = rhs_momentum_y.sc
            if rhs_continuity is None:
                new_rhs_continuity = -div_x.bc - div_y.bc
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
        #for i in range(n_vx):
        #    global_rhs[i] = new_rhs_momentum_x[i]

        #for i in range(n_vy):
        #    global_rhs[n_vx + i] = new_rhs_momentum_y[i]

        #for i in range(n_div):
        #    global_rhs[n_vx + n_vy + i] = new_rhs_continuity[i]

        global_rhs[:n_vx] = new_rhs_momentum_x
        global_rhs[n_vx:n_vx + n_vy] = new_rhs_momentum_y
        global_rhs[n_vx + n_vy:] = new_rhs_continuity

        # Procedure to compute the residual norm
        def residual_norm_test(vx, vy, p):

            temp_vx, temp = vx.duplicate(), vx.duplicate()
            hlm_x.matrix.mult(vx, temp_vx)
            grad_x.matrix.mult(p, temp)
            temp_vx += temp
            n_vx = temp_vx.getSizes()

            temp_vy, temp = vy.duplicate(), vy.duplicate()
            hlm_y.matrix.mult(vy, temp_vy)
            grad_y.matrix.mult(p, temp)
            temp_vy += temp
            n_vy = temp_vy.getSizes()

            temp_div_x = vx.duplicate()
            temp_div_y = vy.duplicate()
            div_x.matrix.mult(vx, temp_div_x)
            div_y.matrix.mult(vy, temp_div_y)
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
            grad_x.matrix.mult(p, temp)
            temp = new_rhs_momentum_x - temp
            vx_next = vx.duplicate()
            ksp_hlm_x.solve(temp, vx_next)

            # computation of vy_next
            temp = vy.duplicate()
            self.grad_y.matrix.mult(p, temp)
            temp = new_rhs_momentum_y - temp
            vy_next = vy.copy()
            ksp_hlm_y.solve(temp, vy_next)

            # computation of p_next
            temp_x = vx.duplicate()
            temp_y = vy.duplicate()
            div_x.matrix.mult(vx_next, temp_x)
            div_y.matrix.mult(vy_next, temp_y)
            if preconditioner is not None:
                delta_p = p.duplicate()
                #nsp.remove(temp_x + temp_y - new_rhs_continuity)
                ksp_pc.solve(temp_x + temp_y - new_rhs_continuity, delta_p)

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
        #vx = sd.Scalar()
        #vy = sd.Scalar()
        #p = sd.Scalar()
        #vx.sc = vx_next.copy()
        #vy.sc = vy_next.copy()
        #p.sc = p_next.copy()


        #return vx, vy, p

