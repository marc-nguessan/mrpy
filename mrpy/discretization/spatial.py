from __future__ import print_function, division

"""This module provides a collection of constructors and selectors that fulfill
a valid representation of the data objects of our DSL for the computation of
convection-reaction-diffusion coupled with low-mach / incompressible flows
numerical simulation in a rectangular domain, as well as the primitive
procedures that help to combine these data objects. It contains all the
procedures needed to perform spatial transformations on the discrete variables.

The primitive data objects are:
    -the scalars, that should be able to provide an array with the discrete
    values over the entire domain of the variable they represent, as well as a
    tag name to identify them (for example the pressure P is a scalar so that
    P.sc gives a vector of the discrete values on the domain of computation, and
    P.tag gives the specific tag of the pressure for the computation

    -the operators, that should be able to provide a matrix-like object that
    is the linear combinations of the discrete spatial operator that it
    represents and a boundary condition vector that results form the computation
    of the operator at the boundary of the domain (for example a gradient
    operator G over the direction x should have a matrix attribute G_x and a
    boundary condition bc_x so that G_xP.sc + bc_x is the discrete gradient of
    teh pressure P over the entire domain. The operators should also have a
    procedure "apply", so that G.apply(P).sc = G_xP.sc + bc_x, i.e. it applies
    the discrete spatial operator to the variable

    - the advection operators, that are special operators designed to compute
    the nonlinear advection operation on a variable. They should provide, in 2D,
    two matrices and two bc vectors, one for each direction. They also have a
    procedure apply that, given a scalar P, an x-velocity component v_x and a
    y-velocity component v_y, returns a scalar representing divergence(v_xP +
    v_yP)

The procedures designed as primitive means of combinations are:
    -add_scalars
    -mul_num_scalar
    -mul_scalars
    -add_operators
    -mul_num_operator
    -mul_operators
    -add_advection_operators
    -mul_operators_to_advection_operator
    -finite_volume_interpolation

These procedures, except for the finite_volume_interpolation procedure, as well
as the apply procedures of the operators and advection operators have the
closure property: they can be applied to their own return values. This will help
build modular higher-order procedures that easily combine the primitive data
objects defined here.
"""

import sys
import petsc4py
petsc4py.init(sys.argv)
import petsc4py.PETSc as petsc
from six.moves import range
import config as cfg
import math
import importlib
from mrpy.mr_utils import mesh

class Scalar(object):
    """Implementation of the Scalar data object.

    It has two main attributes, aa array with the values of variable
    represented, and a name tag that helps identify specific compenents of the
    variable, e.g. its boundary conditions.
    """

    def __init__(self, tree=None):
        self.sc = petsc.Vec().create()
        self.sc.setUp()

        if tree is not None:
            number_of_rows = tree.number_of_leaves
            self.sc.setSizes(number_of_rows, number_of_rows)

            for row in range(number_of_rows):
                self.sc[row] = tree.nvalue[tree.tree_leaves[row]]

        self.tag = None

class Operator(object):
    """Implementation of the Operator data object.

    An operator is designed to be used for a specific variable (for example the
    pressure_gradx instantiation of the Operator class is used to compute the
    x-component of the gradient of the pressure). so it has to be initialized
    with the (Scalar representation of the) variable whose spatial derivative it
    will compute.

    In addition, the class is initialized with a (reference to a) module, that
    creates the matrix of the operator, and the array that gives the values of
    the operator on the boundaries. These are the two main attributes of the
    Operator object: matrix and bc.

    The matrix form depends on the type of boundary conditions that applies to
    the variable, so we use the Scalar tag of the initiating variable to
    determine its bc in the config module.

    The bc attribute also uses the Scalar tag to produce the boundary values
    associated with the spatial derivative computed. For now, we suppose that
    the boundary conditions do not change during the simulation, so we can
    determine them at the beginning of the simulation. This might be changed in
    the future.

    The operator has an "apply" procedure that compute the spatial derivative of
    the initiating variable.

    We should probably do something about the fact that if an Operator is
    initiated with a module but without any scalar, it will return an error (it
    will not be able to read the tag of the scalar).
    """

    def __init__(self, tree=None, axis=None, module=None):

        if module is None:
            self.matrix = petsc.Mat().create()
            self.bc = petsc.Vec().create()
            self.bc.setUp()

        else:
            module = importlib.import_module(module)

            self.matrix = module.create_matrix(tree, axis)

            self.bc = module.create_bc_scalar(tree, axis)

    def apply(self, scalar):

        temp_1, temp_2 = Scalar(), Scalar()
        temp_1.sc, temp_2.sc = scalar.sc.copy(), scalar.sc.copy()

        self.matrix.mult(scalar.sc, temp_1.sc)
        temp_2.sc = self.bc

        return add_scalars(temp_1, temp_2)

class AdvectionOperator(object):
    """Implementation of the Advection Operator data object.

    It is very similar to the Operator class. It needs to be initialized with
    the variable whose advection term it will compute. It also needs the two
    components of the velocity, as well as the two modules needed to compute the
    every part of the advection term. For example, for the temperature variable
    T, if we define adv(T) to be: divergence(Tv_x, Tv_y), then we import the
    divergence_x and divergence_y modules, and we compute adv(T) =
    div_x(Tv_x)+div_y(Tv_y).

    The two bc attributes of the class are also computed at the beginning of the
    simulation, assuming that the boundary conditions of both the variable and
    the velocity will not change over time. This might be changed in the future.

    The operator has an "apply" procedure that compute the advection term of
    the initiating variable.

    We should probably do something about the fact that if an Operator is
    initiated with a module but without any scalars, it will return an error (it
    will not be able to read the tag of the scalars).
    """

    def __init__(self, dimension, tree=None, module=None,
                 tree_velocity_x=None, tree_velocity_y=None, tree_velocity_z=None):

        if module is None:
            if dimension == 1:
                self.matrix_x = petsc.Mat().create()
                self.bc_x = petsc.Vec().create()
                self.bc_x.setUp()

            elif dimension == 2:
                self.matrix_x, self.matrix_y = petsc.Mat().create(), petsc.Mat().create()
                self.bc_x, self.bc_y = petsc.Vec().create(), petsc.Vec().create()
                self.bc_x.setUp(), self.bc_y.setUp()

            elif dimension == 3:
                self.matrix_x, self.matrix_y, self.matrix_z = petsc.Mat().create(), petsc.Mat().create(), petsc.Mat().create()
                self.bc_x, self.bc_y, self.bc_z = petsc.Vec().create(), petsc.Vec().create(), petsc.Vec().create()
                self.bc_x.setUp(), self.bc_y.setUp(), self.bc_z.setUp()

        else:
            if dimension == 1:
                #west = tree.bc["west"][1]*tree_velocity_x.bc["west"][1]
                #east = tree.bc["east"][1]*tree_velocity_x.bc["east"][1]
                def west(coords, t=0.):

                    return tree.bc["west"][1](coords, t)*tree_velocity_x.bc["west"][1](coords, t)

                def east(coords, t=0.):

                    return tree.bc["east"][1](coords, t)*tree_velocity_x.bc["east"][1](coords, t)

                module = importlib.import_module(module)

                self.matrix_x = module.create_matrix(tree, 0)

                self.bc_x = module.create_bc_scalar(tree, 0, west=west, east=east)

            elif dimension == 2:
                #north = tree.bc["north"][1]*tree_velocity_y.bc["north"][1]
                #south = tree.bc["south"][1]*tree_velocity_y.bc["south"][1]
                #west = tree.bc["west"][1]*tree_velocity_x.bc["west"][1]
                #east = tree.bc["east"][1]*tree_velocity_x.bc["east"][1]
                def north(coords, t=0.):

                    return tree.bc["north"][1](coords, t)*tree_velocity_y.bc["north"][1](coords, t)

                def south(coords, t=0.):

                    return tree.bc["south"][1](coords, t)*tree_velocity_y.bc["south"][1](coords, t)

                def west(coords, t=0.):

                    return tree.bc["west"][1](coords, t)*tree_velocity_x.bc["west"][1](coords, t)

                def east(coords, t=0.):

                    return tree.bc["east"][1](coords, t)*tree_velocity_x.bc["east"][1](coords, t)

                module = importlib.import_module(module)

                self.matrix_x, self.matrix_y = module.create_matrix(tree, 0), module.create_matrix(tree, 1)

                self.bc_x = module.create_bc_scalar(tree, 0, west=west, east=east)
                self.bc_y = module.create_bc_scalar(tree, 1, north=north, south=south)

            elif dimension == 3:
                #north = tree.bc["north"][1]*tree_velocity_y.bc["north"][1]
                #south = tree.bc["south"][1]*tree_velocity_y.bc["south"][1]
                #west = tree.bc["west"][1]*tree_velocity_x.bc["west"][1]
                #east = tree.bc["east"][1]*tree_velocity_x.bc["east"][1]
                #forth = tree.bc["forth"][1]*tree_velocity_z.bc["forth"][1]
                #back = tree.bc["back"][1]*tree_velocity_z.bc["back"][1]
                def north(coords, t=0.):

                    return tree.bc["north"][1](coords, t)*tree_velocity_y.bc["north"][1](coords, t)

                def south(coords, t=0.):

                    return tree.bc["south"][1](coords, t)*tree_velocity_y.bc["south"][1](coords, t)

                def west(coords, t=0.):

                    return tree.bc["west"][1](coords, t)*tree_velocity_x.bc["west"][1](coords, t)

                def east(coords, t=0.):

                    return tree.bc["east"][1](coords, t)*tree_velocity_x.bc["east"][1](coords, t)

                def back(coords, t=0.):

                    return tree.bc["back"][1](coords, t)*tree_velocity_z.bc["back"][1](coords, t)

                def forth(coords, t=0.):

                    return tree.bc["forth"][1](coords, t)*tree_velocity_z.bc["forth"][1](coords, t)

                module = importlib.import_module(module)

                self.matrix_x, self.matrix_y, self.matrix_z = module.create_matrix(tree, 0), module.create_matrix(tree, 1), module.create_matrix(tree, 2)

                self.bc_x = module.create_bc_scalar(tree, 0, west=west, east=east)
                self.bc_y = module.create_bc_scalar(tree, 1, north=north, south=south)
                self.bc_z = module.create_bc_scalar(tree, 2, forth=forth, back=back)

    def apply(self, dimension, scalar, velocity_x=None, velocity_y=None,
            velocity_z=None, low_mach=False):

        if low_mach:
            if dimension == 1:
                adv_x = Scalar()
                adv_x.sc = scalar.sc.copy()

                self.matrix_x.mult(scalar.sc, adv_x.sc)

                adv_x = mul_scalars(adv_x, velocity_x)

                temp_x = Scalar()

                temp_x.sc = self.bc_x

                return add_scalars(adv_x, temp_x)

            elif dimension == 2:
                adv_x, adv_y = Scalar(), Scalar()
                adv_x.sc, adv_y.sc = scalar.sc.copy(), scalar.sc.copy()

                self.matrix_x.mult(scalar.sc, adv_x.sc)
                self.matrix_y.mult(scalar.sc, adv_y.sc)

                adv_x = mul_scalars(adv_x, velocity_x)
                adv_y = mul_scalars(adv_y, velocity_y)

                temp_x, temp_y = Scalar(), Scalar()

                temp_x.sc = self.bc_x
                temp_y.sc = self.bc_y

                return add_scalars(add_scalars(adv_x, temp_x), add_scalars(adv_y, temp_y))

            elif dimension == 3:
                adv_x, adv_y, adv_z = Scalar(), Scalar(), Scalar()
                adv_x.sc, adv_y.sc = scalar.sc.copy(), scalar.sc.copy(), scalar.sc.copy()

                self.matrix_x.mult(scalar.sc, adv_x.sc)
                self.matrix_y.mult(scalar.sc, adv_y.sc)
                self.matrix_z.mult(scalar.sc, adv_z.sc)

                adv_x = mul_scalars(adv_x, velocity_x)
                adv_y = mul_scalars(adv_y, velocity_y)
                adv_z = mul_scalars(adv_z, velocity_z)

                temp_x, temp_y, temp_z = Scalar(), Scalar(), Scalar()

                temp_x.sc = self.bc_x
                temp_y.sc = self.bc_y
                temp_z.sc = self.bc_z

                return add_scalars(add_scalars(adv_x, temp_x), add_scalars(adv_y, temp_y), add_scalars(adv_z, temp_z))

        else:
            if dimension == 1:
                adv_x = Scalar()
                adv_x.sc = scalar.sc.copy()

                temp_x = mul_scalars(scalar, velocity_x)

                self.matrix_x.mult(temp_x.sc, adv_x.sc)

                temp_x.sc = self.bc_x

                return add_scalars(adv_x, temp_x)

            elif dimension == 2:
                adv_x, adv_y = Scalar(), Scalar()
                adv_x.sc, adv_y.sc = scalar.sc.copy(), scalar.sc.copy()

                temp_x = mul_scalars(scalar, velocity_x)
                temp_y = mul_scalars(scalar, velocity_y)

                self.matrix_x.mult(temp_x.sc, adv_x.sc)
                self.matrix_y.mult(temp_y.sc, adv_y.sc)

                temp_x.sc = self.bc_x
                temp_y.sc = self.bc_y

                return add_scalars(add_scalars(adv_x, temp_x), add_scalars(adv_y, temp_y))

            elif dimension == 3:
                adv_x, adv_y, adv_z = Scalar(), Scalar(), Scalar()
                adv_x.sc, adv_y.sc = scalar.sc.copy(), scalar.sc.copy(), scalar.sc.copy()

                temp_x = mul_scalars(scalar, velocity_x)
                temp_y = mul_scalars(scalar, velocity_y)

                self.matrix_x.mult(temp_x.sc, adv_x.sc)
                self.matrix_y.mult(temp_y.sc, adv_y.sc)
                self.matrix_z.mult(temp_y.sc, adv_z.sc)

                temp_x.sc = self.bc_x
                temp_y.sc = self.bc_y
                temp_z.sc = self.bc_z

                return add_scalars(add_scalars(adv_x, temp_x), add_scalars(adv_y, temp_y), add_scalars(adv_z, temp_z))

def add_scalars(first_scalar, *other_scalars):
    """Takes as arguments some scalars, and return their sum."""

    scalars_sum = Scalar()

    scalars_sum.sc = first_scalar.sc.copy()

    for scalar in other_scalars:
        scalars_sum.sc.axpy(1, scalar.sc)

    return scalars_sum

def mul_num_scalar(num, scalar):
    """Takes a real number r and a scalar S, and return the scalar rS, that is,
    every value of S multiplied by r."""


    num_scalar_product = Scalar()

    num_scalar_product.sc = scalar.sc.copy()

    num_scalar_product.sc.scale(num)

    return num_scalar_product

def mul_scalars(first_scalar, *other_scalars):
    """Takes as arguments some scalars, and return their (scalar, pointwise)
    product."""

    scalars_product = Scalar()

    scalars_product.sc = first_scalar.sc.copy()

    for scalar in other_scalars:
        scalars_product.sc.pointwiseMult(scalars_product.sc, scalar.sc)

    return scalars_product

def inverse_scalar(scalar):
    """Takes a scalar S, and return the scalar 1/S, that is,
    the pointwise inverse of S."""


    temp = scalar.sc.copy()
    temp.set(1.)

    foo = Scalar()
    foo.sc = temp.copy()
    foo.sc.pointwiseDivide(temp, scalar.sc)

    return foo

def add_operators(first_operator, *other_operators):
    """Takes as arguments some operators, and return their sum."""

    operators_sum = Operator()

    operators_sum.matrix = first_operator.matrix.copy()
    operators_sum.bc = first_operator.bc.copy()

    for operator in other_operators:
        operators_sum.matrix.axpy(1, operator.matrix)
        operators_sum.bc.axpy(1, operator.bc)

    return operators_sum

def mul_num_operator(num, operator):
    """Takes a real number r and an operator O, and return the operator rO, that is,
    every value of O.sc multiplied by r, and O.matrix multiplied by r.
    """

    num_operator_product = Operator()

    num_operator_product.matrix = operator.matrix.copy()
    num_operator_product.bc = operator.bc.copy()

    num_operator_product.matrix.scale(num)
    num_operator_product.bc.scale(num)

    return num_operator_product

def mul_operators(first_operator, *other_operators):
    """Takes as arguments some operators, and returns their product, an operator
    whose matrix is the product of the matrices of the operators given as
    parameters, and whose bc is the bc of the first operator, multiplied by the
    matrices of the subsequent operators.

    Attention has to be paid to the fact that the order of parameters matters,
    given the fact that matrix multiplication is not commutative.
    """

    operators_product = Operator()

    operators_product.matrix = first_operator.matrix.copy()
    operators_product.bc = first_operator.bc.copy()

    for operator in other_operators:
        operators_product.matrix = operator.matrix.matMult(operators_product.matrix)
        temp = operators_product.bc.copy()
        operator.matrix.mult(temp, operators_product.bc)

    return operators_product

def add_advection_operators(dimension, first_advection_operator, *other_advection_operators):
    """Takes as arguments some advection operators, and return their sum."""

    if dimension == 1:
        advection_operators_sum = AdvectionOperator(dimension)

        advection_operators_sum.matrix_x = first_advection_operator.matrix_x.copy()
        advection_operators_sum.bc_x = first_advection_operator.bc_x.copy()

        for advection_operator in other_advection_operators:
            advection_operators_sum.matrix_x.axpy(1, advection_operator.matrix_x)
            advection_operators_sum.bc_x.axpy(1, advection_operator.bc_x)

        return advection_operators_sum

    elif dimension == 2:
        advection_operators_sum = AdvectionOperator(dimension)

        advection_operators_sum.matrix_x = first_advection_operator.matrix_x.copy()
        advection_operators_sum.matrix_y = first_advection_operator.matrix_y.copy()
        advection_operators_sum.bc_x = first_advection_operator.bc_x.copy()
        advection_operators_sum.bc_y = first_advection_operator.bc_y.copy()

        for advection_operator in other_advection_operators:
            advection_operators_sum.matrix_x.axpy(1, advection_operator.matrix_x)
            advection_operators_sum.matrix_y.axpy(1, advection_operator.matrix_y)
            advection_operators_sum.bc_x.axpy(1, advection_operator.bc_x)
            advection_operators_sum.bc_y.axpy(1, advection_operator.bc_y)

        return advection_operators_sum

    elif dimension == 3:
        advection_operators_sum = AdvectionOperator(dimension)

        advection_operators_sum.matrix_x = first_advection_operator.matrix_x.copy()
        advection_operators_sum.matrix_y = first_advection_operator.matrix_y.copy()
        advection_operators_sum.matrix_z = first_advection_operator.matrix_z.copy()
        advection_operators_sum.bc_x = first_advection_operator.bc_x.copy()
        advection_operators_sum.bc_y = first_advection_operator.bc_y.copy()
        advection_operators_sum.bc_z = first_advection_operator.bc_z.copy()

        for advection_operator in other_advection_operators:
            advection_operators_sum.matrix_x.axpy(1, advection_operator.matrix_x)
            advection_operators_sum.matrix_y.axpy(1, advection_operator.matrix_y)
            advection_operators_sum.matrix_z.axpy(1, advection_operator.matrix_z)
            advection_operators_sum.bc_x.axpy(1, advection_operator.bc_x)
            advection_operators_sum.bc_y.axpy(1, advection_operator.bc_y)
            advection_operators_sum.bc_z.axpy(1, advection_operator.bc_z)

        return advection_operators_sum

def mul_operators_to_advection_operator(dimension, advection_operator, *other_operators):
    """Takes as argument an advection operator, and some operators, and returns
    an advection-like operator, whose matrices are the matrics of the
    advection_operator parameter, multiplied by the matrices of the subsequent
    operators, and whose bcs are the bcs of the advection_operator parameter,
    multiplied by the matrices of the subsequent operators.

    The goal is to be able to compute easily the gradient of the advection of
    the temperature for example.

    Attention has to be paid to the fact that the order of parameters matters,
    given the fact that matrix multiplication is not commutative.
    """

    if dimension == 1:
        advection_operator_product = AdvectionOperator(dimension)

        advection_operator_product.matrix_x = advection_operator.matrix_x.copy()
        advection_operator_product.bc_x = advection_operator.bc_x.copy()

        for operator in other_operators:
            advection_operator_product.matrix_x = operator.matrix.matMult(advection_operator_product.matrix_x)
            temp = advection_operator_product.bc_x.copy()
            operator.matrix.mult(temp, advection_operator_product.bc_x)

        return advection_operator_product

    if dimension == 2:
        advection_operator_product = AdvectionOperator(dimension)

        advection_operator_product.matrix_x = advection_operator.matrix_x.copy()
        advection_operator_product.matrix_y = advection_operator.matrix_y.copy()
        advection_operator_product.bc_x = advection_operator.bc_x.copy()
        advection_operator_product.bc_y = advection_operator.bc_y.copy()

        for operator in other_operators:
            advection_operator_product.matrix_x = operator.matrix.matMult(advection_operator_product.matrix_x)
            advection_operator_product.matrix_y = operator.matrix.matMult(advection_operator_product.matrix_y)
            temp = advection_operator_product.bc_x.copy()
            operator.matrix.mult(temp, advection_operator_product.bc_x)
            temp = advection_operator_product.bc_y.copy()
            operator.matrix.mult(temp, advection_operator_product.bc_y)

        return advection_operator_product

    if dimension == 3:
        advection_operator_product = AdvectionOperator(dimension)

        advection_operator_product.matrix_x = advection_operator.matrix_x.copy()
        advection_operator_product.matrix_y = advection_operator.matrix_y.copy()
        advection_operator_product.matrix_z = advection_operator.matrix_z.copy()
        advection_operator_product.bc_x = advection_operator.bc_x.copy()
        advection_operator_product.bc_y = advection_operator.bc_y.copy()
        advection_operator_product.bc_z = advection_operator.bc_z.copy()

        for operator in other_operators:
            advection_operator_product.matrix_x = operator.matrix.matMult(advection_operator_product.matrix_x)
            advection_operator_product.matrix_y = operator.matrix.matMult(advection_operator_product.matrix_y)
            advection_operator_product.matrix_z = operator.matrix.matMult(advection_operator_product.matrix_z)
            temp = advection_operator_product.bc_x.copy()
            operator.matrix.mult(temp, advection_operator_product.bc_x)
            temp = advection_operator_product.bc_y.copy()
            operator.matrix.mult(temp, advection_operator_product.bc_y)
            temp = advection_operator_product.bc_z.copy()
            operator.matrix.mult(temp, advection_operator_product.bc_z)

        return advection_operator_product

def finite_volume_interpolation(tree, function, time=0.):
    """Takes as arguments an analytic function and a tree, and modify the values
    of the tree's leaves to contain the finite volume interpolation of the function over the
    domain.

    This procedure also takes as an optional parameter the time of the
    simulation, so as to be able to use it to compute the finite volume
    interpolation of the function at the time specified.
    """

    if tree.dimension == 2:

        for index in tree.tree_leaves:
            tree.nvalue[index] = 1/4.*(function(tree.ncoord_x[index] - tree.ndx[index]/2., tree.ncoord_y[index] - tree.ndy[index]/2., time) + \
                               function(tree.ncoord_x[index] + tree.ndx[index]/2., tree.ncoord_y[index] - tree.ndy[index]/2., time) + \
                               function(tree.ncoord_x[index] - tree.ndx[index]/2., tree.ncoord_y[index] + tree.ndy[index]/2., time) + \
                               function(tree.ncoord_x[index] + tree.ndx[index]/2., tree.ncoord_y[index] + tree.ndy[index]/2., time))

# This is a temporary function designed to emulate checkerboard modes in
# collocated NS incompressible flow simulation; it should be deleted or
# implemented elsewhere
def create_checkerboard_mode():
    """...

    """

    (xs, xe), (ys, ye) = cfg.main_grid.getRanges()

    temp = Scalar()

    scalar_array = cfg.main_grid.getVecArray(temp.sc)

    for j in range(ys, ye):
        for i in range(xs, xe):
            scalar_array[i, j] = (-1)**(i+j)

    return temp

if __name__ == "__main__":

    tree = mesh.create_new_tree(cfg.dimension, cfg.min_level, cfg.max_level, cfg.stencil_graduation, cfg.stencil_prediction)

    tree.tag = "u"

    mesh.listing_of_leaves(tree)

    print(tree.number_of_leaves)
    print("")

    for index in tree.tree_leaves:
        tree.nvalue[index] = 1
