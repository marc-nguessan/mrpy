import sys, petsc4py
petsc4py.init(sys.argv)
import petsc4py.PETSc as petsc
import mpi4py.MPI as mpi

number_of_rows = 10

def create_matrix(diag_val):

    matrix = petsc.Mat().create()
    size = (number_of_rows, number_of_rows)
    matrix.setSizes((size, size))
    matrix.setUp()

    temp = petsc.Vec().create()
    temp.setSizes(number_of_rows, number_of_rows)
    temp.setUp()
    temp.set(diag_val)
    matrix.setDiagonal(temp)

    return matrix

def test_1():

    A = create_matrix(1)
    B = create_matrix(2)
    C = create_matrix(3)

    matrix_nest = petsc.Mat().createNest(((A, B, C),))

    #mats = [list(mat) for mat in ([A],)]
    #print(mats)
    #mats = [list(mat) for mat in ((A, B, C),)]
    #print(mats)
    matrix_nest.view()

def test_2():

    A = create_matrix(1)
    B = create_matrix(2)
    C = create_matrix(3)

    matrix_nest = petsc.Mat().createNest(([A], [B], [C]))
    matrix_nest.view()

def test_3():

    A = create_matrix(1)
    B = create_matrix(2)
    C = create_matrix(3)

    matrix_nest_1 = petsc.Mat().createNest([[A, B]])
    matrix_nest = petsc.Mat().createNest([[matrix_nest_1, C]])
    matrix_nest.view()

def test_4():

    A = create_matrix(1)
    B = create_matrix(2)
    C = create_matrix(3)

    matrix_nest = petsc.Mat().createNest([[A, B], [C, None]])
    matrix_nest.view()

def test_5():

    A = create_matrix(1)
    B = create_matrix(2)
    C = create_matrix(3)

    matrix_nest = petsc.Mat().createNest([(A, B)])
    matrix_nest.view()

if __name__ == "__main__":
    test_1()
