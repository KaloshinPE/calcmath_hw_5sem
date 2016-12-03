# the task is to solve the system of linear equations via using three different methods: direct, iterative and variation
# I suppose that system has the only one solution
import numpy as np
# parameters
number_of_iterations = 43

# input system
A_start = np.array([[11.0, 2.0, 3.0],
                    [4.0, 15.0, 6.0],
                    [7.0, 8.0, 21.0]])
b_start = np.array([10.0, 11.0, 12.0])


''' Gauss method (direct)'''
def add_line(i_target, i_add, k=1):
    # add [i_add] line of system, multiplied by k, to [i_target] line
    A[i_target] += k*A[i_add]
    b[i_target] += k*b[i_add]


def exchange_lines(i, j):
    if i != j:
        c = np.copy(A[i])
        A[i] = A[j]
        A[j] = c


def find_nonzero_j(i, j):
    # returns number of line after i (i line included) with nonzero j element
    for k in range(A.shape[0])[i:]:
        if A[k][j] != 0:
            return k
        else:
            print "zero column detected"


def solve_triangular_system(A, b):
    # solve system with top-triangular matrix
    solution = list()
    for i in range(A.shape[0]-1, -1, -1):
        solution.append(b[i]/A[i][i])
        for k in range(i):
            b[k] -= A[k][i]*solution[-1]
    solution.reverse()
    return solution

A = np.copy(A_start)
b = np.copy(b_start)

for i in range(A.shape[0]-1):
    exchange_lines(i, find_nonzero_j(i, i))
    map(lambda j: add_line(j, i, -1*A[j][i]/A[i][i]), range(A.shape[0])[i+1:])

direct_solution = solve_triangular_system(A, b)
print "direct method (Gauss):" + str(direct_solution)


''' method Jacobi (iterative)'''
A = np.copy(A_start)
b = np.copy(b_start)


def check_covergence(matrix):
    for i in range(matrix.shape[0]):
        summ = 0
        for j in range(matrix.shape[0]):
            if j!=i:
                summ += matrix[i][j]
        if np.abs(matrix[i][i] <= summ):
            print "condition for covergence is not sutisfied"
            return False
    return True


def reverse_matrix(matrix):
    # calculates revers matrix to diagonal matrix
    reversed_matrix = np.copy(matrix)
    for i in range(matrix.shape[0]):
        reversed_matrix[i][i] = 1/reversed_matrix[i][i]
    return reversed_matrix


if check_covergence(A) :
    L, D, U = np.zeros(A.shape), np.zeros(A.shape), np.zeros(A.shape)
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            if i == j:
                D[i][j] = A[i][j]
            elif i > j:
                L[i][j] = A[i][j]
            elif i<j:
                U[i][j] = A[i][j]

    R = -1*np.dot(reverse_matrix(D), L+U)
    F = np.dot(reverse_matrix(D), b)
    x = np.zeros(A.shape[0])

    for i in range(number_of_iterations):
        x = np.dot(R, x) + F

    print "iterative method (Jacobi): " + str(x) + ", number of iterations: " + str(number_of_iterations) + ", accuracy = " + str(np.linalg.norm(direct_solution - x))


'''method of residual (variation)'''
A = np.copy(A_start)
b = np.copy(b_start)
x = np.zeros(A.shape[0])

for i in range(number_of_iterations):
    r = b - np.dot(A, x)
    if (np.linalg.norm(r) == 0):
        number_of_iterations = i
        break
    x = x + np.dot(np.dot(A, r), r)/np.linalg.norm(np.dot(A, r))**2*r

print "variation method (residual): " + str(x) + ", number of iterations: " + str(
    number_of_iterations) + ", accuracy = " + str(np.linalg.norm(direct_solution - x))



