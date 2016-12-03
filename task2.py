# the task is to solve the system of linear equations via using three different methods: direct, iterative and variation
# I suppose that system has the only one solution
import numpy as np


# input system
A_start = np.array([[1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [7.0, 8.0, 11.0]])
b_start = np.array([10.0, 11.0, 12.0])


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




''' Gauss method (direct)'''
A = np.copy(A_start)
b = np.copy(b_start)

for i in range(A.shape[0]-1):
    exchange_lines(i, find_nonzero_j(i, i))
    map(lambda j: add_line(j, i, -1*A[j][i]/A[i][i]), range(A.shape[0])[i+1:])

print solve_triangular_system(A, b)

''' '''

