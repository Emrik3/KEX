import numpy as np
import numpy.linalg as la
from sympy import *
from choose_word_classes import class_to_index


def frobnorm(A):
    fnorm = 0
    Ashape = shape(A)
    for i in range(Ashape[0]):
        for j in range(Ashape[1]):
            fnorm += A[i, j] ** 2
    return fnorm ** .5


def norminf(A):
    ninf = []
    Ashape = shape(A)
    for i in range(Ashape[0]):
        n = 0
        for j in range(Ashape[1]):
            n += abs(A[i, j])
        ninf.append(n)
    return max(ninf)


def norm1(A):
    ninf = []
    Ashape = shape(A)
    for j in range(Ashape[1]):
        n = 0
        for i in range(Ashape[0]):
            n += abs(A[i, j])
        ninf.append(n)
    return max(ninf)


def norm2(A):
    # Fix this one so that it first checks if real eigenvalues, or maybe take the norm of complex value
    # Could find some theorems here that prove something about the matrix when it is no possible to do this...
    # Maybe just take abs of the complex eigenvalue, could work...
    ATA = A.T * A
    eig = ATA.eigenvals()  # Add the absolute value here!!!
    return float(max(list(eig.keys()))) ** .5


def mostprobfollows(A, classtext):
    for cl in classtext:
        k = 0
        i = class_to_index[cl]

        for j in range(len(A[i])):
            if k < A[i][j]:
                k = A[i][j]
                holdj = j

        key_list = list(class_to_index.keys())
        val_list = list(class_to_index.values())

        position = val_list.index(holdj)
        print(cl + ": " + key_list[position])


def maxlike():
    pass

def distance(A, B, normtype):
    return normtype(np.subtract(A, B))


def running_metrics(A, B):
    print("1-norm: " + str(distance(A, B, norm1)))
    print("infinity-norm: " + str(distance(A, B, norminf)))
    print("frobenius norm: " + str(distance(A, B, frobnorm)))
