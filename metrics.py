import numpy as np
import numpy.linalg as la
from sympy import *
from choose_word_classes import class_to_index
from scipy.stats import wasserstein_distance
from scipy.linalg import svd

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


def tensor1norm(A):
    #https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
    return np.linalg.norm(A.flatten(), ord=1)
def tensor2norm(A):
    return np.linalg.norm(A.flatten(), ord=2)
def tensorinfnorm(A):
    # reshaping tensor to matrix with (n*n)x(n*n) with n=amount of wc and ord specifies norm, gives bad result so
    return np.linalg.norm(A.flatten(), ord=np.inf)
def tensorfrobnorm(A):
    return np.linalg.norm(A) #Apparently calculates frobenius norm, by flattening A to a 2d matrix or something ,ord'fro' doesn't work

def crossentropy(A,B):
    #https://en.wikipedia.org/wiki/Cross-entropy
    return -np.sum(A * np.log(B+10**(-10))) #the same as the likelyhood method? kolla joars

def Kullback_Leibler(A, B):
    #https://en.wikipedia.org/wiki/Kullback–Leibler_divergence
    return np.sum(A * np.log((A+10**(-10))/(B+10**(-10)))) #always
def maxsingularvalues(A):
    #https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html
    return np.max(np.linalg.svd(A, compute_uv=False))
def wasserstein_distance_1d(A, B):
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html
    return wasserstein_distance(A.flatten(), B.flatten())

def maxlike():
    pass

def distance(A, B, normtype):
    return normtype(np.subtract(A, B))

def running_metrics(A, B):
    #print("1-norm: " + str(distance(A, B, norm1)))
    #print("infinity-norm: " + str(distance(A, B, norminf)))
    #print("frobenius norm: " + str(distance(A, B, frobnorm)))
    return distance(A, B, norm1), distance(A,B, tensor2norm), distance(A, B, norminf), distance(A, B, frobnorm), crossentropy(A,B), Kullback_Leibler(A,B), distance(A, B, maxsingularvalues), wasserstein_distance_1d(A, B)
def running_metrics2(A,B):
    return distance(A, B, tensor1norm), distance(A,B, tensor2norm), distance(A,B,tensorinfnorm), distance(A,B,tensorfrobnorm), crossentropy(A,B), Kullback_Leibler(A, B), distance(A, B, maxsingularvalues), wasserstein_distance_1d(A, B)

def running_change_metrics(A, B):
    return distance(A, B, maxsingularvalues), wasserstein_distance_1d(A, B), distance(A, B, tensor1norm), distance(A,B,tensorfrobnorm), crossentropy(A,B), Kullback_Leibler(A, B)
