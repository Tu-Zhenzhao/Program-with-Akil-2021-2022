import numpy as np
from math import *
from numpy import linalg as LA
from scipy import linalg as la
import matplotlib.pyplot as plt

A = np.random.randn(2, 2)
Sym = A @ A.T

eigenval, eigenvect = LA.eig(Sym)

eg = np.array([[5, 4], [4, 5]])
eg_eigenval, eg_eigenvect = LA.eig(eg)

#----------------------------------------------------

#this is for example of Lay's book
B = np.array([[4, 11, 14], [8, 7, -2]])

sym_B = B.T @ B
sym_B2 = B @ B.T

B_eigva, B_eigvect = LA.eig(sym_B)
B2_eigva, B2_eigvect = LA.eig(sym_B2)

B_u, B_lamb, B_vh = LA.svd(B, full_matrices=True)

print(sym_B)
print()
print("Eigenvalue of B:", B_eigva)
print("Eigenvector of B", B_eigvect)
print()
print("Eigenvalue of B2:", B2_eigva)
print("Eigenvector of B2", B2_eigvect)
print()
print("U:", B_u)
print("B_lamb:", B_lamb)
print("B_vh:", B_vh)


