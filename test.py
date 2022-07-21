import numpy as np
from math import *
from numpy import linalg as LA
from scipy import linalg as la
import matplotlib.pyplot as plt

#A = np.random.randn(5, 3)

#b = np.random.randn(3)

#C = A*b # A 'modified' elementwise multiplication

#D = A@b # Matrix multiplication

#n = 4
#A2 = np.random.randn(n,n)
#b2 = np.random.randn(n)

# Solve A2@x = b2

# 1 way
#x1 = np.linalg.solve(A2, b2)

#P, L, U = la.lu(A2)

# x2 = inv(U) @ inv(L) @ inv(P) @ b2

#print(U)
#print(np.linalg.inv(U))

#inv_U = np.linalg.inv(U)

#inv_L = np.linalg.inv(L)

#inv_P = np.linalg.inv(P)

#x2 = inv_U @ inv_L @ inv_P @ b2

#---------------------------------------

#A3 = np.matrix('1, -2; -3, 6')

#np.linalg.det(A3)
#np.linalg.inv(A3)

#P_3, L_3, U_3 = la.lu(A3)

#3print(P_3)

#B = np.random.rand((5,5))
#k = np.random.rand(5)
#C = B @ np.diag(1/k)

#C @ np.diag(k) == B

#---------------------------------------

A = np.random.randn(2,2)
k = np.random.randn(2)

Sym = A @ A.T

eigenval, eigenvect = LA.eig(Sym)

b = np.array([[1, 2], [2, 1]])

eigenval_eg, eigenvect_eg = LA.eig(b)

Dia_eg = np.diag(eigenval_eg)

D = np.diag(eigenval)

S_check = eigenvect @ D @ eigenvect.T


#find S_1 and lamb_1
lamb_1 = eigenval[0] * np.outer(eigenvect[:,[0]], eigenvect[1])

lamb_2 = eigenval[1] * np.outer(eigenvect[:,[1]], eigenvect[0])

proj_1 = np.outer(eigenvect[:,[0]], eigenvect[1])

proj_2 = np.outer(eigenvect[:,[1]], eigenvect[0])


#final S
S = lamb_1 +lamb_2


#set a variable x
x = np.array([[1], [1]])

x_vect = x.flatten()


#find the projections with lambda
S_1_lamb = lamb_1  @ x

S_1_vect_lamb = S_1_lamb.flatten()

S_2_lamb = lamb_2 @ x

S_2_vect_lamb = S_2_lamb.flatten()


#find the projections of S
S_1 = proj_1  @ x

S_1_vect = S_1.flatten()

S_2 = proj_2 @ x

S_2_vect = S_2.flatten()



#plot the vectors
x_pos = [0, 0, 0]
y_pos = [0, 0, 0]

fig, ax = plt.subplots(1)

new_vect = np.array([x_vect, S_1_vect, S_2_vect])

ax.quiver(x_pos,y_pos, new_vect[:,0], new_vect[:,1], scale=1, scale_units='xy',angles = 'xy', color=['r','b','g'])


x_2D = [0, 0]
y_2D = [0, 0]

new_vect_lamb = np.array([S_1_vect_lamb, S_2_vect_lamb])

ax.quiver(x_2D, y_2D, new_vect_lamb[:,0], new_vect_lamb[:,1], angles='xy', scale_units='xy', scale=1, linestyle='dashed', facecolor='none', linewidth=2, width=0.0001, headwidth=300, headlength=500)

ax.set_xlim((-1, new_vect[:,0].max()+1))
ax.set_ylim((-1, new_vect[:,1].max()+1))

plt.grid() 
plt.show()

print()
print()
print()
print()




