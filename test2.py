import numpy as np
from math import *
from numpy import linalg as LA
from scipy import linalg as la
import matplotlib.pyplot as plt

A = np.random.randn(2,2)

Sym = A @ A.T

eigenval, eigenvect = LA.eig(Sym)

eg = np.array([[5, 4], [4, 5]])
eg_eigenval, eg_eigenvect = LA.eig(eg)

# draw ellipse

a = 4

b = 9

t = np.linspace(0, 2*pi, 100)

x = a*np.cos(t)

y = b*np.sin(t)

X_Y = np.array([x, y])

t_rot=pi/4

R_rot = np.array([[cos(t_rot) , -sin(t_rot)],[sin(t_rot) , cos(t_rot)]])

W_Z = R_rot @  X_Y

w = W_Z.T[:,0]

z = W_Z.T[:,1]


plt.plot(x , y, 'r')
plt.plot(w , z, 'b')
plt.grid(color='lightgray',linestyle='--')
plt.show()

# draw vectors
x_pos = [0, 0]
y_pos = [0, 0]

fig, ax = plt.subplots(1)

ax.quiver(x_pos,y_pos, eg_eigenvect[:,0], eg_eigenvect[:,1], scale=1, scale_units='xy',angles = 'xy', color=['r','b','g'])
plt.grid()

plt.plot()

plt.show()






print(Sym)
print(first)
print(second)
