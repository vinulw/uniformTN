
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from progressbar import ProgressBar
from ncon import ncon
import pickle

from uniformTN_states import *
from uniformTN_Hamiltonians import *
from rw_functions import save_obj,load_obj

N_iter = 1000
beta = 0.5 #for vumps
learningRate = 0.1 #for gradDescent/TDVP
D = 2 #bond dimension

#ising
J = 1
X = np.array([[0,1],[1,0]])
Z = np.array([[-1,0],[0,1]])
Y = np.array([[0,1j],[-1j,0]])
heis = J/4 * (np.kron(X,X) + np.kron(Y,Y) + np.kron(Z,Z))

rho = np.load('./examples/test_rho.npy')
H = localH([twoBodyH(rho)])

np.random.seed(1)

from uniformTN_gs_solvers import vumps_1d,gradDescent

#vumps in centre gauge
psi_centre = uMPS_1d_centre(D)
psi_centre.randoInit()
psi_centre,eDensity,leftError,rightError = vumps_1d(psi_centre,H,N_iter,beta,stable_polar=True,stateConsistErrors=True)
data = [psi_centre.Al, psi_centre.Ac, psi_centre.Ar, psi_centre.C]
fname = 'psi_optimised.pkl'
with open(fname, 'wb') as fle:
    pickle.dump(data, fle)
print('Saved')

plt.subplot(1,2,1)
plt.plot(eDensity,label="VUMPS")
plt.xlabel("Iter")
plt.ylabel("E0")
plt.legend()
plt.subplot(1,2,2)
plt.plot(leftError,label="Left error")
plt.plot(rightError,linestyle="--",label="right error")
plt.yscale('log')
plt.legend()
plt.xlabel("Iter")
plt.ylabel("|| Al C - Ac ||")
plt.show()
