# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from FEM import LTE, vistools

# Distances in mm
nodes = np.array([
    [    0, 10e3], # 0
    [ 10e3, 10e3], # 1
    [    0, 0   ], # 2
    [ 10e3, 0   ], # 3
    [ 20e3, 0   ], # 4
])

elements = np.array([
    [0, 3, 2], # 0
    [0, 1, 3], # 1
    [1, 4, 3], # 2
])

BC = {
    0: "x",
    2:"xy",
    3:"y",
    4:"y"
}

F = {
    0: [   0, 1000], # N
    1: [ 500, 0   ]
}

E = 100e3 # MPa
v = 0.3 # Po

D = LTE.strain_matrix(E, v)

# setup mesh
for element in elements:
    LTE(element, nodes)

F, U, s = LTE.force_routine(D, F, BC)

visualiser = vistools(LTE)

fig = visualiser.PlotMesh(Distortion=U, scaling=20e3)
fig.show()