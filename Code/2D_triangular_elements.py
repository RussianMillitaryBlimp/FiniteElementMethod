#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 12:20:14 2021

@author: diego
"""

import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import numpy as np
from FEM import LTE

fig = go.Figure()

def get_D(*args):
    return (E/(1-v*v))*np.array([
        [1, v, 0],
        [v, 1, 0],
        [0, 0, 0.5*(1-v)],
    ])


def get_B(*args):
    
    x = nodes[element][:,0]
    y = nodes[element][:,1]
    
    B = np.array([
        [y[1] - y[2],           0, y[2] - y[0],           0, y[0] - y[1],           0],
        [          0, x[2] - x[1],           0, x[1] - x[2],           0, x[1] - x[0]],
        [x[2] - x[1], y[1] - y[2], x[0] - x[2], y[2] - y[0], x[1] - x[0], y[0] - y[1]]
    ])
    
    return B

def get_detj(*args):
    
    x = nodes[element][:,0]
    y = nodes[element][:,1]
    
    j =     (x[0] - x[2])*(y[1] - y[2]) - \
            (x[1] - x[2])*(y[0] - y[2])
    
    return j

# Distances in m
nodes = np.array([
    [    3, 0    ], # 0
    [    3, 2    ], # 1
    [    0, 2    ], # 2
    [    0, 0    ], # 3
])

elements = np.array([
    [0, 1, 3], # 0
    [2, 3, 1], # 1
])

BC = {
    0: "y",
    2:"xy",
    3:"xy",
}

F = {
    1: [   0,-1000], # N
}

# for i, element in enumerate(elements):
#     x = list(nodes[element][:, 0])
#     x.append(x[0])
#     y = list(nodes[element][:, 1])
#     y.append(y[0])
#     print(x, y)
#     fig.add_trace(go.Scatter(
#         x=x,
#         y=y,
#         mode="lines",
#         name=i
#     ))
#     pass

# pio.renderers.default = 'browser'
# fig.show()

E = 30e6 # Pa
v = 0.25  # Po
t = 1. # thickness

D = get_D(E, v)

b = np.zeros([len(elements), 3, 6])
k = np.zeros([len(elements), 6, 6])

DoF = len(nodes)*2 # number of nodes * dimension

K = np.zeros([DoF, DoF])
f = np.zeros(DoF)
u = np.zeros(DoF)

for n, element in enumerate(elements):
    
    detj = get_detj(element, nodes)
    b[n] = get_B(element, nodes)/detj
    bt = b[n].transpose()
    
    k[n] = 0.25*abs(detj)*t*np.matmul(bt, np.matmul(D, b[n]))
    
    for node in element:
        m = np.where(element==node)[0][0]
        
        K[node*2, element*2] += k[n, m*2, np.array([0, 1, 2])*2]
        K[node*2+1, element*2] += k[n, m*2+1, np.array([0, 1, 2])*2]
        K[node*2, element*2+1] += k[n, m*2, np.array([0, 1, 2])*2 + 1]
        K[node*2+1, element*2+1] += k[n, m*2+1, np.array([0, 1, 2])*2 + 1]


    
for node, bc in BC.items():
    print("Setting Node ", node, " as ", bc)
    if "x" in bc: # 0 2 4
        dex = 2*node
        K[dex, :] = 0
        K[:, dex] = 0
        pass
    if "y" in bc: # 1 3 5
        dex = 2*node + 1
        K[dex, :] = 0
        K[:, dex] = 0
        pass

    
for node, p in F.items():
    print("Setting Node ", node, " with ", p)
    f[2*node]   = p[0]
    f[2*node+1] = p[1]

i = np.where(~K.any(axis=1))[0] # find 0 axes
K = np.delete(K, i, axis=0)
K = np.delete(K, i, axis=1)
f = np.delete(f, i, axis=0)

q = np.matmul(f, np.linalg.inv(K))

j = list(range(DoF))
l = [x for x in j if x not in i] # DoF left

u[l] = q[:]


sig = np.zeros([len(elements), 3, ])

for n, element in enumerate(elements):
    uel = np.zeros(6)
    for node in element:
        m = np.where(element==node)[0][0]
        print(l)
        if 2*node in l:
            uel[2*m] = u[2*node]
        if 2*node+1 in l:
            uel[2*m+1] = u[2*node+1]
    s = np.matmul(D, b[n])
    sig[n] = np.matmul(s, uel)