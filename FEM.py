#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 14:03:29 2021

@author: diego
"""

import numpy as np
import plotly.io as pio
import plotly.graph_objects as go
from functools import lru_cache



class vistools:
    def __init__(self, element_type):
        self.element_type = element_type
        if element_type == LTE:
            self.mesh = element_type.mesh
            self.nodes = element_type.nodes
    
    def PlotMesh(self, Distortion=None, scaling=None):
        fig = go.Figure()
        if self.element_type == LTE:
            for i, element in enumerate(self.element_type.Mesh):
                x = list(element.x)
                x.append(x[0])
                y = list(element.y)
                y.append(y[0])
                fig.add_trace(go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    name=i
                ))
            if Distortion is not None:
                for i, element in enumerate(self.element_type.Mesh):
                    scale=100 if scaling==None else scaling
                    dx = Distortion[list(element.index*2)]*scale
                    dy = Distortion[list(element.index*2+1)]*scale
                    
                    x = [x+d for x, d in zip(element.x,dx)]
                    y = [y+d for y, d in zip(element.y,dy)]
                    
                    x.append(x[0])
                    y.append(y[0])
                    
                    fig.add_trace(go.Scatter(
                        x=x,
                        y=y,
                        mode="lines",
                        fill="toself",
                        name="%d - distorted"%i
                    ))
            
        pio.renderers.default = 'browser'
        return fig
               


class LTE:
    """
    # Linear Triangular Element
    """
    Mesh = []
    def __init__(self, element_nodes, global_nodes, thickness = 1, NoE = None):
        self.nodes = global_nodes[element_nodes]
        self.index = element_nodes
        
        self.x = self.nodes[:,0]
        self.y = self.nodes[:,1]
        
        self.t = thickness
        self.__class__.Mesh.append(self)
        
        if len(self.__class__.Mesh) == 1:
            self.__class__.mesh_nodes(global_nodes)
 
    
    @classmethod
    def mesh(cls): # LTE objects as list
        return cls.Mesh
    
    
    @classmethod # LTE stiffness matrix
    def Stiffness(cls, D, BC=None):
        DoF = cls.DoF()
        K = np.zeros([DoF, DoF])
        
        for element in cls.Mesh:
            for i in element.index:
                m = np.where(element.index==i)[0][0]
                
                K[i*2, element.index*2] += \
                    element.k(D)[m*2, np.array([0, 1, 2])*2]
                K[i*2+1, element.index*2] += \
                    element.k(D)[m*2+1, np.array([0, 1, 2])*2]
                K[i*2, element.index*2+1] += \
                    element.k(D)[m*2, np.array([0, 1, 2])*2 + 1]
                K[i*2+1, element.index*2+1] += \
                    element.k(D)[m*2+1, np.array([0, 1, 2])*2 + 1]
        if BC == None:
            cls.K = K
            return K
        else:
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
            cls.K = K
            return K
    
    
    @classmethod
    def mesh_nodes(cls, global_nodes): # LTE mesh Nodes
        cls.nodes = global_nodes
        return cls.nodes
    
    
    @classmethod
    @lru_cache(maxsize=None)
    def DoF(cls): # DoFs for LTE mesh
        return len(cls.nodes)*2 # number of nodes * dimension
    
    
    @classmethod
    def set_F(cls, f_dict): # set Variable (Force) on mesh
        f = np.zeros(cls.DoF())

        for node, p in f_dict.items():
            print("Setting Node ", node, " with ", p)
            f[2*node]   = p[0]
            f[2*node+1] = p[1]
        
        cls.F = f
        return f
    
    
    @classmethod
    def get_u(cls): # get u from setted Variable 
        u = np.zeros(cls.DoF())
        i = np.where(~cls.K.any(axis=1))[0] # find 0 axes
        K = np.delete(cls.K, i, axis=0)
        K = np.delete(K, i, axis=1)
        f = np.delete(cls.F, i, axis=0)
        
        # catch error here for inverse probs
        q = np.matmul(f, np.linalg.inv(K))

        j = list(range(cls.DoF()))
        l = [x for x in j if x not in i] # DoF left

        u[l] = q[:]
        
        cls.q = l
        cls.u = u
        return u
    
    
    @classmethod
    def get_s(cls, D): # get stress from u and F
        sig = np.zeros([len(cls.Mesh), 3, ])
        for n, element in enumerate(cls.Mesh):
            uel = np.zeros(6)
            for node in element.index:
                m = np.where(element.index==node)[0][0]
                if 2*node   in cls.q:
                    uel[2*m]   = cls.u[2*node]
                if 2*node+1 in cls.q:
                    uel[2*m+1] = cls.u[2*node+1]
            s = np.matmul(D, element.b()/element.det_J())
            sig[n] = np.matmul(s, uel)
        
        return sig
    
    
    @classmethod
    def force_routine(cls, D, F, BC=None):
        cls.DoF()
        cls.Stiffness(D, BC)
        
        F = cls.set_F(F)
        U = cls.get_u()
        s = cls.get_s(D)
        
        return F, U, s
    
    
    def strain_matrix(E, v):
        return (E/(1-v*v))*np.array([
            [1, v, 0],
            [v, 1, 0],
            [0, 0, 0.5*(1-v)],
        ])

    def det_J(self):
        x, y = self.x, self.y
        
        J =     (x[0] - x[2])*(y[1] - y[2]) - \
                (x[1] - x[2])*(y[0] - y[2])
        
        return J

    
    def J(self):
        x, y = self.x, self.y
        J = np.zeros([2, 2])
        
        J[0, 0] = x[0] - x[2]
        J[0, 1] = x[1] - x[2]
        J[1, 0] = y[0] - y[2]
        J[1, 1] = y[1] - y[2]
        
        return J

    
    def b(self):
        x, y = self.x, self.y
        
        b = np.array([
            [y[1]-y[2],        0,y[2]-y[0],        0,y[0]-y[1],        0],
            [        0,x[2]-x[1],        0,x[1]-x[2],        0,x[1]-x[0]],
            [x[2]-x[1],y[1]-y[2],x[0]-x[2],y[2]-y[0],x[1]-x[0],y[0]-y[1]]
        ])
        
        return b

    
    def k(self, D):
        detj = self.det_J()
        B = self.b()/detj
        Bt = B.transpose()
        
        return 0.25*abs(detj)*self.t*np.matmul(Bt, np.matmul(D, B))