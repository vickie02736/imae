#shallow water propagation
"""
Solution of Shallow-water equations using a Python class.
Adapted for Python training course at CNRS from https://github.com/mrocklin/ShallowWater/

Dmitry Khvorostyanov, 2015
CNRS/LMD/IPSL, dmitry.khvorostyanov @ lmd.polytechnique.fr
"""


from pylab import *

import numpy as np
from tqdm import tqdm
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# from scipy.interpolate import griddata
import os
import argparse


parser = argparse.ArgumentParser(description='Shallow Water Simulation')
parser.add_argument('--R', type=int, help='R')
parser.add_argument('--Hp', type=int, help='Hp')
parser.add_argument('--root-path', type=str, default= '../data/', help='root path')
parser.add_argument('--task-name', type=str, help='task name')
parser.add_argument('--iteration-times', type=int, default=10000, help='iteration times')
args = parser.parse_args()

R = args.R
Hp = args.Hp
root_path = args.root_path
task_name = args.task_name
iteration_times = args.iteration_times

def x_to_y(X): # averaging in 2*2 windows (4 pixels)
    dim = X.shape[0]
    dim = 20
    Y = np.zeros((int(dim/2),int(dim/2)))
    for i in range(int(dim/2)):
        for j in range(int(dim/2)):
            Y[i,j] = X[2*i,2*j] + X[2*i+1,2*j] + X[2*i,2*j+1] + X[2*i+1,2*j+1]
            
            Y_noise = np.random.multivariate_normal(np.zeros(100),0.0000 * np.eye(100))
            Y_noise.shape = (10,10)
            Y = Y + Y_noise
    return Y



class shallow(object):

    # domain

    #N = 100
    #L = 1.
    #dx =  L / N
    #dt = dx / 100.

    # Initial Conditions

    #u = zeros((N,N)) # velocity in x direction
    #v = zeros((N,N)) # velocity in y direction

    #h_ini = 1.
    #h = h_ini * ones((N,N)) # pressure deviation (like height)
    #x,y = mgrid[:N,:N]

    time = 0

    plt = []
    fig = []


    def __init__(self, x=[],y=[],h_ini = 1.,u=[],v = [], 
                 # distance (m), time interval (s)
                 dx=0.01,dt=0.0001, N=64,L=1., px=16, py=16, 
                 # Square of radius, height of water colunm 
                 R=64, Hp=0.1, g=1., b=0.): # How define no default argument before?


        # add a perturbation in pressure surface
        

        self.px, self.py = px, py
        self.R = R
        self.Hp = Hp

        

        # Physical parameters

        self.g = g
        self.b = b
        self.L=L
        self.N=N

        # limits for h,u,v
        
        self.dx=dx
        self.dt=dt
        
        self.x,self.y = mgrid[:self.N,:self.N]
        
        self.u=zeros((self.N,self.N))
        self.v=zeros((self.N,self.N))
        
        self.h_ini=h_ini
        
        self.h=self.h_ini * ones((self.N,self.N))
        
        rr = (self.x-px)**2 + (self.y-py)**2
        self.h[rr<R] = self.h_ini + Hp #set initial conditions
        
        self.lims = [(self.h_ini-self.Hp,self.h_ini+self.Hp),(-0.02,0.02),(-0.02,0.02)]
        
        

    def dxy(self, A, axis=0):
        """
        Compute derivative of array A using balanced finite differences
        Axis specifies direction of spatial derivative (d/dx or d/dy)
        dA[i]/dx =  (A[i+1] - A[i-1] )  / 2dx
        """
        return (roll(A, -1, axis) - roll(A, 1, axis)) / (self.dx*2.) # roll: shift the array axis=0 shift the horizontal axis

    def d_dx(self, A):
        return self.dxy(A,1)

    def d_dy(self, A):
        return self.dxy(A,0)


    def d_dt(self, h, u, v):
        """
        http://en.wikipedia.org/wiki/Shallow_water_equations#Non-conservative_form
        """
        for x in [h, u, v]: # type check
           assert isinstance(x, ndarray) and not isinstance(x, matrix)

        g,b,dx = self.g, self.b, self.dx

        du_dt = -g*self.d_dx(h) - b*u
        dv_dt = -g*self.d_dy(h) - b*v

        H = 0 #h.mean() - our definition of h includes this term
        dh_dt = -self.d_dx(u * (H+h)) - self.d_dy(v * (H+h))

        return dh_dt, du_dt, dv_dt


    def evolve(self):
        """
        Evolve state (h, u, v) forward in time using simple Euler method
        x_{N+1} = x_{N} +   dx/dt * d_t
        """

        dh_dt, du_dt, dv_dt = self.d_dt(self.h, self.u, self.v)
        dt = self.dt

        self.h += dh_dt * dt
        self.u += du_dt * dt
        self.v += dv_dt * dt
        self.time += dt

        return self.h, self.u, self.v
    


def simulation(R, Hp_hat, iteration_times, root_path, task_name):
    

    SW = shallow(N=128,px=72,py=80,R=R,Hp=Hp_hat/100,b=0.2)

    # chose a point (x,y) to check the evolution
    x, y = 10,10

    index = 0

    #SW.plot()
    u_vect=np.zeros(iteration_times)
    v_vect=np.zeros(iteration_times)
    h_vect=np.zeros(iteration_times)

    data_list = []
    for i in range(iteration_times):
        SW.evolve()
        u_vect[i]=SW.u[x][y]
        v_vect[i]=SW.v[x][y]
        h_vect[i]=SW.h[x][y]
        #SW.animate()

        if i % 100 == 0:

            U = np.expand_dims(SW.u, axis=0) # convert (128,128) to (1,128,128)
            V = np.expand_dims(SW.v, axis=0) # convert (128,128) to (1,128,128)
            H = np.expand_dims(SW.h, axis=0) # convert (128,128) to (1,128,128)
            
            data = [U, V, H] # aggregate 3 np arrays to one list
            
            data = np.concatenate(data, axis=0) # concatenate 3 np arrays by axis 0, here the shape is (3,128,128)

            data = np.expand_dims(data, axis=0) # convert (3,128,128) to (1,3,128,128)

            data_list.append(data) # append current data into a list


    final_npy = np.concatenate(data_list, axis=0) # concatenate all array in this list, there shape is (t,3,128,128)   

    save_dir = os.path.join(root_path, task_name)
    os.makedirs(save_dir, exist_ok=True)
    file_name = f'R_{R}_Hp_{Hp_hat}'
    save_path = os.path.join(save_dir, file_name)
    np.save(save_path, final_npy)
    

    
# R_list = [36, 49, 64, 72, 81, 90, 100, 110, 121, 132, 144, 160]
# Hp_list_hat = [x for x in range(2, 21)] # Here Hp_hat = Hp*100

# for R in tqdm(R_list): 
#     for Hp in tqdm(Hp_list_hat): 
#         simulation(R, Hp, root_path)

simulation(R, Hp, iteration_times, root_path, task_name)