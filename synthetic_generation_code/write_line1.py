import multiprocessing #:)
import os
import matplotlib.pyplot as plt
import numpy as np
from utils import interpolate, one_shot
from datetime import datetime
from examples.seismic import Model, plot_velocity,TimeAxis, RickerSource, Receiver
from devito import Operator,TimeFunction,Eq, solve
from utils_source import RickerSource_MinimumPhase
from os.path import exists
from functools import partial



#vel_filename =r'V:\PROMAX\Vemund\PhD\2_data_synthetic\realistic_vels_line.sgy'
vel_filename ='./vel/survey1-line1.sgy'

savefolder='wline1'

if os.path.exists(savefolder)==False:
    os.mkdir(savefolder)


dx=1.
dz=1.
v_interp = interpolate(vel_filename, dx, dz)/1000
v_interp = v_interp[100:v_interp.shape[0]-200,:1000]
nx,nz = v_interp.shape

for i in range(75,v_interp.shape[1]):
    v_interp[:,i]=v_interp[:,75]

#%%
# define survey 

dr = 12.5
ds = 37.5

nx_physical = nx*dx
nz_physical = nz*dz
n_receivers = 642



source_matrix = np.arange(1500+322*dr,nx_physical-322*dr-1500, ds)

receiver_vector=np.arange(1500+dr, dr*(n_receivers+1)+1500, dr)
receiver_matrix=np.empty((source_matrix.shape[0],receiver_vector.shape[0]))

for i in range(source_matrix.shape[0]):
    receiver_matrix[i]=receiver_vector+i*ds


#%%

spacing = (dx, dz)  # Grid spacing in m. The domain size is now 1km by 1km
x_range = np.arange(0,nx_physical,dx)

#%%

# Multiprocessing :)
num_proc = multiprocessing.cpu_count() # use all processors
num_proc = 30                          # specify number to use (to be nice)
p = multiprocessing.Pool(num_proc)
p.map(partial(one_shot, receiver_matrix, dx, v_interp, x_range, spacing, source_matrix, n_receivers, savefolder), range(0,source_matrix.shape[0],1))
p.close()

#%%





















