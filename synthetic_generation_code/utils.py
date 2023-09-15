# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 14:42:50 2022

@author: vsthorkildsen
"""

import numpy as np
import segyio
from scipy.interpolate import interp2d, NearestNDInterpolator
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
#from examples.seismic import Model, plot_velocity,TimeAxis, Receiver
#from devito import Operator,TimeFunction,Eq, solve
from utils_source import Ormsby

#%%

def interpolate(filename,dx_out,dz_out, mode='nearest'):
    '''
    This function interpolates a given 2D velocity field. 
    Parameters
    ----------
    filename : .segy or .sgy file
        .segy file of the 2D velocity field. This function does not work for 3D
    dx_out : float
        Lateral sampling of the output velocity field.
    dz_out : float
        Vertical sampling of the output velocity field.
    mode : string, optional
        String specifying the mode of interpolation. Available options are 
        'linear' and 'nearest'. The default is 'nearest'. Note that nearest
        neighbour interpolation will preserve the sharp boundaries better than 
        linear interpolation

    Returns
    -------
    numpy array
        Interpolated velocity field.

    '''
    # load velocity file an some header values
    with segyio.open(filename, ignore_geometry=True) as f:
        dz = segyio.tools.dt(f) / 1000
        z_range = f.samples
        cdpX = f.attributes(segyio.TraceField.CDP_X)[:]/100
        cdpY = f.attributes(segyio.TraceField.CDP_Y)[:]/100
        velocity=f.trace.raw[:]
    
    dx=np.sqrt((cdpX[0]-cdpX[1])**2+(cdpY[0]-cdpY[1])**2) # input lateral sampling
    x_out = np.arange(0, (velocity.shape[0]-1)*dx, dx_out) # ouput lateral grid
    z_out = np.arange(0, z_range[-1], dz_out) # output vertical grid
    
    x_range=np.linspace(0,(velocity.shape[0]-1)*dx,velocity.shape[0])
    
    
    if mode=='linear':
        print('linear interpolation')
        velocity_interpolated = interp2d(x_range, z_range, velocity.T)
        velocity_out = velocity_interpolated(x_out, z_out)
    
    if mode=='nearest':
        print('nearest neighbour interpolation')
        x0,z0=np.meshgrid(x_range,z_range)
        x = x0.ravel()
        y = z0.ravel()
        z = velocity.T.ravel()
        X, Y = np.meshgrid(x_out, z_out)  # 2D grid for interpolation
        interp = NearestNDInterpolator(list(zip(x, y)), z)
        velocity_out = interp(X, Y)
    
    
    return velocity_out.T


#%%

def interpolate_source(file_path, dt_orig, dt_out):    
    '''
    This function is not used. 

    Parameters
    ----------
    file_path : .txt file
        file path for real source signature.
    dt_orig : float
        original time sampling.
    dt_out : float
        output time sampling.

    Returns
    -------
    source_out : TYPE
        DESCRIPTION.

    '''
    source=np.loadtxt(file_path)
    dt=dt_orig
    original_time=np.arange(0,(source.shape[0])*dt,dt)
    dtout=dt_out
    timeout=np.arange(0,(source.shape[0])*dt+dtout,dtout)
    interpolated_source=interp1d(original_time, source, bounds_error=False, fill_value=0.)
    source_out=interpolated_source(timeout)
    return source_out


#%%

def one_shot(receiver_matrix, dx, v_interp, x_range, spacing, source_matrix, n_receivers, savefolder, i):
    '''
    

    Parameters
    ----------
    receiver_matrix : numpy matrix
        receiver locations for ALL shots. The script will extract the correct 
        receiver coordinates (parameter i).
    dx : float
        spatial samling of the velocity field.
    v_interp : numpy matrix
        Interpolated velocity field. The script will automatically extract an area around the active receivers (1000m)
        IMPORTANT: velocity is given in km/s 
    x_range : numpy array
        full lateral range of interpolated velocity field.
    spacing : tuple 
        lateral and vertical sampling of velocity field.
    source_matrix : numpy array
        source locations for ALL shots. The script will extract the correct 
        source coordinates (parameter i).
    n_receivers : int
        number of receivers. Can be removed by using receiver_matrix.shape[0] instead
    savefolder : string
        folder to save shots.
    i : int
        shot number.

    Returns
    -------
    None. 

    '''
    
    # make vector for the active part of the velocity model
    min_x = receiver_matrix[i,0]-1000
    max_x = receiver_matrix[i,-1]+1000
    x_vec=np.arange(min_x,max_x,dx)
    
    # extract velocity and calculate origin of active area
    v_subsampled = np.zeros((x_vec.shape[0],v_interp.shape[1]))
    indexes_in_full_vel = np.where((x_range>min_x)&(x_range<max_x))
    origin = (x_range[indexes_in_full_vel[0][0]], 0.)
    v_subsampled=np.squeeze(v_interp[indexes_in_full_vel,:])
    shape = (v_subsampled.shape[0], v_subsampled.shape[1])  # Number of grid point (nx, nz)
    v = np.float32(v_subsampled)
    
    
    
    
    
    # With the velocity and model size defined, we can create the seismic model that
    # encapsulates this properties. We also define the size of the absorbing layer
    # as 1000 grid points, but with a free surface (fs=True)
    model = Model(vp=v, origin=origin, shape=shape, spacing=spacing,
                  space_order=2, nbl=1000, bcs="damp",fs=True)
    
    
    
    
    
    t0 = 0.     # Simulation starts a t=0
    tn = 1000.  # Simulation last 1 second (1000 ms)
    dt = 0.25   # time sampling of simulation (0.25 ms). It is important to have
                # a smaller time sampling in the modelling than in the output seismic  
    
    time_range = TimeAxis(start=t0, stop=tn, step=dt)   # define object with time sampling
    
    
    # define an ormsby wavelet 
    f0 =[1,4,100,200] 
    src = Ormsby(name='src', grid=model.grid, f0=f0, npoint=1, time_range=time_range)
    
    
    
    
    src.coordinates.data[0, :] = source_matrix[i]
    src.coordinates.data[0, -1] = 15.  # Depth is 15m
    
    rec = Receiver(name='rec', grid=model.grid, npoint=n_receivers, time_range=time_range)
    rec.coordinates.data[:, 0] = receiver_matrix[i,:]
    rec.coordinates.data[:, 1] = 15.  # Depth is 15m
    
    
    # Define the wavefield with the size of the model and the time dimension
    u = TimeFunction(name="u", grid=model.grid, time_order=2, space_order=2)
    
    # We can now write the PDE
    pde = model.m * u.dt2 - u.laplace + model.damp * u.dt
    
    # This discrete PDE can be solved in a time-marching way updating u(t+dt) from the previous time step
    # Devito as a shortcut for u(t+dt) which is u.forward. We can then rewrite the PDE as 
    # a time marching updating equation known as a stencil using customized SymPy functions
    
    stencil = Eq(u.forward, solve(pde, u.forward))
    stencil
    
    
    
    # Finally we define the source injection and receiver read function to generate the corresponding code
    src_term = src.inject(field=u.forward, expr=src * dt**2 / model.m)
    
    # Create interpolation expression for receivers
    rec_term = rec.interpolate(expr=u.forward)
    
    
    
    
    op = Operator([stencil] + src_term + rec_term, subs=model.spacing_map)
    op(time=time_range.num-1, dt=dt)
    
    
    
    # finally we downsample the output data to 2ms from the modelling sample interval of 0.25ms
    np.save(savefolder+'/shot_number_'+str(i),rec.data[::8,:]) #save shot gather
