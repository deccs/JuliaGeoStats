# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import scipy.stats
import scipy.optimize
import scipy.interpolate
from krige import *
from utilities import *
import random

# <codecell>

clstr = read_data( 'cluster.dat' )

# <codecell>

clstr[:10]

# <codecell>

# data
nx = 50 ; xsteps = 30
ny = 50 ; ysteps = 30
xrng = np.linspace(0,nx,num=xsteps)
yrng = np.linspace(0,ny,num=ysteps)

# <codecell>

# path
N = nx * ny
idx = np.arange( N )
random.shuffle( idx )
path = list()
t = 0
for i in range( xsteps ):
    for j in range( ysteps ):
        path.append( [ idx[t], (i,j), (xrng[i],yrng[j]) ] )
        t += 1
path.sort()

# <codecell>

locations = np.array( clstr[['Xlocation','Ylocation']] )
variable = np.array( clstr['Primary'] )
norm, inv, param, mu, sd = to_norm( variable, 1000 )
data = np.vstack((locations.T,norm)).T
#@jit
def sgs( data, bw, path, xsteps, ysteps ):
    M = np.zeros((xsteps,ysteps))
    # lags
    hs = np.arange(0,50,bw)
    for step in path :
        idx, cell, loc = step
        # create the model
        model = SphericalModel( data, hs, bw ).fit()
        kv = krige( data, model, hs, bw, loc, 4 )
        M[cell[0],cell[1]] = kv
        newdata = [ loc[0], loc[1], kv ]
        # add kv to `data`
        data = np.vstack(( data, newdata ))
    return M

# <codecell>

import time

# <codecell>

t0 = time.time()
M = sgs( data, 5, path, xsteps, ysteps )
t1 = time.time()
print (t1-t0)/60.0

# <codecell>

matshow( M[:,::-1].T, cmap=YPcmap )
colorbar();
xs, ys = str(xsteps), str(ysteps)
xlabel('X') ; ylabel('Y') ; title('Raw SGS '+xs+'x'+ys+'\n')
savefig( 'sgs_untransformed_'+xs+'_by_'+ys+'.png', fmt='png', dpi=200 )

# <codecell>

z = from_norm( M.ravel(), inv, param, mu, sd )
z = np.reshape( z, ( xsteps, ysteps ) )
matshow( z[:,::-1].T, cmap=YPcmap )
colorbar() ;
xs, ys = str(xsteps), str(ysteps)
xlabel('X') ; ylabel('Y') ; title('SGS '+xs+'x'+ys+'\n')
savefig( 'sgs_transformed_'+xs+'_by_'+ys+'.png', fmt='png', dpi=200 )

# <codecell>
