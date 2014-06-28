#!/usr/bin/env python

import pandas, scipy
import numpy as np
import scipy.stats
import matplotlib

'''
This file contains the following functions

  - read_data( fn ) -> df 
    returns a data frame

  - cdf( d, bins=12 ) -> f, inv 
    returns two NumPy arrays describing the domain and range of the CDF function and its inverse

  - fit( d ) -> f 
    returns a function that interpolates a CDF from data returned by cdf()

  - to_norm( data, bins=12 ) -> z, inv, param, mu, sd 
    returns the z-score transformed data, and other statistics for the back transformation

  - from_norm( data, inv, param, mu, sd ) -> z 
    performs the back transfromation to take data from a z-score transfrom to its original state
'''

def read_data( fn ):
    '''
    Read files of the type described in the GSLIB manual
    '''
    f = open( fn, "r" )
    # title of the data set
    title = f.readline()
    # number of variables
    nvar = int( f.readline() )
    # variable names
    columns = [ f.readline().strip() for i in range( nvar ) ]
    # make a list for the data
    data = list()
    # for each line of the data
    while True:
        # read a line
        line = f.readline()
        # if that line is empty
        if line == '':
            # the jump out of the while loop
            break
        # otherwise, append that line to the data list
        else:
            data.append( line )
    # strip off the newlines, and split by whitespace
    data = [ i.strip().split() for i in data ]
    # turn a list of list of strings into an array of floats
    data = np.array( data, dtype=np.float )
    # combine the data with the variable names into a DataFrame
    df = pandas.DataFrame( data, columns=columns,  )
    return df
    
def cdf( d, bins=12 ):
    '''
    Input:  (d)    iterable, a data set
            (bins) granularity of CDF
    Output: (f)    NumPy array with (bins) rows and two columns
                   the first column are values in the range of (d)
                   the second column are CDF values
                   alternatively, think of the columns as the
                   domain and range of the CDF function
            (finv) inverse of (f)
    ---------------------------------------------------------
    Calculate the cumulative distribution function
    and its inverse for a given data set
    '''
    N = len(d)
    counts, intervals = np.histogram( d, bins=bins )
    h = np.diff( intervals ) / 2.0
    f, finv = np.zeros((N,2)), np.zeros((N,2))
    idx, k, T = 0, 0, float( np.sum( counts ) )
    for count in counts:
        for i in range( count ):
            x = intervals[idx]+h[0]
            y = np.cumsum( counts[:idx+1] )[-1] / T
            f[k,:] = x, y 
            finv[k,:] = y, x
            k += 1
        idx += 1
    return f, finv
    
def fit( d ):
    x, y = d[:,0], d[:,1]
    def f(t):
        if t <= x.min():
            return y[ np.argmin(x) ]
        elif t >= x.max():
            return y[ np.argmax(x) ]
        else:
            intr = scipy.interpolate.interp1d( x, y )
            return intr(t)
    return f
    
# transform data to normal dist
def to_norm( data, bins=12 ):
    mu = np.mean( data )
    sd = np.std( data )
    z = ( data - mu ) / sd
    f, inv = cdf( z, bins=bins )
    z = scipy.stats.norm(0,1).ppf( f[:,1] )
    z = np.where( z==np.inf, np.nan, z )
    z = np.where( np.isnan( z ), np.nanmax( z ), z )
    param = ( mu, sd )
    return z, inv, param, mu, sd

# transform data from normal dist back
def from_norm( data, inv, param, mu, sd ):
    h = fit( inv )
    f = scipy.stats.norm(0,1).cdf( data )
    z = [ h(i)*sd + mu for i in f ]
    return z

# this is a colormap that ranges from yellow to purple to black
cdict = {'red':   ((0.0, 1.0, 1.0),
                   (0.5, 225/255., 225/255. ),
                   (0.75, 0.141, 0.141 ),
                   (1.0, 0.0, 0.0)),
         'green': ((0.0, 1.0, 1.0),
                   (0.5, 57/255., 57/255. ),
                   (0.75, 0.0, 0.0 ),
                   (1.0, 0.0, 0.0)),
         'blue':  ((0.0, 0.376, 0.376),
                   (0.5, 198/255., 198/255. ),
                   (0.75, 1.0, 1.0 ),
                   (1.0, 0.0, 0.0)) }
YPcmap = matplotlib.colors.LinearSegmentedColormap('my_colormap',cdict,256)
