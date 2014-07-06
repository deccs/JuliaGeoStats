#!/usr/bin/env python

import pandas, scipy
import numpy as np
import scipy.stats
import matplotlib

def readGeoEAS( fn ):
    '''
    Input:  (fn)   filename describing a GeoEAS file
    Output: (data) NumPy array
    --------------------------------------------------
    Read GeoEAS files as described by the GSLIB manual
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
    # df = pandas.DataFrame( data, columns=columns,  )
    return data
    
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
    # the number of data points
    N = len(d)
    # the number of data points in each bin,
    # and the edges for each bin
    counts, intervals = np.histogram( d, bins=bins )
    # find the midpoints for each bin
    h = np.diff( intervals ) / 2.0
    # iitialize arrays for a mapping and it's inverse
    f, finv = np.zeros((N,2)), np.zeros((N,2))
    # initialize some counters and a scalar T (total)
    idx, k, T = 0, 0, float( np.sum( counts ) )
    # for each bin..
    for count in counts:
		# for each data point in that bin..
        for i in range( count ):
			# let x be the midpoint of that bin
            x = intervals[idx]+h[0]
            # let y represent the percentage of the
            # distriution we have covered so far 
            y = np.cumsum( counts[:idx+1] )[-1] / T
            # f : input value --> output percentage describing
            # the number of points less than the input scalar
            # in the modeled distribution; if 5 is 20% into
            # the distribution, then f[5] = 0.20
            f[k,:] = x, y
            # inverse of f
            # finv : input percentage --> output value that 
            # represents the input percentage point of the
            # distribution; if 5 is 20% into the distribution,
            # then finv[0.20] = 5
            finv[k,:] = y, x
            # increment k
            k += 1
        # increment idx
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
    return z, inv, mu, sd

# transform data from normal dist back
def from_norm( data, inv, mu, sd ):
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
