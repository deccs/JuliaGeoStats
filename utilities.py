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
    
def cdf( d ):
    '''
    Input:  (d)    iterable, a data set
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
    N = float( len( d ) )
    # sorted array of data points
    xs = np.sort( d )
    # array of unique data points
    xu = np.unique( xs )
    # number of unique data points
    U = len( xu )
    # initialize an array of U zeros
    cdf = np.zeros((U))
    # for each unique data point..
    for i in range( U ):
        # count the number of points less than
        # this point, and then divide by the
        # total number of data points
    	cdf[i] = len( xs[ xs < xu[i] ] ) / N
    # f : input value --> output percentage describing
    # the number of points less than the input scalar
    # in the modeled distribution; if 5 is 20% into
    # the distribution, then f[5] = 0.20
    f = np.vstack((xu,cdf)).T
    # inverse of f
    # finv : input percentage --> output value that 
    # represents the input percentage point of the
    # distribution; if 5 is 20% into the distribution,
    # then finv[0.20] = 5
    finv = np.fliplr(f)
    return f, finv
    
   
def fit( d ):
    '''
    Input:  (d) NumPy array with two columns,
                a domain and range of a mapping
    Output: (f) function that interpolates the mapping d
    ----------------------------------------------------
    This takes a mapping and returns a function that
    interpolates values that are missing in the original
    mapping, and maps values outside the range* of the
    domain (d) to the maximum or minimum values of the
    range** of (d), respectively.
    ----------------------------------------------------
    *range - here, I mean "maximum minus minimum"
    **range - here I mean the output of a mapping
    '''
    x, y = d[:,0], d[:,1]
    def f(t):
        # return the minimum of the range
        if t <= x.min():
            return y[ np.argmin(x) ]
        # return the maximum of the range
        elif t >= x.max():
            return y[ np.argmax(x) ]
        # otherwise, interpolate linearly
        else:
            intr = scipy.interpolate.interp1d( x, y )
            return intr(t)
    return f
    
def to_norm( z ):
    '''
    Input  (z)   1D NumPy array of observational data
    Output (z)   1D NumPy array of z-score transformed data
           (inv) inverse mapping to retrieve original distribution
    '''
    N = len( z )
    # grab the cumulative distribution function
    f, inv = cdf( z )
    # h will return the cdf of z
    # by interpolating the mapping f
    h = fit( f )
    # ppf will return the inverse cdf
    # of the standard normal distribution
    ppf = scipy.stats.norm(0,1).ppf
    # for each data point..
    for i in range( N ):
        # h takes z to a value in [0,1]
        p = h( z[i] )
        # ppf takes p (in [0,1]) to a z-score
        z[i] = ppf( p )
    # convert positive infinite values
    posinf = np.isposinf( z )
    z = np.where( posinf, np.nan, z )
    z = np.where( np.isnan( z ), np.nanmax( z ), z )
    # convert negative infinite values
    neginf = np.isneginf( z )
    z = np.where( neginf, np.nan, z )
    z = np.where( np.isnan( z ), np.nanmin( z ), z )
    return z, inv

def from_norm( data, inv ):
    '''
    Input:  (data) NumPy array of zscore data
            (inv)  mapping that takes zscore data back
                   to the original distribution
    Output: (z)    Data that should conform to the 
                   distribution of the original data
    '''
    # convert to a column vector
    d = data.ravel()
    # create an interpolating function 
    # for the inverse cdf, mapping zscores
    # back to the original data distribution
    h = fit( inv )
    # convert z-score data to cdf values in [0,1]
    f = scipy.stats.norm(0,1).cdf( d )
    # use inverse cdf to map [0,1] values to the
    # original distribution, then add the mu and sd
    z = np.array( [ h(i) for i in f ] )
    # reshape the data
    z = np.reshape( z, data.shape )
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
