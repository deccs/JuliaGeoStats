#!/usr/bin/env julia

using Distance

include("utilities.jl")

function SVh( P, h, bw )
    # calculate the pairwise distances
    pd = pairwise( Euclidean(), P[:,1:2]' )
    # get the dimensions of P
    rows, cols = size( P )
    # create an array for the distances
    Z = Float64[]
    # for each pair of data..
    for i in 1:rows
        for j in i+1:rows
            if( pd[i,j] >= h-bw )&&( pd[i,j] <= h+bw )
                # if the two points are within the right 
                # distance of each other, push the squared 
                # difference of the variable into Z
                push!( Z, ( P[i,3] - P[j,3] )^2.0 )
            end
        end
    end
    # return half of the average squared deviation
    return sum( Z ) / ( 2.0 * length( Z ) )
end

function SV( P, hs, bw )
    sv = Float64[]
    hx = Float64[]
    # for an array of distances, h out of hs..
    for h in hs
        # calculate the semivariogram at a specific distance h
        s = SVh( P, h, bw )
        # if the value of the semivariogram is not NaN
        if s > 0
            # push the semivariogram value into sv
            push!( sv, s )
            # push the distance h into hx
            push!( hx, h )
        end
    end
    # concatenate hx and sv
    sv = hcat( hx, sv )
    return sv
end

function C( P, h, bw )
    c0 = var( P[:,3] )
    if h == 0
        return c0
    end
    return c0 - SVh( P, h, bw )
end

function opt( fct, x, y, C0, parameterRange=None, meshSize=1000 )
    if parameterRange == None
        parameterRange = ( x[2], x[end] )
    end
    mse = zeros( meshSize )
    a = linspace( parameterRange[1], parameterRange[2], meshSize )
    @parallel for i in 1:meshSize
        mse[i] = mean( ( y - fct( x, a[i], C0 ) ).^2.0 )
    end
    # indmin() == argmin()
    return a[ indmin( mse ) ]
end

@everywhere function spherical( h, a, C0 )
    if typeof( h ) == Float64
        if h <= a
            return C0*( 1.5*h/a - 0.5*(h/a)^3.0 )
        else
            return C0
        end
    else
        a = ones( length( h ) ) * a
        C0 = ones( length( h ) ) * C0
        return map( spherical, h, a, C0 )
    end
end

function cvmodel( P, model, hs, bw )
    sv = SV( P, hs, bw )
    C0 = C( P, hs[1], bw )
    param = opt( model, sv[:,1], sv[:,2], C0 )
    covfct = h -> C0 - model( h, param, C0 )
    return covfct
end

function krige( P, model, hs, bw, u, N )

    # calculate the covariance function
    covfct = cvmodel( P, model, hs, bw )

    # mean of the variable
    mu = mean( P[:,3] )
    
    # distance between u and each data point in P
    d = sqrt( ( P[:,1]-u[1] ).^2.0 + ( P[:,2]-u[2] ).^2.0 )
    d = reshape( d, size( P, 1 ), 1 )
    
    # add these distances to P
    P = hcat( P, d )
    
    # sort P by these distances
    # take the first N of them
    d = reshape( d, size(P,1) )
    s = sortperm( d )
    P = P[ s[1:N], 1:size(P,2) ]

    # apply the covariance model to the distances
    k = covfct( P[:,4] )
    println( k )
    
    # form a matrix of distances between existing data points
    K = pairwise( Euclidean(), P[:,1:2]' )
    # apply the covariance model to these distances
    N, N = size( K )
    K = reshape( K, N*N )
    K = covfct( K )
    
    # reshape into an array
    K = reshape( K, N, N )

    # calculate the kriging weights
    weights = inv( K ) * k

    # calculate the residuals
    residuals = P[:,3] - mu

    # calculate the estimation
    estimation = weights' * residuals + mu
    
    return estimation[1]
end
