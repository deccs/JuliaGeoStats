#!/usr/bin/env julia

include("krige.jl")

function gridpath( xdim, ydim )
    xrng = linspace( xdim... )
    yrng = linspace( ydim... )
    N = xdim[3] * ydim[3]
    idx = [1:N]
    shuffle!( idx )
    cell = Tuple[]
    loc  = Tuple[]
    for i in 1:xdim[3]
  	for j in 1:ydim[3]
	    push!( cell, (int(i),int(j)) )
	    push!( loc,  (xrng[i],yrng[j]) )
	end
    end
    cell = cell[idx]
    loc = loc[idx]
    return cell, loc
end

function sgs( data, model, hs, bw, xs, ys, pad )
    if ys == None
	ys = xs
    end
    xdim = ( minimum( data[:,1] )-pad, maximum( data[:,1] )+pad, xs )
    ydim = ( minimum( data[:,2] )-pad, maximum( data[:,2] )+pad, ys )
    cells, locs = gridpath( xdim, ydim )
    newdata = zeros( Float64, 1, 3 )
    M = zeros( Float64, xs, ys )
    N = length( cells )
    for t in 1:N
	cell, loc = cells[t], locs[t]
	kv = krige( data, model, hs, bw, loc, 4 )
	M[cell[1],cell[2]] = kv
	newdata[1,:] = [ loc[1], loc[2], kv ]
	data = vcat( data, newdata )
    end
    return M
end


function static_krige( P, covfct, u, N )

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
