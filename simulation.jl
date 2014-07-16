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
