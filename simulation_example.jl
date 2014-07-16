#!/usr/bin/env julia

include("utilities,jl")
include("simulation.jl")

P = readGeoEAS("cluster.dat")

P = P[:,1:3]

hs, bw = linspace( 10, 50, 5 ), 5.0

xs, yx = 45, 45

pad = 0

M = sgs( P, spherical, hs, bw, xs, ys, pad )

print( M )
