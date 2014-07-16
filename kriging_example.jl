#!/usr/bin/env julia

P = readGeoEAS("cluster.dat")

P = P[:,1:3]

hs = linspace( 10, 50, 5 )

bw = 5.0

u = [ 25.0, 25.0 ]

k = krige( P[:,1:3], spherical, hs, bw, u, 4 )

println( k )
