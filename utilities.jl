#!/usr/bin/env julia

function readGeoEAS( fn )
    f = open( fn, "r" )
    title = readline( f )
    nvar = int( readline( f ) )
    columns = [ strip( readline( f ) ) for i in 1:nvar ]
    tmp = ASCIIString[]
    while !eof( f )
        push!( tmp, readline( f ) )
    end
    tmp = [ split( i ) for i in tmp ]
	M = size( tmp, 1 )
	N = size( tmp[1], 1 )
	data = zeros( Float64, M, N )
	for i in 1:M
	    for j in 1:N
	        data[i,j] = float( tmp[i][j] )
	    end
	end
	return data
end
