using Distance

function SVh( P, h, bw )
    # calculate the pairwise distances
    pd = pairwise( Euclidean(), P'[1:2,:] )
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

function opt( fct, x, y, C0, parameterRange=None, meshSize=10 )
    if parameterRange == None
        parameterRange = ( x[2], x[end] )
    end
    mse = zeros( meshSize )
    a = linspace( parameterRange[1], parameterRange[2], meshSize )
    for i in 1:meshSize
        mse[i] = mean( ( y - fct( x, a[i], C0 ) ).^2.0 )
    end
    # indmin() == argmin()
    return a[ indmin( a ) ]
end

function spherical( h, a, C0 )
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
    covfct = h -> model( h, param, C0 )
    return covfct
end
