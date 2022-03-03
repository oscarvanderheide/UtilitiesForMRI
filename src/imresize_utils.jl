# Image resizing utilities

export anti_aliasing_filter

function anti_aliasing_filter(n::Integer; fact::Integer=1, T::DataType=Float32)
    kmax = T(pi)
    k = kmax*coord_norm(n)
    filt = zeros(T, n)
    i1 = abs.(k) .<= kmax/2^fact-kmax/2^(fact+1)
    i2 = (abs.(k) .> kmax/2^fact-kmax/2^(fact+1)) .&& (abs.(k) .<= kmax/2^fact+kmax/2^(fact+1))
    filt[i1] .= 1
    filt[i2] .= (cos.(range(-T(pi), T(pi); length=length(findall(i2)))).+1)/2
    return filt
end

function anti_aliasing_filter(n::NTuple{3,Integer}; fact::Integer=1, T::DataType=Float32)
    filt_x = reshape(anti_aliasing_filter(n[1]; fact=fact, T=T), :, 1, 1)
    filt_y = reshape(anti_aliasing_filter(n[2]; fact=fact, T=T), 1, :, 1)
    filt_z = reshape(anti_aliasing_filter(n[3]; fact=fact, T=T), 1, 1, :)
    return filt_x.*filt_y.*filt_z
end

anti_aliasing_filter(X::RegularCartesianSpatialSampling{T}; fact::Integer=1) where {T<:Real} = anti_aliasing_filter(X.n; fact=fact, T=T)

function downscale(u::AbstractArray{CT,3}, X::RegularCartesianSpatialSampling{T}; fact::Integer=1) where {T<:Real,CT<:RealOrComplex{T}}
    x, y, z = coord(X)
    itp = interpolate((vec(x), vec(y), vec(z)), u, Gridded(Linear()))
    u_ = ifft(ifftshift(anti_aliasing_filter(X; fact=fact)).*fft(u))
    X_h = downscale(X; fact=fact)
    u_h = similar(u_, size(X_h))
    x_h, y_h, z_h = coord(X_h)
    @inbounds for i = 1:X_h.n[1], j = 1:X_h.n[2], k = 1:X_h.n[3]
        u_h[i, j, k] = itp(x_h[i], y_h[j], z_h[k])
    end
    return u_h
end

function upscale(u::AbstractArray{CT,3}, X::RegularCartesianSpatialSampling{T}; fact::Integer=1) where {T<:Real,CT<:RealOrComplex{T}}
    x, y, z = coord(X)
    itp = extrapolate(interpolate((vec(x), vec(y), vec(z)), u, Gridded(Linear())), T(0))
    X_h = upscale(X; fact=fact)
    u_h = similar(u, size(X_h))
    x_h, y_h, z_h = coord(X_h)
    @inbounds for i = 1:X_h.n[1], j = 1:X_h.n[2], k = 1:X_h.n[3]
        u_h[i, j, k] = itp(x_h[i], y_h[j], z_h[k])
    end
    return u_h
end