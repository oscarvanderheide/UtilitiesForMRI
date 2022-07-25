# Image resizing utilities

export downscale, anti_aliasing_filter



function downscale(d::AbstractArray{CT,2}, K::KSpaceCartesianSampling{T}; fact::Integer=1, flat::Bool=false, coeff::Real=1) where {T<:Real,CT<:RealOrComplex{T}}
    (fact == 0) && (return d)
    i_pe, i_r = downscale_phase_encode_index(K; fact=fact, readout=true)
    return anti_aliasing_filter(K; fact=fact, flat=flat, coeff=coeff).*d[i_pe, i_r]/T(sqrt(2.0^(3*fact)))
end

function downscale(u::AbstractArray{CT,3}, X::RegularCartesianSpatialSampling{T}; fact::Integer=1, flat::Bool=false, coeff::Real=1) where {T<:Real,CT<:RealOrComplex{T}}
    (fact == 0) && (return u)
    x, y, z = coord(X)
    itp = interpolate((vec(x), vec(y), vec(z)), u, Gridded(Linear()))
    u_ = ifft(ifftshift(anti_aliasing_filter(X; fact=fact, flat=flat, coeff=coeff)).*fft(u))
    X_h = downscale(X; fact=fact)
    u_h = similar(u_, size(X_h))
    x_h, y_h, z_h = coord(X_h)
    @inbounds for i = 1:X_h.n[1], j = 1:X_h.n[2], k = 1:X_h.n[3]
        u_h[i, j, k] = itp(x_h[i], y_h[j], z_h[k])
    end
    return u_h
end

function upscale(u::AbstractArray{CT,3}, X::RegularCartesianSpatialSampling{T}; fact::Integer=1) where {T<:Real,CT<:RealOrComplex{T}}
    (fact == 0) && (return u)
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


## Filter

function anti_aliasing_filter(T::DataType=Float32, n::Integer; factor::Integer=1, flat::Bool=false, coeff::Real=1)
    kmax = T(pi)
    k = kmax*coord_norm(n)
    filt = zeros(T, n)
    if ~flat
        i1 = abs.(k) .<= kmax/2^fact-kmax/2^(fact+1)
        i2 = (abs.(k) .> kmax/2^fact-kmax/2^(fact+1)) .&& (abs.(k) .<= kmax/2^fact+kmax/2^(fact+1))
        filt[i2] .= ((cos.(range(-T(pi), T(pi); length=length(findall(i2)))).+1)/2).^T(coeff)
    else
        i1 = abs.(k) .<= kmax/2^fact
    end
    filt[i1] .= 1
    return filt
end

function anti_aliasing_filter(K::KSpaceCartesianSampling{T}; fact::Integer=1, flat::Bool=false, coeff::Real=1) where {T<:Real}
    nt, nk = size(K)
    perm = (K.phase_encoding..., K.readout)
    filt = reshape(permutedims(anti_aliasing_filter(K.X.n; fact=fact, flat=flat, coeff=coeff), perm), nt, nk)[K.subsampling,:]
    i_pe, i_r = downscale_phase_encode_index(K; fact=fact, readout=true)
    return filt[i_pe, i_r]
end

function anti_aliasing_filter(n::NTuple{3,Integer}; fact::Integer=1, flat::Bool=false, coeff::Real=1, T::DataType=Float32)
    filt_x = reshape(anti_aliasing_filter(n[1]; fact=fact, flat=flat, coeff=coeff, T=T), :, 1, 1)
    filt_y = reshape(anti_aliasing_filter(n[2]; fact=fact, flat=flat, coeff=coeff, T=T), 1, :, 1)
    filt_z = reshape(anti_aliasing_filter(n[3]; fact=fact, flat=flat, coeff=coeff, T=T), 1, 1, :)
    return filt_x.*filt_y.*filt_z
end

anti_aliasing_filter(X::RegularCartesianSpatialSampling{T}; fact::Integer=1, flat::Bool=false, coeff::Real=1) where {T<:Real} = anti_aliasing_filter(X.n; fact=fact, flat=flat, coeff=coeff, T=T)