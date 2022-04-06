# k-space trajectory utilities

export KSpaceCartesianSampling
export kspace_sampling, kspace_Cartesian_sampling, downscale, downscale_phase_encode_index


## k-space trajectory

coord(K::AbstractKSpaceSampling) = K.K

struct KSpaceFixedSizeSampling{T}<:AbstractKSpaceFixedSizeSampling{T}
    K::AbstractArray{T,3} # Array (nt,nk,3)
end

Base.getindex(K::AbstractKSpaceFixedSizeSampling, t::Integer) = K.K[t,:,:]
Base.size(K::AbstractKSpaceFixedSizeSampling) = size(K.K)[1:2]

kspace_sampling(K::AbstractArray{T,3}) where {T<:Real} = KSpaceFixedSizeSampling{T}(K)

struct KSpaceCartesianSampling{T}<:AbstractKSpaceCartesianSampling{T}
    X::RegularCartesianSpatialSampling{T}
    K::AbstractArray{T,3} # Array (nt,nk,3)
    phase_encoding::NTuple{2,Integer}
    readout::Integer
    subsampling::AbstractVector{<:Integer}
end

function kspace_Cartesian_sampling(X::RegularCartesianSpatialSampling{T}; phase_encoding::NTuple{2,Integer}=(1,2), subsampling::Union{Nothing,AbstractVector{<:Integer}}=nothing) where {T<:Real}

    # Set phase-encode/readout dimensions
    nx, ny, nz = X.n
    hx, hy, hz = X.h
    ((phase_encoding == (1,2)) || (phase_encoding == (2,1))) && (readout = 3)
    ((phase_encoding == (1,3)) || (phase_encoding == (3,1))) && (readout = 2)
    ((phase_encoding == (2,3)) || (phase_encoding == (3,2))) && (readout = 1)
    nt = (X.n[phase_encoding[1]], X.n[phase_encoding[2]]); nk = X.n[readout]
    perm = (phase_encoding..., readout)

    # Full k-space grid
    kx = T(pi)/hx*collect(coord_norm(nx))
    ky = T(pi)/hy*collect(coord_norm(ny))
    kz = T(pi)/hz*collect(coord_norm(nz))
    Kx = reshape(permutedims(repeat(reshape(kx,:,1,1); outer=(1,ny,nz)), perm), prod(nt), nk)
    Ky = reshape(permutedims(repeat(reshape(ky,1,:,1); outer=(nx,1,nz)), perm), prod(nt), nk)
    Kz = reshape(permutedims(repeat(reshape(kz,1,1,:); outer=(nx,ny,1)), perm), prod(nt), nk)
    K = cat(Kx, Ky, Kz; dims=3)

    # Subsampling and reordering
    isnothing(subsampling) && (subsampling = 1:prod(nt))
    K = K[subsampling, :, :]

    return KSpaceCartesianSampling{T}(X, K, phase_encoding, readout, subsampling)

end

function coord_norm(n::Integer)
    (mod(n,2) == 0) ? (c = -div(n, 2):div(n, 2)-1) : (c = -div(n-1, 2):div(n-1, 2))
    return c/norm(c, Inf)
end

function downscale(K::KSpaceCartesianSampling{T}; fact::Integer=1) where {T<:Real}
    (fact == 0) && (return K)
    X_h = downscale(K.X; fact=fact)
    phase_encode_idx_h, readout_idx_h = downscale_phase_encode_index(K; fact=fact, readout=true)
    K_h = K.K[phase_encode_idx_h, readout_idx_h, :]
    return KSpaceCartesianSampling{T}(X_h, K_h, K.phase_encoding, K.readout, K.subsampling)
end

function downscale_phase_encode_index(K::KSpaceCartesianSampling{T}; fact::Integer=1, readout::Bool=false) where {T<:Real}
    k_max = T(pi)./K.X.h
    pe = K.phase_encoding
    r = K.readout
    phase_encode_idx_h = findall((K.K[:,1,pe[1]] .< k_max[pe[1]]/2^fact) .&& (K.K[:,1,pe[1]] .>= -k_max[pe[1]]/2^fact) .&& (K.K[:,1,pe[2]] .< k_max[pe[2]]/2^fact) .&& (K.K[:,1,pe[2]] .>= -k_max[pe[2]]/2^fact))
    readout_idx_h = findall((K.K[1,:,r] .< k_max[r]/2^fact) .&& (K.K[1,:,r] .>= -k_max[r]/2^fact))
    readout ? (return (phase_encode_idx_h, readout_idx_h)) : (return phase_encode_idx_h)
end

function downscale(d::AbstractArray{CT,2}, K::KSpaceCartesianSampling{T}; fact::Integer=1, flat::Bool=false, coeff::Real=1) where {T<:Real,CT<:RealOrComplex{T}}
    (fact == 0) && (return d)
    i_pe, i_r = downscale_phase_encode_index(K; fact=fact, readout=true)
    return anti_aliasing_filter(K; fact=fact, flat=flat, coeff=coeff).*d[i_pe, i_r]/T(sqrt(2.0^(3*fact)))
end