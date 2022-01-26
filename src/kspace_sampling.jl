# k-space trajectory utilities

export KSpaceOrderedSampling, KSpaceFixedSizeSampling
export kspace_sampling


## k-space trajectory type (general)

struct KSpaceOrderedSampling{T}<:AbstractKSpaceOrderedSampling{T}
    K::AbstractArray{T,2} # size(K) = (∑_t n_kt, 3)
    idx_t::AbstractVector{AbstractVector{<:Integer}}
end

function kspace_sampling(K::AbstractArray{KT,1}) where {T<:Real, KT<:AbstractArray{T,2}}
    nt = length(K)
    nk_t = Vector{Int64}(undef, nt)
    idx_t = Vector{Vector{Int64}}(undef, nt)
    ntot = 0
    @inbounds for t = 1:nt
        nk_t[t]  = size(K[t], 1)
        idx_t[t] = ntot+1:ntot+nk_t[t]
        ntot += nk_t[t]
    end
    K_arr = similar(K[1], sum(nk_t), 3)
    @inbounds for t = 1:nt
        K_arr[idx_t[t], :] = K[t]
    end
    return KSpaceOrderedSampling{T}(K_arr, idx_t)
end

Base.getindex(K::KSpaceOrderedSampling, t::Integer) = K[K.idx_t[t]]

Base.size(K::KSpaceOrderedSampling) = sum(length.(K.idx_t))

coord(K::KSpaceOrderedSampling) = K.K


## k-space trajectory type (fixed size)

struct KSpaceFixedSizeSampling{T}<:AbstractKSpaceOrderedSampling{T}
    K::AbstractArray{T,3} # size(K) = (nt, nk, 3)
end

Base.getindex(K::KSpaceFixedSizeSampling, t::Integer) = K.K[t,:,:]

Base.size(K::KSpaceFixedSizeSampling) = size(K.K)[1:2]

coord(K::KSpaceFixedSizeSampling) = K.K

kspace_sampling(K::AbstractArray{T,3}) where {T<:Real} = KSpaceFixedSizeSampling{T}(K)

function kspace_sampling(X::RegularCartesianSpatialSampling{T}; phase_encode::Symbol=:xy, readout::Symbol=:z) where {T<:Real}

    # Input check
    (readout == :x) && ((phase_encode == :xy) || (phase_encode == :yx) || (phase_encode == :xz) || (phase_encode == :zx)) && error("Incompatible readout/phase-encode dimensions")
    (readout == :y) && ((phase_encode == :xy) || (phase_encode == :yx) || (phase_encode == :yz) || (phase_encode == :zy)) && error("Incompatible readout/phase-encode dimensions")
    (readout == :z) && ((phase_encode == :xz) || (phase_encode == :zx) || (phase_encode == :yz) || (phase_encode == :zy)) && error("Incompatible readout/phase-encode dimensions")

    # Set phase-encode/readout dimensions
    nx, ny, nz = X.n
    hx, hy, hz = X.h
    (readout == :x) && (nk = nx; pk = 1)
    (readout == :y) && (nk = ny; pk = 2)
    (readout == :z) && (nk = nz; pk = 3)
    (phase_encode == :xy) && (nt = (nx, ny); pt = (1, 2))
    (phase_encode == :yx) && (nt = (ny, nx); pt = (2, 1))
    (phase_encode == :xz) && (nt = (nx, nz); pt = (1, 3))
    (phase_encode == :zx) && (nt = (nz, nx); pt = (3, 1))
    (phase_encode == :yz) && (nt = (ny, nz); pt = (2, 3))
    (phase_encode == :zy) && (nt = (nz, ny); pt = (3, 2))
    perm = (pt..., pk)

    # Mesh k-space grid
    kx = T(pi)/hx*collect(coord_norm(nx))
    ky = T(pi)/hy*collect(coord_norm(ny))
    kz = T(pi)/hz*collect(coord_norm(nz))
    Kx = reshape(permutedims(repeat(reshape(kx,:,1,1); outer=(1,ny,nz)), perm), prod(nt), nk)
    Ky = reshape(permutedims(repeat(reshape(ky,1,:,1); outer=(nx,1,nz)), perm), prod(nt), nk)
    Kz = reshape(permutedims(repeat(reshape(kz,1,1,:); outer=(nx,ny,1)), perm), prod(nt), nk)
    K = cat(Kx, Ky, Kz; dims=3)

    return KSpaceFixedSizeSampling{T}(K)

end

function coord_norm(n::Integer)
    (mod(n,2) == 0) ? (c = -div(n, 2):div(n, 2)-1) : (c = -div(n-1, 2):div(n-1, 2))
    return c/norm(c, Inf)
end