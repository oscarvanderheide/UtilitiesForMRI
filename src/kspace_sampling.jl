# k-space trajectory utilities

export KSpaceOrderedSampling, KSpaceCartesianSampling, KSpaceCartesianSamplingRBM
export kspace_ordered_sampling, kspace_sampling, phase_shift_orig, phase_shift


## k-space trajectory type (general)

struct KSpaceOrderedSampling{T}<:AbstractKSpaceOrderedSampling{T}
    X::AbstractCartesianSpatialSampling{T}
    K::AbstractArray{AbstractArray{T,2},1} # size(k[t]) = (n_kt, 3), k[t] = [kx ky kz]
end

kspace_ordered_sampling(X::AbstractCartesianSpatialSampling{T}, K::AbstractArray{AbstractArray{T,2},1}; o::NTuple{3,T}=(T(0),T(0),T(0))) where {T<:Real} = KSpaceOrderedSampling{T}(K, o)

Base.getindex(K::KSpaceOrderedSampling, i::Integer) = K.K[i]


## k-space trajectory type (fixed size)

Base.size(K::AbstractKSpaceFixedSizeSampling) = size(K.K)
Base.getindex(K::AbstractKSpaceFixedSizeSampling, t) = (K.K[:,t,1], K.K[:,t,2], K.K[:,t,3])


## k-space trajectory type (Cartesian)

struct KSpaceCartesianSampling{T}<:AbstractKSpaceFixedSizeSampling{T}
    X::RegularCartesianSpatialSampling{T}
    K::AbstractArray{T,3} # size(K) = (nk, nt, 3)
    phase_shift::Union{Nothing,AbstractArray{Complex{T},2}} # size(phase_shift) = (nk, nt)
end

phase_shift_orig(K::KSpaceCartesianSampling) = isnothing(K.phase_shift) ? exp.(1im*(K.K[:,:,1]*K.X.o[1]+K.K[:,:,2]*K.X.o[2]+K.K[:,:,3]*K.X.o[3])) : K.phase_shift

function kspace_sampling(X::RegularCartesianSpatialSampling{T}; phase_encode::Symbol=:xy, readout::Symbol=:z, phase_shift::Bool=true) where T

    # Input check
    (readout == :x) && ((phase_encode == :xy) || (phase_encode == :yx) || (phase_encode == :xz) || (phase_encode == :zx)) && error("Incompatible readout/phase-encode dimensions")
    (readout == :y) && ((phase_encode == :xy) || (phase_encode == :yx) || (phase_encode == :yz) || (phase_encode == :zy)) && error("Incompatible readout/phase-encode dimensions")
    (readout == :z) && ((phase_encode == :xz) || (phase_encode == :zx) || (phase_encode == :yz) || (phase_encode == :zy)) && error("Incompatible readout/phase-encode dimensions")

    # Set phase-encode/readout dimensions
    nx, ny, nz = X.n
    (readout == :x) && (nk = nx; pk = 1)
    (readout == :y) && (nk = ny; pk = 2)
    (readout == :z) && (nk = nz; pk = 3)
    (phase_encode == :xy) && (nt = (nx, ny); pt = (1, 2))
    (phase_encode == :yx) && (nt = (ny, nx); pt = (2, 1))
    (phase_encode == :xz) && (nt = (nx, nz); pt = (1, 3))
    (phase_encode == :zx) && (nt = (nz, nx); pt = (3, 1))
    (phase_encode == :yz) && (nt = (ny, nz); pt = (2, 3))
    (phase_encode == :zy) && (nt = (nz, ny); pt = (3, 2))
    perm = (pk, pt...)

    # Mesh k-space grid
    kx_max, ky_max, kz_max = T(pi)./X.h
    kx = collect(range(-kx_max, kx_max; length=nx))
    ky = collect(range(-ky_max, ky_max; length=ny))
    kz = collect(range(-kz_max, kz_max; length=nz))
    Kx = reshape(permutedims(repeat(reshape(vec(kx),:,1,1); outer=(1,ny,nz)), perm), nk, prod(nt))
    Ky = reshape(permutedims(repeat(reshape(vec(ky),1,:,1); outer=(nx,1,nz)), perm), nk, prod(nt))
    Kz = reshape(permutedims(repeat(reshape(vec(kz),1,1,:); outer=(nx,ny,1)), perm), nk, prod(nt))
    K = cat(Kx, Ky, Kz; dims=3)

    # Phase shift (w.r.t. origin)
    phase_shift ? (phase_shift = exp.(1im*(K[:,:,1]*X.o[1]+K[:,:,2]*X.o[2]+K[:,:,3]*X.o[3]))) : (phase_shift = nothing)

    return KSpaceCartesianSampling{T}(X, K, phase_shift)

end


## Interaction of rigid-body motion w/ Cartesian sampling type

struct KSpaceCartesianSamplingRBM{T}<:AbstractKSpaceFixedSizeSampling{T}
    Kc::KSpaceCartesianSampling{T}
    K::AbstractArray{T,3} # size(K) = (nk, nt, 3)
    phase_shift::Union{Nothing,AbstractArray{Complex{T},2}} # size(phase_shift) = (nk, nt)
    θ::AbstractArray{T,2}
end

function (K::KSpaceCartesianSampling{T})(θ::AbstractArray{T,2}) where {T<:Real}

    # Simplifying notation
    ox, oy, oz = K.X.o
    Kx, Ky, Kz = K.K[:,:,1], K.K[:,:,2], K.K[:,:,3]
    τx, τy, τz, ϕxy, ϕxz, ϕyz = θ[:,1]', θ[:,2]', θ[:,3]', θ[:,4]', θ[:,5]', θ[:,6]'

    # Translation
    phase_shift = exp.(-1im*(Kx.*(τx-ox)+Ky.*(τy-oy)+Kz.*(τz-oz)))

    # Rotation
    cxy = cos.(-ϕxy); sxy = sin.(-ϕxy)
    Kx_ = cxy.*Kx-sxy.*Ky
    Ky_ = sxy.*Kx+cxy.*Ky
    Kx = Kx_; Ky = Ky_;
    cxz = cos.(-ϕxz); sxz = sin.(-ϕxz)
    Kx_ = cxz.*Kx-sxz.*Kz
    Kz_ = sxz.*Kx+cxz.*Kz
    Kx = Kx_; Kz = Kz_;
    cyz = cos.(-ϕyz); syz = sin.(-ϕyz)
    Ky_ = cyz.*Ky-syz.*Kz
    Kz_ = syz.*Ky+cyz.*Kz
    Ky = Ky_; Kz = Kz_;

    return KSpaceCartesianSamplingRBM{T}(K, cat(Kx, Ky, Kz; dims=3), phase_shift, θ)

end

phase_shift(K::KSpaceCartesianSamplingRBM) = isnothing(K.phase_shift) ? exp.(-1im*(K.K[:,:,1].*(K.θ[:,1]'.-K.X.o[1])+K.K[:,:,2].*(K.θ[:,2]'.-K.X.o[2])+K.K[:,:,3].*(K.θ[:,3]'.-K.X.o[3]))) : K.phase_shift