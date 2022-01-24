# NFFT utilities

export NFFTstd
export nfft_linop


# ## Load Python library

# const pynufft = PyNULL()
# function __init__()
#     pushfirst!(PyVector(pyimport("sys")."path"), joinpath(dirname(pathof(UtilitiesForMRI)), "pysrc"))
#     copy!(pynufft, pyimport("cufinufft_interface"))
# end


## NFFT linear operator (uniform to non-uniform sampling)

struct NFFTstd{T}<:AbstractNFFT{T,3,2}
    X::RegularCartesianSpatialSampling
    K::KSpaceFixedSizeSampling
    tol::Real
    phase_shift::Union{T,AbstractArray{T,2}}
end

function nfft_linop(X::RegularCartesianSpatialSampling{T}, K::KSpaceFixedSizeSampling{T}; tol::T=T(1e-6)) where {T<:Real}
    phase_shift, _ = process_nfft_pars(K, nothing)
    return NFFTstd{complex(T)}(X, K, tol, phase_shift)
end
nfft_linop(X::RegularCartesianSpatialSampling{T}; phase_encode::Symbol=:xy, readout::Symbol=:z, tol::T=T(1e-6)) where {T<:Real} = nfft_linop(X, kspace_sampling(X; phase_encode=phase_encode, readout=readout); tol=tol)

phase_shift(F::NFFTstd) = F.phase_shift
k_coord(F::NFFTstd; norm::Bool=true) = coord(F.K; norm=norm)

AbstractLinearOperators.domain_size(F::NFFTstd) = F.X.n
AbstractLinearOperators.range_size(F::NFFTstd) = size(F.K)

function AbstractLinearOperators.matvecprod(F::NFFTstd{Complex{T}}, u::AbstractArray{Complex{T},3}) where {T<:Real}
    Kx, Ky, Kz = k_coord(F; norm=true)
    return reshape(nufft3d2(vec(Kx), vec(Ky), vec(Kz), -1, F.tol, u), range_size(F)).*phase_shift(F)/T(sqrt(prod(domain_size(F))))
end

function AbstractLinearOperators.matvecprod_adj(F::NFFTstd{Complex{T}}, d::AbstractArray{Complex{T},2}) where {T<:Real}
    Kx, Ky, Kz = k_coord(F; norm=true)
    return nufft3d1(vec(Kx), vec(Ky), vec(Kz), vec(d.*conj(phase_shift(F))), 1, F.tol, domain_size(F)...)[:,:,:,1]/T(sqrt(prod(domain_size(F))))
end


## Perturbed NFFT by rigid-body motion

function (F::NFFTstd{Complex{T}})(θ::AbstractArray{T,2}) where {T<:Real}
    (size(θ,1) !== size(F.K)[1]) && error("Incompatible time dimension")
    phase_shift, Kθ = process_nfft_pars(F.K, θ)
    return NFFTstd{Complex{T}}(F.X, Kθ, F.tol, phase_shift)
end