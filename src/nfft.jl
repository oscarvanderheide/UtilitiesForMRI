# NFFT utilities

export NFFTstd, JacobianNFFTFixedSizeSampling
export nfft_linop, Jacobian, ∂


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


# ## Delayed evaluations

# struct DelayedFun{FT}
#     f::FT
# end

# (f::FT)() = DelayedFun{FT}(f)

# struct DelayedEval{FT,XT}
#     f::FT
#     x::XT
# end

# Base.:*(f::DelayedFun{AT}, u::XT) where {AT,XT} = DelayedEval{AT,XT}(f, u)


# ## Derivatives w.r.t. rigid-body motion parameters

# struct JacobianNFFTFixedSizeSampling{T}<:AbstractLinearOperator{T,2,2}
#     Fu::DelayedEval{NFFTFixedSizeSampling{T},AbstractArray{T,3}}
#     θ::AbstractArray
# end

# function Jacobian(Fu::DelayedEval{NFFTCartesianSampling{Complex{T}},AbstractArray{Complex{T},3}}, θ::AbstractArray{T,2}) where {T<:Real}

#     # Coordinate transform under rigid-body motion
#     Kθ = Fu.F.K(θ)

#     # NFFT plan
#     plan = finufft_makeplan(2, prod(range_size(Fu.F)), -1, 4, Fu.F.tol, dtype=real(T))
#     k = (vec(Kθ.K[:,:,1]), vec(Kθ.K[:,:,2]), vec(Kθ.K[:,:,3]))
#     finufft_setpts!(plan, k...)

#     # NFFT of u and derivatives thereof
#     x, y, z = coord(Fu.F.X)
#     out = finufft_exec(plan, cat(Fu.u, -1im*Fu.u.*x, -1im*Fu.u.*y, -1im*Fu.u.*z; dims=4)); finufft_destroy!(plan);
#     d     = reshape(out[:, 1], range_size(Fu.F))
#     ∂kx_d = reshape(out[:, 2], range_size(Fu.F))
#     ∂ky_d = reshape(out[:, 3], range_size(Fu.F))
#     ∂kz_d = reshape(out[:, 4], range_size(Fu.F))

#     reshape(nufft3d2(vec(F.K.K[:,:,1]), vec(F.K.K[:,:,2]), vec(F.K.K[:,:,3]), -1, F.tol, u), range_size(F)).*phase_shift(F.K)/T(sqrt(prod(domain_size(F))))

#     JacobianNFFTFixedSizeSampling{Complex{T}}(Fu, θ)
# end

# ∂(Fu::NFFTFixedSizeSampling_delayedeval{Complex{T}}, θ::AbstractArray{T,2}) where {T<:Real} = Jacobian(Fu, θ)

# AbstractLinearOperators.domain_size(J::JacobianNFFTFixedSizeSampling) = size(J.θ)
# AbstractLinearOperators.range_size(J::JacobianNFFTFixedSizeSampling) = size(J.Fu.u)

# function AbstractLinearOperators.matvecprod(J::JacobianNFFTFixedSizeSampling{Complex{T}}, θ::AbstractArray{Complex{T},3}) where {T<:Real}
#     reshape(nufft3d2(vec(F.K.K[:,:,1]), vec(F.K.K[:,:,2]), vec(F.K.K[:,:,3]), -1, F.tol, u), range_size(F)).*F.K.phase_shift/T(sqrt(prod(domain_size(F))))
# end

# AbstractLinearOperators.matvecprod_adj(F::AbstractNFFTFixedSizeSampling{Complex{T}}, d::AbstractArray{Complex{T},2}) where {T<:Real} = nufft3d1(vec(F.K.K[:,:,1]), vec(F.K.K[:,:,2]), vec(F.K.K[:,:,3]), vec(d.*conj(F.K.phase_shift)), 1, F.tol, size(F.X)...)[:,:,:,1]/T(sqrt(prod(domain_size(F))))