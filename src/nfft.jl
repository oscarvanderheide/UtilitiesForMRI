# NFFT utilities

export NFFTLinOp, NFFTParametericLinOp, NFFTParametericDelayedEval, JacobianNFFTEvaluated
export nfft_linop, nfft, Jacobia, ∂, sparse_matrix_GaussNewton


## NFFT linear operator

struct NFFTLinOp{T}<:AbstractLinearOperator{Complex{T},3,2}
    X::RegularCartesianSpatialSampling{T}
    K::KSpaceFixedSizeSampling{T}
    tol::T
end

nfft_linop(X::RegularCartesianSpatialSampling{T}, K::KSpaceFixedSizeSampling{T}; tol::T=T(1e-6)) where {T<:Real} = NFFTLinOp{T}(X, K, tol)

nfft_linop(X::RegularCartesianSpatialSampling{T}; phase_encode::Symbol=:xy, readout::Symbol=:z, tol::T=T(1e-6)) where {T<:Real} = nfft_linop(X, kspace_sampling(X; phase_encode=phase_encode, readout=readout); tol=tol)

k_coord(F::NFFTLinOp) = coord(F.K)

AbstractLinearOperators.domain_size(F::NFFTLinOp) = F.X.n
AbstractLinearOperators.range_size(F::NFFTLinOp) = size(F.K)

function AbstractLinearOperators.matvecprod(F::NFFTLinOp{T}, u::AbstractArray{Complex{T},3}) where {T<:Real}
    K = k_coord(F)
    return reshape(nufft3d2(vec(K[:,:,1]*F.X.h[1]), vec(K[:,:,2]*F.X.h[2]), vec(K[:,:,3]*F.X.h[3]), -1, F.tol, u), range_size(F))/T(sqrt(prod(domain_size(F))))
end

function AbstractLinearOperators.matvecprod_adj(F::NFFTLinOp{T}, d::AbstractArray{Complex{T},2}) where {T<:Real}
    K = k_coord(F)
    return nufft3d1(vec(K[:,:,1]*F.X.h[1]), vec(K[:,:,2]*F.X.h[2]), vec(K[:,:,3]*F.X.h[3]), vec(d), 1, F.tol, domain_size(F)...)[:,:,:,1]/T(sqrt(prod(domain_size(F))))
end


## Parameteric linear operator

struct NFFTParametericLinOp{T}
    X::RegularCartesianSpatialSampling{T}
    K::KSpaceFixedSizeSampling{T}
    tol::T
end

nfft(X::RegularCartesianSpatialSampling{T}, K::KSpaceFixedSizeSampling{T}; tol::T=T(1e-6)) where {T<:Real} = NFFTParametericLinOp{T}(X, K, tol)

function (F::NFFTParametericLinOp{T})(θ::AbstractArray{T,2}) where {T<:Real}
    (size(θ,1) !== size(F.K)[1]) && error("Incompatible time dimension")
    Pτ = phase_shift_linop(F.K, -θ[:,1:3])
    Rφ = rotation_linop(-θ[:,4:end])
    Kφ = kspace_sampling(Rφ*coord(F.K))
    F = nfft_linop(F.X, Kφ; tol=F.tol)
    return Pτ*F
end

(F::NFFTParametericLinOp)() = F


## Parameteric delayed evaluation

struct NFFTParametericDelayedEval{T}
    F::NFFTParametericLinOp{T}
    u::AbstractArray{Complex{T},3}
end

Base.:*(F::NFFTParametericLinOp{T}, u::AbstractArray{Complex{T},3}) where {T<:Real} =  NFFTParametericDelayedEval{T}(F, u)

(Fu::NFFTParametericDelayedEval{T})(θ::AbstractArray{T,2}) where {T<:Real} = Fu.F(θ)*Fu.u


## Jacobian of nfft evaluated

struct JacobianNFFTEvaluated{T}<:AbstractLinearOperator{Complex{T},2,2}
    J::AbstractArray{Complex{T},3}
end

function Jacobian(Fu::NFFTParametericDelayedEval{T}, θ::AbstractArray{T,2}) where {T<:Real}

    # Simplifying notation
    u = Fu.u
    X = Fu.F.X
    K = Fu.F.K
    tol = Fu.F.tol
    τ = θ[:,1:3]
    φ = θ[:,4:end]
    P = phase_shift(K) # phase-shift
    R = rotation() # rotation

    # NFFT of u and derivatives thereof
    Kφ, ∂Kφ = ∂(R()*K, -φ)
    Fφ = nfft_linop(X, kspace_sampling(Kφ); tol=tol)
    x, y, z = coord(X)
    Fu = Fφ*u
    ∇Fu = -im*cat(Fφ*(u.*x), Fφ*(u.*y), Fφ*(u.*z); dims=3)

    # Computing rigid-body motion Jacobian
    d, Pτ, ∂Pτu = ∂(P()*Fu, -τ)
    J = cat(-∂Pτu.J, -(Pτ*dot(∇Fu, ∂Kφ)); dims=3)

    return d, Pτ*Fφ, JacobianNFFTEvaluated{T}(J)

end

∂(Fu::NFFTParametericDelayedEval{T}, θ::AbstractArray{T,2}) where {T<:Real} = Jacobian(Fu, θ)

AbstractLinearOperators.domain_size(∂Fu::JacobianNFFTEvaluated) = (size(∂Fu.J,1),6)
AbstractLinearOperators.range_size(∂Fu::JacobianNFFTEvaluated) = size(∂Fu.J)[1:2]
AbstractLinearOperators.matvecprod(∂Fu::JacobianNFFTEvaluated{T}, Δθ::AbstractArray{Complex{T},2}) where {T<:Real} = sum(∂Fu.J.*reshape(Δθ, :, 1, 6); dims=3)[:,:,1]
Base.:*(∂Fu::JacobianNFFTEvaluated{T}, Δθ::AbstractArray{T,2}) where {T<:Real} = ∂Fu*complex(Δθ)
AbstractLinearOperators.matvecprod_adj(∂Fu::JacobianNFFTEvaluated{T}, Δd::AbstractArray{Complex{T},2}) where {T<:Real} = sum(conj(∂Fu.J).*reshape(Δd, size(Δd)..., 1); dims=2)[:,1,:]


## Other utilities

function sparse_matrix_GaussNewton(J::JacobianNFFTEvaluated{T}) where {T<:Real}
    J = J.J
    GN = similar(J, T, size(J, 1), 6, 6)
    @inbounds for i = 1:6, j = 1:6
        GN[:,i,j] = real(sum(conj(J[:,:,i]).*J[:,:,j]; dims=2)[:,1])
    end
    h(i,j) = spdiagm(0 => GN[:,i,j])
    return [h(1,1) h(1,2) h(1,3) h(1,4) h(1,5) h(1,6);
            h(2,1) h(2,2) h(2,3) h(2,4) h(2,5) h(2,6);
            h(3,1) h(3,2) h(3,3) h(3,4) h(3,5) h(3,6);
            h(4,1) h(4,2) h(4,3) h(4,4) h(4,5) h(4,6);
            h(5,1) h(5,2) h(5,3) h(5,4) h(5,5) h(5,6);
            h(6,1) h(6,2) h(6,3) h(6,4) h(6,5) h(6,6)]
end