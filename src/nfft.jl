# NFFT utilities

export NFFTType2LinOp, SampledCartesianNFFTType2LinOp, NFFTParametericLinOp, NFFTParametericDelayedEval, JacobianNFFTEvaluated
export nfft_linop, nfft, Jacobian, ∂, sparse_matrix_GaussNewton


## General type-2 NFFT linear operator

struct NFFTType2LinOp{T}<:AbstractLinearOperator{Complex{T},3,2}
    X::CartesianSpatialGeometry{T}
    K::AbstractKSpaceSampling{T}
    phase_shift_origin::AbstractVector{Complex{T},2}
    norm_constant::T
    tol::T
end

function nfft_linop(X::CartesianSpatialGeometry{T}, K::AbstractKSpaceSampling{T}; tol::T=T(1e-6), norm_constant::T=prod(spacing(X))) where {T<:Real}
    kh_coordinates = coord(K; normalization=spacing(X))
    o_norm = origin(X; wrt_center=true)./spacing(X)
    phase_shift_origin = exp.(-im*(kh_coordinates[:,:,1]*o_norm[1]+kh_coordinates[:,:,2]*o_norm[2]+kh_coordinates[:,:,3]*o_norm[3]))
    return NFFTType2LinOp{T}(X, K, phase_shift_origin, norm_constant, tol)
end

AbstractLinearOperators.domain_size(F::NFFTType2LinOp) = size(F.X)
AbstractLinearOperators.range_size(F::NFFTType2LinOp) = size(F.K)

AbstractLinearOperators.matvecprod(F::NFFTType2LinOp{T}, u::AbstractArray{Complex{T},3}) where {T<:Real} = reshape(vec(F.phase_shift_origin).*nufft3d2(vec(F.k_coordinates_norm[:,:,1]), vec(F.k_coordinates_norm[:,:,2]), vec(F.k_coordinates_norm[:,:,3]), -1, F.tol, u), range_size(F))*F.norm_constant

AbstractLinearOperators.matvecprod_adj(F::NFFTType2LinOp{T}, d::AbstractVector{Complex{T},2}) where {T<:Real} = nufft3d1(vec(F.k_coordinates_norm[:,:,1]), vec(F.k_coordinates_norm[:,:,2]), vec(F.k_coordinates_norm[:,:,3]), vec(conj(F.phase_shift_origin).*d), 1, F.tol, domain_size(F)...)*F.norm_constant


## Parameteric linear operator

struct NFFTType2ParametericLinOp{T}
    X::CartesianSpatialGeometry{T}
    K::AbstractKSpaceSampling{T}
    phase_shift_origin::AbstractVector{Complex{T},2}
    norm_constant::T
    tol::T
end

function nfft(X::CartesianSpatialGeometry{T}, K::AbstractKSpaceSampling{T}; tol::T=T(1e-6)) where {T<:Real}
    kh_coordinates = coord(K; normalization=spacing(X))
    o_norm = origin(X; wrt_center=true)./spacing(X)
    phase_shift_origin = exp.(-im*(kh_coordinates[:,1]*o_norm[1]+kh_coordinates[:,2]*o_norm[2]+kh_coordinates[:,3]*o_norm[3]))
    return NFFTType2ParametericLinOp{T}(X, K, phase_shift_origin, norm_constant, tol)
end

function (F::NFFTType2ParametericLinOp{T})(θ::AbstractArray{T,2}) where {T<:Real}
    (size(θ,1) !== size(F.K)[1]) && error("Incompatible time dimension")
    Pτ = phase_shift_linop(F.K, θ[:,1:3])
    Rφ = rotation_linop(θ[:,4:end])
    Kφ = kspace_sampling(Rφ*coord(F.K))
    F = nfft_linop(F.X, Kφ; tol=F.tol)
    return Pτ*F
end

(F::NFFTParametericLinOp)() = F

downscale(F::NFFTParametericLinOp{T}; fact::Integer=1) where {T<:Real} = NFFTParametericLinOp{T}(downscale(F.X; fact=fact), downscale(F.K; fact=fact), F.tol)


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
    Kφ, ∂Kφ = ∂(R()*K, φ)
    Fφ = nfft_linop(X, kspace_sampling(Kφ); tol=tol)
    x, y, z = coord(X)
    Fu = Fφ*u
    ∇Fu = -im*cat(Fφ*(u.*x), Fφ*(u.*y), Fφ*(u.*z); dims=3)

    # Computing rigid-body motion Jacobian
    d, Pτ, ∂Pτu = ∂(P()*Fu, τ)
    J = cat(∂Pτu.J, Pτ*dot(∇Fu, ∂Kφ); dims=3)

    return d, Pτ*Fφ, JacobianNFFTEvaluated{T}(J)

end

∂(Fu::NFFTParametericDelayedEval{T}, θ::AbstractArray{T,2}) where {T<:Real} = Jacobian(Fu, θ)

AbstractLinearOperators.domain_size(∂Fu::JacobianNFFTEvaluated) = (size(∂Fu.J,1),6)
AbstractLinearOperators.range_size(∂Fu::JacobianNFFTEvaluated) = size(∂Fu.J)[1:2]
AbstractLinearOperators.matvecprod(∂Fu::JacobianNFFTEvaluated{T}, Δθ::AbstractArray{Complex{T},2}) where {T<:Real} = sum(∂Fu.J.*reshape(Δθ, :, 1, 6); dims=3)[:,:,1]
Base.:*(∂Fu::JacobianNFFTEvaluated{T}, Δθ::AbstractArray{T,2}) where {T<:Real} = ∂Fu*complex(Δθ)
AbstractLinearOperators.matvecprod_adj(∂Fu::JacobianNFFTEvaluated{T}, Δd::AbstractArray{Complex{T},2}) where {T<:Real} = sum(conj(∂Fu.J).*reshape(Δd, size(Δd)..., 1); dims=2)[:,1,:]


## Other utilities

function sparse_matrix_GaussNewton(J::JacobianNFFTEvaluated{T}; W::Union{Nothing,AbstractLinearOperator}=nothing, H::Union{Nothing,AbstractLinearOperator}=nothing) where {T<:Real}
    J = J.J
    GN = similar(J, T, size(J, 1), 6, 6)
    if ~isnothing(W)
        WJ = similar(J)
        @inbounds for i = 1:6
            WJ[:, :, i] = W*J[:,:,i]
        end
    else
        WJ = J
    end
    if ~isnothing(H)
        HWJ = similar(J)
        @inbounds for i = 1:6
            HWJ[:, :, i] = H*WJ[:,:,i]
        end
    else
        HWJ = J
    end
    @inbounds for i = 1:6, j = 1:6
        GN[:,i,j] = vec(real(sum(conj(WJ[:,:,i]).*HWJ[:,:,j]; dims=2)))
    end
    h(i,j) = spdiagm(0 => GN[:,i,j])
    return [h(1,1) h(1,2) h(1,3) h(1,4) h(1,5) h(1,6);
            h(2,1) h(2,2) h(2,3) h(2,4) h(2,5) h(2,6);
            h(3,1) h(3,2) h(3,3) h(3,4) h(3,5) h(3,6);
            h(4,1) h(4,2) h(4,3) h(4,4) h(4,5) h(4,6);
            h(5,1) h(5,2) h(5,3) h(5,4) h(5,5) h(5,6);
            h(6,1) h(6,2) h(6,3) h(6,4) h(6,5) h(6,6)]
end