# NFFT-Jacobian utilities

export JacobianNFFTstd
export Jacobian, ∂


## Delayed NFFT evaluations: eye-candy abstraction for Jacobian computation
## Gives meaning to: ∂(F()*u, θ)

struct FunctionalNFFT{FT}
    f::FT
end

(f::AbstractNFFT)() = FunctionalNFFT{typeof(FT)}(f)

struct DelayedNFFTEval{FT,XT}
    f::FT
    u::XT
end

Base.:*(f::FunctionalNFFT{FT}, u::XT) where {FT,XT} = DelayedNFFTEval{FT,XT}(f, u)


## Derivatives w.r.t. rigid-body motion parameters

struct JacobianNFFTstd{T}<:AbstractLinearOperator{T,2,2}
    J::AbstractArray{T,3} # size(J) = (nt, nk, 6)
end

function Jacobian(Fu::DelayedNFFTEval{NFFTstd{Complex{T}},AbstractArray{Complex{T},3}}, θ::AbstractArray{T,2}) where {T<:Real}

    # Simplifying notation
    F = Fu.F
    u = Fu.u
    X = Fu.F.X
    K = Fu.F.K
    tol = Fu.F.tol
    nt, nk = range_size(F)
    c = T(sqrt(prod(domain_size(F))))

    # Coordinate transform under rigid-body motion θ
    phase_shift, Kθ, ∂τ, ∂ϕxy, ∂ϕxz, ∂ϕyz = process_nfft_pars(K, θ; derivatives=true)

    # NFFT plan
    plan = finufft_makeplan(2, nt*nk, -1, 4, tol, dtype=T)
    k = (vec(Kθ[:,:,1]), vec(Kθ[:,:,2]), vec(Kθ[:,:,3]))
    finufft_setpts!(plan, k...)

    # NFFT of u and derivatives thereof
    x, y, z = coord(X)
    out = finufft_exec(plan, cat(u, -im*u.*x, -im*u.*y, -im*u.*z; dims=4)); finufft_destroy!(plan);
    d   = reshape(out[:, 1], nt, nk)/c
    ∇kd = reshape(out[:, 2:4], nt, nk, 3)/c

    # Computing rigid-body motion Jacobian
    J = similar(d, nt, nk, 6)
    J[:,:,1:3] .= ∂τ.*d
    J[:,:,4]   .= -phase_shift.*sum(∇kd.*∂ϕxy; dims=3)[:,:,1]
    J[:,:,5]   .= -phase_shift.*sum(∇kd.*∂ϕxz; dims=3)[:,:,1]
    J[:,:,6]   .= -phase_shift.*sum(∇kd.*∂ϕyz; dims=3)[:,:,1]

    return JacobianNFFTstd{Complex{T}}(J)
end

∂(Fu::DelayedNFFTEval{NFFTstd{Complex{T}},AbstractArray{Complex{T},3}}, θ::AbstractArray{T,2}) where {T<:Real} = Jacobian(Fu, θ)

AbstractLinearOperators.domain_size(J::JacobianNFFTstd) = (size(J.J)[1],size(J.J)[3])
AbstractLinearOperators.range_size(J::JacobianNFFTstd) = (size(J.J)[1],size(J.J)[2])

AbstractLinearOperators.matvecprod(J::JacobianNFFTstd{Complex{T}}, θ::AbstractArray{Complex{T},2}) where {T<:Real} = sum(J.J.*reshape(θ, :, 1, 6); dims=3)[:,:,1]
Base.:*(J::JacobianNFFTstd{Complex{T}}, θ::AbstractArray{T,2}) where {T<:Real} = sum(J.J.*reshape(θ, :, 1, 6); dims=3)[:,:,1]

AbstractLinearOperators.matvecprod_adj(J::JacobianNFFTstd{Complex{T}}, d::AbstractArray{Complex{T},2}) where {T<:Real} = sum(conj(J.J).*reshape(d, size(d)..., 1); dims=2)[:,1,:]


# Gauss-Newton-related utilities

struct JacobianNFFTstdAdjoint{T}<:AbstractLinearOperator{T,2,2}
    J::JacobianNFFTstd{T}
end

Base.adjoint(J::JacobianNFFTstd{T}) where T = JacobianNFFTstdAdjoint{T}(J)
Base.adjoint(J::JacobianNFFTstdAdjoint) = J.J

AbstractLinearOperators.domain_size(J::JacobianNFFTstdAdjoint) = range_size(J.J)
AbstractLinearOperators.range_size(J::JacobianNFFTstdAdjoint) = domain_size(J.J)
AbstractLinearOperators.matvecprod(J::JacobianNFFTstdAdjoint{Complex{T}}, d::AbstractArray{Complex{T},2}) where {T<:Real} = J.J'*d
AbstractLinearOperators.matvecprod_adj(J::JacobianNFFTstdAdjoint{Complex{T}}, θ::AbstractArray{Complex{T},2}) where {T<:Real} = J.J*θ
Base.:*(J::JacobianNFFTstdAdjoint{Complex{T}}, θ::AbstractArray{T,2}) where {T<:Real} = J.J*θ

function Base.:*(Jadj::JacobianNFFTstdAdjoint{T}, J::JacobianNFFTstd{T}) where {T<:Complex}
    nt, nk, _ = size(J.J)
    Harr = sum(reshape(conj(Jadj.J), nt,nk,6,1).*reshape(J.J, nt,nk,1,6); dims=2)[:,1,:,:]
    h(i,j) = spdiagm(Harr[:,i,j])
    return [h(1,1) h(1,2) h(1,3) h(1,4) h(1,5) h(1,6);
            h(2,1) h(2,2) h(2,3) h(2,4) h(2,5) h(2,6);
            h(3,1) h(3,2) h(3,3) h(3,4) h(3,5) h(3,6);
            h(4,1) h(4,2) h(4,3) h(4,4) h(4,5) h(4,6);
            h(5,1) h(5,2) h(5,3) h(5,4) h(5,5) h(5,6);
            h(6,1) h(6,2) h(6,3) h(6,4) h(6,5) h(6,6)]
end