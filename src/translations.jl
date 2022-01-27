# Translation utilities

export PhaseShiftLinOp, JacobianPhaseShiftEvaluated, PhaseShiftParametericLinOp, PhaseShiftParametericDelayedEval
export phase_shift_linop, phase_shift, Jacobian, ∂


## Linear operator

struct PhaseShiftLinOp{T}<:AbstractLinearOperator{Complex{T},2,2}
    phase_shift::AbstractArray{Complex{T},2}
end

function phase_shift_linop(K::KSpaceFixedSizeSampling{T}, τ::AbstractArray{T,2}) where {T<:Real}
    phase_shift  = exp.(im*sum(coord(K).*reshape(τ,:,1,3); dims=3)[:,:,1])
    return PhaseShiftLinOp{T}(phase_shift)
end

AbstractLinearOperators.domain_size(P::PhaseShiftLinOp) = size(P.phase_shift)
AbstractLinearOperators.range_size(P::PhaseShiftLinOp) = domain_size(P)

AbstractLinearOperators.matvecprod(P::PhaseShiftLinOp{T}, d::AbstractArray{Complex{T},2}) where {T<:Real} = P.phase_shift.*d
Base.:*(P::PhaseShiftLinOp{T}, d::AbstractArray{Complex{T},3}) where {T<:Real} = P.phase_shift.*d
AbstractLinearOperators.matvecprod_adj(P::PhaseShiftLinOp{T}, d::AbstractArray{Complex{T},2}) where {T<:Real} = conj(P.phase_shift).*d


## Parameteric linear operator

struct PhaseShiftParametericLinOp{T}
    K::KSpaceFixedSizeSampling{T}
end

phase_shift(K::KSpaceFixedSizeSampling{T}) where {T<:Real} = PhaseShiftParametericLinOp{T}(K)

(P::PhaseShiftParametericLinOp{T})(τ::AbstractArray{T,2}) where {T<:Real} = phase_shift_linop(P.K, τ)

(P::PhaseShiftParametericLinOp)() = P


## Parameteric delayed evaluation

struct PhaseShiftParametericDelayedEval{T}
    P::PhaseShiftParametericLinOp{T}
    d::AbstractArray{Complex{T},2}
end

Base.:*(P::PhaseShiftParametericLinOp{T}, d::AbstractArray{Complex{T},2}) where {T<:Real} =  PhaseShiftParametericDelayedEval{T}(P, d)

(Pd::PhaseShiftParametericDelayedEval{T})(τ::AbstractArray{T,2}) where {T<:Real} = phase_shift_linop(τ, Pd.P.K)*Pd.d


## Jacobian of phase-shift evaluated

struct JacobianPhaseShiftEvaluated{T}<:AbstractLinearOperator{Complex{T},2,2}
    J::AbstractArray{Complex{T},3}
end

function Jacobian(Pd::PhaseShiftParametericDelayedEval{T}, τ::AbstractArray{T,2}) where {T<:Real}
    Pτ = phase_shift_linop(Pd.P.K, τ)
    Pτd = Pτ*Pd.d
    J = im*coord(Pd.P.K).*reshape(Pτd, size(Pτd)..., 1)
    return Pτd, Pτ, JacobianPhaseShiftEvaluated{T}(J)
end

∂(Pd::PhaseShiftParametericDelayedEval{T}, τ::AbstractArray{T,2}) where {T<:Real} = Jacobian(Pd, τ)

AbstractLinearOperators.domain_size(∂Pd::JacobianPhaseShiftEvaluated) = (size(∂Pd.J,1),3)
AbstractLinearOperators.range_size(∂Pd::JacobianPhaseShiftEvaluated) = size(∂Pd.J)[1:2]

AbstractLinearOperators.matvecprod(∂Pd::JacobianPhaseShiftEvaluated{T}, Δτ::AbstractArray{CT,2}) where {T<:Real,CT<:RealOrComplex{T}} = sum(∂Pd.J.*reshape(Δτ, :, 1, 3); dims=3)[:,:,1]
Base.:*(∂Pd::JacobianPhaseShiftEvaluated{T}, Δτ::AbstractArray{T,2}) where {T<:Real} = AbstractLinearOperators.matvecprod(∂Pd, Δτ)
AbstractLinearOperators.matvecprod_adj(∂Pd::JacobianPhaseShiftEvaluated{T}, Δd::AbstractArray{Complex{T},2}) where {T<:Real} = sum(conj(∂Pd.J).*reshape(Δd, size(Δd)..., 1); dims=2)[:,1,:]