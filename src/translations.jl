# Translation utilities

export PhaseShiftLinOp, JacobianPhaseShiftEvaluated, PhaseShiftParametericLinOp, PhaseShiftParametericDelayedEval
export phase_shift_linop, phase_shift, Jacobian, ∂


## Linear operator

struct PhaseShiftLinOp{T}<:AbstractLinearOperator{T,2,2}
    phase_shift::AbstractArray{T,2}
end

function phase_shift_linop(K::KSpaceFixedSizeSampling{T}, τ::AbstractArray{T,2}) where {T<:Real}
    phase_shift  = exp.(im*sum(coord(K).*reshape(τ,:,1,3); dims=3)[:,:,1])
    return PhaseShiftLinOp{complex(T)}(phase_shift)
end

AbstractLinearOperators.domain_size(P::PhaseShiftLinOp) = size(P.phase_shift)
AbstractLinearOperators.range_size(P::PhaseShiftLinOp) = domain_size(P)

AbstractLinearOperators.matvecprod(P::PhaseShiftLinOp{T}, d::AbstractArray{T,2}) where {T<:Complex} = P.phase_shift.*d
AbstractLinearOperators.matvecprod_adj(P::PhaseShiftLinOp{T}, d::AbstractArray{T,2}) where {T<:Complex} = conj(P.phase_shift).*d


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

struct JacobianPhaseShiftEvaluated{T}<:AbstractLinearOperator{T,2,2}
    ∂Pd::AbstractArray{T,3}
end

function Jacobian(Pd::PhaseShiftParametericDelayedEval{T}, τ::AbstractArray{T,2}) where {T<:Real}
    Pτ = phase_shift_linop(Pd.P.K, τ)
    Pτd = Pτ*Pd.d
    ∂Pτd = im*coord(Pd.P.K).*reshape(Pτd, size(Pτd)..., 1)
    return Pτd, JacobianPhaseShiftEvaluated{complex(T)}(∂Pτd)
end

∂(Pd::PhaseShiftParametericDelayedEval{T}, τ::AbstractArray{T,2}) where {T<:Real} = Jacobian(Pd, τ)

AbstractLinearOperators.domain_size(∂Pd::JacobianPhaseShiftEvaluated) = (size(∂Pd.∂Pd,1),3)
AbstractLinearOperators.range_size(∂Pd::JacobianPhaseShiftEvaluated) = size(∂Pd.∂Pd)[1:2]

AbstractLinearOperators.matvecprod(∂Pd::JacobianPhaseShiftEvaluated{T}, Δτ::AbstractArray{T,2}) where {T<:Complex} = sum(∂Pd.∂Pd.*reshape(Δτ, :, 1, 3); dims=3)[:,:,1]
Base.:*(∂Pd::JacobianPhaseShiftEvaluated{Complex{T}}, Δτ::AbstractArray{T,2}) where {T<:Real} = ∂Pd*complex(Δτ)
AbstractLinearOperators.matvecprod_adj(∂Pd::JacobianPhaseShiftEvaluated{T}, Δd::AbstractArray{T,2}) where {T<:Complex} = sum(conj(∂Pd.∂Pd).*reshape(Δd, size(Δd)..., 1); dims=2)[:,1,:]