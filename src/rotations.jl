# Rotation utilities

export RotationLinOp, RotationParametericLinOp, RotationParametericDelayedEval, JacobianRotationEvaluated
export rotation_linop, rotation, Jacobian, ∂


## Linear operator

struct RotationLinOp{T}<:AbstractLinearOperator{T,3,3}
    R::AbstractArray{T,3}
    c::Union{Nothing,AbstractArray{T,2}}
    s::Union{Nothing,AbstractArray{T,2}}
end

rotation_linop(R::AbstractArray{T,3}; c::Union{Nothing,AbstractArray{T,2}}=nothing, s::Union{Nothing,AbstractArray{T,2}}=nothing) where {T<:Real} = RotationLinOp{T}(R, c, s)

function rotation_linop(φ::AbstractArray{T,2}; derivative::Bool=false) where {T<:Real}
    c = cos.(φ)
    s = sin.(φ)
    Rout = rotation_matrix(c, s; derivative=derivative)
    ~derivative ? (return RotationLinOp{T}(Rout, c, s)) : (return (RotationLinOp{T}(Rout[1], c, s), Rout[2]))
end

function rotation_matrix(c::AbstractArray{T,2}, s::AbstractArray{T,2}; derivative::Bool=false) where {T<:Real}
    c1 = c[:,1]; s1 = s[:,1]
    c2 = c[:,2]; s2 = s[:,2]
    c3 = c[:,3]; s3 = s[:,3]

    R = similar(c1, length(c1), 3, 3)
    R[:,1,1] .= c1.*c2;            R[:,1,2] .= -c2.*s1;           R[:,1,3] .= -s2;
    R[:,2,1] .= c3.*s1-c1.*s2.*s3; R[:,2,2] .= c1.*c3+s1.*s2.*s3; R[:,2,3] .= -c2.*s3;
    R[:,3,1] .= c1.*c3.*s2+s1.*s3; R[:,3,2] .= c1.*s3-c3.*s1.*s2; R[:,3,3] .= c2.*c3;

    if derivative
        ∂R = similar(c1, length(c1), 3, 3, 3)

        ∂R[:,1,1,1] .= -s1.*c2;            ∂R[:,1,2,1] .= -c2.*c1;            ∂R[:,1,3,1] .= 0;
        ∂R[:,2,1,1] .= c3.*c1+s1.*s2.*s3;  ∂R[:,2,2,1] .= -s1.*c3+c1.*s2.*s3; ∂R[:,2,3,1] .= 0;
        ∂R[:,3,1,1] .= -s1.*c3.*s2+c1.*s3; ∂R[:,3,2,1] .= -s1.*s3-c3.*c1.*s2; ∂R[:,3,3,1] .= 0;

        ∂R[:,1,1,2] .= -c1.*s2;     ∂R[:,1,2,2] .= s2.*s1;      ∂R[:,1,3,2] .= -c2;
        ∂R[:,2,1,2] .= -c1.*c2.*s3; ∂R[:,2,2,2] .= s1.*c2.*s3;  ∂R[:,2,3,2] .= s2.*s3;
        ∂R[:,3,1,2] .= c1.*c3.*c2;  ∂R[:,3,2,2] .= -c3.*s1.*c2; ∂R[:,3,3,2] .= -s2.*c3;

        ∂R[:,1,1,3] .= 0;                  ∂R[:,1,2,3] .= 0;                  ∂R[:,1,3,3] .= 0;
        ∂R[:,2,1,3] .= -s3.*s1-c1.*s2.*c3; ∂R[:,2,2,3] .= -c1.*s3+s1.*s2.*c3; ∂R[:,2,3,3] .= -c2.*c3;
        ∂R[:,3,1,3] .= -c1.*s3.*s2+s1.*c3; ∂R[:,3,2,3] .= c1.*c3+s3.*s1.*s2;  ∂R[:,3,3,3] .= -c2.*s3;
    end

    ~derivative ? (return R) : (return (R, ∂R))
end

AbstractLinearOperators.domain_size(R::RotationLinOp) = (size(R.R,1), "nk", 3)
AbstractLinearOperators.range_size(R::RotationLinOp) = domain_size(R)

function AbstractLinearOperators.matvecprod(R::RotationLinOp{T}, K::AbstractArray{T,3}) where {T<:Real}
    nt, nk, _ = size(K)
    return sum(reshape(R.R, nt, 1, 3, 3).*reshape(K, nt, nk, 1, 3); dims=4)[:,:,:,1]
end

function AbstractLinearOperators.matvecprod_adj(R::RotationLinOp{T}, K::AbstractArray{T,3}) where {T<:Real}
    nt, nk, _ = size(K)
    return sum(reshape(R.R, nt, 1, 3, 3).*reshape(K, nt, nk, 3, 1); dims=3)[:,:,1,:]
end


## Parameteric linear operator

struct RotationParametericLinOp end

rotation() = RotationParametericLinOp()

(::RotationParametericLinOp)(φ::AbstractArray{T,2}) where {T<:Real} = rotation_linop(φ)

(R::RotationParametericLinOp)() = R


## Parameteric delayed evaluation

struct RotationParametericDelayedEval{T}
    K::KSpaceFixedSizeSampling{T}
end

Base.:*(::RotationParametericLinOp, K::KSpaceFixedSizeSampling{T}) where {T<:Real} =  RotationParametericDelayedEval{T}(K)

(RK::RotationParametericDelayedEval{T})(φ::AbstractArray{T,2}) where {T<:Real} = rotation_linop(φ)*coord(RK.K)


## Jacobian of rotation evaluated

struct JacobianRotationEvaluated{T}<:AbstractLinearOperator{Complex{T},2,3}
    K::KSpaceFixedSizeSampling{T}
    ∂R::AbstractArray{T,4}
end

function Jacobian(RK::RotationParametericDelayedEval{T}, φ::AbstractArray{T,2}) where {T<:Real}
    Rφ, ∂Rφ = rotation_linop(φ; derivative=true)
    return Rφ*coord(RK.K), JacobianRotationEvaluated{T}(RK.K, ∂Rφ)
end

∂(RK::RotationParametericDelayedEval{T}, φ::AbstractArray{T,2}) where {T<:Real} = Jacobian(RK, φ)

AbstractLinearOperators.domain_size(∂RK::JacobianRotationEvaluated) = (size(∂RK.∂R,1),3)
AbstractLinearOperators.range_size(∂RK::JacobianRotationEvaluated) = (size(∂RK.K)...,3)

function AbstractLinearOperators.matvecprod(∂RK::JacobianRotationEvaluated{T}, Δφ::AbstractArray{CT,2}) where {T<:Real,CT<:Union{T,Complex{T}}}
    ΔR = sum(∂RK.∂R.*reshape(Δφ, :, 1, 1, 3); dims=4)[:,:,:,1]
    return rotation_linop(ΔR)*coord(∂RK.K)
end
Base.:*(∂RK::JacobianRotationEvaluated{T}, Δφ::AbstractArray{T,2}) where {T<:Real} = AbstractLinearOperators.matvecprod(∂RK, Δφ)

function AbstractLinearOperators.matvecprod_adj(∂RK::JacobianRotationEvaluated{T}, ΔRK::AbstractArray{CT,3}) where {T<:Real,CT<:Union{T,Complex{T}}}
    nt, nk, _ = size(ΔRK)
    KΔRK = sum(ΔRK.*reshape(coord(∂RK.K), nt, nk, 1, 3); dims=2)[:,1,:,:] 
    return sum(∂RK.∂R.*reshape(KΔRK, nt, 3, 3, 1); dims=2:3)[:,1,1,:]
end

function LinearAlgebra.dot(∇u::AbstractArray{Complex{T},3}, ∂RK::JacobianRotationEvaluated{T}) where {T<:Real}
    nt, nk, _ = size(∇u)
    ∂R = ∂RK.∂R
    K = coord(∂RK.K)
    J = similar(∇u, nt, nk, 3)
    ∂jRK = similar(∇u, nt, nk, 3)
    @inbounds for j = 1:3
        ∂jRK .= sum(reshape(∂R[:,:,:,j], nt, 1, 3, 3).*reshape(K, nt, nk, 1, 3); dims=4)[:,:,:,1]
        J[:,:,j] .= sum(∇u.*∂jRK; dims=3)[:,:,1]
    end
    return J
end