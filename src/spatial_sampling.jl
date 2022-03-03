# Rigid-body motion utilities

export RegularCartesianSpatialSampling
export spatial_sampling, coord, upscale, downscale


## Cartesian domain type

struct RegularCartesianSpatialSampling{T}<:AbstractCartesianSpatialSampling{T}
    n::NTuple{3,Integer}
    h::AbstractVector{T}
end

spatial_sampling(n::NTuple{3,Integer}; h::AbstractVector{T}=T.([1,1,1])) where {T<:Real} = RegularCartesianSpatialSampling{T}(n, h)


## Utils

Base.size(X::RegularCartesianSpatialSampling) = X.n

function coord(X::RegularCartesianSpatialSampling{T}; mesh::Bool=false) where {T<:Real}
    idx_orig = idx_orig_default(X.n)
    x = reshape((collect(1:X.n[1]).-idx_orig[1])*X.h[1], :,1,1)
    y = reshape((collect(1:X.n[2]).-idx_orig[2])*X.h[2], 1,:,1)
    z = reshape((collect(1:X.n[3]).-idx_orig[3])*X.h[3], 1,1,:)
    if mesh
        x = repeat(reshape(vec(x),:,1,1); outer=(1,X.n[2],X.n[3]))
        y = repeat(reshape(vec(y),1,:,1); outer=(X.n[1],1,X.n[3]))
        z = repeat(reshape(vec(z),1,1,:); outer=(X.n[1],X.n[2],1))
    end
    return x, y, z
end

idx_orig_default(n::Integer) = div(n,2)+1
idx_orig_default(n::NTuple{3,Integer}) = idx_orig_default.(n)

upscale(X::RegularCartesianSpatialSampling{T}; fact::Integer=1) where {T<:Real} = spatial_sampling(Integer.(X.n.*2.0^fact); h=X.h.*T(2)^-fact)

function downscale(X::RegularCartesianSpatialSampling{T}; fact::Integer=1) where {T<:Real}
    (fact == 0) && (X_h = X)
    (mod.(X.n,2^fact) == (0,0,0)) && (X_h = spatial_sampling(Integer.(X.n.*2.0^-fact); h=X.h.*T(2)^fact))
    return X_h
end