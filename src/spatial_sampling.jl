# Rigid-body motion utilities

export RegularCartesianSpatialSampling
export spatial_sampling, coord


## Cartesian domain type

struct RegularCartesianSpatialSampling{T}<:AbstractCartesianSpatialSampling{T}
    n::NTuple{3,Integer}
    h::NTuple{3,T}
end

spatial_sampling(n::NTuple{3,Integer}; h::NTuple{3,T} = T.((1,1,1))) where {T<:Real} = RegularCartesianSpatialSampling{T}(n, h)


## Utils

Base.size(X::RegularCartesianSpatialSampling) = X.n

function coord(X::RegularCartesianSpatialSampling{T}) where {T<:Real}
    idx_orig = idx_orig_default(X.n)
    x = reshape((collect(1:X.n[1]).-idx_orig[1])*X.h[1], :,1,1)
    y = reshape((collect(1:X.n[2]).-idx_orig[2])*X.h[2], 1,:,1)
    z = reshape((collect(1:X.n[3]).-idx_orig[3])*X.h[3], 1,1,:)
    return x, y, z
end

idx_orig_default(n::Integer) = div(n,2)+1
idx_orig_default(n::NTuple{3,Integer}) = idx_orig_default.(n)