# Rigid-body motion utilities

export RegularCartesianSpatialSampling
export spatial_sampling, coord


## Cartesian domain type

struct RegularCartesianSpatialSampling{T}<:AbstractCartesianSpatialSampling{T}
    n::NTuple{3,Integer}
    h::NTuple{3,T}
    idx_orig::NTuple{3,T}
end

idx_orig_default(n::Integer) = div(n, 2)+1

spatial_sampling(n::NTuple{3,Integer}; h::NTuple{3,T} = T.((1,1,1)), idx_orig::NTuple{3,T}=T.(idx_orig_default.(n))) where {T<:Real} = RegularCartesianSpatialSampling{T}(n, h, idx_orig)


## Utils

Base.size(X::RegularCartesianSpatialSampling) = X.n

function coord(X::RegularCartesianSpatialSampling{T}) where {T<:Real}
    x = reshape((collect(1:X.n[1]).-X.idx_orig[1])*X.h[1], :,1,1)
    y = reshape((collect(1:X.n[2]).-X.idx_orig[2])*X.h[2], 1,:,1)
    z = reshape((collect(1:X.n[3]).-X.idx_orig[3])*X.h[3], 1,1,:)
    return x, y, z
end

normalized_shift_nfft(X::RegularCartesianSpatialSampling{T}) where {T<:Real} = idx_orig_default.(X.n).-X.idx_orig