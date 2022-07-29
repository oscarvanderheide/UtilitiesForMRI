# k-space geometry utilities

export StructuredKSpaceSampling, kspace_sampling, coord, aligned_readout_sampling


## k-space sampling (= ordered wave-number coordinates)

struct StructuredKSpaceSampling{T}<:AbstractKSpaceSampling{T}
    coordinates_norm::AbstractArray{T,3} # normalized k-space coordinates; coord[:,:,i] = coord_norm[:,:,i]./norm[i]
    normalization::NTuple{3,T}           # convenience normalization factors due to nfft implementation
end

function kspace_sampling(coordinates::AbstractArray{T,3}; normalization::Union{Nothing,NTuple{3,T}}=nothing) where {T<:Real}
    isnothing(normalization) ? (coordinates_norm = coordinates) : (coordinates_norm = coordinates.*reshape([normalization...],1,1,3))
    return StructuredKSpaceSampling{T}(coordinates_norm, normalization)
end

Base.size(K::StructuredKSpaceSampling) = (size(K.coordinates_norm,1),size(K.coordinates_norm,2))

function Base.getindex(K::StructuredKSpaceSampling{T}, t::Integer; normalization::Union{Nothing,NTuple{3,T}}=nothing) where {T<:Real}
    isnothing(normalization) && (return K.coordinates_norm[t,:,:]./reshape([K.normalization...],1,3))
    (normalization == K.normalization) && (return K.coordinates_norm[t,:,:])
    return K.coordinates_norm.*reshape([normalization./K.normalization...],1,3)
end

function coord(K::StructuredKSpaceSampling{T}; normalization::Union{Nothing,NTuple{3,T}}=nothing) where {T<:Real}
    isnothing(normalization) && (return K.coordinates_norm./reshape([K.normalization...],1,1,3))
    (normalization == K.normalization) && (return K.coordinates_norm)
    return K.coordinates_norm.*reshape([normalization./K.normalization...],1,1,3)
end

function kspace_sampling(X::CartesianSpatialGeometry{T}, phase_encoding_dims::NTuple{2,Integer}; phase_encode_sampling::Union{Colon,AbstractVector{<:Integer}}=Colon(), readout_sampling::Union{Colon,AbstractVector{<:Integer}}=Colon()) where {T<:Real}
    nt = length(phase_encode_sampling); nk = length(readout_sampling)
    perm = dims_permutation(phase_encoding_dims)
    k_pe1, k_pe2, k_r = k_coord(X; mesh=false)[perm]
    n_pe1 = length(k_pe1); n_pe2 = length(k_pe2)
    k_pe = reshape(cat(repeat(reshape(k_pe1,:,1); outer=(1,n_pe2)), repeat(reshape(k_pe2,1,:); outer=(n_pe1,1)); dims=3),:,2)[phase_encode_sampling,:]
    k_r = k_r[readout_sampling]
    k_coordinates = cat(repeat(reshape(k_pe, nt,1,2); outer=(1,nk,1)), repeat(reshape(k_r, 1,nk,1); outer=(nt,1,1)); dims=3)[:,:,invperm(perm)]
    return kspace_sampling(k_coordinates; normalization=spacing(X))
end

function readout_dim(phase_encoding::NTuple{2, Integer})
    ((phase_encoding == (1, 2)) || (phase_encoding == (2, 1))) && (readout = 3)
    ((phase_encoding == (1, 3)) || (phase_encoding == (3, 1))) && (readout = 2)
    ((phase_encoding == (2, 3)) || (phase_encoding == (3, 2))) && (readout = 1)
    return readout
end

dims_permutation(phase_encoding_dims::NTuple{2,Integer}) = [phase_encoding_dims..., readout_dim(phase_encoding_dims)]