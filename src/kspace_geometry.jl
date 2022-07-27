# k-space geometry utilities

export AlignedReadoutSampling, aligned_readout_sampling
export CartesianKSpaceGeometry, kspace_geometry, SampledCartesianKSpaceGeometry, sample, coord


## Subsampling schemes

struct AlignedReadoutSampling<:AbstractCartesianKSpaceSampling
    dims::NTuple{3,Integer}
    phase_encoding::Union{Colon,AbstractVector{<:Integer}} # phase-encoding k-space subsampling
    readout::Union{Colon,AbstractVector{<:Integer}}        # readout k-space subsampling
end

aligned_readout_sampling(phase_encoding_dims::NTuple{2,Integer}; phase_encode_sampling::Union{Colon,AbstractVector{<:Integer}}=Colon(), readout_sampling::Union{Colon,AbstractVector{<:Integer}}=Colon()) = AlignedReadoutSampling((phase_encoding_dims..., readout_dim(phase_encoding_dims)), phase_encode_sampling, readout_sampling)

function Base.size(sampling::AlignedReadoutSampling)
    (sampling.phase_encoding isa Colon) ? (nt = nothing) : (nt = length(sampling.phase_encoding))
    (sampling.readout isa Colon)        ? (nk = nothing) : (nk = length(sampling.readout))
    return nt, nk
end

function readout_dim(phase_encoding::NTuple{2, Integer})
    ((phase_encoding == (1, 2)) || (phase_encoding == (2, 1))) && (readout = 3)
    ((phase_encoding == (1, 3)) || (phase_encoding == (3, 1))) && (readout = 2)
    ((phase_encoding == (2, 3)) || (phase_encoding == (3, 2))) && (readout = 1)
    return readout
end

dims_permutation(sampling::AlignedReadoutSampling) = [sampling.dims...]


## Cartesian k-space geometry (fully sampled)

struct CartesianKSpaceGeometry{T}<:AbstractCartesianKSpaceGeometry{T}
    X::CartesianSpatialGeometry{T}
end

kspace_geometry(X::CartesianSpatialGeometry) = CartesianKSpaceGeometry(X)

Base.size(K::CartesianKSpaceGeometry) = (prod(size(K.X)), 1)

function Base.getindex(K::CartesianKSpaceGeometry, t::Integer; angular::Bool=true)
    kx, ky, kz = k_coord(K.X; mesh=false, angular=angular)
    I = CartesianIndices(size(K.X))[t]
    return [kx[I[1]], ky[I[2]], kz[I[3]]]
end 

function coord(K::CartesianKSpaceGeometry{T}; angular::Bool=true) where {T<:Real}
    kx, ky, kz = k_coord(K.X; mesh=true, angular=angular)
    return reshape([vec(kx) vec(ky) vec(kz)], :, 1, 3)
end


## Cartesian k-space geometry (subsampled)

struct SampledCartesianKSpaceGeometry{T}<:AbstractCartesianKSpaceGeometry{T}
    K::CartesianKSpaceGeometry{T}
    sampling::AbstractCartesianKSpaceSampling
end

sample(K::CartesianKSpaceGeometry{T}, sampling::AbstractCartesianKSpaceSampling) where {T<:Real} = SampledCartesianKSpaceGeometry{T}(K, sampling)

function Base.size(K::SampledCartesianKSpaceGeometry)
    nt, nk = size(K.sampling)
    isnothing(nt) && (nt = prod(size(K.K.X)[[K.sampling.dims[1:2]...]]))
    isnothing(nk) && (nk = size(K.K.X)[K.sampling.dims[3]])
    return nt, nk
end

function Base.getindex(K::SampledCartesianKSpaceGeometry{T}, t::Integer; angular::Bool=true) where {T<:Real}
    _, nk = size(K)
    k_pe, k_r = phase_encode_coordinates(K; angular=angular)
    return [k_pe[t,1]*ones(T, nk) k_pe[t,2]*ones(T, nk) k_r][:, invperm(dims_permutation(K))]
end

function coord(K::SampledCartesianKSpaceGeometry; angular::Bool=true, phase_encoded::Bool=false)
    nt, nk = size(K)
    k_pe, k_r = phase_encode_coordinates(K; angular=angular)
    phase_encoded && (return (k_pe, k_r))
    k_pe = repeat(reshape(k_pe, :, 1, 2); outer=(1, nk, 1))
    k_r  = repeat(reshape(k_r,  1, :, 1); outer=(nt, 1, 1))
    return cat(k_pe, k_r; dims=3)[:, :, invperm(dims_permutation(K))]
end

dims_permutation(K::SampledCartesianKSpaceGeometry) = dims_permutation(K.sampling)

function phase_encode_coordinates(K::SampledCartesianKSpaceGeometry; angular::Bool=true)
    k_pe1, k_pe2, k_r = k_coord(K.K.X; mesh=false, angular=angular)[dims_permutation(K)]
    k_pe = reshape(cat(repeat(reshape(k_pe1, :, 1); outer=(1,length(k_pe2))), repeat(reshape(k_pe2, 1, :); outer=(length(k_pe1),1)); dims=3), :, 2)[K.sampling.phase_encoding, :]
    return k_pe, k_r[K.sampling.readout]
end