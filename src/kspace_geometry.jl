# k-space trajectory utilities

export FixedReadoutKSpaceGeometry, AlignedReadoutKSpaceGeometry, kspace_geometry, readout_dim, coord, downscale


## k-space trajectory

struct FixedReadoutKSpaceGeometry{T}<:AbstractFixedReadoutKSpaceGeometry{T}
    coordinates::AbstractArray{T, 3} # k-space coordinates (nt, nk, 3)
end

fixed_readout_kspace_geometry(coordinates::AbstractArray{T, 3}) where {T<:Real} = FixedReadoutKSpaceGeometry{T}(coordinates)

coord(K::FixedReadoutKSpaceGeometry) = K.coordinates

Base.getindex(K::AbstractFixedReadoutKSpaceGeometry, t::Integer) = coord(K)[t, :, :]

Base.size(K::AbstractFixedReadoutKSpaceGeometry) = size(coord(K))[1:2]

struct AlignedReadoutKSpaceGeometry{T}<:AbstractFixedReadoutKSpaceGeometry{T}
    phase_encoding::NTuple{2, Integer}
    readout::Integer
    coordinates_phase_encoding::AbstractArray{T, 2} # phase-encoding k-space coordinates (nt, 2)
    coordinates_readout::AbstractVector{T}          # readout k-space coordinates (nk, )
end

function kspace_geometry(X::CartesianSpatialGeometry{T}; phase_encoding::NTuple{2, Integer}=(1, 2), phase_encode_subsampling::Union{Nothing,AbstractVector{<:Integer}}=nothing, readout_subsampling::Union{Nothing,AbstractVector{<:Integer}}=nothing) where {T<:Real}

    # k-space coordinates
    readout = readout_dim(phase_encoding)
    perm = [phase_encoding..., readout]
    k_pe1, k_pe2, k_r = k_coord(X; mesh=false)[perm]
    n_pe1 = length(k_pe1); n_pe2 = length(k_pe2)
    k_pe1 = repeat(reshape(k_pe1, :, 1); outer=(1, n_pe2))
    k_pe2 = repeat(reshape(k_pe2, 1, :); outer=(n_pe1, 1))
    k_pe = [vec(k_pe1) vec(k_pe2)]

    # Subsampling
    ~isnothing(phase_encode_subsampling) && (k_pe = k_pe[phase_encode_subsampling, :])
    ~isnothing(readout_subsampling) && (k_r = k_r[readout_subsampling])

    return AlignedReadoutKSpaceGeometry{T}(phase_encoding, readout, k_pe, k_r)

end

function readout_dim(phase_encoding::NTuple{2, Integer})
    ((phase_encoding == (1, 2)) || (phase_encoding == (2, 1))) && (readout = 3)
    ((phase_encoding == (1, 3)) || (phase_encoding == (3, 1))) && (readout = 2)
    ((phase_encoding == (2, 3)) || (phase_encoding == (3, 2))) && (readout = 1)
    return readout
end

function coord(K::AlignedReadoutKSpaceGeometry; angular::Bool=false)
    nt, nk = size(K)
    k_pe = K.coordinates_phase_encoding
    k_r  = K.coordinates_readout
    k_pe = repeat(reshape(k_pe, :, 1, 2); outer=(1, nk, 1))
    k_r  = repeat(reshape(k_r,  1, :, 1); outer=(nt, 1, 1))
    k_coords = cat(k_pe, k_r; dims=3)[:, :, invperm(dims_permutation(K))]
    angular ? (return 2*T(pi)*k_coords) : (return k_coords)
end

Base.size(K::AlignedReadoutKSpaceGeometry) = (size(K.coordinates_phase_encoding, 1), length(K.coordinates_readout))

function Base.getindex(K::AlignedReadoutKSpaceGeometry, t::Integer; angular::Bool=false)
    _, nk = size(K)
    k_coords = cat(repeat(reshape(K.coordinates_phase_encoding[t, :], 1, 2); outer=(nk, 1)), reshape(K.coordinates_readout, :, 1); dims=2)[:, invperm(dims_permutation(K))]
    angular ? (return 2*T(pi)*k_coords) : (return k_coords)
end

dims_permutation(K::AlignedReadoutKSpaceGeometry) = [K.phase_encoding..., K.readout]


## Multiscale behavior

function downscale(K::AlignedReadoutKSpaceGeometry{T}, factor::Union{Integer,NTuple{3,Integer}}; scale_index::Bool=false) where {T<:Real}

    # Checking input
    (factor isa Integer) && (factor = (factor, factor, factor))
    factor = factor[dims_permutation(K)]

    # k-space coordinates & limits
    k_pe = K.coordinates_phase_encoding
    k_r  = K.coordinates_readout
    k_pe1_min = minimum(K.coordinates_phase_encoding[:, 1])/2^factor[1]
    k_pe2_min = minimum(K.coordinates_phase_encoding[:, 2])/2^factor[2]
    k_r_min   = minimum(K.coordinates_readout)/2^factor[3]
    k_pe1_max = maximum(K.coordinates_phase_encoding[:, 1])/2^factor[1]
    k_pe2_max = maximum(K.coordinates_phase_encoding[:, 2])/2^factor[2]
    k_r_max   = maximum(K.coordinates_readout)/2^factor[3]

    # Scaling
    pe_idx = findall((k_pe[:, 1] .<= k_pe1_max) .&& (k_pe[:, 1] .>= k_pe1_min) .&& (k_pe[:, 2] .<= k_pe2_max) .&& (k_pe[:, 2] .>= k_pe2_min))
    r_idx  = findall((k_r .<= k_r_max) .&& (k_r .>= k_r_min))
    K_h = AlignedReadoutKSpaceGeometry{T}(K.phase_encoding, K.readout, k_pe[pe_idx, :], k_r[r_idx])

    scale_index ? (return (K_h, pe_idx, r_idx)) : (return K_h)

end