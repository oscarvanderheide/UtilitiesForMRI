# Resizing/Downscaling utilities

export downscale, subsample, upscale


## Spatial geometry

function downscale(X::CartesianSpatialGeometry, factor::Union{Integer,NTuple{3,Integer}})

    # Checking input
    factor isa Integer && (factor = (factor, factor, factor))    
    (factor == (0, 0, 0)) && (return X)

    # Calculating samples
    nsamples = div.(X.nsamples, 2 .^factor)
    nsamples = nsamples.+(mod.(nsamples, 2) .!= mod.(X.nsamples, 2))

    return spatial_geometry(X.field_of_view, nsamples; origin=X.origin)

end


## k-space geometry

downscale(K::CartesianKSpaceGeometry, factor::Union{Integer,NTuple{3,Integer}}) = kspace_geometry(downscale(K.X, factor))

function downscale(K::SampledCartesianKSpaceGeometry{T}, factor::Union{Integer,NTuple{3,Integer}}) where {T<:Real}

    # Checking input
    (factor isa Integer) && (factor = (factor, factor, factor))
    (factor == (0, 0, 0)) && (return K)

    # k-space coordinates & limits
    k_pe, k_r = phase_encode_coordinates(K)
    k_pe1_max, k_pe2_max, k_r_max = (Nyquist(K.K.X)./2 .^factor)[dims_permutation(K)]

    # Scaling
    pe_idx = findall((k_pe[:,1] .< k_pe1_max) .&& (k_pe[:,1] .>= -k_pe1_max) .&& (k_pe[:,2] .< k_pe2_max) .&& (k_pe[:,2] .>= -k_pe2_max))
    r_idx  = findall((k_r .< k_r_max) .&& (k_r .>= -k_r_max))

    # Subsampling scheme
    sampling = aligned_readout_sampling(K.sampling.dims[1:2]; phase_encode_sampling=K.sampling.phase_encoding[pe_idx], readout_sampling=K.sampling.readout[r_idx])

    return sample(K.K, sampling)

end


## Data array

function downscale(K::SampledCartesianKSpaceGeometry{T}, d::AbstractArray{CT,2}, factor::Union{Integer,NTuple{3,Integer}}) where {T<:Real,CT<:RealOrComplex{T}}

    Kq = downscale(K, factor)
    pe_idx = Kq.subsampling.phase_encoding
    r_idx = Kq.subsampling.readout
    return d[pe_idx, r_idx]

end

subsample(d::AbstractArray{CT,2}, Kq::SampledCartesianKSpaceGeometry{T}) where {T<:Real,CT<:RealOrComplex{T}} = d[Kq.sampling.phase_encoding, Kq.sampling.readout]

function downscale(u::AbstractArray{CT,3}, factor::Union{Integer,NTuple{3,Integer}}) where {T<:Real,CT<:RealOrComplex{T}}

    nx, ny, nz = size(u)
    idx_x = div(nx,2)+1-div(nx,2^(factor[1]+1)):div(nx,2)+div(nx,2^(factor[1]+1))+mod(nx,2)
    idx_y = div(ny,2)+1-div(ny,2^(factor[2]+1)):div(ny,2)+div(ny,2^(factor[2]+1))+mod(ny,2)
    idx_z = div(nz,2)+1-div(nz,2^(factor[3]+1)):div(nz,2)+div(nz,2^(factor[3]+1))+mod(nz,2)
    C = (length(idx_x)*length(idx_y)*length(idx_z))/(nx*ny*nz)
    return C*ifft(ifftshift(fftshift(fft(u))[idx_x,idx_y,idx_z]))

end

function upscale(u::AbstractArray{CT,3}, factor::Union{Integer,NTuple{3,Integer}}) where {T<:Real,CT<:RealOrComplex{T}}

    nx, ny, nz = size(u)
    (mod(nx,2) == 0) ? (nxq = 2^factor[1]*nx) : (nxq = 2^factor[1]*(nx-1)+1)
    (mod(ny,2) == 0) ? (nyq = 2^factor[2]*ny) : (nyq = 2^factor[2]*(ny-1)+1)
    (mod(nz,2) == 0) ? (nzq = 2^factor[3]*nz) : (nzq = 2^factor[3]*(nz-1)+1)
    idx_xq = div(nxq,2)+1-div(nxq,2^(factor[1]+1)):div(nxq,2)+div(nxq,2^(factor[1]+1))+mod(nxq,2)
    idx_yq = div(nyq,2)+1-div(nyq,2^(factor[2]+1)):div(nyq,2)+div(nyq,2^(factor[2]+1))+mod(nyq,2)
    idx_zq = div(nzq,2)+1-div(nzq,2^(factor[3]+1)):div(nzq,2)+div(nzq,2^(factor[3]+1))+mod(nzq,2)
    Uq = zeros(CT, nxq, nyq, nzq)
    Uq[idx_xq,idx_yq,idx_zq] .= fftshift(fft(u))
    C = (nxq*nyq*nzq)/(nx*ny*nz)
    return C*ifft(ifftshift(Uq))

end