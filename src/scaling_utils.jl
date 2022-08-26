# Resizing/Downscaling utilities

export resample, subsample, subsampling_index


## Spatial geometry

resample(X::CartesianSpatialGeometry, n::NTuple{3,Integer}) = spatial_geometry(X.field_of_view, n; origin=X.origin)


## k-space geometry

function subsample(K::CartesianStructuredKSpaceSampling{T}, k_max::NTuple{3,T}) where {T<:Real}

    # k-space coordinates & limits
    k_pe, k_r = coord_phase_encoding(K), coord_readout(K)
    k_pe1_max, k_pe2_max, k_r_max = k_max[dims_permutation(K)]

    # Scaling
    pe_idx = findall((k_pe[:,1] .< k_pe1_max) .&& (k_pe[:,1] .>= -k_pe1_max) .&& (k_pe[:,2] .< k_pe2_max) .&& (k_pe[:,2] .>= -k_pe2_max))
    r_idx  = findall((k_r .< k_r_max) .&& (k_r .>= -k_r_max))

    return CartesianStructuredKSpaceSampling{T}(K.spatial_geometry, K.permutation_dims, K.idx_phase_encoding[pe_idx], K.idx_readout[r_idx])

end

subsample(K::CartesianStructuredKSpaceSampling{T}, X::CartesianSpatialGeometry{T}) where {T<:Real} = subsample(K, Nyquist(X))


## Data array

function subsample(K::CartesianStructuredKSpaceSampling{T}, d::AbstractArray{CT,2}, Kq::CartesianStructuredKSpaceSampling{T}; norm_constant::Union{Nothing,T}=nothing) where {T<:Real,CT<:RealOrComplex{T}}

    subidx_pe_q, subidx_r_q = subsampling_index(K, Kq)
    isnothing(norm_constant) ? (return d[subidx_pe_q, subidx_r_q]) : (return d[subidx_pe_q, subidx_r_q]*norm_constant)

end

function subsampling_index(K::CartesianStructuredKSpaceSampling{T}, Kq::CartesianStructuredKSpaceSampling{T}) where {T<:Real}

    nt_global = prod(K.spatial_geometry.nsamples[[K.permutation_dims[1:2]...]])
    nk_global = K.spatial_geometry.nsamples[K.permutation_dims[3]]
    nt_local, nk_local = size(K)
    subidx_pe_q = Vector{Integer}(undef,nt_global); subidx_pe_q[K.idx_phase_encoding] = 1:nt_local; subidx_pe_q = subidx_pe_q[Kq.idx_phase_encoding]
    subidx_r_q = Vector{Integer}(undef,nk_global); subidx_r_q[K.idx_readout] = 1:nk_local; subidx_r_q = subidx_r_q[Kq.idx_readout]

    return subidx_pe_q, subidx_r_q

end


# Reconstruction array

function resample(u::AbstractArray{CT,3}, n_scale::NTuple{3,Integer}) where {T<:Real,CT<:RealOrComplex{T}}

    # FFT
    n = size(u)
    U_scale = zeros(CT, n_scale); idx_scale = Vector{UnitRange{Integer}}(undef,3)
    U       = fftshift(fft(u));   idx       = Vector{UnitRange{Integer}}(undef,3)
    @inbounds for i = 1:3
        if n[i] <= n_scale[i]
            idx[i]       = 1:n[i]
            idx_scale[i] = div(n_scale[i],2)+1-div(n[i],2):div(n_scale[i],2)+1+div(n[i],2)-(mod(n[i],2)==0)
        else
            idx[i]       = div(n[i],2)+1-div(n_scale[i],2):div(n[i],2)+1+div(n_scale[i],2)-(mod(n_scale[i],2)==0)
            idx_scale[i] = 1:n_scale[i]
        end
    end
    U_scale[idx_scale...] .= U[idx...]
    
    return T(prod(n_scale)/prod(n))*ifft(ifftshift(U_scale))

end


# Filter utilities

function noringing_filter_1d(T::DataType, n::Integer)
    ;
end