# Resizing/Downscaling utilities

export rescale


## Spatial geometry

rescale(X::CartesianSpatialGeometry, n::NTuple{3,Integer}) = spatial_geometry(X.field_of_view, n; origin=X.origin)


## k-space geometry

function rescale(K::CartesianStructuredKSpaceSampling{T}, k_max::NTuple{3,T}) where {T<:Real}

    # k-space coordinates & limits
    k_pe, k_r = coord_phase_encoding(K), coord_readout(K)
    k_pe1_max, k_pe2_max, k_r_max = k_max[dims_permutation(K)]

    # Scaling
    pe_idx = findall((k_pe[:,1] .< k_pe1_max) .&& (k_pe[:,1] .>= -k_pe1_max) .&& (k_pe[:,2] .< k_pe2_max) .&& (k_pe[:,2] .>= -k_pe2_max))
    r_idx  = findall((k_r .< k_r_max) .&& (k_r .>= -k_r_max))

    return CartesianStructuredKSpaceSampling{T}(K.spatial_geometry, K.permutation_dims, K.idx_phase_encoding[pe_idx], K.idx_readout[r_idx])

end

rescale(K::CartesianStructuredKSpaceSampling{T}, X::CartesianSpatialGeometry{T}) where {T<:Real} = rescale(K, Nyquist(X))


## Data array

rescale(d::AbstractArray{CT,2}, K::CartesianStructuredKSpaceSampling{T}) where {T<:Real,CT<:RealOrComplex{T}} = d[K.idx_phase_encoding, K.idx_readout]


# Reconstruction array

function rescale(u::AbstractArray{CT,3}, n_scale::NTuple{3,Integer}) where {T<:Real,CT<:RealOrComplex{T}}

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