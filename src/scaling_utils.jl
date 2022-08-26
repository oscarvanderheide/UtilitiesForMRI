# Resizing/Downscaling utilities

export resample, subsample, subsampling_index, noringing_filter_1d, noringing_filter


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

function subsample(K::CartesianStructuredKSpaceSampling{T}, d::AbstractArray{CT,2}, Kq::CartesianStructuredKSpaceSampling{T}; norm_constant::Union{Nothing,T}=nothing, damping_factor::Union{T,Nothing}=nothing) where {T<:Real,CT<:RealOrComplex{T}}

    # Check input
    (K == Kq) && (isnothing(norm_constant) ? (return d) : (return d*norm_constant))

    # Subsampling indexes
    subidx_pe_q, subidx_r_q = subsampling_index(K, Kq)

    # Normalization
    isnothing(norm_constant) ? (d_subsampled = d[subidx_pe_q, subidx_r_q]) : (d_subsampled = d[subidx_pe_q, subidx_r_q]*norm_constant)

    # Filtering
    if ~isnothing(damping_factor)
        filter = noringing_filter(K, Kq; damping_factor=damping_factor)
        d_subsampled .*= filter[subidx_pe_q, subidx_r_q]
    end

    return d_subsampled

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

function resample(u::AbstractArray{CT,3}, n_scale::NTuple{3,Integer}; damping_factor::Union{T,Nothing}=nothing) where {T<:Real,CT<:RealOrComplex{T}}

    # Check sizes
    n = size(u)
    (n == n_scale) && (return u)

    # Computing FFT
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

    # Filtering
    if ~isnothing(damping_factor)
        filter = noringing_filter(T, n, n_scale; damping_factor=damping_factor)
        U_scale .*= filter[idx...]
    end
    
    return T(prod(n_scale)/prod(n))*ifft(ifftshift(U_scale))

end


# Filter utilities

function noringing_filter_1d(T::DataType, n::Integer, n_cutoff::Integer; damping_factor::Union{Real,Nothing}=nothing)
    (isnothing(damping_factor) || (n <= n_cutoff)) && (return ones(T,n))
    ((damping_factor < 0) || (damping_factor > 1)) && error("Damping factor must be between 0 and 1")
    x = convert.(T, (-div(n,2):div(n,2)-(mod(n,2)==0)))
    x_cutoff = convert(T,div(n_cutoff,2)-(mod(n_cutoff,2)==0))
    σ2 = -x_cutoff^2/(2*log(convert(T,damping_factor)))
    return exp.(-x.^2/(2*σ2))
end

function noringing_filter(T::DataType, n::NTuple{N,Integer}, n_cutoff::NTuple{N,Integer}; damping_factor::Union{Real,Nothing}=nothing) where N
    f = Vector{Array{T,N}}(undef,N)
    @inbounds for i = 1:N
        sz = ones(Integer,N); sz[i] = n[i]
        f[i] = Array{T,N}(undef,sz...)
        f[i][:] .= noringing_filter_1d(T, n[i], n_cutoff[i]; damping_factor=damping_factor)
    end
    return .*(f...)
end

function noringing_filter(K::CartesianStructuredKSpaceSampling{T}, Kq::CartesianStructuredKSpaceSampling{T}; damping_factor::Union{T,Nothing}=nothing) where {T<:Real}
    nt, nk = size(K)
    isnothing(damping_factor) && (return ones(T,nt,nk))
    k_pe,  k_r  = coord_phase_encoding(K),  coord_readout(K)
    absk2_pe = sum(k_pe.^2; dims=2); absk2_r = k_r.^2
    kq_pe, kq_r = coord_phase_encoding(Kq), coord_readout(Kq)
    max_abskq2_pe = maximum(sum(kq_pe.^2; dims=2)); max_abskq2_r = maximum(kq_r)^2
    σ2_pe = -max_abskq2_pe/(2*log(damping_factor)); σ2_r = -max_abskq2_r/(2*log(damping_factor))
    return reshape(exp.(-absk2_pe/(2*σ2_pe)), nt, 1).*reshape(exp.(-absk2_r/(2*σ2_r)), 1, nk)
end