# Main functions

## Spatial geometry utilities

These are the main utilities to build spatial discretization objects.

```@docs
spatial_geometry(field_of_view::NTuple{3,T}, nsamples::NTuple{3,Integer}; origin::NTuple{3,T}=field_of_view./2) where {T<:Real}
```

## ``k``-space geometry utilities

These are the main utilities to specify ``k``-space acquisition trajectories, where the ordering is typically dictated by the order of acquisition (e.g., a proxy for time).

```@docs
kspace_sampling(permutation_dims::NTuple{3,Integer}, coord_phase_encoding::AbstractArray{T,2}, coord_readout::AbstractVector{T}) where {T<:Real}
```

```@docs
kspace_sampling(X::CartesianSpatialGeometry{T}, phase_encoding_dims::NTuple{2,Integer}; phase_encode_sampling::Union{Nothing,AbstractVector{<:Integer}}=nothing, readout_sampling::Union{Nothing,AbstractVector{<:Integer}}=nothing) where {T<:Real}
```

## Resampling utilities

We list the main functionalities for subsampling/upsampling of spatial geometries, ``k``-space geometries, and 3D images:

## Image-quality metrics

Convenience functions to compute slice-based PSNR and SSIM metrics for 3D images (these relies on the package `ImageQualityMetrics` ):

```@docs
psnr(u_noisy::AbstractArray{T,3}, u_ref::AbstractArray{T,3}; slices::Union{Nothing,NTuple{N,VolumeSlice}}=nothing, orientation::Orientation=standard_orientation()) where {T<:Real,N}
```

```@docs
ssim(u_noisy::AbstractArray{T,3}, u_ref::AbstractArray{T,3}; slices::Union{Nothing,NTuple{N,VolumeSlice}}=nothing, orientation::Orientation=standard_orientation()) where {T<:Real,N}
```

## Visualization tools

```@docs
orientation(perm::NTuple{3,Integer}; reverse::NTuple{3,Bool}=(false,false,false))
```

```@docs
standard_orientation()
```

```@docs
volume_slice(dim, n; window=nothing)
```

```@docs
select(u::AbstractArray{T,3}, slice::VolumeSlice; orientation::Orientation=standard_orientation()) where {T<:Real}
```

```@docs
plot_volume_slices(u::AbstractArray{T,3};
    slices::Union{Nothing,NTuple{N,VolumeSlice}}=nothing,
    spatial_geometry::Union{Nothing,CartesianSpatialGeometry{T}}=nothing,
    cmap::String="gray",
    vmin::Union{Nothing,Real}=nothing, vmax::Union{Nothing,Real}=nothing,
    xlabel::Union{Nothing,AbstractString}=nothing, ylabel::Union{Nothing,AbstractString}=nothing,
    cbar_label::Union{Nothing,AbstractString}=nothing,
    title::Union{Nothing,AbstractString}=nothing,
    savefile::Union{Nothing,String}=nothing,
    orientation::Orientation=standard_orientation()) where {T<:Real,N}
```