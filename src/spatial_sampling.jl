# Rigid-body motion utilities

export RegularCartesianSpatialSampling, spatial_sampling, field_of_view, spacing, Nyquist_frequency, coord, upscale, downscale


## Cartesian domain type

struct RegularCartesianSpatialSampling{T}<:AbstractCartesianSpatialSampling{T}
    field_of_view::NTuple{3,T}  # field of view size
    nsamples::NTuple{3,Integer} # number of spatial samples
    origin::NTuple{3,T}         # origin wrt to (0,0,0)
end

spatial_sampling(field_of_view::NTuple{3,T}, nsamples::NTuple{3,Integer}; origin::NTuple{3,T}=field_of_view./2) where {T<:Real} = RegularCartesianSpatialSampling{T}(field_of_view, nsamples, origin)


## Utils

field_of_view(X::RegularCartesianSpatialSampling) = X.field_of_view

spacing(X::RegularCartesianSpatialSampling) = X.field_of_view./X.nsamples

Nyquist_frequency(X::RegularCartesianSpatialSampling) = 1 ./(2 .*spacing(X))

Base.size(X::RegularCartesianSpatialSampling) = X.nsamples

function coord(X::RegularCartesianSpatialSampling{T}; mesh::Bool=false) where {T<:Real}
# Returns cell barycenter coordinate

    h = spacing(X)
    x = reshape((collect(1:X.nsamples[1]).-T(0.5))*h[1].-X.origin[1], :,1,1)
    y = reshape((collect(1:X.nsamples[2]).-T(0.5))*h[2].-X.origin[2], 1,:,1)
    z = reshape((collect(1:X.nsamples[3]).-T(0.5))*h[3].-X.origin[3], 1,1,:)
    if mesh
        x = repeat(x; outer=(1,X.nsamples[2],X.nsamples[3]))
        y = repeat(y; outer=(X.nsamples[1],1,X.nsamples[3]))
        z = repeat(z; outer=(X.nsamples[1],X.nsamples[2],1))
    end
    return x, y, z

end

upscale(X::RegularCartesianSpatialSampling; factor::Integer=2) = spatial_sampling(X.field_of_view, X.nsamples.*factor; origin=X.origin)

downscale(X::RegularCartesianSpatialSampling; factor::Integer=2) = spatial_sampling(X.field_of_view, Integer.(round.(X.nsamples./factor)); origin=X.origin)