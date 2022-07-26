# Rigid-body motion utilities

export CartesianSpatialGeometry, spatial_geometry, field_of_view, spacing, coord, k_coord, Nyquist


## Cartesian domain type

struct CartesianSpatialGeometry{T}<:AbstractSpatialGeometry{T}
    field_of_view::NTuple{3,T}  # field of view size
    nsamples::NTuple{3,Integer} # number of spatial samples
    origin::NTuple{3,T}         # origin wrt to (0,0,0)
end

spatial_geometry(field_of_view::NTuple{3,T}, nsamples::NTuple{3,Integer}; origin::NTuple{3,T}=field_of_view./2) where {T<:Real} = CartesianSpatialGeometry{T}(field_of_view, nsamples, origin)


## Utils

field_of_view(X::CartesianSpatialGeometry) = X.field_of_view

spacing(X::CartesianSpatialGeometry) = X.field_of_view./X.nsamples

Base.size(X::CartesianSpatialGeometry) = X.nsamples

function coord(X::CartesianSpatialGeometry{T}; mesh::Bool=false) where {T<:Real}
# Returns cell barycenter coordinate

    nx, ny, nz = X.nsamples
    hx, hy, hz = spacing(X)
    ox, oy, oz = X.origin
    x = ((1:nx).-T(0.5))*hx.-ox
    y = ((1:ny).-T(0.5))*hy.-oy
    z = ((1:nz).-T(0.5))*hz.-oz
    if mesh
        x = repeat(reshape(x, :,1,1); outer=(1,ny,nz))
        y = repeat(reshape(y, 1,:,1); outer=(nx,1,nz))
        z = repeat(reshape(z, 1,1,:); outer=(nx,ny,1))
    end
    return x, y, z

end

function k_coord(X::CartesianSpatialGeometry{T}; mesh::Bool=true, angular::Bool=true) where {T<:Real}
    Lx, Ly, Lz = X.field_of_view
    nx, ny, nz = X.nsamples
    kx = k_coord(Lx, nx; angular=angular); ky = k_coord(Ly, ny; angular=angular); kz = k_coord(Lz, nz; angular=angular)
    if mesh
        kx = repeat(reshape(kx, :, 1, 1); outer=(1, ny, nz))
        ky = repeat(reshape(ky, 1, :, 1); outer=(nx, 1, nz))
        kz = repeat(reshape(kz, 1, 1, :); outer=(nx, ny, 1))
    end
    return kx, ky, kz
end

function k_coord(L::T, n::Integer; angular::Bool=true) where {T<:Real}
    k = T.(-div(n,2):div(n,2))[1:n]/L
    angular ? (return 2*T(pi)*k) : (return k)
end

Nyquist(X::CartesianSpatialGeometry; angular::Bool=true) = (Nyquist(X.field_of_view[1], X.nsamples[1]; angular=angular), Nyquist(X.field_of_view[2], X.nsamples[2]; angular=angular), Nyquist(X.field_of_view[3], X.nsamples[3]; angular=angular))

Nyquist(L::T, n::Integer; angular::Bool=true) where {T<:Real} = angular ? 2*T(pi)*div(n,2)/L : div(n,2)/L