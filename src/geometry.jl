#: Define domain geometry types


export DomainCartesian2D, geometry_cartesian_2D, extent, spacing, ktransform


# Geometry abstract type

abstract type AbstractDomainGeometry{T} end
abstract type AbstractDomainCartesian2D{T}<:AbstractDomainGeometry{T} end


# Cartesian geometry

struct DomainCartesian2D{T}<:AbstractDomainCartesian2D{T}
    origin::Tuple{Int64,Int64}
    size::Tuple{Int64,Int64}
    spacing::Tuple{T,T}
    unit::Union{String,Nothing}
end

geometry_cartesian_2D(ox::Int64, oy::Int64, nx::Int64, ny::Int64, dx::T, dy::T, unit::String) where T = DomainCartesian2D{T}((ox,oy), (nx,ny), (dx,dy), unit)
geometry_cartesian_2D(ox::Int64, oy::Int64, nx::Int64, ny::Int64, dx::T, dy::T) where T = DomainCartesian2D{T}((ox,oy), (nx,ny), (dx,dy), nothing)
geometry_cartesian_2D(nx::Int64, ny::Int64, dx::T, dy::T) where T = DomainCartesian2D{T}((1,1), (nx,ny), (dx,dy), nothing)


# Geometry utils

Base.size(geom::DomainCartesian2D) = geom.size

spacing(geom::DomainCartesian2D) = geom.spacing

extent(geom::DomainCartesian2D) = (-geom.spacing[1]*(geom.origin[1]-1), geom.spacing[1]*(geom.size[1]-geom.origin[1]), -geom.spacing[2]*(geom.origin[2]-1), geom.spacing[2]*(geom.size[2]-geom.origin[2]))

function ktransform(geom::DomainCartesian2D{T}; centered::Bool=true) where T
    if centered
        return geometry_cartesian_2D(div(geom.size[1],2)+1, div(geom.size[2],2)+1, geom.size[1], geom.size[2], T(1/(geom.size[1]*geom.spacing[1])), T(1/(geom.size[2]*geom.spacing[2])))
    else
        return geometry_cartesian_2D(1, 1, geom.size[1], geom.size[2], T(1/(geom.size[1]*geom.spacing[1])), T(1/(geom.size[2]*geom.spacing[2])))
    end
end