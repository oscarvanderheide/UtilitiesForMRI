#: MR-data acquisition geometry

export MRacqgeomGridded2D, MRacqgeomFullgrid2D, MRacqgeomCartesiangrid2D, inject, restrict, acqgeom_fullgrid, acqgeom_cartesiangrid


# Abstract

abstract type AbstractMRacqgeom{T} end
abstract type AbstractMRacqgeomGridded2D{T}<:AbstractMRacqgeom{T} end


# Concrete

## Gridded

struct MRacqgeomGridded2D{T}<:AbstractMRacqgeomGridded2D{T}
    geom::DomainCartesian2D{T}
    index::Array{Int64,1}
end
Base.size(md::MRacqgeomGridded2D) = length(md.index)

function inject(d::AbstractArray{Complex{T},1}, md::MRacqgeomGridded2D{T}) where T
    u = typeof(d)(undef, size(md.geom))
    u .= T(0)
    u[md.index] .= d
    return u
end
restrict(u::AbstractArray{Complex{T},2}, md::MRacqgeomGridded2D{T}) where T = u[md.index]

## Full grid

struct MRacqgeomFullgrid2D{T}<:AbstractMRacqgeomGridded2D{T}
    geom::DomainCartesian2D{T}
end
Base.size(md::MRacqgeomFullgrid2D) = size(md.geom)

inject(d::AbstractArray{Complex{T},2}, ::MRacqgeomFullgrid2D{T}) where T = d
restrict(u::AbstractArray{Complex{T},2}, ::MRacqgeomFullgrid2D{T}) where T = u

## Cartesian grid

struct MRacqgeomCartesiangrid2D{T}<:AbstractMRacqgeomGridded2D{T}
    geom::DomainCartesian2D{T}
    index_x::Array{Int64,1}
    index_y::Array{Int64,1}
end
Base.size(md::MRacqgeomCartesiangrid2D) = (length(md.index_x), length(md.index_y))

function inject(d::AbstractArray{Complex{T},2}, md::MRacqgeomCartesiangrid2D{T}) where T
    u = typeof(d)(undef, size(md.geom))
    u .= T(0)
    u[md.index_x, md.index_y] = d
    return u
end
restrict(u::AbstractArray{Complex{T},2}, md::MRacqgeomCartesiangrid2D{T}) where T = u[md.index_x, md.index_y]


# Constructors

## Full

acqgeom_fullgrid(geom::DomainCartesian2D{T}) where T = MRacqgeomFullgrid2D{T}(geom)

## Cartesian

function acqgeom_cartesiangrid(geom::DomainCartesian2D{T}, kx::AbstractArray{T,1}, ky::AbstractArray{T,1}) where T
    dk = spacing(geom)
    o = origin(geom)
    index_x = Int64.(round.(kx/dk[1])).+o[1]
    index_y = Int64.(round.(ky/dk[2])).+o[2]
    return MRacqgeomCartesiangrid2D{T}(geom, index_x, index_y)
end