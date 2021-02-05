export MRacqgeom_gridded, MRacqgeom_fullgrid, MRacqgeom_cartesiangrid, MRdata_gridded, MRdata_fullgrid, MRdata_cartesiangrid


# MR acquisition geometry types

## Abstract

abstract type AbstractMRacqgeom end
abstract type AbstractMRacqgeom_gridded<:AbstractMRacqgeom end

## Gridded

struct MRacqgeom_gridded<:AbstractMRacqgeom_gridded
    geom::DomainCartesianK
    index::Array{Int64,1}
end
Base.isequal(md1::MRacqgeom_gridded, md2::MRacqgeom_gridded) = (md1.geom == md2.geom) && (md1.index == md2.index)
join_metadata(md1::MRacqgeom_gridded, md2::MRacqgeom_gridded) = (md1 == md2) ? (return md1) : (throw(ArgumentError("Metadata mismatch")))

## Full grid

struct MRacqgeom_fullgrid<:AbstractMRacqgeom_gridded
    geom::DomainCartesianK
end
Base.isequal(md1::MRacqgeom_fullgrid, md2::MRacqgeom_fullgrid) = (md1.geom == md2.geom)
join_metadata(md1::MRacqgeom_fullgrid, md2::MRacqgeom_fullgrid) = (md1 == md2) ? (return md1) : (throw(ArgumentError("Metadata mismatch")))

## Cartesian grid

struct MRacqgeom_cartesiangrid<:AbstractMRacqgeom_gridded
    geom::DomainCartesianK
    index_x::Array{Int64,1}
    index_y::Array{Int64,1}
end
Base.isequal(md1::MRacqgeom_cartesiangrid, md2::MRacqgeom_cartesiangrid) = (md1.geom == md2.geom) && (md1.index_x == md2.index_x) && (md1.index_y == md2.index_y)
join_metadata(md1::MRacqgeom_cartesiangrid, md2::MRacqgeom_cartesiangrid) = (md1 == md2) ? (return md1) : (throw(ArgumentError("Metadata mismatch")))


# MR-data types

## Abstract

abstract type AbstractMRdata{T,MDT}<:AbstractMetaDataArray{Array{Complex{T},1},MDT,Complex{T},1} end
abstract type AbstractMRdata_gridded{T}<:AbstractMRdata{T,MRacqgeom_gridded} end
abstract type AbstractMRdata_fullgrid{T}<:AbstractMRdata{T,MRacqgeom_fullgrid} end
abstract type AbstractMRdata_cartesiangrid{T}<:AbstractMRdata{T,MRacqgeom_cartesiangrid} end

## Gridded

struct MRdata_gridded{T}<:AbstractMRdata_gridded{T}
    array::Array{Complex{T},1}
    geom::DomainCartesianK
    index::Array{Int64,1}
    description::Union{Nothing,String}
    ground_truth::Union{Nothing,AbstractArray{T}}
end
MRdata_gridded(array::Array{Complex{T},1}, md::MRacqgeom_gridded) where T = MRdata_gridded{T}(array, md.geom, md.index, nothing, nothing)
raw_input(d::MRdata_gridded) = d.array
meta_data(d::MRdata_gridded) = MRacqgeom_gridded(d.geom, d.index)

struct CuMRdata_gridded{T}<:AbstractMRdata_gridded{T}
    array::CuArray{Complex{T},1}
    geom::DomainCartesianK
    index::Array{Int64,1}
    description::Union{Nothing,String}
    ground_truth::Union{Nothing,AbstractArray{T}}
end
CuMRdata_gridded(array::CuArray{Complex{T},1}, md::MRacqgeom_gridded) where T = CuMRdata_gridded{T}(array, md.geom, md.index, nothing, nothing)
raw_input(d::CuMRdata_gridded) = d.array
meta_data(d::CuMRdata_gridded) = MRacqgeom_gridded(d.geom, d.index)
MRdata_gridded(array::CuArray{Complex{T},1}, md::MRacqgeom_gridded) where T = CuMRdata_gridded{T}(array, md)

## Full grid

struct MRdata_fullgrid{T}<:AbstractMRdata_fullgrid{T}
    array::Array{Complex{T},1}
    geom::DomainCartesianK
    description::Union{Nothing,String}
    ground_truth::Union{Nothing,AbstractArray{T}}
end
MRdata_fullgrid(array::Array{Complex{T},1}, md::MRacqgeom_fullgrid) where T = MRdata_fullgrid{T}(array, md.geom, nothing, nothing)
raw_input(d::MRdata_fullgrid) = d.array
meta_data(d::MRdata_fullgrid) = MRacqgeom_fullgrid(d.geom)

struct CuMRdata_fullgrid{T}<:AbstractMRdata_fullgrid{T}
    array::CuArray{Complex{T},1}
    geom::DomainCartesianK
    description::Union{Nothing,String}
    ground_truth::Union{Nothing,AbstractArray{T}}
end
CuMRdata_fullgrid(array::CuArray{Complex{T},1}, md::MRacqgeom_fullgrid) where T = CuMRdata_fullgrid{T}(array, md.geom, nothing, nothing)
raw_input(d::CuMRdata_fullgrid) = d.array
meta_data(d::CuMRdata_fullgrid) = MRacqgeom_fullgrid(d.geom)
MRdata_fullgrid(array::CuArray{Complex{T},1}, md::MRacqgeom_fullgrid) where T = CuMRdata_fullgrid{T}(array, md)

## Cartesian grid

struct MRdata_cartesiangrid{T}<:AbstractMRdata_cartesiangrid{T}
    array::Array{Complex{T},1}
    geom::DomainCartesianK
    index_x::Array{Int64,1}
    index_y::Array{Int64,1}
    description::Union{Nothing,String}
    ground_truth::Union{Nothing,AbstractArray{T}}
end
MRdata_cartesiangrid(array::Array{Complex{T},1}, md::MRacqgeom_cartesiangrid) where T = MRdata_cartesiangrid{T}(array, md.geom, md.index_x, md.index_y, nothing, nothing)
raw_input(d::MRdata_cartesiangrid) = d.array
meta_data(d::MRdata_cartesiangrid) = MRacqgeom_cartesiangrid(d.geom, d.index_x, d.index_y)

struct CuMRdata_cartesiangrid{T}<:AbstractMRdata_cartesiangrid{T}
    array::CuArray{Complex{T},1}
    geom::DomainCartesianK
    index_x::Array{Int64,1}
    index_y::Array{Int64,1}
    description::Union{Nothing,String}
    ground_truth::Union{Nothing,AbstractArray{T}}
end
CuMRdata_cartesiangrid(array::CuArray{Complex{T},1}, md::MRacqgeom_cartesiangrid) where T = CuMRdata_cartesiangrid{T}(array, md.geom, md.index_x, md.index_y, nothing, nothing)
raw_input(d::CuMRdata_cartesiangrid) = d.array
meta_data(d::CuMRdata_cartesiangrid) = MRacqgeom_cartesiangrid(d.geom, d.index_x, d.index_y)
MRdata_cartesiangrid(array::CuArray{Complex{T},1}, md::MRacqgeom_cartesiangrid) where T = CuMRdata_cartesiangrid{T}(array, md)


# Injection/restriction

## Full grid

inject(d::MRdata_fullgrid) = scalar_field(copy(d.array), d.geom)
restrict(u::AbstractScalarField2D, md::MRacqgeom_fullgrid) = MRdata_fullgrid(copy(u.array), md)
restrict(u::AbstractScalarField2D, d::AbstractMRdata_fullgrid) = restrict(u, meta_data(d))

## Cartesian grid

function inject(d::MRdata_cartesiangrid)
    u = zeros_as(d.array, d.md.geom.nkx, d.md.geom.nky)
    u[d.index_x, d.index_y] .= reshape(d.array, length(d.index_x), length(d.index_y))
    return scalar_field(vec(u), d.geom)
end
restrict(u::AbstractScalarField2D, md::MRacqgeom_cartesiangrid) = MRdata_cartesiangrid(vec(reshape(u.array, u.geom.nkx, u.geom.nky)[md.index_x, md.index_y]), md)
restrict(u::AbstractScalarField2D, d::AbstractMRdata_cartesiangrid) = restriction(u, meta_data(d))


# Utils

function MRacqgeom_cartesiangrid(geom::DomainCartesianK, kx::AbstractArray{T,1}, ky::AbstractArray{T,1}) where {T<:Real}
    index_x = Int64.(round.(kx./geom.dkx)).+geom.okx
    index_y = Int64.(round.(ky./geom.dky)).+geom.oky
    return MRacqgeom_cartesiangrid(geom, index_x, index_y)
end

MRacqgeom_cartesiangrid(geom::DomainCartesian, kx::AbstractArray{T,1}, ky::AbstractArray{T,1}) where {T<:Real} = MRacqgeom_cartesiangrid(transform(geom), kx, ky)


# Plotting

function PyPlot.imshow(d::AbstractMRdata; title=nothing, cmap=nothing, vmin=nothing, vmax=nothing, save::Bool=false, fname="./pic.png", dpi=300, xlabel=L"k$_x$ (mm$^{-1}$)", ylabel=L"k$_y$ (mm$^{-1}$)", transparent=true, bbox_inches="tight", preproc::Function=x->real.(x))
    u = inject(d)
    PyPlot.imshow(u; title=title, cmap=cmap, vmin=vmin, vmax=vmax, save=save, fname=fname, dpi=dpi, xlabel=xlabel, ylabel=ylabel, transparent=transparent, bbox_inches=bbox_inches, preproc=preproc)
end