export MRmd_gridded, MRmd_cartesian


# MR metadata types

abstract type AbstractMRmetadata end
abstract type AbstractMRmd_gridded<:AbstractMRmetadata end

struct MRmd_gridded<:AbstractMRmd_gridded
    geom::DomainCartesian
    index::Array{Int64,1}
end
Base.isequal(md1::MRmd_gridded, md2::MRmd_gridded) = (md1.geom == md2.geom) && (md1.index == md2.index)
join_metadata(md1::MRmd_gridded, md2::MRmd_gridded) = (md1 == md2) ? (return md1) : (throw(ArgumentError("Metadata mismatch")))


# MR data types

abstract type AbstractMRdata{T,MDT}<:AbstractMetaDataArray{Array{Complex{T},1},MDT,Complex{T},1} end
abstract type AbstractMRdata_gridded{T}<:AbstractMRdata{T,MRmd_gridded} end

struct MRdata_gridded{T}<:AbstractMRdata_gridded{T}
    array::Array{Complex{T},1}
    md::MRmd_gridded
    description::Union{Nothing,String}
    ground_truth::Union{Nothing,AbstractArray{T}}
end
MRdata_gridded(array::Array{Complex{T},1}, md::MRmd_gridded) where T = MRdata_gridded{T}(array, md, nothing, nothing)
raw_input(d::MRdata_gridded) = d.array
meta_data(d::MRdata_gridded) = d.md

struct CuMRdata_gridded{T}<:AbstractMRdata_gridded{T}
    array::CuArray{Complex{T},1}
    md::MRmd_gridded
    description::Union{Nothing,String}
    ground_truth::Union{Nothing,AbstractArray{T}}
end
CuMRdata_gridded(array::CuArray{Complex{T},1}, md::MRmd_gridded) where T = CuMRdata_gridded{T}(array, md, nothing, nothing)
raw_input(d::CuMRdata_gridded) = d.array
meta_data(d::CuMRdata_gridded) = d.md

MRdata_gridded(array::CuArray{Complex{T},1}, md::MRmd_gridded) where T = CuMRdata_gridded{T}(array, md, nothing, nothing)