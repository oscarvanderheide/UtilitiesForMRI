# Abstract types


## Spatial sampling types

abstract type AbstractSpatialSampling{T} end
abstract type AbstractCartesianSpatialSampling{T}<:AbstractSpatialSampling{T} end


## k-space sampling types

abstract type AbstractKSpaceSampling{T} end
abstract type AbstractKSpaceFixedSizeSampling{T}<:AbstractKSpaceSampling{T} end
abstract type AbstractKSpaceCartesianSampling{T}<:AbstractKSpaceFixedSizeSampling{T} end