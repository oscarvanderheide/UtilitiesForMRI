# Abstract types

abstract type AbstractSpatialGeometry{T} end
abstract type AbstractCartesianSpatialGeometry{T}<:AbstractSpatialGeometry{T} end
abstract type AbstractKSpaceSampling{T} end