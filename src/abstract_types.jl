# Abstract types

abstract type AbstractSpatialGeometry{T} end
abstract type AbstractKSpaceGeometry{T} end
abstract type AbstractCartesianKSpaceGeometry{T}<:AbstractKSpaceGeometry{T} end
abstract type AbstractKSpaceSampling end
abstract type AbstractCartesianKSpaceSampling<:AbstractKSpaceSampling end