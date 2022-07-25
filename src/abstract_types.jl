# Abstract types

abstract type AbstractSpatialGeometry{T} end
abstract type AbstractKSpaceGeometry{T} end
abstract type AbstractFixedReadoutKSpaceGeometry{T}<:AbstractKSpaceGeometry{T} end