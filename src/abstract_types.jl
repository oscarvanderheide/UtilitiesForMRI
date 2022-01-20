# Abstract types


## Spatial sampling types

abstract type AbstractSpatialSampling{T} end
abstract type AbstractCartesianSpatialSampling{T}<:AbstractSpatialSampling{T} end


## k-space sampling types

abstract type AbstractKSpaceOrderedSampling{T} end
abstract type AbstractKSpaceFixedSizeSampling{T}<:AbstractKSpaceOrderedSampling{T} end


## NFFT types

abstract type AbstractNFFT{T,NX,NK}<:AbstractLinearOperator{T,NX,NK} end