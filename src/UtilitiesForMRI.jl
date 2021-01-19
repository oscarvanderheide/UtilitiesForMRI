module UtilitiesForMRI

using LinearAlgebra, CUDA, Flux, FFTW, PyPlot, VectorFields, AbstractLinearOperators, MetaDataArrays
import MetaDataArrays: raw_data, meta_data, join_metadata

# Fourier linear operator
include("./fft.jl")

# MRI data types
include("./data.jl")

end