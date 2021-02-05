using LinearAlgebra, CUDA, Flux, Test, JLD, UtilitiesForMRI, PyPlot
CUDA.allowscalar(false)

# Load raw input
@load "./data/BrainWebA_T1spiral2.jld"