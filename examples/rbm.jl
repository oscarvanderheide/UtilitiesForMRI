using UtilitiesForMRI, LinearAlgebra, PyPlot

# Cartesian domain
n = (64, 64, 64)
o = n./2.0
h = (1.0, 1.0, 1.0)
X = spatial_sampling(o, n, h)

# Fourier operator (w/ standard k-space sampling)
nk = n[3]
nt = n[1]*n[2]
readout = :z
phase_encode = :xy
F = nfft_linop(X; readout=readout, phase_encode=phase_encode)

# Rigid-body perturbation
θ = zeros(Float64, nt, 6)
θ[:, 1] .= 20.0
θ[:, 2] .= 0.0
θ[:, 3] .= 0.0
θ[:, 4] .= pi/8
θ[:, 5] .= 0.0
θ[:, 6] .= 0.0
Fθ = F(θ)

# Rigid-body motion
u = zeros(ComplexF64, n); u[33-5:33+5, 33-5:33+5, 33-5:33+5] .= 1.0
u_rbm = F'*Fθ*u