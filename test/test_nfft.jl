using UtilitiesForMRI, LinearAlgebra, CUDA, Test
CUDA.allowscalar(false)

# Cartesian domain
n = (256,256,256)
o = n./2.0
h = (1.0, 1.0, 1.0)
X = spatial_sampling(o, n, h)

# Fourier operator (w/ standard k-space sampling)
readout = :z
phase_encode = :xy
F = nfft_linop(X; readout=readout, phase_encode=phase_encode)

# Adjoint test
nk = n[3]
nt = n[1]*n[2]
d = randn(ComplexF64, nk, nt)
u = randn(ComplexF64, n)
@test dot(F*u, d) ≈ dot(u, F'*d) rtol=1e-6

# Rigid-body perturbation
θ = randn(Float64, nt, 6)
Fθ = F(θ)
@test dot(Fθ*u, d) ≈ dot(u, Fθ'*d) rtol=1e-6