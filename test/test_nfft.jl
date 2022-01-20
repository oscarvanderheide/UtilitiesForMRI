using UtilitiesForMRI, LinearAlgebra, CUDA, Test
CUDA.allowscalar(false)

# Cartesian domain
n = (256, 256, 256)
idx_orig = (1, 1, 129)
h = (1.0, 1.0, 1.0)
X = spatial_sampling(n; h=h, idx_orig=idx_orig)

# Fourier operator (w/ standard k-space sampling)
readout = :z
phase_encode = :xy
F = nfft_linop(X; readout=readout, phase_encode=phase_encode)

# Adjoint test
nt = n[1]*n[2]
nk = n[3]
d = randn(ComplexF64, nt, nk)
u = randn(ComplexF64, n)
@test dot(F*u, d) ≈ dot(u, F'*d) rtol=1e-6

# Rigid-body perturbation
θ = randn(Float64, nt, 6)
Fθ = F(θ)
@test dot(Fθ*u, d) ≈ dot(u, Fθ'*d) rtol=1e-6