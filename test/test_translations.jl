using UtilitiesForMRI, LinearAlgebra, CUDA, Test
CUDA.allowscalar(false)

# Cartesian domain
n = (256, 256, 256)
h = (abs(randn()), abs(randn()), abs(randn()))
X = spatial_sampling(Float64, n; h=h)

# Cartesian sampling in k-space
phase_encoding = (1,2)
K = kspace_Cartesian_sampling(X; phase_encoding=phase_encoding)

# Adjoint test (linear operator)
P = phase_shift(K)
nt, nk = size(K)
τ = randn(Float64, nt, 3)
Pτ = P(τ)
d = randn(ComplexF64, nt, nk)
e = randn(ComplexF64, nt, nk)
@test dot(Pτ*d, e) ≈ dot(d, Pτ'*e) rtol=1e-6

# Jacobian
Pτd, Pτ, ∂Pτd = ∂(P()*d, τ)
@test Pτd ≈ Pτ*d rtol=1e-6

# Adjoint test (Jacobian)
Δτ = randn(ComplexF64, nt, 3); Δτ *= norm(τ)/norm(Δτ)
Δd_ = ∂Pτd*Δτ; Δd = randn(ComplexF64, size(Δd_)); Δd *= norm(Δd_)/norm(Δd)
@test dot(∂Pτd*Δτ, Δd) ≈ dot(Δτ, ∂Pτd'*Δd) rtol=1e-6

# Gradient test
Δτ = randn(Float64, nt, 3); Δτ *= norm(τ)/norm(Δτ)
t = 1f-6
Pτd_p1 = P(τ+0.5*t*Δτ)*d
Pτd_m1 = P(τ-0.5*t*Δτ)*d
@test (Pτd_p1-Pτd_m1)/t ≈ ∂Pτd*Δτ rtol=1e-5