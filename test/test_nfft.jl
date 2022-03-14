using UtilitiesForMRI, LinearAlgebra, Test, Random

# Cartesian domain
n = (256,256,256)
h = (abs(randn()), abs(randn()), abs(randn()))
X = spatial_sampling(Float64, n; h=h)

# Cartesian sampling in k-space
phase_encoding = (1,2)
subsampling = (1:256^2)[randperm(256^2)][1:32]
K = kspace_Cartesian_sampling(X; phase_encoding=phase_encoding, subsampling=subsampling)

# Fourier operator
tol = 1e-6
F = nfft(X, K; tol=tol)

# Adjoint test (linear operator)
nt, nk = size(K)
θ = 1e-3*pi*randn(Float64, nt, 6)
Fθ = F(θ)
u = randn(ComplexF64, n)
d = randn(ComplexF64, nt, nk)
@test dot(Fθ*u, d) ≈ dot(u, Fθ'*d) rtol=1e-6

# Jacobian & consistency test
d, _, ∂Fθu = ∂(F()*u, θ)
@test d ≈ Fθ*u rtol=1e-6

# Adjoint test (Jacobian)
Δθ = randn(Float64, nt, 6); Δθ *= norm(θ)/norm(Δθ)
ΔF_ = ∂Fθu*Δθ; ΔF = randn(ComplexF64, size(ΔF_)); ΔF *= norm(ΔF_)/norm(ΔF)
@test dot(∂Fθu*Δθ, ΔF) ≈ dot(Δθ, ∂Fθu'*ΔF) rtol=1e-6

# Gradient test
Δθ = randn(Float64, nt, 6); Δθ *= norm(θ)/norm(Δθ)
t = 1e-6
Fθu_p1 = F(θ+0.5*t*Δθ)*u
Fθu_m1 = F(θ-0.5*t*Δθ)*u
@test (Fθu_p1-Fθu_m1)/t ≈ ∂Fθu*Δθ rtol=1e-3

# Sparse matrix Gauss-Newton
H = sparse_matrix_GaussNewton(∂Fθu) # ∂Fθu'*∂Fθu
Δθ = randn(Float64, nt, 6); Δθ *= norm(θ)/norm(Δθ)
@test H*vec(Δθ) ≈ vec(real(∂Fθu'*(∂Fθu*Δθ))) rtol=1e-6