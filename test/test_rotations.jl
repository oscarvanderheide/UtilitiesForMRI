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
R = rotation()
nt, nk = size(K)
φ = randn(Float64, nt, 3)
Rφ = R(φ)
k = randn(Float64, nt, nk, 3)
l = randn(Float64, nt, nk, 3)
@test dot(Rφ*k, l) ≈ dot(k, Rφ'*l) rtol=1e-6

# Jacobian
RφK, ∂RφK = ∂(R()*K, φ)
@test RφK ≈ Rφ*coord(K) rtol=1e-6

# Adjoint test (Jacobian)
Δφ = randn(Float64, nt, 3); Δφ *= norm(φ)/norm(Δφ)
ΔR_ = ∂RφK*Δφ; ΔR = randn(ComplexF64, size(ΔR_)); ΔR *= norm(ΔR_)/norm(ΔR)
@test dot(∂RφK*Δφ, ΔR) ≈ dot(Δφ, ∂RφK'*ΔR) rtol=1e-6

# Gradient test
Δφ = randn(Float64, nt, 3); Δφ *= norm(φ)/norm(Δφ)
t = 1e-6
RφK_p1 = R(φ+0.5*t*Δφ)*coord(K)
RφK_m1 = R(φ-0.5*t*Δφ)*coord(K)
@test (RφK_p1-RφK_m1)/t ≈ ∂RφK*Δφ rtol=1e-5