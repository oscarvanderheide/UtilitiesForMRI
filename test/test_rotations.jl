using UtilitiesForMRI, LinearAlgebra, Test, Random

# Cartesian domain
n = (256, 257, 256)
L = (1.0, 1.5, 2.1)
X = spatial_geometry(L, n)
K0 = kspace_geometry(X)

# Cartesian sampling in k-space
# phase_encoding = (1,2); readout = 3
phase_encoding = (1,3); readout = 2
pe_subs = randperm(prod(n[[phase_encoding...]]))[1:100]
r_subs = randperm(n[readout])[1:61]
sampling_scheme = aligned_readout_sampling(phase_encoding; phase_encode_sampling=pe_subs, readout_sampling=r_subs)
K = sample(K0, sampling_scheme)

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
ΔR_ = ∂RφK*Δφ; ΔR = randn(Float64, size(ΔR_)); ΔR *= norm(ΔR_)/norm(ΔR)
@test dot(∂RφK*Δφ, ΔR) ≈ dot(Δφ, ∂RφK'*ΔR) rtol=1e-6

# Gradient test
Δφ = randn(Float64, nt, 3); Δφ *= norm(φ)/norm(Δφ)
t = 1e-6
RφK_p1 = R(φ+0.5*t*Δφ)*coord(K)
RφK_m1 = R(φ-0.5*t*Δφ)*coord(K)
@test (RφK_p1-RφK_m1)/t ≈ ∂RφK*Δφ rtol=1e-5