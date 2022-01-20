using UtilitiesForMRI, Test

# Cartesian domain
n = (64, 64, 64)
idx_orig = (5, 7, 33)
h = (0.25f0, 1f0, 0.5f0)
X = spatial_sampling(n; h=h, idx_orig=idx_orig)

# Coordinates
x, y, z = coord(X)
coord_norm(n) = (mod(n,2) == 0) ? (return -n/2:n/2-1) : (return -(n-1)/2:(n-1)/2)
x_norm, y_norm, z_norm = coord_norm(n[1]), coord_norm(n[2]), coord_norm(n[3])

# Test
τ = shift_nfft(X)
x_ = x_norm.*h[1].+τ[1]
y_ = y_norm.*h[2].+τ[2]
z_ = z_norm.*h[3].+τ[3]
@test vec(x) ≈ vec(x_) rtol=1f-6
@test vec(y) ≈ vec(y_) rtol=1f-6
@test vec(z) ≈ vec(z_) rtol=1f-6

# Shift to default origin
u = zeros(ComplexF32, n); u[idx_orig...] = 1
u_shift = shift_nfft(u, X)