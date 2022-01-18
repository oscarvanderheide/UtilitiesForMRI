using UtilitiesForMRI, Test

# Cartesian domain
n = (128, 129, 130)
idx_orig = (32f0, 33f0, 34f0)
h = (0.25f0, 1f0, 0.5f0)
X = spatial_sampling(n; h=h, idx_orig=idx_orig)

# Coordinates
x, y, z = coord(X)
coord_norm(n) = (mod(n,2) == 0) ? (return -n/2:n/2-1) : (return -(n-1)/2:(n-1)/2)
x_norm, y_norm, z_norm = coord_norm(n[1]), coord_norm(n[2]), coord_norm(n[3])

# Test
ints = UtilitiesForMRI.normalized_shift_nfft(X)
x_ = (x_norm.+ints[1]).*h[1]
y_ = (y_norm.+ints[2]).*h[2]
z_ = (z_norm.+ints[3]).*h[3]
@test vec(x) ≈ vec(x_) rtol=1f-6 
@test vec(y) ≈ vec(y_) rtol=1f-6 
@test vec(z) ≈ vec(z_) rtol=1f-6 