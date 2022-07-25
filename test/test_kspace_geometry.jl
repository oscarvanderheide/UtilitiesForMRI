using UtilitiesForMRI, Random, Test
Random.seed!(123)

# k-space
fov = (1f0, 2f0, 3f0)
n = (128, 131, 101)
X = spatial_geometry(fov, n)
phase_encoding = (1, 2); readout = readout_dim(phase_encoding)
pe_subs = randperm(prod(n[[phase_encoding...]]))[1:100]
r_subs = randperm(n[readout])[1:61]
K = kspace_geometry(X; phase_encoding=phase_encoding, phase_encode_subsampling=pe_subs, readout_subsampling=r_subs)

# Plotting
using PyPlot
figure()
nt, _ = size(K)
for t = 1:nt
    plot3D(K[t][:,1],  K[t][:,2],  K[t][:,3], "b.")
end

# Check consistency
t = 3
Kt = K[3]
@test Kt ≈ coord(K)[t, :, :]

# Downscale
factor = (0, 1, 3)
K_, pe_idx, r_idx = downscale(K, factor; scale_index=true)
@test coord(K_) ≈ coord(K)[pe_idx, r_idx, :]
nt, _ = size(K_)
for t = 1:nt
    plot3D(K_[t][:,1],  K_[t][:,2],  K_[t][:,3], "r.")
end

# Consistency check
fov = (1f0, 2f0, 3f0)
n = (256, 257, 259)
X = spatial_geometry(fov, n)
K = kspace_geometry(X; phase_encoding=(1, 2))
factor = (2, 3, 4)
X_ = downscale(X, factor)
K_ = kspace_geometry(X_; phase_encoding=(1, 2))
K__ = downscale(K, factor)
@test coord(K_) ≈ coord(K__)