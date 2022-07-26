using UtilitiesForMRI, Random, Test
Random.seed!(123)

# k-space
fov = (1f0, 2f0, 3f0)
n = (128, 131, 101)
X = spatial_geometry(fov, n)
K0 = kspace_geometry(X)

# Subsampling scheme
phase_encoding = (2, 3)
readout = 1
pe_subs = randperm(prod(n[[phase_encoding...]]))[1:100]
r_subs = randperm(n[readout])[1:61]
sampling_scheme = aligned_readout_sampling(phase_encoding; phase_encode_sampling=pe_subs, readout_sampling=r_subs)
K = sample(K0, sampling_scheme)

# # Plotting
# using PyPlot
# figure()
# nt, _ = size(K)
# for t = 1:nt
#     plot3D(K[t][:,1],  K[t][:,2],  K[t][:,3], "b.")
# end

# Check consistency
t = 3
Kt = K[3]
@test Kt ≈ coord(K)[t, :, :]