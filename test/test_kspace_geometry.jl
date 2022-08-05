using UtilitiesForMRI, Random, Test
Random.seed!(123)

# k-space
fov = (1f0, 2f0, 3f0)
n = (256, 257, 256)
X = spatial_geometry(fov, n)

# Subsampling scheme
phase_encoding_dims = (2, 3); readout = 1
pe_subs = randperm(prod(n[[phase_encoding_dims...]]))[1:100]
r_subs = randperm(n[readout])[1:20]
K = kspace_sampling(X, phase_encoding_dims; phase_encode_sampling=pe_subs, readout_sampling=r_subs)

# Check consistency
t = 3
Kt = K[3]
@test Kt ≈ coord(K)[t,:,:]
@test reshape(coord(K),:,3) ≈ coord(destructure(K))

# # Plotting
# using PyPlot
# figure()
# nt, _ = size(K)
# for t = 1:nt
#     plot3D(K[t][:,1],  K[t][:,2],  K[t][:,3], "b.")
# end