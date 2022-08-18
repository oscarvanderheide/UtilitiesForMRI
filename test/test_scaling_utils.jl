using UtilitiesForMRI, LinearAlgebra, Test, Random
Random.seed!(123)

# Cartesian domain
n = (256, 257, 256)
fov = (1.0, 2.0, 2.1)
o = (0.8, 1.0, 1.6)
X = spatial_geometry(fov, n; origin=o)

# Down-scaling
Xh = rescale(X, div.(n,2).+1)

# Subsampling scheme
phase_encoding_dims = (2, 3); readout = 1
pe_subs = randperm(prod(n[[phase_encoding_dims...]]))[1:100]
r_subs = randperm(n[readout])[1:20]
K = kspace_sampling(X, phase_encoding_dims; phase_encode_sampling=pe_subs, readout_sampling=r_subs)

# Down-scaling
factor = (2, 0, 3)
n_scale = div.(n,2 .^factor).+1
Xh = rescale(X, n_scale)
Kh = rescale(K, Xh)
# using PyPlot
# for t = 1:size(K)[1]
#     plot3D(K[t][:,1], K[t][:,2], K[t][:,3], "b.")
# end
# for t = 1:size(Kh)[1]
#     plot3D(Kh[t][:,1], Kh[t][:,2], Kh[t][:,3], "r.")
# end

# Consistency check
K0 = kspace_sampling(X, phase_encoding_dims)
nt, nk = size(K0)
@test coord(Kh) ≈ coord(K0)[Kh.idx_phase_encoding, Kh.idx_readout,:]

# Consistency check
nt, nk = size(K0)
d = randn(ComplexF64, nt, nk)
@test d[Kh.idx_phase_encoding, Kh.idx_readout] ≈ rescale(d, Kh)

# Downsampling array
n = (256, 256, 256)
fov = (1.0, 1.0, 1.0)
u = zeros(ComplexF64, n); u[129-60:129+60,129-60:129+60,129-60:129+60] .= 1
factor = (2, 1, 3)
n_scale = div.(n,2 .^factor).+1
uh = rescale(u, n_scale)
# using PyPlot
# subplot(1, 2, 1)
# imshow(abs.(u[:,:,129]); extent=(0, fov[2], fov[1], 0))
# colorbar()
# subplot(1, 2, 2)
# imshow(abs.(uh[:,:,div(size(uh,3),2)+1]); extent=(0, fov[2], fov[1], 0))
# colorbar()

# Upsampling array
n = (64, 64, 64)
fov = (1.0, 1.0, 1.0)
u = zeros(ComplexF64, n); u[33-10:33+10,33-10:33+10,33-10:33+10] .= 1
factor = (2, 1, 3)
n_scale = n.*2 .^factor.+1
uh = rescale(u, n_scale)
# using PyPlot
# subplot(1, 2, 1)
# imshow(abs.(u[:,:,33]); extent=(0, fov[2], fov[1], 0))
# colorbar()
# subplot(1, 2, 2)
# imshow(abs.(uh[:,:,div(size(uh,3),2)+1]); extent=(0, fov[2], fov[1], 0))
# colorbar()
@test rescale(uh, n) ≈ u