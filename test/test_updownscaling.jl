using UtilitiesForMRI, LinearAlgebra, Test

# Cartesian domain
n = (256, 256, 256)
h = [abs(randn()), abs(randn()), abs(randn())]
X = spatial_sampling(n; h=h)

# Cartesian sampling in k-space
readout = :z
phase_encode = :xy
K = kspace_sampling(X; readout=readout, phase_encode=phase_encode)

# Random input
nt, nk = size(K)
d = randn(ComplexF64, nt, nk)

# Down-scaling
d_ = downscale_data(d, K; fact=1)

# Test
n2 = div.(n,2)
n4 = div.(n,4)
cidx = div.(n,2).+1
@test reshape(reshape(d, n)[cidx[1]-n4[1]:cidx[1]+n4[1]-1,cidx[2]-n4[2]:cidx[2]+n4[2]-1,cidx[3]-n4[3]:cidx[3]+n4[3]-1], n2[1]*n2[2],n2[3]).*sqrt(prod(n4)/prod(n2)) ≈ d_ rtol=1e-6

# Cartesian domain
n = (16, 16, 16)
h = [abs(randn()), abs(randn()), abs(randn())]
X = spatial_sampling(n; h=h)

# Cartesian sampling in k-space
readout = :z
phase_encode = :xy
K = kspace_sampling(X; readout=readout, phase_encode=phase_encode)

# Random input
nt, nk = size(K)
θ = randn(Float64, nt, 6); reshape(θ,n[1],n[2],6)[div(n[1],2)+1,div(n[2],2)+1,:].=10

# Up-scaling
θ_ = upscale_motion_pars(θ, K; fact=1)

# Test
n2 = n.*2
cidx = div.(n2,2).+1
@test reshape(reshape(θ_, n2[1],n2[2],6)[cidx[1]-div(n[1],2):cidx[1]+div(n[1],2)-1,cidx[2]-div(n[2],2):cidx[2]+div(n[2],2)-1,:], n[1]*n[2],6).*reshape([0.5,0.5,0.5,1,1,1], 1,:) ≈ θ rtol=1e-6

#
c = coord(X; mesh=true)
c1 = coord(upscale(X;fact=1); mesh=true)
c2 = coord(upscale(X;fact=2); mesh=true)

# Consistency check
n = (64,64,64)
h = [1f0,1f0,1f0]
X = spatial_sampling(n; h=h)
K = kspace_sampling(X; phase_encode=:xy, readout=:z)
F = nfft_linop(X; phase_encode=:xy, readout=:z, tol=1f-6)
u = zeros(ComplexF32, n); u[33-5:33+5,33-5:33+5,33-5:33+5] .= 1
d = F*u
X = spatial_sampling(Integer.(n./2); h=h)
F = nfft_linop(X; phase_encode=:xy, readout=:z, tol=1f-6)
u1 = F'*downscale_data(d, K; fact=1)
X = spatial_sampling(Integer.(n./4); h=h)
F = nfft_linop(X; phase_encode=:xy, readout=:z, tol=1f-6)
u2 = F'*downscale_data(d, K; fact=2)