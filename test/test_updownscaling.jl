using UtilitiesForMRI, LinearAlgebra, Test

# Cartesian domain
n = (256, 256, 256)
h = (abs(randn()), abs(randn()), abs(randn()))
X = spatial_sampling(Float64, n; h=h)

# Cartesian sampling in k-space
K = kspace_Cartesian_sampling(X; phase_encoding=(1,2))

# Random input
nt, nk = size(K)
d = randn(ComplexF64, nt, nk)

# Down-scaling
d_ = downscale(d, K; fact=1, flat=true)

# Test
n2 = div.(n,2)
n4 = div.(n,4)
cidx = div.(n,2).+1
@test reshape(reshape(d, n)[cidx[1]-n4[1]:cidx[1]+n4[1]-1,cidx[2]-n4[2]:cidx[2]+n4[2]-1,cidx[3]-n4[3]:cidx[3]+n4[3]-1], n2[1]*n2[2],n2[3]).*sqrt(prod(n4)/prod(n2)) ≈ d_ rtol=1e-6

# Consistency check
n = (128,128,128)
h = (abs(randn()), abs(randn()), abs(randn()))
X = spatial_sampling(Float64, n; h=h)
F = nfft_linop(X; phase_encoding=(1,2), tol=1e-6)
u = zeros(ComplexF64, n); u[65-10:65+10,65-10:65+10,65-10:65+10] .= 1
d = F*u
F_h = downscale(F; fact=1)
flat = false
coeff = 10
d_h = downscale(d, F.K; fact=1, flat=flat, coeff=coeff)
u_h = F_h'*d_h
F_2h = downscale(F; fact=2)
d_2h = downscale(d, F.K; fact=2, flat=flat, coeff=coeff)
u_2h = F_2h'*d_2h

# Downscaling images with anti-aliasing
u = zeros(ComplexF64, n); u[65-10:65+10,65-10:65+10,65-10:65+10] .= 1; u[65,65,65] = 10
u_h  = downscale(u, X; fact=1)
u_2h = downscale(u, X; fact=2)
u_h_ = upscale(u, X; fact=1)