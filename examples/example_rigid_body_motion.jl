using UtilitiesForMRI, LinearAlgebra, PyPlot

# Cartesian domain
n = (64, 64, 64)
fov = (64.0, 64.0, 64.0)
o = fov./2 .-(10.0, 10.0, 00.0)
X = spatial_geometry(fov, n; origin=o)

# Cartesian sampling in k-space
phase_encoding = (1,2)
K = kspace_sampling(X, phase_encoding)
nt, nk = size(K)

# Fourier operator
tol = 1e-6
F = nfft_linop(X, K; tol=tol)

# Rigid-body perturbation
θ = zeros(Float64, nt, 6)
θ[:, 1] .= 0
θ[:, 2] .= 0
θ[:, 3] .= 0
θ[:, 4] .= 0
θ[:, 5] .= 0
θ[:, 6] .= 0

# Volume
u = zeros(ComplexF64, n); u[33-5:33+5, 33-5:33+5, 33-5:33+5] .= 1

# Rigid-body motion
# u_rbm = F0'*F(θ)*u
# figure()
for θxy = range(0,2*pi; length=20)
    θ[:, 4] .= θxy
    u_rbm = F'*F(θ)*u
    imshow(real(u_rbm)[:,:,33])
    pause(0.1)
end