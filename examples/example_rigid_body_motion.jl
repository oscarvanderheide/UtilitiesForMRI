using UtilitiesForMRI, LinearAlgebra, PyPlot

# Cartesian domain
n = (64, 64, 64)
h = (1.0, 1.0, 1.0)
# o = (0.0, 0.0, 0.0)
o = (-5.0, -5.0, 0.0)
X = spatial_sampling(Float32, n; h=h, o=o)

# Cartesian sampling in k-space
phase_encoding = (1,2)
K = kspace_Cartesian_sampling(X; phase_encoding=phase_encoding)
nt, nk = size(K)

# Fourier operator
tol = 1f-6
F = nfft(X, K; tol=tol)
F0 = F(zeros(Float32, nt, 6))

# Rigid-body perturbation
θ = zeros(Float32, nt, 6)
θ[:, 1] .= 0
θ[:, 2] .= 0
θ[:, 3] .= 0
θ[:, 4] .= 0
θ[:, 5] .= 0
θ[:, 6] .= 0

# Volume
u = zeros(ComplexF32, n); u[33-5:33+5, 33-5:33+5, 33-5:33+5] .= 1

# Rigid-body motion
# u_rbm = F0'*F(θ)*u
figure()
for θxy = range(0,2*pi; length=20)
    θ[:, 4] .= θxy
    u_rbm = F0'*F(θ)*u
    imshow(real(u_rbm)[:,:,33])
    pause(0.1)
end