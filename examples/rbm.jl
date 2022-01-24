using UtilitiesForMRI, LinearAlgebra, PyPlot

# Cartesian domain
n = (64,64,64)
h = (1.0,1.0,1.0)
X = spatial_sampling(n; h=h)

# Fourier operator (w/ standard k-space sampling)
nt = n[1]*n[2]
nk = n[3]
readout = :z
phase_encode = :xy
F = nfft_linop(X; readout=readout, phase_encode=phase_encode)

# Rigid-body perturbation
θ = zeros(Float64, nt, 6)
θ[:, 1] .= 0.0
θ[:, 2] .= 0.0
θ[:, 3] .= 0.0
θ[:, 4] .= 0.0
θ[:, 5] .= 0.0
θ[:, 6] .= 0.0

# Rigid-body motion
figure()
u = zeros(ComplexF64, n); u[33-5:33+5, 33-5:33+5, 33-5:33+5] .= 1.0
for θxy = range(0,2*pi; length=100)
    θ[:, 4] .= θxy
    u_rbm = F'*F(θ)*u
    imshow(real(u_rbm)[:,:,33])
    pause(0.1)
end