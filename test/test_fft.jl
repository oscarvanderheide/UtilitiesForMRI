using LinearAlgebra, CUDA, Flux, Test, TestImages, PyPlot, VectorFields, UtilitiesForMRI
CUDA.allowscalar(false)

# Random input
T = Float32
n = (1001, 2001)
h = (abs(randn(T)), abs(randn(T)))
geom = geometry_cartesian(n..., h...)

# Operators
flag_gpu = true
# flag_gpu = false
F = Fourier_linop(T, geom; gpu=flag_gpu)

# Adjoint test (F)
u = randn_scalar(T, geom) |> gpu
geom_k = (F*u).geom
v = randn_scalar(Complex{T}, geom_k) |> gpu
a = real(dot(F*u, v))
b = dot(u, adjoint(F)*v)
@test a ≈ b rtol=1f-3

# Isometry test (F)
u = randn_scalar(T, geom) |> gpu
v = F*u
@test u ≈ adjoint(F)*F*u rtol=1f-3
@test v ≈ F*adjoint(F)*v rtol=1f-3

# Plot
geom = geometry_cartesian(512, 512, 1f0, 1f0)
u = scalar_field(vec(T.(testimage("mandril_gray"))), geom)
figure(); imshow(u; save=true, cmap="gray", transparent=false, fname="./data/mandrill.png")
F = Fourier_linop(T, geom; gpu=false)
u_ = F*u
figure(); imshow(u_; preproc=x->abs.(x), vmin=0f0, vmax=1f0, save=true, xlabel="kx", ylabel="ky", cmap="gray", transparent=false, fname="./data/mandrill_fft.png")