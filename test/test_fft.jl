using LinearAlgebra, CUDA, Flux, Test, TestImages, PyPlot, UtilitiesForMRI
CUDA.allowscalar(false)

# Random input
T = Float32
n = (1024, 2048)
h = (abs(randn(T)), abs(randn(T)))
geom = geometry_cartesian_2D(n..., h...)

# Operators
orth=true
centered=false
F = Fourier_transform(geom; orth=orth, centered=centered)

# Adjoint test (F)
u = randn(T, geom.size) |> gpu
v = randn(Complex{T}, geom.size) |> gpu
a = real(dot(F*u, v))
b = dot(u, adjoint(F)*v)
@test a ≈ b rtol=1f-3

# Isometry test (F)
u = randn(T, geom.size) |> gpu
v = F*u
if orth
    @test u ≈ adjoint(F)*F*u rtol=1f-3
    @test v ≈ F*adjoint(F)*v rtol=1f-3
end