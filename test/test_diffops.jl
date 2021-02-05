using DifferentialOperatorsForTV, VectorFields, LinearAlgebra, CUDA, Flux, Test
CUDA.allowscalar(false)

# Geometry
n = (1024, 2048)
h = (abs(randn(Float32)), abs(randn(Float32)))
geom = geometry_cartesian(n..., h...)

# Operators
# flag_gpu = true
flag_gpu = false
T = Float32
Dx = horz_derivative_linop(T, geom)
Dy = vert_derivative_linop(T, geom)
∇ = gradient_linop(T, geom)

# Coherency test
u = randn_scalar(T, geom); flag_gpu && (u = u |> gpu)
∇u = ∇*u
∇u_ = [Dx*u; Dy*u]
@test ∇u ≈ ∇u_ rtol=1f-3

# Adjoint test (Dx)
u = randn_scalar(T, geom) |> gpu
v = randn_scalar(T, geom) |> gpu
a = dot(Dx*u, v)
b = dot(u, adjoint(Dx)*v)
@test a ≈ b rtol = 1f-3

# Adjoint test (Dy)
u = randn_scalar(T, geom) |> gpu
v = randn_scalar(T, geom) |> gpu
a = dot(Dy*u, v)
b = dot(u, adjoint(Dy)*v)
@test a ≈ b rtol = 1f-3

# Adjoint test (∇)
u = randn_scalar(T, geom) |> gpu
v = randn_vector(T, geom) |> gpu
a = dot(∇*u, v)
b = dot(u, adjoint(∇)*v)
@test a ≈ b rtol = 1f-3