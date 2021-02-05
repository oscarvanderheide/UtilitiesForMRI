using DifferentialOperatorsForTV, VectorFields, BenchmarkTools, CUDA, Flux
CUDA.allowscalar(false)

# Geometry
n = (1024, 2048)
h = (abs(randn(Float32)), abs(randn(Float32)))
geom = geometry_cartesian(n..., h...)

# Operator
T = Float32
∇ = gradient_linop(T, geom)

# Inputs
u_cpu = randn_scalar(T, geom)
u_gpu = randn_scalar(T, geom) |> gpu

# Timings
@benchmark ∇*u_cpu
@benchmark ∇*u_gpu