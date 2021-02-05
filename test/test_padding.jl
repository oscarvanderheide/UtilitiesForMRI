using DifferentialOperatorsForTV, VectorFields, LinearAlgebra, CUDA, Flux, Test
CUDA.allowscalar(false)

# Size
n = (5, 5)

# Padding
p = (1, 2, 3, 4)
n_ext = n.+(p[1]+p[2],p[3]+p[4])

# Adjoint test
u = randn(Float32, n..., 1, 1) |> gpu
v = randn(Float32, n_ext..., 1, 1) |> gpu
@test dot(DifferentialOperatorsForTV.pad_zero(u, p), v) ≈ dot(u, DifferentialOperatorsForTV.restrict_ignore(v, p)) rtol=1f-3
@test dot(DifferentialOperatorsForTV.pad_copy(u, p), v) ≈ dot(u, DifferentialOperatorsForTV.restrict_sum(v, p)) rtol=1f-3
@test dot(DifferentialOperatorsForTV.pad_periodic(u, p), v) ≈ dot(u, DifferentialOperatorsForTV.restrict_periodic(v, p)) rtol=1f-3

# Adjoint test
u = randn(Float32, n..., 2, 1) |> gpu
v = randn(Float32, n_ext..., 2, 1) |> gpu
@test dot(DifferentialOperatorsForTV.pad_zero(u, p), v) ≈ dot(u, DifferentialOperatorsForTV.restrict_ignore(v, p)) rtol=1f-3
@test dot(DifferentialOperatorsForTV.pad_copy(u, p), v) ≈ dot(u, DifferentialOperatorsForTV.restrict_sum(v, p)) rtol=1f-3
@test dot(DifferentialOperatorsForTV.pad_periodic(u, p), v) ≈ dot(u, DifferentialOperatorsForTV.restrict_periodic(v, p)) rtol=1f-3