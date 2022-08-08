using UtilitiesForMRI, LinearAlgebra, Test, Random
Random.seed!(123)

# nt = 256
# t = Float64.(1:nt)
# D = derivative1d_motionpars_linop(t, 2; pars=(true,false,true,false,true,true))
# θ = zeros(Float64, nt, 6); θ[129-30:129+30,:] .= 1; θ[1:10,1] .= 1; θ[end-9:end,1] .= 1
# Dθ = reshape(D*vec(θ), :, 6)

# nt = 256+1
# t = Float64.(0:nt-1)
# ti = Float64.(0:2:nt-1); nti = length(ti)
# Itp = interpolation1d_motionpars_linop((ti,ti,t,ti,ti,t), t)
# θ = randn(Float64, 4*nti+2*nt)
# Iθ = reshape(Itp*θ, nt, 6)

# Extrapolation linear operator
fov = (1f0, 2f0, 3f0)
n = (256, 256, 256)
X = spatial_geometry(fov, n)
phase_encoding_dims = (2, 3); readout = 1
pe_subs = randperm(prod(n[[phase_encoding_dims...]]))[1:6000]
I = extrapolate_motionpars_linop(n[[phase_encoding_dims...]], pe_subs, nothing; T=Float32, kernel_size=1, dist_fcn=r2->exp.(-r2/(2*1^2)))
d = randn(Float32, length(pe_subs))
d_ = reshape(I*d, n[[phase_encoding_dims...]])
# using PyPlot
# imshow(d_)

# Consistency w/ all-parameter extrapolation operator
I6 = extrapolate_motionpars_linop(n[[phase_encoding_dims...]], pe_subs, nothing; T=Float32, kernel_size=10, dist_fcn=r2->exp.(-r2/(2*2^2)), all_pars=true)
@test vec(repeat(I*d; outer=(1,6))) ≈ I6*vec(repeat(d; outer=(1,6))) rtol=1f-6