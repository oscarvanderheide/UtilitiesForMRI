using UtilitiesForMRI, LinearAlgebra, Test, PyPlot

nt = 256
t = Float64.(1:nt)
D = derivative1d_motionpars_linop(t, 2; pars=(true,false,true,false,true,true))
θ = zeros(Float64, nt, 6); θ[129-30:129+30,:] .= 1; θ[1:10,1] .= 1; θ[end-9:end,1] .= 1
Dθ = reshape(D*vec(θ), :, 6)

nt = 256+1
t = Float64.(0:nt-1)
ti = Float64.(0:2:nt-1); nti = length(ti)
Itp = interpolation1d_motionpars_linop((ti,ti,t,ti,ti,t), t)
θ = randn(Float64, 4*nti+2*nt)
Iθ = reshape(Itp*θ, nt, 6)