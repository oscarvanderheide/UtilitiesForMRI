# Motion parameter utilities

export derivative1d_linop, derivative1d_motionpars_linop, interpolation1d_linop, interpolation1d_motionpars_linop

function derivative1d_linop(t::AbstractVector{T}, order::Integer) where {T<:Real}
    (order != 1) && (order != 2) && error("Order not supported (only 1 or 2)")
    nt = length(t)
    Δt = t[2:end]-t[1:end-1]
    if order == 1
        d_0 = [-T(1)./Δt; T(0)]
        d_p1 = T(1)./Δt;
        return spdiagm(nt, nt, 0 => d_0, 1 => d_p1)
    elseif order == 2
        Δt_m1 = Δt[1:end-1]
        Δt_p1 = Δt[2:end]
        Δt_mean = (Δt_m1+Δt_p1)/T(2)
        d_m1 = T(1)./(Δt_m1.*Δt_mean)
        d_p1 = [T(0); T(1)./(Δt_p1.*Δt_mean)]
        d_0  = [T(0); -d_m1-d_p1[2:end]]
        return spdiagm(nt, nt, -1 => d_m1, 0 => d_0, 1 => d_p1)
    end
end

function derivative1d_motionpars_linop(t::AbstractVector{T}, order::Integer; pars::NTuple{6,Bool}=(true, true, true, true, true, true)) where {T<:Real}
    D = derivative1d_linop(t, order)
    Id = sparse(I, length(t), length(t))
    A = []
    for i = 1:6
        pars[i] ? push!(A, D) : push!(A, Id)
    end
    return cat(A...; dims=(1,2))
end

function interpolation1d_linop(t::AbstractVector{T}, ti::AbstractVector{T}; interp::Symbol=:linear) where {T<:Real}
    (interp != :linear) && error("Only linear interpolation supported")
    nt = length(t)
    nti = length(ti)
    I = repeat(reshape(1:nti, :, 1); outer=(1,2))
    J = Array{Int64,2}(undef, nti, 2)
    V = Array{T,2}(undef, nti, 2)
    for i = 1:nti
        if t[1] == ti[i]
            J[i,:] = [1 2]
            V[i,:] = [T(1) T(0)]
        else
            idx = maximum(findall(t.<ti[i]))
            J[i,:] = [idx idx+1]
            Δt = t[idx+1]-t[idx]
            V[i,:] = [(t[idx+1]-ti[i])/Δt (ti[i]-t[idx])/Δt]
        end
    end
    return sparse(vec(I),vec(J),vec(V),nti,nt)
end

function interpolation1d_motionpars_linop(t::NTuple{6,AbstractVector{T}}, ti::NTuple{6,AbstractVector{T}}; interp::Symbol=:linear) where {T<:Real}
    Ip = []
    for i=1:6
        push!(Ip, interpolation1d_linop(t[i], ti[i]; interp=interp))
    end
    return cat(Ip...; dims=(1,2))
end

interpolation1d_motionpars_linop(t::AbstractVector{T}, ti::NTuple{6,AbstractVector{T}}; interp::Symbol=:linear) where {T<:Real} = interpolation1d_motionpars_linop((t,t,t,t,t,t), ti; interp=interp)

interpolation1d_motionpars_linop(t::NTuple{6,AbstractVector{T}}, ti::AbstractVector{T}; interp::Symbol=:linear) where {T<:Real} = interpolation1d_motionpars_linop(t, (ti,ti,ti,ti,ti,ti); interp=interp)