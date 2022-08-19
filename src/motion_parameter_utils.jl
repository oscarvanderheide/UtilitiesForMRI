# Motion parameter utilities

export derivative1d_linop, derivative1d_motionpars_linop, interpolation1d_linop, interpolation1d_motionpars_linop, extrapolate_motionpars_linop, fill_gaps

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
        if     ti[i] < t[1]
            J[i, :] .= [1; 1]
            V[i, :] .= [T(0); T(0)]
        elseif ti[i] > t[end]
            J[i, :] .= [1; 1]
            V[i, :] .= [T(0); T(0)]
        elseif ti[i] == t[1] 
            J[i, :] .= [1; 2]
            V[i, :] .= [T(1); T(0)]
        elseif ti[i] == t[end] 
            J[i, :] .= [nt-1; nt]
            V[i, :] .= [T(0); T(1)]
        else
            idx_ = findall(t .< ti[i])
            if length(idx_) != 0
                idx = maximum(idx_)
                J[i,:] .= [idx; idx+1]
                Δt = t[idx+1]-t[idx]
                V[i,:] .= [(t[idx+1]-ti[i])/Δt; (ti[i]-t[idx])/Δt]
            else
                J[i, :] .= [1; 1]
                V[i, :] .= [T(0); T(0)]
            end
        end
    end
    return sparse(vec(I), vec(J), vec(V), nti, nt)
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

interpolation1d_motionpars_linop(t::AbstractVector{T}, ti::AbstractVector{T}; interp::Symbol=:linear) where {T<:Real} = interpolation1d_motionpars_linop((t,t,t,t,t,t), (ti,ti,ti,ti,ti,ti); interp=interp)

function extrapolate_motionpars_linop(coord_q::AbstractArray{T,2}, coord::AbstractArray{T,2}; kernel_size::Integer=1, dist_fcn::Function=r2->exp.(-r2/2), all_pars::Bool=false) where {T<:Real}
    nt = size(coord,1); nt_q = size(coord_q,1)
    Is = repeat(reshape(1:nt,1,:); outer=(kernel_size,1))
    Js = Array{Integer,2}(undef,kernel_size,nt)
    Vs = Array{T,2}(undef,kernel_size,nt)
    @inbounds for i = 1:nt
        dist2_i = sum((coord[i:i,:].-coord_q).^2; dims=2)[:,1]
        Js[:,i], dist2_min = mink(dist2_i, kernel_size)
        Vs[:,i] = dist_fcn(dist2_min)
        # (dist2_min[1] == 0) && (Vs[2:end,i] .= 0)
        # i_cutoff = findfirst(cumsum(Vs[:,i]) .> 1)
        # ~isnothing(i_cutoff) && (Vs[i_cutoff:end,i] .= 0)
        Vs[:,i] ./= sum(Vs[:,i])
    end
    A = sparse(vec(Is), vec(Js), vec(Vs), nt, nt_q)
    all_pars ? (return kron(Matrix{T}(I, 6, 6), A)) : (return A)
end

function extrapolate_motionpars_linop(n::NTuple{2,Integer}, idx_q::AbstractVector{<:Integer}, idx::Union{Nothing,AbstractVector{<:Integer}}; T::DataType=Float32, kernel_size::Integer=1, dist_fcn::Function=r2->exp.(-r2/2), all_pars::Bool=false)
    isnothing(idx) && (idx = 1:prod(n))
    x = repeat(reshape(T.(1:n[1]),:,1); outer=(1,n[2]))
    y = repeat(reshape(T.(1:n[2]),1,:); outer=(n[1],1))
    coord = [vec(x) vec(y)]
    return extrapolate_motionpars_linop(coord[idx_q,:], coord[idx,:]; kernel_size=kernel_size, dist_fcn=dist_fcn, all_pars=all_pars)
end

function mink(a::AbstractArray{T}, k::Integer) where {T<:Real}
    b = partialsortperm(a, 1:k)
    return b, a[b]
end

# function interp_linear_filling(n::NTuple{2,Integer}, θ::AbstractArray{T,2}, fact::Integer; keep_low_freqs::Bool=true, extrapolate::Bool=false) where {T<:Real}
#     (fact == 0) && (return θ)

#     # Find indexes corresponding to low-frequency region corners
#     n1, n2 = n
#     k_max = T(pi)
#     k1 = range(-k_max, k_max; length=n1)
#     k2 = range(-k_max, k_max; length=n2)
#     i1 = findfirst(k1 .>= -k_max/2^fact)
#     i2 = findlast( k1 .<=  k_max/2^fact)
#     j1 = findfirst(k2 .>= -k_max/2^fact)
#     j2 = findlast( k2 .<=  k_max/2^fact)

#     # Setting parameters to mean value in the central portion of the k-space
#     θ_ = deepcopy(reshape(θ, n1, n2, 6))
#     @inbounds for j = j1:j2, p = 1:6
#         θ_[i1:i2, j, p] .= sum(θ_[i1:i2, j, p])/length(i1:i2)
#     end

#     # Interpolate in between low-frequency lines
#     @inbounds for j = j1:j2-1, p = 1:6
#         t = range(T(0), T(1); length=n2-i2+i1-1)
#         θ_[i2+1:end, j,   p] .= vec(θ_[i2, j, p].+t[1:n2-i2].*(θ_[i1, j+1, p]-θ_[i2, j, p]))
#         θ_[1:i1-1,   j+1, p] .= vec(θ_[i2, j, p].+t[n2-i2+1:end].*(θ_[i1, j+1, p]-θ_[i2, j, p]))
#     end

#     # Extrapolate everywhere else
#     if extrapolate
#         θ_[:,      1:j1-1, :] .= reshape(θ_[i1, j1, :], 1, 1, 6)
#         θ_[1:i1-1, j1,     :] .= reshape(θ_[i1, j1, :], 1, 6)
#         θ_[i2+1:end, j2,       :] .= reshape(θ_[i2, j2, :], 1, 6)
#         θ_[:,        j2+1:end, :] .= reshape(θ_[i2, j2, :], 1, 1, 6)
#     end

#     # Restore low frequencies if required
#     keep_low_freqs && (θ_[i1:i2, j1:j2, :] .= reshape(θ, n1, n2, 6)[i1:i2, j1:j2, :])

#     return reshape(θ_, :, 6)
# end

function fill_gaps(idx_local::AbstractVector{<:Integer}, θ_local::AbstractArray{T,2}, nt::Integer; average::Bool=false, extrapolate::Bool=false) where {T<:Real}

    # Determine gap indexes
    block_idx_1 = [idx_local[1]]; block_idx_2 = []
    @inbounds for i = 1:length(idx_local)-1
        (idx_local[i+1]-idx_local[i] > 1) && (push!(block_idx_2, idx_local[i]); push!(block_idx_1, idx_local[i+1]))
    end
    (length(block_idx_1) > length(block_idx_2)) && push!(block_idx_2, idx_local[end])

    # Setting parameters to mean value within contiguous blocks
    np = size(θ_local, 2)
    θ = similar(θ_local, T, nt, np); fill!(θ, 0); θ[idx_local, :] = θ_local
    @inbounds for i = eachindex(block_idx_1)
        θ[block_idx_1[i]:block_idx_2[i], :] .= sum(θ[block_idx_1[i]:block_idx_2[i], :])/length(block_idx_1[i]:block_idx_2[i])
    end

    # Interpolate values within gaps
    @inbounds for i = 1:length(block_idx_2)-1
        gap_idxs = block_idx_2[i]:block_idx_1[i+1]
        t = reshape(range(T(0), T(1); length=length(gap_idxs)), :, 1)
        θ[gap_idxs, :] .= θ[gap_idxs[1:1], :].+t.*(θ[gap_idxs[end:end], :]-θ[gap_idxs[1:1], :])
    end

    # Extrapolate start/end
    if extrapolate
        (block_idx_1[1] > 1) && (θ[1:block_idx_1[1]-1, :] .= reshape(θ[block_idx_1[1], :], 1, :))
        (block_idx_2[end] < nt) && (θ[block_idx_2[end]+1:end, :] .= reshape(θ[block_idx_2[end], :], 1, :))
    end

    # Restore contiguous block values if required
    ~average && (θ[idx_local, :] .= θ_local)

    return θ

end