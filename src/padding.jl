#: Padding utils


function pad_zero(u::DT, p::NTuple{4,Int64}) where {T,DT<:AbstractArray{T,4}}
    nx, ny, nc, nb = size(u)
    o1 = zeros_as(u, p[1], ny, nc, nb)
    o2 = zeros_as(u, p[2], ny, nc, nb)
    pu = cat(o1, u, o2; dims=1)
    o1 = zeros_as(pu, size(pu, 1), p[3], nc, nb)
    o2 = zeros_as(pu, size(pu, 1), p[4], nc, nb)
    return cat(o1, pu, o2; dims=2)
end

restrict_ignore(u::DT, p::NTuple{4,Int64}) where {T,DT<:AbstractArray{T,4}} = u[p[1]+1:end-p[2], p[3]+1:end-p[4], :, :]

function pad_copy(u::DT, p::NTuple{4,Int64}) where {T,DT<:AbstractArray{T,4}}
    pu = cat(repeat(u[1:1, :, :, :], outer=(p[1],1,1,1)), u, repeat(u[end:end, :, :, :], outer=(p[2],1,1,1)); dims=1)
    return cat(repeat(pu[:, 1:1, :, :],outer=(1,p[3],1,1)), pu, repeat(pu[:, end:end, :, :], outer=(1,p[4],1,1)); dims=2)
end

function restrict_sum(u::DT, p::NTuple{4,Int64}) where {T,DT<:AbstractArray{T,4}}
    ru = cat(sum(u[1:p[1]+1, :, :, :]; dims=1), u[p[1]+2:end-p[2]-1, :, :, :], sum(u[end-p[2]:end, :, :, :]; dims=1); dims=1)
    return cat(sum(ru[:, 1:p[3]+1, :, :]; dims=2), ru[:, p[3]+2:end-p[4]-1, :, :], sum(ru[:, end-p[4]:end, :, :]; dims=2); dims=2)
end

function pad_periodic(u::DT, p::NTuple{4,Int64}) where {T,DT<:AbstractArray{T,4}}
    nx, ny, nc, nb = size(u)
    o1 = u[end-p[1]+1:end,:,:,:]
    o2 = u[1:p[2],:,:,:]
    pu = cat(o1, u, o2; dims=1)
    o1 = pu[:,end-p[3]+1:end,:,:]
    o2 = pu[:,1:p[4],:,:]
    return cat(o1, pu, o2; dims=2)
end

function restrict_periodic(u::DT, p::NTuple{4,Int64}) where {T,DT<:AbstractArray{T,4}}
    nx_ext, ny_ext, nc, nb = size(u)
    ru_ = u[:, p[3]+1:end-p[4], :, :]
    ru_[:, 1:p[4], :, :] += u[:, end-p[4]+1:end, :, :]
    ru_[:, end-p[3]+1:end, :, :] += u[:, 1:p[3], :, :]
    ru = ru_[p[1]+1:end-p[2], :, :, :]
    ru[1:p[2], :, :, :] += ru_[end-p[2]+1:end, :, :, :]
    ru[end-p[1]+1:end, :, :, :] += ru_[1:p[1], :, :, :]
    return ru
end