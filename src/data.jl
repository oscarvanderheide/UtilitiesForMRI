export MRIdata, restriction_op, plot_data


# MRI data type

struct MRIdata{T} <: AbstractArray{Complex{T},1}
    description::Union{Nothing,String}             # Description of data (type, acquisition, ...)
    resolution::Tuple{T,T}                         # Grid stepsize (in meters)
    grid_size::NTuple{2,Int64}                     # Grid size (in pixels)
    k_index::Array{Int64,2}                        # Absolute index in k-space
    data::AbstractArray{Complex{T},1}              # Actual data (as complex array)
    groundtruth::Union{Nothing,AbstractArray{T,2}} # Groundtruth (if available)
end

size(d::MRIdata{T}) where T = size(d.data)
IndexStyle(::Type{<:MRIdata}) = IndexLinear()
getindex(d::MRIdata{T}, i::Int64) where T = getindex(d.data, i)
setindex!(d::MRIdata{T}, i::Int64) where T = setindex!(d.data, i)
show(io::IO, d::MRIdata{T}) where T = show(io, d.data)
show(io::IO, mime::MIME"text/plain", d::MRIdata{T}) where T = show(io, mime, d.data)


# Restriction operator

restriction(u::ScalarField2D{T}, index::Array{Int64,1}) where T = d[index]
function injection(d::AbstractArray{T,1}, size::NTuple{4,Int64}, index::Array{Int64,1}) where T
    d[index]
end


# Algebra

# ...


# GPU utils
gpu(d::MRIdata{T}) where T = MRIdata{T}(d.description, d.resolution, d.grid_size, d.kspace_index, gpu(d.data))
cpu(d::MRIdata{T}) where T = MRIdata{T}(d.description, d.resolution, d.grid_size, d.kspace_index, cpu(d.data))


# Plot

function plot_data(d::MRIdata{T}; proc::Function=identity, title::Union{Nothing,String}=nothing, resolution::Union{Nothing,Tuple{RT,RT}}=nothing, colorbar::Bool=true, cmap::String="gray", save::Bool=false, filename::String="figure", dpi::Int64=300) where {T,RT}
    data = zeros(T,d.grid_size)
    for i = 1:size(d.kspace_index, 2)
        data[d.k_index[1,i]+d.k_origin[1]-1,d.k_index[2,i]+d.k_origin[2]-1] = d.data[i];
    end
    fig = figure()
    title !== nothing && PyPlot.title(title)
    d.resolution !== nothing && (resolution = d.resolution)
    resolution !== nothing ? (xlabel("x (m)"); ylabel("y (m)")) : (resolution = (1,1); xlabel("x (idx)"); ylabel("y (idx)"))
    x = resolution[1]*(0:grid_size[1]-1)
    y = resolution[2]*(0:grid_size[2]-1)
    imshow(proc(data), extent=(x[1], x[end], y[end], y[1]), cmap=cmap, aspect="equal")
    colorbar && PyPlot.colorbar()
    save && savefig(string("./", filename, ".png"), transparent = true, bbox_inches = "tight", pad_inches = 0, dpi = dpi)
end