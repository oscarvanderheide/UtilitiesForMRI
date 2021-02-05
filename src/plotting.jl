#: Plotting utils

function PyPlot.imshow(u::Array{T,2}, geom::DomainCartesian2D{T}; title=nothing, cmap=nothing, vmin=nothing, vmax=nothing, save::Bool=false, fname="./pic.png", dpi=300, xlabel="x", ylabel="y", transparent=true, bbox_inches="tight") where T

    (size(geom) !== size(u)) && throw(ArgumentError("Array and geometry size inconsistent"))
    ext = extent(geom)
    imshow(u; extent=ext, vmin=vmin, vmax=vmax, cmap=cmap)
    PyPlot.title(title)
    colorbar()
    PyPlot.xlabel(xlabel); PyPlot.ylabel(ylabel)
    save && savefig(fname; dpi=dpi, transparent=transparent, bbox_inches=bbox_inches)

end