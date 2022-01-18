#: Plotting utils

function PyPlot.imshow(u::Array{T,2}, geom::DomainCartesian2D{T}; title=nothing, cmap=nothing, vmin=nothing, vmax=nothing, save::Bool=false, fname="./image.png", dpi=300, xlabel="x", ylabel="y", transparent=true, bbox_inches="tight") where T

    (size(geom) !== size(u)) && throw(ArgumentError("Array and geometry size inconsistent"))
    ext = extent(geom)
    imshow(u; extent=ext, vmin=vmin, vmax=vmax, cmap=cmap)
    PyPlot.title(title)
    colorbar()
    PyPlot.xlabel(xlabel); PyPlot.ylabel(ylabel)
    save && savefig(fname; dpi=dpi, transparent=transparent, bbox_inches=bbox_inches)

end

function PyPlot.imshow(d::Array{T,2}, md::MRacqgeomCartesiangrid2D{T}; title=nothing, cmap=nothing, vmin=nothing, vmax=nothing, save::Bool=false, fname="./data.png", dpi=300, xlabel=L"k$_x$", ylabel=L"k$_y$", transparent=true, bbox_inches="tight") where T
    u = inject(d.+0im, md)
    PyPlot.imshow(real.(u), md.geom; title=title, cmap=cmap, vmin=vmin, vmax=vmax, save=save, fname=fname, dpi=dpi, xlabel=xlabel, ylabel=ylabel, transparent=transparent, bbox_inches=bbox_inches)
end