export Orientation, VolumeSlice, select, plot_volume_slice, plot_volume_slices, plot_parameters


# Orientation type (custom to standard)

struct Orientation
    perm::NTuple{3,Integer}
    reverse::NTuple{3,Bool}
end

standard_orientation() = Orientation((1,2,3), (false,false,false))


# Volume slice type

struct VolumeSlice
    dim::Integer
    n::Integer
end

function select(u::AbstractArray{T,3}, slice::VolumeSlice; orientation::Union{Nothing,Orientation}=nothing) where {T<:Real}
    n = size(u)
    perm_inv = invperm(orientation.perm)
    dim = perm_inv[slice.dim]
    sl = Array{Any,1}(undef,3)
    for i = 1:3
        if i == dim
            orientation.reverse[i] ? (slice_n = n[i]-slice.n+1) : (slice_n = slice.n)
            sl[i] = slice_n:slice_n
        else
            orientation.reverse[i] ? (sl[i] = n[i]:-1:1) : (sl[i] = 1:n[i])
        end
    end
    return dropdims(permutedims(u[sl...], orientation.perm); dims=slice.dim)
end

function dims(slice::VolumeSlice)
    slice.dim == 1 && (return (2,3))
    slice.dim == 2 && (return (1,3))
    slice.dim == 3 && (return (1,2))
end


# Plot utils

function plot_volume_slice(u::AbstractArray{T,2};
    extent::Union{Nothing,NTuple{4,<:Real}}=nothing,
    cmap::String="gray",
    vmin::Union{Nothing,Real}=nothing, vmax::Union{Nothing,Real}=nothing,
    xlabel::Union{Nothing,AbstractString}=nothing, ylabel::Union{Nothing,AbstractString}=nothing,
    cbar_label::Union{Nothing,AbstractString}=nothing,
    title::Union{Nothing,AbstractString}=nothing,
    savefile::Union{Nothing,String}=nothing) where {T<:Real}

    figure(); ax = gca()
    imshow(u[:,end:-1:1]'; extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
    ~isnothing(xlabel) ? PyPlot.xlabel(xlabel) : ax.axes.xaxis.set_visible(false)
    ~isnothing(ylabel) ? PyPlot.ylabel(ylabel) : ax.axes.yaxis.set_visible(false)
    ~isnothing(cbar_label) && colorbar(label=cbar_label)
    PyPlot.title(title)
    ~isnothing(savefile) && savefig(savefile, dpi=300, transparent=false, bbox_inches="tight")

end

function plot_volume_slices(u::AbstractArray{T,3};
    slices::Union{Nothing,NTuple{N,VolumeSlice}}=nothing,
    spatial_geometry::Union{Nothing,CartesianSpatialGeometry{T}}=nothing,
    cmap::String="gray",
    vmin::Union{Nothing,Real}=nothing, vmax::Union{Nothing,Real}=nothing,
    xlabel::Union{Nothing,AbstractString}=nothing, ylabel::Union{Nothing,AbstractString}=nothing,
    cbar_label::Union{Nothing,AbstractString}=nothing,
    title::Union{Nothing,AbstractString}=nothing,
    savefile::Union{Nothing,String}=nothing,
    orientation::Orientation=standard_orientation()) where {T<:Real,N}

    if isnothing(slices)
        nx, ny, nz = size(u)
        slices = (VolumeSlice(1, div(nx,2)+1), VolumeSlice(2, div(ny,2)+1), VolumeSlice(3, div(nz,2)+1))
    end

    for n = 1:length(slices)
        isnothing(savefile) ? (savefile_slice=nothing) : (savefile_slice = string(savefile[1:end-4], "_slice", n, savefile[end-3:end]))
        u_slice = select(u, slices[n]; orientation=orientation)
        x, y = coord(spatial_geometry; mesh=false)[[orientation.perm...]][[dims(slices[n])...]]
        extent = (x[1], x[end], y[end], y[1])
        plot_volume_slice(u_slice; extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, xlabel=xlabel, ylabel=ylabel, cbar_label=cbar_label, title=title, savefile=savefile_slice)
    end

end

function plot_parameters(t::AbstractVector,
                         θ::AbstractArray, θ_ref::Union{Nothing,AbstractArray};
                         plot_flag::AbstractVector{Bool}=[true,true,true,true,true,true],
                         vmin::AbstractArray=[nothing,nothing,nothing,nothing,nothing,nothing],
                         vmax::AbstractArray=[nothing,nothing,nothing,nothing,nothing,nothing],
                         fmt1::Union{Nothing,AbstractString}=nothing, fmt2::Union{Nothing,AbstractString}=nothing,
                         linewidth1=2, linewidth2=1,
                         xlabel::Union{Nothing,AbstractString}="t", ylabel::Union{Nothing,AbstractVector}=[L"$\tau_x$ (mm)", L"$\tau_y$ (mm)", L"$\tau_z$ (mm)", L"$\theta_{xy}$ ($^{\circ}$)", L"$\theta_{xz}$ ($^{\circ}$)", L"$\theta_{yz}$ ($^{\circ}$)"],
                         filepath="", ext=".png")

    nplots = count(plot_flag)
    _, ax = subplots(nplots, 1)
    # _, ax = subplots(1, nplots)
    c = 1
    for i = 1:6
        if plot_flag[i]
            (i >= 4) ? (C = 180/pi) : (C = 1)
            ax[c].plot(t, C*θ[:,i],     fmt1, linewidth=linewidth1, label="Estimated")
            ~isnothing(θ_ref) && ax[c].plot(t, C*θ_ref[:,i], fmt2, linewidth=linewidth2, label="Reference")
            (c == 1) && ax[c].legend(loc="upper right")
            ~isnothing(xlabel) && (i == 6) && ax[c].set(xlabel=xlabel)
            (i < 6) && ax[c].get_xaxis().set_ticks([])
            ~isnothing(ylabel[i]) && ax[c].set(ylabel=ylabel[i])

            # Axes limit
            if isnothing(vmin[i])
                ~isnothing(θ_ref) ? (vmin_i = minimum(C*θ_ref[:,i])) : (vmin_i = minimum(C*θ[:,i]))
            else
                vmin_i = vmin[i]
            end
            if isnothing(vmax[i])
                ~isnothing(θ_ref) ? (vmax_i = maximum(C*θ_ref[:,i])) : (vmax_i = maximum(C*θ[:,i]))
            else
                vmax_i = vmax[i]
            end
            Δv = vmax_i-vmin_i
            ax[c].set(ylim=[vmin_i-0.1*Δv, vmax_i+0.1*Δv])

            c += 1
        end
    end
    # mngr = get_current_fig_manager()
    # mngr.window.setGeometry(50, 100, 700, 1000); pause(0.1)
    savefig(string(filepath, ext), dpi=300, transparent=false, bbox_inches="tight")

end