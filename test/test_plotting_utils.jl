using UtilitiesForMRI

# Cartesian domain
n = (4, 4, 4)
fov = (3.0, 1.0, 2.0)
o = (0.5, 0.5, 0.5)
X = spatial_geometry(fov, n; origin=o)

u = zeros(Float64, n)
u[2,2,2] = 1
u[3,2,2] = 2
u[2,3,2] = 3
u[3,3,2] = 4
u[2,2,3] = 5
u[3,2,3] = 6
u[2,3,3] = 7
u[3,3,3] = 8
plot_volume_slices(u; spatial_geometry=X)

θ = randn(Float32, 10, 6)
orientation = Orientation((1,2,3),(false,false,false))
using PyPlot
plot_parameters(1:10, θ, nothing; xlabel="t", ylabel=[L"$\tau_x$ (mm)", L"$\tau_y$ (mm)", L"$\tau_z$ (mm)", L"$\theta_{xy}$ ($^{\circ}$)", L"$\theta_{xz}$ ($^{\circ}$)", L"$\theta_{yz}$ ($^{\circ}$)"], savefile="./prova.png", orientation=orientation)