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
plot_volume_slices(u; X=X, savefile="test.png")