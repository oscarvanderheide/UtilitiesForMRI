using UtilitiesForMRI, LinearAlgebra, Test

# Cartesian domain
n = (256, 257, 256)
fov = (1.0, 2.0, 2.1)
o = (0.8, 1.0, 1.6)
X = spatial_sampling(fov, n; origin=o)

# Features
field_of_view(X)
h = spacing(X)
x, y, z = coord(X)
f1, f2, f3 = Nyquist_frequency(X)

# Down-scaling
X_2h = downscale(X; factor=2)
X_4h = downscale(X; factor=4)
X_ = upscale(X_4h; factor=4)
x_4h, y_4h, z_4h = coord(X_4h)