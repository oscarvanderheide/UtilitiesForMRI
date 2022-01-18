using UtilitiesForMRI, Test

# Geometry size
n = (512, 512)
d = (0.1f0, 0.1f0)
geom = geometry_cartesian_2D(n..., d...)

# Utils
size(geom)
extent(geom)
kspace_transform(geom)