using LinearAlgebra, Test, UtilitiesForMRI

# Random input
T = Float32
# n = (1025, 2049)
n = (1024, 2048)
h = (1f0, 1f0)
geom = geometry_cartesian_2D(n..., h...)

# Transform
geom_k = kspace_transform(geom)
klim = extent(geom_k)

# Acquisition geometry cartesian grid
kx = (klim[1]:geom_k.spacing[1]:klim[2])
ky = (klim[3]:geom_k.spacing[2]:klim[4])
d_geom = acqgeom_cartesiangrid(geom_k, kx, ky)

# Acquisition geometry full grid
d_geom = acqgeom_fullgrid(geom_k)