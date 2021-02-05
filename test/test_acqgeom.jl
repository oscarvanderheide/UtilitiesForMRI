using LinearAlgebra, Test, UtilitiesForMRI, VectorFields

# Random input
T = Float32
# n = (1025, 2049)
n = (1024, 2048)
h = (1f0, 1f0)
geom = geometry_cartesian(n..., h...)

# Transform
geom_k = transform(geom)
klim = extent(geom_k)

# Acquisition geometry
kx = (klim[1]:geom_k.dkx:klim[2])
ky = (klim[3]:geom_k.dky:klim[4])
d_geom = MRacqgeom_cartesiangrid(geom, kx, ky)