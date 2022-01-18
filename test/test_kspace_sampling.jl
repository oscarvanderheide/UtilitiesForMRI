using UtilitiesForMRI, PyPlot

# Cartesian domain
n = (128,128,128)
o = n./2f0
h = (1f0, 1f0, 1f0)
X = spatial_sampling(o, n, h)

# Cartesian sampling in k-space
readout = :y
phase_encode = :xz
K = kspace_sampling(X; readout=readout, phase_encode=phase_encode)

# Plot
figure()
plot3D(K[1]..., ".")
plot3D(K[10]..., ".")
plot3D(K[20]..., ".")
plot3D(K[128*(10-1)+1]..., ".")
plot3D(K[128*(20-1)+1]..., ".")
plot3D(K[128^2]..., ".")