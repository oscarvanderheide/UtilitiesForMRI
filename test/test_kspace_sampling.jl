using UtilitiesForMRI, PyPlot

# Cartesian domain
n = (128,128,128)
h = (1f0, 2f0, 3f0)
X = spatial_sampling(n; h=h)

# Cartesian sampling in k-space
readout = :z
phase_encode = :xy
K = kspace_sampling(X; readout=readout, phase_encode=phase_encode)

# Plot
figure()
plot3D(K[1][:,1],  K[1][:,2],  K[1][:,3], ".")
plot3D(K[10][:,1], K[10][:,2], K[10][:,3], ".")
plot3D(K[20][:,1], K[20][:,2], K[20][:,3], ".")
plot3D(K[128*(10-1)+1][:,1], K[128*(10-1)+1][:,2], K[128*(10-1)+1][:,3], ".")
plot3D(K[128*(20-1)+1][:,1], K[128*(20-1)+1][:,2], K[128*(20-1)+1][:,3], ".")
plot3D(K[128^2][:,1], K[128^2][:,2], K[128^2][:,3], ".")