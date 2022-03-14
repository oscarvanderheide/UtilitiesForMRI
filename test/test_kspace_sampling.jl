using UtilitiesForMRI, Random

# Cartesian domain
n = (128,128,128)
h = (1f0, 1f0, 1f0)
X = spatial_sampling(Float32, n; h=h)

# Cartesian sampling (dense)
phase_encoding = (1,2)
K = kspace_Cartesian_sampling(X; phase_encoding=phase_encoding)
# using PyPlot
# figure()
# plot3D(K[1][:,1],  K[1][:,2],  K[1][:,3], ".")
# plot3D(K[10][:,1], K[10][:,2], K[10][:,3], ".")
# plot3D(K[20][:,1], K[20][:,2], K[20][:,3], ".")
# plot3D(K[128*(10-1)+1][:,1], K[128*(10-1)+1][:,2], K[128*(10-1)+1][:,3], ".")
# plot3D(K[128*(20-1)+1][:,1], K[128*(20-1)+1][:,2], K[128*(20-1)+1][:,3], ".")
# plot3D(K[128^2][:,1], K[128^2][:,2], K[128^2][:,3], ".")

# Cartesian sampling (randomized)
subsampling = (1:128^2)[randperm(128^2)][1:32]
K = kspace_Cartesian_sampling(X; phase_encoding=(1,3), subsampling=subsampling)
# figure()
# for i = 1:32
#     plot3D(K[i][:,1], K[i][:,2], K[i][:,3], ".")
# end

# for i = 1:50:128^2
#     plot3D(K[i][:,1], K[i][:,2], K[i][:,3], ".")
# end