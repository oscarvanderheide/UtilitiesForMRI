using TestImages, PyPlot, UtilitiesForMRI

geom = geometry_cartesian_2D(512, 512, 1f0, 1f0)
u = Float32.(testimage("mandril_gray"))

figure()
imshow(u, geom; save=true, cmap="gray", transparent=false, fname="./data/mandrill.png")

F = Fourier_transform(geom; orth=true, centered=false)
u_ = F*u

figure()
imshow(abs.(u_), geom_out(F); vmin=0f0, vmax=1f0, save=true, xlabel="kx", ylabel="ky", cmap="gray", transparent=false, fname="./data/mandrill_fft.png")