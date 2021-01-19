#: Fourier operator

export Fourier_linop


function Fourier_linop(T::DataType, geom::DomainCartesian; gpu::Bool=false)
    ~(T<:Real) && throw(ArgumentError("Element type must be real"))

    # Domain/range types
    ~gpu ? (DT = ScalarField2D{T}) : (DT = CuScalarField2D{T})
    ~gpu ? (RT = ScalarField2D{Complex{T}}) : (RT = CuScalarField2D{Complex{T}})

    # Domain/range sizes
    domain_size = size(geom)
    geom_k = geometry_cartesian(div(geom.nx, 2)+1, div(geom.ny, 2)+1, geom.nx, geom.ny, T(1/(geom.nx*geom.dx)), T(1/(geom.ny*geom.dy)))

    # *, adj
    norm_const = T(sqrt(geom.nx*geom.ny))
    matvecprod(u) = scalar_field(vec(fftshift(fft(ifftshift(reshape(u.array, geom.nx, geom.ny))))), geom_k)/norm_const
    matvecprod_adj(u) = scalar_field(real(vec(fftshift(bfft(ifftshift(reshape(u.array, geom_k.nx, geom_k.ny)))))), geom)/norm_const

    return linear_operator(DT, RT, domain_size, domain_size, matvecprod, matvecprod_adj)

end