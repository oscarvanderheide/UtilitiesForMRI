#: Linear operator

export horz_derivative_2D, vert_derivative_2D, gradient_2D


# Differential operators

function horz_derivative_2D(T::DataType, geom::DomainCartesian; pad_type::String="periodic")

    # Domain/range types
    DT = AbstractArray{T,2}

    # Domain/range sizes
    domain_size = size(geom)

    # Convolution
    Dx = reshape([T(1)/geom.dx T(-1)/geom.dx], 2, 1, 1, 1)
    cdims = DenseConvDims((geom.nx+1,geom.ny,1,1), (2,1,1,1); stride=(1,1), padding=(0,0))
    Dx_gpu = Dx |> gpu

    # Padding
    if pad_type == "periodic"
        pad_fun = pad_periodic
        res_fun = restrict_periodic
    elseif pad_type == "zero"
        pad_fun = pad_zero
        res_fun = restrict_ignore
    elseif pad_type == "copy"
        pad_fun = pad_copy
        res_fun = restrict_sum
    else
        throw(ArgumentError("Requested pad type not implemented"))
    end

    # *, adj
    function matvecprod(u::ScalarField2D)
        u_ext = pad_fun(reshape(raw_data(u), geom.nx, geom.ny, 1, 1), (0,1,0,0))
        return scalar_field(vec(conv(u_ext, Dx, cdims)), geom)
    end
    function matvecprod(u::CuScalarField2D)
        u_ext = pad_fun(reshape(raw_data(u), geom.nx, geom.ny, 1, 1), (0,1,0,0))
        return scalar_field(vec(conv(u_ext, Dx_gpu, cdims)), geom)
    end
    function matvecprod_adj(u::ScalarField2D)
        v = ∇conv_data(reshape(raw_data(u), geom.nx, geom.ny, 1, 1), Dx, cdims)
        return scalar_field(vec(res_fun(v, (0,1,0,0))), geom)
    end
    function matvecprod_adj(u::CuScalarField2D)
        v = ∇conv_data(reshape(raw_data(u), geom.nx, geom.ny, 1, 1), Dx_gpu, cdims)
        return scalar_field(vec(res_fun(v, (0,1,0,0))), geom)
    end

    return linear_operator(DT, DT, domain_size, domain_size, matvecprod, matvecprod_adj)

end

function vert_derivative_linop(T::DataType, geom::DomainCartesian; pad_type::String="periodic")

    # Domain/range types
    DT = AbstractScalarField2D{T}

    # Domain/range sizes
    domain_size = size(geom)

    # Convolution
    Dy = reshape([T(1)/geom.dy T(-1)/geom.dy], 1, 2, 1, 1)
    cdims = DenseConvDims((geom.nx,geom.ny+1,1,1), (1,2,1,1); stride=(1,1), padding=(0,0))
    Dy_gpu = Dy |> gpu

    # Padding
    if pad_type == "periodic"
        pad_fun = pad_periodic
        res_fun = restrict_periodic
    elseif pad_type == "zero"
        pad_fun = pad_zero
        res_fun = restrict_ignore
    elseif pad_type == "copy"
        pad_fun = pad_copy
        res_fun = restrict_sum
    else
        throw(ArgumentError("Requested pad type not implemented"))
    end

    # *, adj
    function matvecprod(u::ScalarField2D)
        u_ext = pad_fun(reshape(raw_data(u), geom.nx, geom.ny, 1, 1), (0,0,0,1))
        return scalar_field(vec(conv(u_ext, Dy, cdims)), geom)
    end
    function matvecprod(u::CuScalarField2D)
        u_ext = pad_fun(reshape(raw_data(u), geom.nx, geom.ny, 1, 1), (0,0,0,1))
        return scalar_field(vec(conv(u_ext, Dy_gpu, cdims)), geom)
    end
    function matvecprod_adj(u::ScalarField2D)
        v = ∇conv_data(reshape(raw_data(u), geom.nx, geom.ny, 1, 1), Dy, cdims)
        return scalar_field(vec(res_fun(v, (0,0,0,1))), geom)
    end
    function matvecprod_adj(u::CuScalarField2D)
        v = ∇conv_data(reshape(raw_data(u), geom.nx, geom.ny, 1, 1), Dy_gpu, cdims)
        return scalar_field(vec(res_fun(v, (0,0,0,1))), geom)
    end

    return linear_operator(DT, DT, domain_size, domain_size, matvecprod, matvecprod_adj)

end

function gradient_linop(T::DataType, geom::DomainCartesian; pad_type::String="periodic")

    # Domain/range types
    DT = AbstractScalarField2D{T}
    RT = AbstractVectorField2D{T}

    # Domain/range sizes
    domain_size = size(geom)
    range_size = (domain_size..., 2)

    # Convolution
    D = cat(reshape([T(0) T(1)/geom.dx; T(0) T(-1)/geom.dx], 2, 2, 1, 1), reshape([T(0) T(0); T(1)/geom.dy T(-1)/geom.dy], 2, 2, 1, 1); dims=4)
    D_gpu = D |> gpu
    cdims = DenseConvDims((geom.nx+1,geom.ny+1,1,1), (2,2,1,2); stride=(1,1), padding=(0,0))

    # Padding
    if pad_type == "periodic"
        pad_fun = pad_periodic
        res_fun = restrict_periodic
    elseif pad_type == "zero"
        pad_fun = pad_zero
        res_fun = restrict_ignore
    elseif pad_type == "copy"
        pad_fun = pad_copy
        res_fun = restrict_sum
    else
        throw(ArgumentError("Requested pad type not implemented"))
    end

    # *, adj
    function matvecprod(u::ScalarField2D)
        u_ext = pad_fun(reshape(raw_data(u), geom.nx, geom.ny, 1, 1), (0,1,0,1))
        return vector_field(reshape(conv(u_ext, D, cdims), :, 2), geom)
    end
    function matvecprod(u::CuScalarField2D)
        u_ext = pad_fun(reshape(raw_data(u), geom.nx, geom.ny, 1, 1), (0,1,0,1))
        return vector_field(reshape(conv(u_ext, D_gpu, cdims), :, 2), geom)
    end
    function matvecprod_adj(u::VectorField2D)
        v = ∇conv_data(reshape(raw_data(u), geom.nx, geom.ny, 2, 1), D, cdims)
        return scalar_field(vec(res_fun(v, (0,1,0,1))), geom)
    end
    function matvecprod_adj(u::CuVectorField2D)
        v = ∇conv_data(reshape(raw_data(u), geom.nx, geom.ny, 2, 1), D_gpu, cdims)
        return scalar_field(vec(res_fun(v, (0,1,0,1))), geom)
    end

    return linear_operator(DT, RT, domain_size, range_size, matvecprod, matvecprod_adj)

end