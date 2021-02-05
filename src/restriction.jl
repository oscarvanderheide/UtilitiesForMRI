#: Restriction operator

export data_restriction_linop


function data_restriction_linop(T::DataType, md::AbstractMRacqgeom)
    ~(T<:Real) && throw(ArgumentError("Element type must be real"))

    # Domain/range types
    DT = AbstractScalarField2D{Complex{T}}
    RT = AbstractMRdata{T}

    # Domain/range sizes
    domain_size = size(geom)
    range_size = (length(md.index_x), length(md.index_y))

    # *, adj
    matvecprod(u::AbstractScalarField2D) = restriction(u, md)
    matvecprod_adj(d::AbstractMRdata) = inject(d)

    return linear_operator(DT, RT, domain_size, range_size, matvecprod, matvecprod_adj)

end

data_restriction_linop(T::DataType, d::AbstractMRdata) = data_restriction_linop(T, meta_data(d))