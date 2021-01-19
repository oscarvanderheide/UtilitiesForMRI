#: Restriction operator

export data_restriction_linop


function data_restriction_linop(T::DataType, md::MRmd_gridded; gpu::Bool=false)
    ~(T<:Real) && throw(ArgumentError("Element type must be real"))

    # Domain/range types
    ~gpu ? (DT = ScalarField2D{Complex{T}}) : (DT = CuScalarField2D{Complex{T}})
    ~gpu ? (RT = MRdata_gridded{T}) : (RT = CuMRdata_gridded{T})

    # Domain/range sizes
    domain_size = size(geom)
    range_size = length(md.index)

    # *, adj
    matvecprod(u::ScalarField2D{Complex{T}}) where {T<:Real} = MRdata_gridded{T}(u.array[md.index], md)
    matvecprod(u::CuScalarField2D{Complex{T}}) where {T<:Real} = CuMRdata_gridded{T}(u.array[md.index], md)
    function matvecprod_adj(d::MRdata_gridded{T}) where {T<:Real}
        array = zeros_scalar(Complex{T}, md.geom; gpu=false); array[md.index] = d.array
        return ScalarField2D{Complex{T}}(array, md.geom)
    end
    function matvecprod_adj(d::CuMRdata_gridded{T}) where {T<:Real}
        array = zeros_scalar(Complex{T}, md.geom; gpu=true); array[md.index] = d.array
        return CuScalarField2D{Complex{T}}(array, md.geom)
    end

    return linear_operator(DT, RT, domain_size, range_size, matvecprod, matvecprod_adj)

end