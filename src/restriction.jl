#: Restriction operator

export restriction_linop


# Restriction type

struct Restriction{T,GEOMT}<:AbstractLinearOperator{AbstractArray{Complex{T},2},AbstractArray{Complex{T}}}
    acq_geom::GEOMT
end

AbstractLinearOperators.domain_size(R::Restriction) = size(R.acq_geom.geom)
AbstractLinearOperators.range_size(R::Restriction) = size(R.acq_geom)
AbstractLinearOperators.matvecprod(R::Restriction{T,GEOMT}, u::AbstractArray{Complex{T},2}) where {T,GEOMT} = restrict(u, R.acq_geom)
AbstractLinearOperators.matvecprod_adj(R::Restriction{T,GEOMT}, d::AbstractArray{Complex{T}}) where {T,GEOMT} = inject(d, R.acq_geom)


# Constructor

restriction_linop(acq_geom::GEOMT) where {T,GEOMT<:AbstractMRacqgeomGridded2D{T}} = Restriction{T,GEOMT}(acq_geom)