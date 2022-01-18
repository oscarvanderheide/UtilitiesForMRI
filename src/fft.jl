#: Fourier operator

export Fourier_transform, FourierTransform, FourierTransformCentered, FourierTransformOrthogonal, FourierTransformOrthogonalCentered, geom, geom_out


# Abstract type

"""Concrete FourierTransform types must have a geom method"""
abstract type AbstractFourierTransform{T}<:AbstractLinearOperator{AbstractArray{T,2},AbstractArray{Complex{T},2}} end

AbstractLinearOperators.domain_size(F::AbstractFourierTransform) = size(geom(F))
AbstractLinearOperators.range_size(F::AbstractFourierTransform) = size(geom(F))


# Concrete types

## Standard

struct FourierTransform{T}<:AbstractFourierTransform{T}
    geom::DomainCartesian2D{T}
end

geom(F::FourierTransform) = F.geom
AbstractLinearOperators.matvecprod(::FourierTransform{T}, u::AbstractArray{T,2}) where T = fft(u)
AbstractLinearOperators.matvecprod_adj(::FourierTransform{T}, v::AbstractArray{Complex{T},2}) where T = real(bfft(v))

geom_out(F::FourierTransform) = kspace_transform(F.geom; centered=false)

## Centered

struct FourierTransformCentered{T}<:AbstractFourierTransform{T}
    geom::DomainCartesian2D{T}
end

geom(F::FourierTransformCentered) = F.geom
AbstractLinearOperators.matvecprod(::FourierTransformCentered{T}, u::AbstractArray{T,2}) where T = fftshift(fft(ifftshift(u)))
AbstractLinearOperators.matvecprod_adj(::FourierTransformCentered{T}, v::AbstractArray{Complex{T},2}) where T = real(fftshift(bfft(ifftshift(u))))

geom_out(F::FourierTransformCentered) = kspace_transform(F.geom; centered=true)

## Orthogonal

struct FourierTransformOrthogonal{T}<:AbstractFourierTransform{T}
    geom::DomainCartesian2D{T}
end

geom(F::FourierTransformOrthogonal) = F.geom
AbstractLinearOperators.matvecprod(F::FourierTransformOrthogonal{T}, u::AbstractArray{T,2}) where T = fft(u)/sqrt(T(prod(size(F.geom))))
AbstractLinearOperators.matvecprod_adj(F::FourierTransformOrthogonal{T}, v::AbstractArray{Complex{T},2}) where T = real(bfft(v)/sqrt(T(prod(size(F.geom)))))

geom_out(F::FourierTransformOrthogonal) = kspace_transform(F.geom; centered=false)

## Orthogonal/Centered

struct FourierTransformOrthogonalCentered{T}<:AbstractFourierTransform{T}
    geom::DomainCartesian2D{T}
end

geom(F::FourierTransformOrthogonalCentered) = F.geom
AbstractLinearOperators.matvecprod(F::FourierTransformOrthogonalCentered{T}, u::AbstractArray{T,2}) where T = fftshift(fft(ifftshift(u)))/sqrt(T(prod(size(F.geom))))
AbstractLinearOperators.matvecprod_adj(F::FourierTransformOrthogonalCentered{T}, v::AbstractArray{Complex{T},2}) where T = real(ifftshift(bfft(fftshift(v)))/sqrt(T(prod(size(F.geom)))))

geom_out(F::FourierTransformOrthogonalCentered) = kspace_transform(F.geom; centered=true)


# Constructors

function Fourier_transform(geom::DomainCartesian2D{T}; orth::Bool=false, centered::Bool=false) where T
    (~orth && ~centered) && return FourierTransform{T}(geom)
    (~orth &&  centered) && return FourierTransformCentered{T}(geom)
    ( orth && ~centered) && return FourierTransformOrthogonal{T}(geom)
    ( orth &&  centered) && return FourierTransformOrthogonalCentered{T}(geom)
end