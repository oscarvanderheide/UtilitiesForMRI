# Rigid-body motion utilities


function process_nfft_pars(K::KSpaceFixedSizeSampling{T}, θ::AbstractArray{T,2}) where {T<:Real}

    # Simplifying notation
    Kx, Ky, Kz = coord(K; norm=false)
    τx = θ[:,1]; τy = θ[:,2]; τz = θ[:,3]; ϕxy = θ[:,4]; ϕxz = θ[:,5]; ϕyz = θ[:,6]

    # Translation
    phase_shift = exp.(-1im*(Kx.*τx+Ky.*τy+Kz.*τz))

    # Rotation
    cxy = cos.(ϕxy); sxy = sin.(ϕxy);
    cxz = cos.(ϕxz); sxz = sin.(ϕxz);
    cyz = cos.(ϕyz); syz = sin.(ϕyz);
    Kx_ = cxy.*Kx+sxy.*Ky
    Ky_ = -sxy.*Kx+cxy.*Ky
    Kx = Kx_; Ky = Ky_;
    Kx_ = cxz.*Kx+sxz.*Kz
    Kz_ = -sxz.*Kx+cxz.*Kz
    Kx = Kx_; Kz = Kz_;
    Ky_ = cyz.*Ky+syz.*Kz
    Kz_ = -syz.*Ky+cyz.*Kz
    Ky = Ky_; Kz = Kz_;
    Kθ = KSpaceFixedSizeSampling{T}(K.h, cat(Kx*K.h[1], Ky*K.h[2], Kz*K.h[3]; dims=3))

    return phase_shift, Kθ

end

process_nfft_pars(K::KSpaceFixedSizeSampling{T}, θ::Nothing) where {T<:Real} = (complex(T)(1), K)