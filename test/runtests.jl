using UtilitiesForMRI, Test

@testset "UtilitiesForMRI.jl" begin
    include("./test_kspace_sampling.jl")
    include("./test_translations.jl")
    include("./test_rotations.jl")
    include("./test_nfft.jl")
    include("./test_motionpars_utils.jl")
end