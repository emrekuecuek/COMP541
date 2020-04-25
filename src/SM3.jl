module SM3

export
    SM3
    sm3
    sm3!
    
using Knet, LinearAlgebra, AutoGrad: full
include("optimizer.jl")
include("helperfunctions.jl")

end