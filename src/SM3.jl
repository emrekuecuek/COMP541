module SM3
    
using Knet, LinearAlgebra, AutoGrad: full
include("optimizer.jl")
include("helperfunctions.jl")

export SM3, sm3, sm3!

end