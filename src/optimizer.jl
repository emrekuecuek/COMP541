mutable struct SM3
    lr::AbstractFloat
    eps::AbstractFloat
    gclip::AbstractFloat
    dims
#     momentum::AbstractArray{AbstractFloat}
    accumulators
end

for T in (Array{Float32},Array{Float64},KnetArray{Float32},KnetArray{Float64}); @eval begin
    function Knet.update!(w::$T, g, p::SM3)
        Knet.gclip!(g, p.gclip)
        g = full(g)
        if p.accumulators==nothing; 
            p.dims=size(w);
            p.accumulators=[KnetArray(zeros(Float32, _shape_for_broadcasting(p.dims, i))) for i in 1:length(p.dims)];
        end
        accumulator = _compute_past_accumulator(p.accumulators, p.dims)
        accumulator .+= g.*g
        #TODO: Add momentum tensor for scaled gradient
        axpy!(-p.lr, g./(sqrt.(accumulator .+ p.eps)), w)
        #TODO: Add accumulator updates
        p.accumulators = _accumulator_updater(p.accumulators, p.dims, accumulator)
    end
end;end

SM3(; lr=0.001, eps=1e-30, gclip=0.0) = SM3(lr, eps, gclip, nothing, nothing)
sm3(f,d; lr=0.001, eps=1e-30, gclip=0.0,o...) = Knet.minimize(f,d,SM3(lr, eps, gclip, nothing, nothing);o...)
sm3!(x...;o...) = for y in sm3(x...;o...); end
Knet.clone(a::SM3) = SM3(a.lr, a.eps, a.gclip, nothing, nothing)