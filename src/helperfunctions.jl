function _shape_for_broadcasting(dims, desired)
    rank = length(dims)
    return tuple([i==desired ? dims[i] : 1 for i in 1:rank]...)
end

function _compute_past_accumulator(accumulators, dims)
    rank = length(dims)
    accumulators_for_broadcasting = [
        reshape(accumulators[i], _shape_for_broadcasting(dims, i))
        for i in 1:rank]
    
    result = accumulators_for_broadcasting[1]
#     return result
    # Check if min is doing for number of elmns.
    for i in 1:rank
        result = min.(result, accumulators_for_broadcasting[i])
    end
    return result
    
end

function _accumulator_updater(accumulators, dims, update_tensor)
    rank = length(dims)
    for i in 1:rank
        max_dims = []
#       max_dims = [i!=j ? j : for j in 1:rank]
#       TODO: Make this by array comprehension.
        for j in 1:rank
            if i!=j
                append!(max_dims, j)
            end
        end
        accumulators[i] = max.(accumulators[i], maximum(update_tensor, dims=tuple(max_dims...)))
    end
    return accumulators
end