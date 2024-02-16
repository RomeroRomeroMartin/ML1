module ModelSelection

##########################################################

# This module provides functions for splitting the
# data with different standard techniques, such as
# HoldOut and cross-validation.

###########################################################

using Statistics;
using Flux;
using Random;

# including preprocessing module
using Preprocessing;

################################## - HOLD OUT - #######################################


function holdOut(N::Int, P::Real)
    @assert ((P>=0.) & (P<=1.));
    indices = randperm(N)
    n_train = Int(round((1-P)*N))
    return (indices[1:n_train], indices[n_train + 1:end])
end

function holdOut(N::Int, Pval::Real, Ptest::Real) 
    @assert ((Pval>=0.) & (Pval<=1.))
    @assert ((Ptest>=0.) & (Ptest<=1.))

    train_indices, temp_indices = holdOut(N, Pval + Ptest)
    val_indices, test_indices = holdOut(length(temp_indices), Ptest / (Pval + Ptest))

    return (train_indices, temp_indices[val_indices], temp_indices[test_indices])
end


#################### - Cross validation functions - ###################################

function crossvalidation(N::Int64, k::Int64)
    folds = collect(1:k) # vector with k sorted elements, from 1 to k
    reps = ceil(Int64, N / k) # Calculate the number of repetitions needed to make the length >= N
    
    repeated_vector = repeat(folds, reps) # Repeat the sorted_vector to make its length >= N
    truncated_vector = repeated_vector[1:N] # Take the first N values
    
    shuffle!(truncated_vector)
    return truncated_vector
end

function crossvalidation(targets::AbstractArray{Bool, 2}, k::Int64)
    N, n_classes = size(targets)
    index_vector = Array{Int64,1}(undef, N);
    
    for c in 1:n_classes
        index_vector[findall(x -> x, targets[:,c])] .= crossvalidation(sum(targets[:,c]), k)
    end
    
    return index_vector
end

function crossvalidation(targets::AbstractArray{<:Any, 1}, k::Int64)
    one_hot_targets = oneHotEncoding(targets) # Convert the targets to one-hot encoding
    return crossvalidation(one_hot_targets, k) # Call the second crossvalidation function
end



export holdOut, crossvalidation;  # Exports all functions

end  # Module end