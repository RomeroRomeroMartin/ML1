module Evaluation

##########################################################

# This module provides different utilities to 
# evaluate the models with various metrics.

###########################################################


using Statistics;
using Flux;

# including preprocessing module
using Preprocessing;

function classifyOutputs(outputs::AbstractArray{<:Real,2}; 
                        threshold::Real=0.5) 
   numOutputs = size(outputs, 2);
    @assert(numOutputs!=2)
    if numOutputs==1
        return outputs.>=threshold;
    else
        # Look for the maximum value using the findmax funtion
        (_,indicesMaxEachInstance) = findmax(outputs, dims=2);
        # Set up then boolean matrix to everything false while max values aretrue.
        outputs = falses(size(outputs));
        outputs[indicesMaxEachInstance] .= true;
        # Defensive check if all patterns are in a single class
        @assert(all(sum(outputs, dims=2).==1));
        return outputs;
    end;
end;


function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1}) 
    mean(outputs.==targets);
end;


function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}) 
    @assert(all(size(outputs).==size(targets)));
    if (size(targets,2)==1)
        return accuracy(outputs[:,1], targets[:,1]);
    else
        return mean(all(targets .== outputs, dims=2));
    end;
end;

function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1};      
                threshold::Real=0.5)
    accuracy(outputs.>=threshold, targets);
end;


function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2};
                threshold::Real=0.5)
    @assert(all(size(outputs).==size(targets)));
    if (size(targets,2)==1)
        return accuracy(outputs[:,1], targets[:,1]);
    else
        return accuracy(classifyOutputs(outputs; threshold=threshold), targets);
    end;
end;



############### Confusion Matrices ####################################33


function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    @assert length(outputs) == length(targets) "Outputs and targets must have the same length."

    # Calculate confusion matrix elements
    TP = sum((outputs .== true) .& (targets .== true))
    TN = sum((outputs .== false) .& (targets .== false))
    FP = sum((outputs .== true) .& (targets .== false))
    FN = sum((outputs .== false) .& (targets .== true))

    # Calculate metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    error_rate = 1 - accuracy

    # Handle edge cases for metrics
    sensitivity = (TP + FN == 0) ? 1.0 : TP / (TP + FN)
    specificity = (TN + FP == 0) ? 1.0 : TN / (TN + FP)
    positive_predictive_value = (TP + FP == 0) ? 1.0 : TP / (TP + FP)
    negative_predictive_value = (TN + FN == 0) ? 1.0 : TN / (TN + FN)
    denominator_f_score = (positive_predictive_value + sensitivity)
    f_score = (denominator_f_score == 0) ? 0.0 : (2 * positive_predictive_value * sensitivity) / denominator_f_score

    # Create confusion matrix
    confusion = Array{Int64,2}([TN FP; FN TP])

    return accuracy, error_rate, sensitivity, specificity, positive_predictive_value, negative_predictive_value, f_score, confusion
end

function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    # Binarize the outputs using the threshold
    binarized_outputs = outputs .>= threshold

    # Use the previously defined confusionMatrix function
    return confusionMatrix(binarized_outputs, targets)
end

using LinearAlgebra

function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    # Check if number of columns is equal for both matrices
    if size(outputs, 2) != size(targets, 2)
        throw(ArgumentError("Outputs and targets must have the same number of columns"))
    end

    numClasses = size(outputs, 2)

    # If only one column, then it's a binary classification
    if numClasses == 1
        return confusionMatrix(vec(outputs), vec(targets))
    end

    # Initialize metrics
    sensitivity = zeros(numClasses)
    specificity = zeros(numClasses)
    PPV = zeros(numClasses)
    NPV = zeros(numClasses)
    F_score = zeros(numClasses)
    confusion = zeros(Int, numClasses, numClasses)

    validClasses = 0 # Counter for classes with instances

    # Iterate over each class
    for i in 1:numClasses
        if sum(targets[:, i]) != 0 # Check if there are patterns in this class
            _, _, sens, spec, ppv, npv, f, _ = confusionMatrix(outputs[:, i], targets[:, i])
            sensitivity[i] = sens
            specificity[i] = spec
            PPV[i] = ppv
            NPV[i] = npv
            F_score[i] = f
            validClasses += 1
        end

        # Construct the confusion matrix
        for j in 1:numClasses
            confusion[i, j] = sum(outputs[:, i] .& targets[:, j])
        end
    end

    # Aggregate metrics
    if weighted
        weights = sum(targets, dims=1) / size(targets, 1)
        sensitivity = dot(sensitivity, weights)
        specificity = dot(specificity, weights)
        PPV = dot(PPV, weights)
        NPV = dot(NPV, weights)
        F_score = dot(F_score, weights)
    else
        sensitivity = sum(sensitivity) / validClasses
        specificity = sum(specificity) / validClasses
        PPV = sum(PPV) / validClasses
        NPV = sum(NPV) / validClasses
        F_score = sum(F_score) / validClasses
    end

    acc = accuracy(outputs, targets)
    error_rate = 1 - acc
    return acc, error_rate, sensitivity, specificity, PPV, NPV, F_score, confusion
end

function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    bool_outputs = classifyOutputs(outputs)
    return confusionMatrix(bool_outputs, targets, weighted=weighted)
end

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    # Ensure all output classes are included in target classes
    @assert(all([in(output, unique(targets)) for output in outputs]))

    # Get unique classes from both outputs and targets
    classes = unique([outputs; targets])

    # Convert outputs and targets to one-hot encoded form
    encoded_outputs = oneHotEncoding(outputs, classes)
    encoded_targets = oneHotEncoding(targets, classes)

    # Call the confusionMatrix function
    return confusionMatrix(encoded_outputs, encoded_targets, weighted=weighted)
end



export classifyOutputs, accuracy, confusionMatrix;  # Exports all functions

end  # Module end


