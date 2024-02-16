#Corssvalidation function
function crossvalidation(N::Int64, k::Int64)
    #TODO
    # Step 1: Create a vector with k sorted elements
    subsets = collect(1:k)
    # Step 2: Repeat the vector until its length is greater than or equal to N
    subsets = repeat(subsets, outer = ceil(Int, N / k))
    # Step 3: Take the first N values of this vector
    subsets = subsets[1:N]
    # Step 4: Shuffle the vector
    shuffle!(subsets)

    return subsets
end

#Crossvalidation function
function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    #TODO
    N = size(targets, 1)
    indices = collect(1:N)
    if size(targets, 2) == 1
        loop=2
        for class in 1:loop
            # Step 1: Take the number of elements belonging to that class
            if class==1
                value=0
                num_elements=count(!, targets)
            else
                value=1
                num_elements=count(targets)
            end
            # Step 2: Call the crossvalidation function
            class_indices = crossvalidation(num_elements, k)

            # Step 3: Update the index vector positions
            if class==1
                indices[1:num_elements]=class_indices
            else
                indices[end-num_elements+1:end]=class_indices
            end
        end
        
    else
        loop=size(targets, 2)
        for class in 1:loop
            # Step 1: Take the number of elements belonging to that class
            num_elements = sum(targets[:, class])
            # Step 2: Call the crossvalidation function
            class_indices = crossvalidation(num_elements, k)
            # Step 3: Update the index vector positions
            indices[targets[:, class]] = class_indices;
        end
    end
    shuffle!(indices)  # Shuffle the final index vector

    return indices

end


# One-hot encoding function
function oneHotEncoding(labels)
    unique_labels = unique(labels)
    encoded_targets = [labels .== label for label in unique_labels]
    return hcat(encoded_targets...)
end
    
#Crossvalidation function for one-hot encoded targets
function crossvalidation(targets::AbstractArray{<:Any, 1}, k)
    println("Onehot")
    encoded_targets = oneHotEncoding(targets)
    return crossvalidation(encoded_targets, k)
end


function oneHotEncoding(feature::AbstractArray{<:Any,1},      
    classes::AbstractArray{<:Any,1})
# First we are going to set a line as defensive to check values
@assert(all([in(value, classes) for value in feature]));

# Second defensive statement, check the number of classes
numClasses = length(classes);
@assert(numClasses>1)

if (numClasses==2)
    # Case with only two classes
    oneHot = reshape(feature.==classes[1], :, 1);
else
    #Case with more than two clases
    oneHot =  BitArray{2}(undef, length(feature), numClasses);
    for numClass = 1:numClasses
        oneHot[:,numClass] .= (feature.==classes[numClass]);
    end;
end;
return oneHot;
end;

#Define a function equivalent to 
# function oneHotEncoding(feature::AbstractArray{<:Any,1})

oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature, unique(feature));

#function oneHotEncoding(feature::AbstractArray{Bool,1})
#    return reshape(feature, (length(feature), 1))
#end;

# It is prefirable an overload of the method
oneHotEncoding(feature::AbstractArray{Bool,1}) = reshape(feature, :, 1);

function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})
    return minimum(dataset, dims=1), maximum(dataset, dims=1)
end;

# Alternative more compact definition
#calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2}) = ( minimum(dataset, dims=1), maximum(dataset, dims=1) );

function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2})
    return mean(dataset, dims=1), std(dataset, dims=1)
end;

function normalizeMinMax!(dataset::AbstractArray{<:Real,2},      
    normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    minValues = normalizationParameters[1];
    maxValues = normalizationParameters[2];
    dataset .-= minValues;
    dataset ./= (maxValues .- minValues);
    # eliminate any atribute that do not add information
    dataset[:, vec(minValues.==maxValues)] .= 0;
    return dataset;
end;

function normalizeMinMax!(dataset::AbstractArray{<:Real,2})
    normalizeMinMax!(dataset , calculateMinMaxNormalizationParameters(dataset));
end;

function normalizeMinMax( dataset::AbstractArray{<:Real,2},      
    normalizationParameters::NTuple{2, AbstractArray{<:Real,2}}) 
normalizeMinMax!(copy(dataset), normalizationParameters);
end;

function normalizeMinMax( dataset::AbstractArray{<:Real,2})
    normalizeMinMax!(copy(dataset), calculateMinMaxNormalizationParameters(dataset));
end;

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2},      
    normalizationParameters::NTuple{2, AbstractArray{<:Real,2}}) 
    avgValues = normalizationParameters[1];
    stdValues = normalizationParameters[2];
    dataset .-= avgValues;
    dataset ./= stdValues;
    # Remove any atribute that do not have information
    dataset[:, vec(stdValues.==0)] .= 0;
    return dataset; 
end;


function normalizeZeroMean!(dataset::AbstractArray{<:Real,2})
    normalizeZeroMean!(dataset , calculateZeroMeanNormalizationParameters(dataset));   
end;
function normalizeZeroMean( dataset::AbstractArray{<:Real,2},      
    normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
normalizeZeroMean!(copy(dataset), normalizationParameters);
end;

function normalizeZeroMean( dataset::AbstractArray{<:Real,2}) 
    normalizeZeroMean!(copy(dataset), calculateZeroMeanNormalizationParameters(dataset));
end;

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

using Statistics;

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

using Random
function holdOut(N::Int, P::Real)
    #TODO
    if P < 0 || P > 1
        throw(ArgumentError("P must be between 0 and 1"))
    end
    
    indices = randperm(N)
    
    train_size = Int((1 - P) * N)
    
    train_indices = indices[1:train_size]
    test_indices = indices[train_size+1:end]
    
    return (train_indices, test_indices)
end

function holdOut(N::Int, Pval::Real, Ptest::Real) 
    #TODO
   if Pval < 0 || Pval > 1 || Ptest < 0 || Ptest > 1 || Pval + Ptest > 1
        throw(ArgumentError("Invalid values for Pval and Ptest"))
    end
    
    indices = randperm(N)
    
    train_size = Int((1 - Pval-Ptest) * N)
    val_size=Int(Pval*N)
    train_indices = indices[1:train_size]
    val_indices=indices[train_size+1:train_size+val_size]
    test_indices = indices[train_size+val_size+1:end]
    
    return (train_indices, val_indices,test_indices)
end

using Random

function holdOut(N, P)
    indices = randperm(N)
    test_size = round(Int, N * P)
    test_indices = indices[1:test_size]
    train_indices = indices[test_size+1:end]
    return (train_indices, test_indices)
end

function holdOut(N, Pval, Ptest)
    # Calculate the remaining percentage for the training set
    Ptrain = 1.0 - Pval - Ptest
    
    # Generate training set indices
    train_indices, temp_indices = holdOut(N, Ptrain)
    
    # Calculate the remaining indices for validation and test sets
    Premaining = 1.0 - Ptrain
    Pval_adjusted = Pval / Premaining
    Ptest_adjusted = Ptest / Premaining
    
    # Generate validation and test set indices
    val_indices, test_indices = holdOut(length(temp_indices), Pval_adjusted)
    
    # Adjust indices to match the original range
    val_indices = temp_indices[val_indices]
    test_indices = temp_indices[test_indices]
    
    return (train_indices, val_indices, test_indices)
end

"""function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    #TODO"""

function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    tp = sum(outputs .== true .& targets .== true)
    tn = sum(outputs .== false .& targets .== false)
    fp = sum(outputs .== true .& targets .== false)
    fn = sum(outputs .== false .& targets .== true)

    accuracy = (tp + tn) / length(outputs)
    
    error_rate = (fp + fn) / length(outputs)

    sensitivity = tp / max(tp + fn, 1)  # Avoid division by zero
    specificity = tn / max(tn + fp, 1)  # Avoid division by zero

    ppv_denom = max(tp + fp, 1)  # Avoid division by zero
    npv_denom = max(tn + fn, 1)  # Avoid division by zero

    positive_predictive_value = tp / ppv_denom
    negative_predictive_value = tn / npv_denom

    f_score_denom = max(positive_predictive_value + sensitivity, 1)  # Avoid division by zero
    f_score = 2 * positive_predictive_value * sensitivity / f_score_denom

    confusion_matrix = [tp fp; fn tn]

    return accuracy, error_rate, sensitivity, specificity, positive_predictive_value, negative_predictive_value, f_score, confusion_matrix
end

"""function confusionMatrix(outputs::AbstractArray{<:Real,1},targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    #TODO"""

function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}, threshold::Real=0.5)
    predictions = outputs .>= threshold
    return confusionMatrix(predictions, targets)
end

"""function printConfusionMatrix(outputs::AbstractArray{Bool,1},targets::AbstractArray{Bool,1})
    #TODO"""
function printConfusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    accuracy, error_rate, sensitivity, specificity, ppv, npv, f_score, confusion_matrix = confusionMatrix(outputs, targets)

    println("Confusion Matrix:")
    println("True Positive: ", confusion_matrix[1, 1])
    println("False Positive: ", confusion_matrix[1, 2])
    println("False Negative: ", confusion_matrix[2, 1])
    println("True Negative: ", confusion_matrix[2, 2])

    println("\nMetrics:")
    println("Accuracy: ", accuracy)
    println("Error Rate: ", error_rate)
    println("Sensitivity: ", sensitivity)
    println("Specificity: ", specificity)
    println("Positive Predictive Value: ", ppv)
    println("Negative Predictive Value: ", npv)
    println("F-Score: ", f_score)
end

"""function printConfusionMatrix(outputs::AbstractArray{<:Real,1},targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    #TODO"""
function printConfusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}, threshold::Real=0.5)
    accuracy, error_rate, sensitivity, specificity, ppv, npv, f_score, confusion_matrix = confusionMatrix(outputs, targets, threshold)

    println("Confusion Matrix:")
    println("True Positive: ", confusion_matrix[1, 1])
    println("False Positive: ", confusion_matrix[1, 2])
    println("False Negative: ", confusion_matrix[2, 1])
    println("True Negative: ", confusion_matrix[2, 2])

    println("\nMetrics:")
    println("Accuracy: ", accuracy)
    println("Error Rate: ", error_rate)
    println("Sensitivity: ", sensitivity)
    println("Specificity: ", specificity)
    println("Positive Predictive Value: ", ppv)
    println("Negative Predictive Value: ", npv)
    println("F-Score: ", f_score)
end


function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    
    numClasses = size(outputs, 2)
    
    if numClasses == 1
        # If there is only one class, treat it as binary classification
        return confusionMatrix(outputs[:, 1], targets[:, 1])
    elseif numClasses == 2
        # If there are two classes, treat it as binary classification
        return confusionMatrix(outputs, targets)
    end
    sensitivity = zeros(Float64, numClasses)
    specificity = zeros(Float64, numClasses)
    ppv = zeros(Float64, numClasses)
    npv = zeros(Float64, numClasses)
    f1 = zeros(Float64, numClasses)
    
    confusion_matrix = zeros(Int, numClasses, numClasses)
    
    for class_num in 1:numClasses
        
        class_outputs = outputs[:, class_num]
        class_targets = targets[:, class_num]
        #println(class_outputs)
        #@assert(any(class_outputs))
        
         # Calculate confusion matrix for the current class
         _, _, sensitivity[class_num], specificity[class_num], ppv[class_num], npv[class_num], f1[class_num], _ = confusionMatrix(class_outputs, class_targets)

    end
     # Fill the overall confusion matrix
    for i in 1:numClasses
        for j in 1:numClasses
            confusion_matrix[i, j] += sum((outputs[:, i] .== true) .& (targets[:, j] .== true))
        end
    end

    # Calculate accuracy and error rate
    accuracy_value = accuracy(outputs, targets)
    error_rate = 1.0 - accuracy_value
    # Calculate the macro or weighted average
    if weighted
        # Weighted average
        weights = sum(targets, dims=1)
        weights /= sum(weights)
        
        sensitivity = dot(sensitivity, weights)
        specificity = dot(specificity, weights)
        ppv = dot(ppv, weights)
        npv = dot(npv, weights)
        f1 = dot(f1, weights)
    else
        # Macro average
        sensitivity = mean(sensitivity)
        specificity = mean(specificity)
        ppv = mean(ppv)
        npv = mean(npv)
        f1 = mean(f1)
    end
    return accuracy_value, error_rate, sensitivity, specificity, ppv, npv, f1, confusion_matrix

end



function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    boolean_outputs = classifyOutputs(outputs)
    return confusionMatrix(boolean_outputs, targets; weighted=weighted)
end

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    @assert(all([in(output, unique(targets)) for output in outputs]))


    # Get unique classes
    unique_classes = unique(outputs)

    # Encode outputs and targets using one-hot encoding
    encoded_outputs = oneHotEncoding(outputs, unique_classes)
    encoded_targets = oneHotEncoding(targets, unique_classes)

    # Call the confusionMatrix function
    return confusionMatrix(encoded_outputs, encoded_targets; weighted=weighted)
end






