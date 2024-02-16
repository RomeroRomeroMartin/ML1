module Preprocessing

##########################################################

# This module provides utilities for
# data preprocessing: Encodings, normalization...

###########################################################

using Statistics;
using Flux;

### Encoding

function one_hot_encoding(data::Vector{T}) where T
    values = unique(data)
    numClasses = length(values)

    if length(values) == 2
        encoded = data .== values[1]
        return reshape(encoded, :, 1)
    else
        oneHot = Array{Bool,2}(undef, length(targets), numClasses);
        for numClass = 1:numClasses
            oneHot[:,numClass] .= (targets.==values[numClass]);
        end
        return oneHot
    end
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


oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature, unique(feature));

oneHotEncoding(feature::AbstractArray{Bool,1}) = reshape(feature, :, 1);


##### Normalization ###



function Min_Max_Scaler(data::Array{Float32, 2})
    # Dim = 1 (Each row) Dim = 2 (Each column)
    # In this case each value should be taken per column, as each column represents one atribute/feature
    
    # It is needed to convert the resultin matrix into vectors in order to use de broadcasting operations
    # Moreover, it's necessary to transpose the vectors to get the correct shape
    
    min_values = vec(minimum(inputs, dims=1))'
    max_values = vec(maximum(inputs, dims=1))'
    mean_values = vec(mean(inputs, dims=1))'
    std_dev_values = vec(std(inputs, dims=1))'
    
    println("Min Values: ", min_values)
    println("Max Values: ", max_values)
    println("Mean Values: ", mean_values)
    println("Standard Deviation Values: ", std_dev_values)

    
    min_max_ranges = max_values .- min_values
    
    # Find the columns where the minimum value is equal to the maximum value
    equal_min_max = (min_max_ranges .== 0)
    
    # Count the number of columns to be removed
    # It can be done just as a sum as equal_min_max is a vector of ones and zeros
    num_removed = sum(equal_min_max)
    println("\nNumber of attributes removed: ", num_removed)

    # Remove the uninformative columns
    informative_cols = (.!equal_min_max)'
    filtered_data = data[:, informative_cols]
    
    # Normalize the filtered data 
    normalized_data = (filtered_data .- min_values[:,informative_cols]) ./ min_max_ranges[:,informative_cols]
    
    return normalized_data
end



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


export one_hot_encoding, oneHotEncoding, Min_Max_Scaler, calculateMinMaxNormalizationParameters, calculateZeroMeanNormalizationParameters, normalizeMinMax!, normalizeMinMax, normalizeZeroMean!; # Exports all functions

end  # Module end