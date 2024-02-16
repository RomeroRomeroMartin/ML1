module ANN_Modeling

##########################################################

# This module provides functionalities for building, 
# training, and evaluation ANNs using Flux.

###########################################################


using Statistics;

using Flux
using Flux.Losses
using Flux: crossentropy, binarycrossentropy, params
using Random

# Including needed custom modules
using ModelSelection;
using Evaluation;


function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int;
                    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology))) 
    ann=Chain();
    numInputsLayer = numInputs;
    for numHiddenLayer in 1:length(topology)
        numNeurons = topology[numHiddenLayer];
        ann = Chain(ann..., Dense(numInputsLayer, numNeurons, transferFunctions[numHiddenLayer]));
        numInputsLayer = numNeurons;
    end;
    if (numOutputs == 1)
        ann = Chain(ann..., Dense(numInputsLayer, 1, σ));
    else
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity));
        ann = Chain(ann..., softmax);
    end;
    return ann;
end;   


###################### Training ##########################3
#=
function trainClassANN(topology::AbstractArray{<:Int,1},      
                    dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
                    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
                    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01) 

    @assert size(dataset[1], 1) == size(dataset[2], 1) "Training inputs and targets must have the same number of rows."
    
    numEpoch = 0;trainingLoss=minLoss + 1
    inputs = dataset[1]; targets = dataset[2];
    n_patterns, n_features = size(inputs)
    n_classes = size(targets, 2)
    ann = buildClassANN(n_features, topology, n_classes, transferFunctions=transferFunctions)

    loss(m, x, y) = (n_classes == 1) ? Losses.binarycrossentropy(m(x), y) : Losses.crossentropy(m(x), y);
    
    trainingLosses = []
    opt_state = Flux.setup(Adam(learningRate), ann)
    while (numEpoch<maxEpochs) && (trainingLoss>minLoss)
    # Training. Matrixes must be transposed (each pattern in a column)
        Flux.train!(loss, ann, [(inputs', targets')], opt_state);
        numEpoch += 1;
        # Calculate the loss values for this cycle
        trainingLoss = loss(ann, inputs', targets');
        # Store the loss values for this cycle
        push!(trainingLosses, trainingLoss);
        println("Epoch ", numEpoch, ": loss: ", trainingLoss);
    end
    return ann
end                                      


function trainClassANN(topology::AbstractArray{<:Int,1},      
                    (inputs, targets)::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};      
                    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),      
                    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)
    @assert size(inputs, 1) == size(inputs, 1) "Training inputs and targets must have the same number of rows."

     trainClassANN(topology, (inputs, reshape(targets, length(targets), 1)); 
        maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate);
end;

=#

#################### Training with Early Stopping #########################333


function trainClassANN(topology::AbstractArray{<:Int,1},
        trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
        validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=
                (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)),
        testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=
                (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)),
        transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
        maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
        maxEpochsVal::Int=20, showText::Bool=false)

    @assert size(trainingDataset[1], 1) == size(trainingDataset[2], 1) "Training inputs and targets must have the same number of rows."
    @assert size(validationDataset[1], 1) == size(validationDataset[2], 1) "Validation inputs and targets must have the same number of rows."
    @assert size(testDataset[1], 1) == size(testDataset[2], 1) "Test inputs and targets must have the same number of rows."


    (trainingInputs, trainingTargets) = trainingDataset
    (validationInputs, validationTargets) = validationDataset
    (testInputs, testTargets) = testDataset
    
    n_classes = size(trainingTargets, 2)

    ann = buildClassANN(size(trainingInputs, 2), topology, size(trainingTargets, 2);
                        transferFunctions=transferFunctions)

    loss(m, x, y) = (n_classes == 1) ? Losses.binarycrossentropy(m(x), y) : Losses.crossentropy(m(x), y);

    trainingLosses = Float32[]
    validationLosses = Float32[]
    testLosses = Float32[]

    bestAnn = deepcopy(ann)
    bestValidationLoss = Inf
    epochsWithoutImprovement = 0

    opt = Flux.setup(Adam(learningRate), ann)

    for epoch in 1:maxEpochs
        Flux.train!(loss, ann, [(trainingInputs', trainingTargets')], opt)
        currTrainingLoss = loss(ann, trainingInputs', trainingTargets')
        currValidationLoss = length(validationInputs) > 0 ?
                             loss(ann, validationInputs', validationTargets') : Inf
        currTestLoss = length(testInputs) > 0 ?
                       loss(ann, testInputs', testTargets') : Inf
        
        #Tracking losses
        push!(trainingLosses, currTrainingLoss)
        push!(validationLosses, currValidationLoss)
        push!(testLosses, currTestLoss)

        if showText
            println("Epoch: $epoch, Training Loss: $currTrainingLoss, Validation Loss: $currValidationLoss, Test Loss: $currTestLoss")
        end
        
        # Performance checking - Deepcopy if it's the best
        if (currValidationLoss < bestValidationLoss) || (bestValidationLoss == Inf) # If is Inf, there is no val set
            bestValidationLoss = currValidationLoss
            epochsWithoutImprovement = 0
            if (bestValidationLoss == Inf) && (epoch != maxEpochs)
                continue;
            end
            bestAnn = deepcopy(ann)
        else
            epochsWithoutImprovement += 1
        end
        
        # Early stopping check
        if epochsWithoutImprovement >= maxEpochsVal
            #println("Early stopping triggered at epoch $epoch.")
            break
        end

        if length(validationInputs) > 0 && epochsWithoutImprovement >= maxEpochsVal
            #println("Early stopping triggered at epoch $epoch.")
            break
        end

        if currTrainingLoss <= minLoss
            break
        end
    end

    return bestAnn, trainingLosses, validationLosses, testLosses
end


function trainClassANN(
    topology::AbstractArray{<:Int,1},
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=
            (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)),
    testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=
            (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
    maxEpochsVal::Int=20, showText::Bool=false)

    @assert size(trainingDataset[1], 1) == size(trainingDataset[2], 1) "Training inputs and targets must have the same number of rows."
    @assert size(validationDataset[1], 1) == size(validationDataset[2], 1) "Validation inputs and targets must have the same number of rows."
    @assert size(testDataset[1], 1) == size(testDataset[2], 1) "Test inputs and targets must have the same number of rows."

    # Reshape targets for training, validation, and test datasets
    reshapedTrainingDataset = (trainingDataset[1], reshape(trainingDataset[2], :, 1))
    reshapedValidationDataset = (validationDataset[1], reshape(validationDataset[2], :, 1))
    reshapedTestDataset = (testDataset[1], reshape(testDataset[2], :, 1))

    # Call the previously defined trainClassANN function
    return trainClassANN(
        topology, reshapedTrainingDataset;
        validationDataset=reshapedValidationDataset,
        testDataset=reshapedTestDataset,
        transferFunctions=transferFunctions,
        maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate,
        maxEpochsVal=maxEpochsVal, showText=showText
    )
end


######################################################################################################################
#                                        Training with cross-validation                                              #
######################################################################################################################


function trainClassANN(topology::AbstractArray{<:Int,1}, 
                       trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}, 
                       kFoldIndices::Array{Int64,1};
                       transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), 
                       maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, 
                       repetitionsTraining::Int=1, validationRatio::Real=0.0, 
                       maxEpochsVal::Int=20)

    N = size(trainingDataset[1], 1)
    k = maximum(kFoldIndices)
    acc_metrics = zeros(Float64, k); err_rate_metrics = zeros(Float64, k);
    sensitivity_metrics = zeros(Float64, k); specificity_metrics = zeros(Float64, k);
    ppv_metrics = zeros(Float64, k); npv_metrics = zeros(Float64, k);
    f_score_metrics = zeros(Float64, k);


    for fold in 1:k
        train_indices = findall(x -> x != fold, kFoldIndices)
        test_indices = findall(x -> x == fold, kFoldIndices)
        
        train_indices, val_indices = holdOut(length(train_indices), validationRatio) #Spliting train data

        train_set = (trainingDataset[1][train_indices, :], trainingDataset[2][train_indices, :])
        val_set = (trainingDataset[1][val_indices, :], trainingDataset[2][val_indices, :])
        test_set = (trainingDataset[1][test_indices, :], trainingDataset[2][test_indices, :])

        fold_acc = Float64[]; fold_err_rate = Float64[]; fold_sensitivity = Float64[];
        fold_specificity = Float64[]; fold_PPV = Float64[]; fold_acc_NPV = Float64[];
        fold_acc_F_score = Float64[];

        for _ in 1:repetitionsTraining
            trained_model, _, _, _ = trainClassANN(topology, train_set,
                                                   validationDataset = val_set,
                                                   transferFunctions = transferFunctions,
                                                   maxEpochs = maxEpochs, minLoss = minLoss,
                                                   learningRate = learningRate, maxEpochsVal = maxEpochsVal)

            # Calculating accuracy for this iteration
            predicted_outputs = trained_model(test_set[1]')'
            #test_metric = accuracy(predicted_outputs, test_set[2])
            acc, error_rate, sensitivity, specificity, ppv, npv, f_score, _ = confusionMatrix(predicted_outputs, test_set[2]);

            push!(fold_acc, acc)
            push!(fold_err_rate, error_rate)
            push!(fold_sensitivity, sensitivity)
            push!(fold_specificity, specificity)
            push!(fold_PPV, ppv)
            push!(fold_acc_NPV, npv)
            push!(fold_acc_F_score, f_score)

        end

        acc_metrics[fold] = mean(fold_acc)
        err_rate_metrics[fold] = mean(fold_err_rate)
        sensitivity_metrics[fold] = mean(fold_sensitivity)
        specificity_metrics[fold] = mean(fold_specificity)
        ppv_metrics[fold] = mean(fold_PPV)
        npv_metrics[fold] = mean(fold_acc_NPV)
        f_score_metrics[fold] = mean(fold_acc_F_score)
    end

    metrics_dict = Dict(
        "acc" => (mean(acc_metrics), std(acc_metrics)),
        "err_rate" => (mean(err_rate_metrics), std(err_rate_metrics)),
        "sensitivity" => (mean(sensitivity_metrics), std(sensitivity_metrics)),
        "specificity" => (mean(specificity_metrics), std(specificity_metrics)),
        "ppv" => (mean(ppv_metrics), std(ppv_metrics)),
        "npv" => (mean(npv_metrics), std(npv_metrics)),
        "f_score" => (mean(f_score_metrics), std(f_score_metrics))
    )

    return metrics_dict
end

function trainClassANN(topology::AbstractArray{<:Int,1}, 
        trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}, 
        kFoldIndices::Array{Int64,1};
        transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), 
        maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, repetitionsTraining::Int=1, 
        validationRatio::Real=0.0, maxEpochsVal::Int=20)
    
    targets_matrix = reshape(trainingDataset[2], :, 1)
    
    new_trainingDataset = (trainingDataset[1], targets_matrix)
    
    return trainClassANN(topology, new_trainingDataset, kFoldIndices;
                         transferFunctions=transferFunctions, 
                         maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate, 
                         repetitionsTraining=repetitionsTraining, 
                         validationRatio=validationRatio, maxEpochsVal=maxEpochsVal)
end




function trainClassANN_final(topology::AbstractArray{<:Int,1}, 
                       trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}, 
                       testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
                       transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), 
                       maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, 
                       repetitionsTraining::Int=1, validationRatio::Real=0.0, 
                       maxEpochsVal::Int=20)

    # Returns a dict with all the metrics and the aggregated confusion matrix (A sum of the repetitionTraining conf matrices)


    n_classes = size(testDataset[2], 2) # Number of classes is the number length of the oneHot vector
    
    train_indices, val_indices = holdOut(size(trainingDataset[1], 1), validationRatio) #Spliting train data

    train_set = (trainingDataset[1][train_indices, :], trainingDataset[2][train_indices, :])
    val_set = (trainingDataset[1][val_indices, :], trainingDataset[2][val_indices, :])
    
    accs = Float64[]; err_rates = Float64[]; sensitivities = Float64[];
    specificities = Float64[]; PPVs = Float64[]; NPVs = Float64[];
    F_scores = Float64[];

    # matriz of zeros to acumulate the computed matrices
    agg_confusion_matrix = zeros(Float64, n_classes, n_classes)


    for _ in 1:repetitionsTraining
        trained_model, _, _, _ = trainClassANN(topology, train_set,
                                                validationDataset = val_set,
                                                transferFunctions = transferFunctions,
                                                maxEpochs = maxEpochs, minLoss = minLoss,
                                                learningRate = learningRate, maxEpochsVal = maxEpochsVal)

        # Calculating accuracy for this iteration
        predicted_outputs = trained_model(testDataset[1]')'
        #test_metric = accuracy(predicted_outputs, testDataset[2])
        acc, error_rate, sensitivity, specificity, ppv, npv, f_score, conf = confusionMatrix(predicted_outputs, testDataset[2]);

        push!(accs, acc)
        push!(err_rates, error_rate)
        push!(sensitivities, sensitivity)
        push!(specificities, specificity)
        push!(PPVs, ppv)
        push!(NPVs, npv)
        push!(F_scores, f_score)
        agg_confusion_matrix = agg_confusion_matrix .+ conf

    end


    metrics_dict = Dict(
        "acc" => (mean(accs), std(accs)),
        "err_rate" => (mean(err_rates), std(err_rates)),
        "sensitivity" => (mean(sensitivities), std(sensitivities)),
        "specificity" => (mean(specificities), std(specificities)),
        "ppv" => (mean(PPVs), std(PPVs)),
        "npv" => (mean(NPVs), std(NPVs)),
        "f_score" => (mean(F_scores), std(F_scores))
    )

    return metrics_dict, agg_confusion_matrix
end



export buildClassANN, trainClassANN, trainClassANN_final;  # Exports all functions

end  # Module end
