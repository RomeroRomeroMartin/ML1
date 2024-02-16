__precompile__(false)
module Sk_Modeling

##########################################################

# This module provides functionalities for modeling with 
# sklearn utilities. It covers full training and evaluation
# functions of different base models and also for the ensemble
# StackingClassifier

###########################################################
using Statistics;
using Flux;
using Random;

using ScikitLearn

@sk_import neural_network: MLPClassifier
@sk_import svm: SVC
@sk_import tree: DecisionTreeClassifier
@sk_import neighbors: KNeighborsClassifier


# Import stacking for ensembles
@sk_import ensemble: StackingClassifier


# including Preprocessing module
using Preprocessing;

# including Evaluation module
using Evaluation;

# including Ploting module
using Plotting;
using ANN_Modeling


#################################################################################
#                                    Hold-Out                                   #
#################################################################################


function modelHoldOut(modelType::Symbol,
    modelHyperparameters::Dict,
    trainingDataset::Tuple, 
    testDataset::Tuple)

    # If ANN the targets must be oneHot encoded
    @assert modelType != :ANN || (trainingDataset isa Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}})
    @assert modelType != :ANN || (testDataset isa Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}})

    
    if modelType == :ANN
        
        repetitionsTraining =  modelHyperparameters["repetitionsTraining"] # Non deterministic model
        topology = modelHyperparameters["architecture"]
        activation = modelHyperparameters["activation"]
        learning_rate_init = modelHyperparameters["learning_rate"]
        validation_ratio = modelHyperparameters["validation_ratio"]
        n_iter_no_change = modelHyperparameters["n_iter_no_change"]
        max_iter = modelHyperparameters["max_iter"]
        minLoss = modelHyperparameters["minLoss"]

        return trainClassANN_final(topology, 
                       trainingDataset, 
                       testDataset;
                       transferFunctions = activation, 
                       maxEpochs = max_iter, minLoss = minLoss, learningRate = learning_rate_init, 
                       repetitionsTraining = repetitionsTraining, validationRatio = validation_ratio, 
                       maxEpochsVal = n_iter_no_change);
    

    end
   
    model = createModel(modelType, modelHyperparameters)
    train_outs = vec(trainingDataset[2])
            
    # Training the model
    fit!(model, trainingDataset[1], train_outs);

        
    # Getting predictions
    predicted_outputs = predict(model, testDataset[1])
    expected_outs = vec(testDataset[2])

    acc, error_rate, sensitivity, specificity, ppv, npv, f_score, conf = confusionMatrix(predicted_outputs, expected_outs);


    metrics_dict = Dict(
        "acc" => acc,
        "err_rate" => error_rate,
        "sensitivity" => sensitivity,
        "specificity" => specificity,
        "ppv" => ppv,
        "npv" => npv,
        "f_score" => f_score
    )
    
  
    return metrics_dict, conf
end

#################################################################################
#                                 Cross-Validation                              #
#################################################################################


function modelCrossValidation(modelType::Symbol,
    modelHyperparameters::Dict,
    inputs::AbstractArray{<:Real,2},
    targets::AbstractArray{<:Any,1},
    crossValidationIndices::Array{Int64,1})

    train_set = tuple()

    N = size(inputs, 1)
    k = maximum(crossValidationIndices)
    
    if modelType == :ANN

        targets = oneHotEncoding(targets)
        
        repetitionsTraining = modelHyperparameters["repetitionsTraining"] # Non deterministic model
        topology = modelHyperparameters["architecture"]
        activation = modelHyperparameters["activation"]
        learning_rate_init = modelHyperparameters["learning_rate"]
        validation_ratio = modelHyperparameters["validation_ratio"]
        n_iter_no_change = modelHyperparameters["n_iter_no_change"]
        max_iter = modelHyperparameters["max_iter"]
        minLoss = modelHyperparameters["minLoss"]

        return trainClassANN(topology, 
                       (inputs, targets), 
                       crossValidationIndices;
                       transferFunctions = activation, 
                       maxEpochs = max_iter, minLoss=minLoss, learningRate = learning_rate_init, 
                       repetitionsTraining=repetitionsTraining, validationRatio = validation_ratio, 
                       maxEpochsVal = n_iter_no_change)
    end

    # Other models (No ANNs):
    targets = reshape(targets, :, 1) # A matrix is needed

    #Lists of metrics for the mean of each fold
    acc_metrics = zeros(Float64, k); err_rate_metrics = zeros(Float64, k);
    sensitivity_metrics = zeros(Float64, k); specificity_metrics = zeros(Float64, k);
    ppv_metrics = zeros(Float64, k); npv_metrics = zeros(Float64, k);
    f_score_metrics = zeros(Float64, k);


    for fold in 1:k
        train_indices = findall(x -> x != fold, crossValidationIndices)
        test_indices = findall(x -> x == fold, crossValidationIndices)
        
        train_set = (inputs[train_indices, :], targets[train_indices, :])
        test_set = (inputs[test_indices, :], targets[test_indices, :])

        # Model creation  
        model = createModel(modelType, modelHyperparameters)
            
        # Training the model
        train_outs = vec(train_set[2]) # Targest must be vectors for sklearn models
        fit!(model, train_set[1], train_outs);
            
        # Getting predictions
        predicted_outputs = predict(model, test_set[1])
        expected_outs = vec(test_set[2])

        acc, error_rate, sensitivity, specificity, ppv, npv, f_score, _ = confusionMatrix(predicted_outputs, expected_outs);


        acc_metrics[fold] = acc
        err_rate_metrics[fold] = error_rate
        sensitivity_metrics[fold] = sensitivity
        specificity_metrics[fold] = specificity
        ppv_metrics[fold] = ppv
        npv_metrics[fold] = npv
        f_score_metrics[fold] = f_score
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


#################################################################################
#                       Collecting metrcis for printing                         #
#################################################################################

# Wrapper function to execute the entire process
function evaluateAndPrintMetricsRanking(model, hyperparameters_array, norm_inputs, targets, kFoldIndices, estimators = nothing)
    
    @assert model != :Stacking || (estimators isa Vector{Vector{Symbol}}) "When using Stacking an array of estimators of type AbstractArray{AbstractArray{Symbol}} must be given"
    
    all_metrics = collectMetrics(model, hyperparameters_array, norm_inputs, targets, kFoldIndices, estimators)
    printMetricsRanking(all_metrics) #This calls to Ploting module
    return all_metrics
end

# Collect metrics for each set of hyperparameters
function collectMetrics(model, hyperparameters_array, norm_inputs, targets, kFoldIndices, estimators)
    
    all_metrics = []
    for (i, hyperparameters) in enumerate(hyperparameters_array)
        println("Training with set of hyperparameters $i")
        if model == :Stacking #Special treatment for ensemble
            targets = reshape(targets, :, 1)
            metrics = trainClassEnsemble(estimators[i], 
                            hyperparameters, # TYPE CHANGED
                            (norm_inputs, targets),    
                            kFoldIndices)
        else
            metrics = modelCrossValidation(model, hyperparameters, norm_inputs, targets, kFoldIndices)
        end
        push!(all_metrics, (i, metrics))
    end
    return all_metrics
end


#################################################################################
#                                   Ensembles                                   #
#################################################################################


function trainClassEnsemble(estimators::AbstractArray{Symbol,1}, 
                            modelsHyperParameters::AbstractArray{Dict{Symbol, Any}, 1}, # TYPE CHANGED
                            trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Int,2}},    
                            kFoldIndices::Array{Int64,1})

    N, numClasses = size(trainingDataset[2])
    k = maximum(kFoldIndices)
    
    #Lists of metrics for the mean of each fold
    acc_metrics = zeros(Float64, k); err_rate_metrics = zeros(Float64, k);
    sensitivity_metrics = zeros(Float64, k); specificity_metrics = zeros(Float64, k);
    ppv_metrics = zeros(Float64, k); npv_metrics = zeros(Float64, k);
    f_score_metrics = zeros(Float64, k);

    for fold = 1:k
        train_indices = findall(x -> x != fold, kFoldIndices)
        test_indices = findall(x -> x == fold, kFoldIndices)

        train_set = (trainingDataset[1][train_indices, :], reshape(trainingDataset[2][train_indices, :], :))
        test_set = (trainingDataset[1][test_indices, :], reshape(trainingDataset[2][test_indices, :], :))
        
        
        # Creating base models
        baseModels = []
        for (i, (estimator, hyperParams)) in enumerate(zip(estimators, modelsHyperParameters))
            model = createModel(estimator, hyperParams)
            fit!(model, train_set...)
            push!(baseModels, (string(estimator, i), model))  # Add index to the estimator name
        end

        # Creating the stacking ensemble
        final_estimator = SVC()
        ensemble = StackingClassifier(estimators=baseModels, final_estimator=final_estimator)
        fit!(ensemble, train_set...)

        # Evaluating the ensemble
        predicted_outputs = predict(ensemble, test_set[1])
        expected_outs = test_set[2]
        acc, error_rate, sensitivity, specificity, ppv, npv, f_score, _ = confusionMatrix(predicted_outputs, expected_outs);

        acc_metrics[fold] = acc
        err_rate_metrics[fold] = error_rate
        sensitivity_metrics[fold] = sensitivity
        specificity_metrics[fold] = specificity
        ppv_metrics[fold] = ppv
        npv_metrics[fold] = npv
        f_score_metrics[fold] = f_score
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

function trainClassEnsemble(estimators::AbstractArray{Symbol,1}, 
                            modelsHyperParameters::AbstractArray{Dict{Symbol, Any}, 1}, # TYPE CHANGED
                            trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Int,2}},    
                            testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Int,2}})

    N, numClasses = size(trainingDataset[2])


    train_set = (trainingDataset[1], vec(trainingDataset[2]))
    test_set = (testDataset[1], vec(testDataset[2]))
        
        
    # Creating base models
    baseModels = []
    for (i, (estimator, hyperParams)) in enumerate(zip(estimators, modelsHyperParameters))
        model = createModel(estimator, hyperParams)
        fit!(model, train_set...)
        push!(baseModels, (string(estimator, i), model))  # Add index to the estimator name
    end

    # Creating the stacking ensemble
    final_estimator = SVC()
    ensemble = StackingClassifier(estimators=baseModels, final_estimator=final_estimator)
    fit!(ensemble, train_set...)

    # Evaluating the ensemble
    predicted_outputs = predict(ensemble, test_set[1])
    expected_outs = test_set[2]
    acc, error_rate, sensitivity, specificity, ppv, npv, f_score, conf = confusionMatrix(predicted_outputs, expected_outs);


    metrics_dict = Dict(
        "acc" => acc,
        "err_rate" => error_rate,
        "sensitivity" => sensitivity,
        "specificity" => specificity,
        "ppv" => ppv,
        "npv" => npv,
        "f_score" => f_score
    )
    
  
    return metrics_dict, conf
end

# Helper function to create a model based on symbol and hyperparameters
function createModel(estimator::Symbol, hyperParams::Dict)
    if estimator == :ANN
        return MLPClassifier(; hyperParams...)
    
    elseif estimator == :DecisionTree
        # example params:
        # max_depth
        # criterion
        
        return DecisionTreeClassifier(; hyperParams...)
    
    elseif estimator == :SVM
        # example params:
        # kernel
        # degree
        # gamma
        # C
        return SVC(; hyperParams...)
    
    elseif estimator == :KNN
        # example params:
        # n_neighbors
        # metric => "mahalanobis"
        # metric_params
        # weights
        return KNeighborsClassifier(; hyperParams...)
    end
end

    
function trainClassEnsemble(baseEstimator::Symbol, 
    modelsHyperParameters::Dict,
    NumEstimators::Int,  # <--------------------------------- Originally: NumEstimators::Int=100, it gives error.
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}},     
    kFoldIndices::Array{Int64,1})
    
    # Create arrays of estimators and hyperparameters
    estimators = fill(baseEstimator, NumEstimators)
    modelsHyperParameters = fill(modelsHyperParameters, NumEstimators)

    # Call the original trainClassEnsemble function
    return trainClassEnsemble(estimators, modelsHyperParameters, trainingDataset, kFoldIndices)
end


export modelCrossValidation, evaluateAndPrintMetricsRanking, collectMetrics, trainClassEnsemble, modelHoldOut;  # Exports all functions
end  # Module end