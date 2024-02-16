module Plotting

##########################################################

# This module provides functionalities for printing 
# and plotting different metrics of evaluation.

###########################################################

using Statistics;
using Evaluation;

using Flux
using Plots
using StatsPlots


 
function plot_losses(losses_dict)
    plots_array = []
    
    for (topology, losses) in losses_dict
        trainingLosses, validationLosses, testLosses = losses

        p = plot(title="Losses vs Epochs for Topology: $topology", size=(900,600))
        
        plot!(p, trainingLosses, label="Training", legend=:topright)
        
        plot!(p, validationLosses, label="Validation", legend=:topright, linestyle=:dash)
        
        plot!(p, testLosses, label="Test", legend=:topright, linestyle=:dot)
        
        push!(plots_array, p)
    end

    # Show all plots
    plot(plots_array...)
end


#####################################################################################
#                            Confusion Matrices                                     #
#####################################################################################

function printMatrix(conf_matrix)
    # prints a regular given matrix
    println("-"^50)  # Separator

    for i in 1:size(conf_matrix, 1)
        for j in 1:size(conf_matrix, 2)
            print(string(conf_matrix[i, j], "    ")[1:4])  # Four spaces for alignment
        end
        println()  #New line
    end
    
    println("-"^50) 
end

function printConfusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)

    # Get metrics using the confusionMatrix function with real-valued outputs
    accuracy, error_rate, sensitivity, specificity, ppv, npv, f_score, conf_matrix = confusionMatrix(outputs, targets, weighted=weighted)

    # Print the metrics and confusion matrix
    println("Confusion Matrix:")
    printMatrix(conf_matrix)
    println("Accuracy: ", accuracy)
    println("Error Rate: ", error_rate)
    println("Sensitivity: ", sensitivity)
    println("Specificity: ", specificity)
    println("Positive Predictive Value: ", ppv)
    println("Negative Predictive Value: ", npv)
    println("F-Score: ", f_score)
    return accuracy, error_rate, sensitivity, specificity, ppv, npv, f_score, conf_matrix
end

function text_color(value, minval, maxval)
    # Function to compute the color of the text in plotConfusionMatrix
    # Normalized value
    normalized = (value - minval) / (maxval - minval)
    return normalized < 0.5 ? :black : :white
end


function plotConfusionMatrix(model::Symbol, conf_matrix; labels=["Class $i" for i in 1:size(conf_matrix, 1)], f_size=(500, 400), color=:blues)
    
    
    model_name = get_model_name(model)
    minval, maxval = extrema(conf_matrix) # Used to normalise cell values and select the proper color for legibility
    
    
    # Generates the heatmap with the y axis fliped
    heatmap_plot = heatmap(conf_matrix,
                           color=color,  # Color schema
                           xticks=(1:4, labels),
                           yticks=(1:4, labels),
                           xlabel="True labels",
                           ylabel="Output labels",
                           title= model_name * " Confusion Matrix",
                           size=f_size,
                           yflip=true)  # Flips y axis

    # Annotating the values of each cell
    # Using textColor to select a proper color for visualization
    for i in 1:size(conf_matrix, 1)
        for j in 1:size(conf_matrix, 2)
            annotate!(j, i, text(string(conf_matrix[i, j]), text_color(conf_matrix[i, j], minval, maxval), :center, 10))
        end
    end

    return heatmap_plot
end


function plotNormalizedConfusionMatrix(model::Symbol, conf_matrix; labels=["Class $i" for i in 1:size(conf_matrix, 1)], f_size=(500, 400), color=:blues)

    model_name = get_model_name(model);
    # Getting the normalized confusion matrix
    # dividing each cell by the sum of its row (number of instances of that class)
    conf_matrix = conf_matrix ./ sum(conf_matrix, dims=2)
    
    # Generates the heatmap with the y axis fliped
    heatmap_plot = heatmap(conf_matrix,
                           color=color,  # Esquema de colores
                           xticks=(1:4, labels),
                           yticks=(1:4, labels),
                           xlabel="True labels",
                           ylabel="Output labels",
                           title= model_name * " Normalized Confusion Matrix",
                           size=f_size,
                           yflip=true)  # Invierte el eje y

    # Annotating the values of each cell
    for i in 1:size(conf_matrix, 1)
        for j in 1:size(conf_matrix, 2)
            rounded_number = round(conf_matrix[i, j], digits=3)
            # Eassy handling of the color
            annotate!(j, i, text(string(rounded_number), conf_matrix[i, j] < 0.5 ? :black : :white, :center, 10))
        end
    end

    return heatmap_plot
end

function get_model_name(model::Symbol)
    model_name = ""
    if(model == :ANN)
        model_name = "ANN"
    elseif(model == :KNN)
        model_name = "KNN"
    elseif(model == :SVM)
        model_name = "SVM"
    elseif(model == :DecisionTree)
        model_name = "Decision Tree"
    else
        model_name = "Stacking"
    end
    return model_name
end

#####################################################################################
#                                    Bar plots                                      #
#####################################################################################

function plot_acc_comparison(model::Symbol, all_metrics)
    model_name = get_model_name(model);
        
    accs = []
    std_devs = []
    architectures = []
    for metrics in all_metrics
        push!(accs, metrics[2]["acc"][1])
        push!(std_devs, metrics[2]["acc"][2])
        push!(architectures, metrics[1])
    end

    gr()
    # Create a horizontal bar plot with error bars
    bar(accs, yerr=std_devs, label=false)

    # Add labels and title
    ylabel!("Accuracy")
    xlabel!("Architectures")
    xticks!(1:length(architectures), string.(architectures))
    title!(model_name * " Accuracies with Standard Deviations")
end

function plot_final_comparison(title, best_results)

    experiments = ["Exp. "*string(i) for i in 1:length(best_results)]

    accs = [result[1] for result in best_results] # Collect the accs
    std_devs = [result[2] for result in best_results] # Collect the std_devs
    
    gr()
    # Create a horizontal bar plot with error bars
    bar(accs, yerr=std_devs, label=false)

    # Add labels and title
    ylabel!("Accuracy")
    xlabel!("Experiments")
    xticks!(1:length(experiments), experiments)
    title!(title)
end


############################# Collecting metrics #################################


function printMetricsSummary(metrics_dict::Dict{String, Tuple{Float64, Float64}})
    sorted_metrics = ["acc", "err_rate", "sensitivity", "specificity", "ppv", "npv", "f_score"]
    
    println("\n--------------- METRICS SUMMARY ---------------")
    
    for key in sorted_metrics
        if haskey(metrics_dict, key)
            println("\n----- ", key, " -----")
            println("Mean:       ", round(metrics_dict[key][1], digits=4))
            println("Std. Dev.:  ", round(metrics_dict[key][2], digits=4))
        end
    end
    
    println("\n----------------------------------------------")
end

function printMetricsSummary(metrics_dict::Dict{String, Float64})
    sorted_metrics = ["acc", "err_rate", "sensitivity", "specificity", "ppv", "npv", "f_score"]
    
    println("\n--------------- METRICS SUMMARY ---------------")
    
    for key in sorted_metrics
        if haskey(metrics_dict, key)
            println()
            println("-", key, " ==> ",round(metrics_dict[key][1], digits=4))
        end
    end
    
    println("\n----------------------------------------------")
end
    
#In case we have more than one set of parameters to test and compare:

# Sort the metrics, descending by default, ascending for error rate
function sortMetrics(all_metrics, metric_name, descending=true)
    sort!(all_metrics, by=x -> x[2][metric_name][1], rev=descending)
end

# Print the sorted metrics
function printMetricsRanking(all_metrics)
    metrics_to_print = ["acc", "sensitivity", "specificity", "ppv", "npv", "f_score"]
    for metric_name in metrics_to_print
        sortMetrics(all_metrics, metric_name)
        println("\n----- $metric_name" * " means -----")
        for (index, metrics) in all_metrics
            println("Hyperparams set $index -> $(round(metrics[metric_name][1], digits=3)) ± $(round(metrics[metric_name][2], digits=3))")
        end
    end
    
    # Error rate is sorted in ascending order
    println("\n----- err_rate -----")
    sortMetrics(all_metrics, "err_rate", false)
    for (index, metrics) in all_metrics
        println("Hyperparams set $index -> $(round(metrics["err_rate"][1], digits=3)) ± $(round(metrics["err_rate"][2], digits=3))")
    end
end
    
    
#####################################################################################
#                         Dimensionality Reduction                                  #
#####################################################################################
    

function draw_results(x, y, var::Tuple{Int64,Int64} ,colors, target_names=nothing)
    # Creates a dispersion diagram, 
    # it helps to visulize the proyection of two components
    num_classes = length(unique(colors))

    if !isnothing(target_names)
        @assert num_classes == length(target_names)
        label = target_names
    else
        label = [string(i) for i in 1:num_classes]
    end

    fig = plot()
    for i in 1:num_classes
        scatter!(fig, x[y[:, i], var[1]], x[y[:, i], var[2]], markercolor=colors[i], label=label[i])
    end
    display(fig) # ADDED
end

export plot_losses, printMatrix, printMetricsSummary, printMetricsRanking, draw_results, plot_acc_comparison, plot_final_comparison, printConfusionMatrix, plotConfusionMatrix, plotNormalizedConfusionMatrix;  # Exports all functions

end  # Module end




