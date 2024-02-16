# Mobile Phone Price Prediction Using Machine Learning

## Abstract
This project tackles the challenge of predicting mobile phone prices in a saturated market, helping consumers make informed decisions based on value for money. Leveraging a commonly used dataset, the study employs machine learning algorithms to estimate prices based on various phone features. The algorithms tested include Artificial Neural Networks (ANN), k-Nearest Neighbors (kNN), Support Vector Machines (SVM), Decision Trees (DT), and Stacking. The standout result is the ANN model, which excels in accuracy without the need for dimensionality reduction. This model's performance is validated through rigorous cross-validation and testing, demonstrating its utility in guiding consumers in the dynamic mobile phone market. The project is implemented in Julia, a high-level, high-performance programming language.

## Project Structure
### Key Files and Folders
- `data_expl_and_preproc.ipynb`: The main notebook for data exploration and preprocessing. It includes oneHotEncoding, dataset splitting into train and test sets, normalization, and three approaches to preprocessing. The final datasets are also stored here.
- `final_evaluation.ipynb`: This is the main executable file for the final evaluation of the best obtained models during the experimental phase. In this file, the HoldOut at the beginning of the study is used to train the model with the whole train dataset, while the test data is used for the first time to obtain all metrics and the normalized confusion matrix to analyze the final performance of the models on new unseen data. 
- `modules/`: A directory containing all necessary Julia (.jl) functions, organized into modules for different aspects of the project:
  - `modules/ANN_Modeling.jl`: Functions for creating, training, and evaluating ANN models using Flux.
  - `modules/Evaluation.jl`: Utilities for model evaluation with various metrics.
  - `modules/ModelSelection.jl`: Functions for data splitting using standard techniques like HoldOut and cross-validation.
  - `modules/Plotting.jl`: Functionalities for printing and plotting different evaluation metrics.
  - `modules/Preprocessing.jl`: Functions for data preprocessing including encodings, normalization, etc.
  - `modules/Sk_modeling.jl`: Utilities for modeling with sklearn, covering training and evaluation of different base models and the ensemble StackingClassifier.
- `No Dimensionality Reduction (NDR)/`, `Principal Components Analysis (PCA)/`, and `Features Selection (FSelection)/`: Each folder contains:
  - `data.h5`: The data preprocessed according to the specific approach of the folder.
  - Jupyter notebooks for training and evaluating models using ANN, kNN, DT, SVM, and Stacking Ensemble techniques.

### Implementation
The project is implemented in Julia, chosen for its efficiency and capability to handle complex data processing and machine learning tasks. Each part of the project, from data preprocessing to model evaluation, is modularized for clarity and ease of use.

### Usage
To utilize the project:
1. Start with `data_expl_and_preproc.ipynb` for data exploration and preprocessing.
2. Proceed to the respective folders (NDR, PCA, FSelection) depending on the preprocessing approach you are interested in.
3. Explore the model training and evaluation notebooks within these folders to understand the performance of different machine learning models on the preprocessed data.

## Conclusion
This project demonstrates the effective use of various machine learning techniques to predict mobile phone prices, with the ANN model showing exceptional performance. It serves as a valuable tool for consumers and researchers in understanding and navigating the complex mobile phone market.
