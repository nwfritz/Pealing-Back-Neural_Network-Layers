# Pealing-Back-Neural_Network-Layers
Investigating Tertiary Lymphoid Structures (TLS) and Their Genetic Signatures Using Fully Connected Neural Networks (FCNNs)

# Project Overview 
Tertiary lymphoid structures (TLS) are organized aggregates of immune cells that form in non-lymphoid tissues during chronic inflammation, infection, or cancer. Unlike primary and secondary lymphoid organs, TLS develop in situ, often within tumor microenvironments, and are associated with improved outcomes in cancer immunotherapy. Their formation and function are driven by specific molecular signals and gene expression patterns, making them valuable biomarkers for disease progression and therapeutic response.

Gene expression profiling offers a powerful way to understand TLS by identifying key genes and pathways involved in their formation and anti-tumor immune interactions. However, the high dimensionality of these datasets presents challenges, necessitating advanced computational methods. Fully Connected Neural Networks (FCNNs) are well-suited for this task, as they excel at learning complex, non-linear relationships in high-dimensional data.

This project uses FCNNs to classify spatial transcriptomics regions as TLS+ or TLS- based on gene expression profiles. Key areas of focus include designing network architectures to improve accuracy, addressing overfitting due to class imbalance, and applying interpretability techniques to identify the most influential genes driving TLS prediction. By combining machine learning with biological insights, this study aims to advance our understanding of TLS and their role in cancer immunity.
# Data
![](https://github.com/user-attachments/assets/039425b3-6f2d-4303-9e5a-dcb43d07b55f)
Data for this task was taken off the Hugging Face data set page. It is a small sub-section of the STimage-1K4M: A histopathology image-gene expression dataset for spatial transcriptomic (ST). Spatial transcriptomics is a cutting-edge technique that combines gene expression profiling with spatial localization, allowing researchers to study where specific genes are expressed within the architecture of a tissue. Although data was taken from an ST data set it was not utilized in this way. The data used came from Human Kidenies labeled GSE175540_GSM5924030, through GSE175540_GSM5924035. I downloaded the gene_expression and annotation CSV's for each of them. Each section of data represents a section of the same Human Kidney. The annotation CSV held the key to whether each section led to TLS+ or-. I merged these data sets to get one overarching data set which contained over 23,000 individual sections of data containing some variation of 17,943 genes that could have been expressed resulting in TLS+/-. 

<img width="747" alt="image" src="https://github.com/user-attachments/assets/4be88138-9474-4b92-9ff0-9060ed868f0a" />

The ratio of TLS to NO_TLS is 1:27

<img width="399" alt="image" src="https://github.com/user-attachments/assets/5e1a60a6-d0d0-4b01-8039-4a72eb9a0682" />

This data was cleaned for NaN and shuffled before being split into Training, Testing, and Validation sets. 

<img width="500" alt="image" src="https://github.com/user-attachments/assets/64b12dcc-718f-423f-a9e2-2a4ee46bfb6f" />

# Fully Connected Neural Networks
Fully Connected Neural Networks (FCNNs) are a class of deep learning models well-suited for high-dimensional data like gene expression profiles. FCNNs consist of layers of neurons where every neuron in one layer is connected to every neuron in the next. They are particularly effective in learning complex, non-linear relationships in data, making them a valuable tool for tasks such as TLS classification.

In this project, FCNNs are employed to classify spatial regions as TLS+ or TLS- based on their gene expression profiles. The focus is on:

Architecture Design: Investigating how changes in the network's depth, width, and regularization affect performance.
Interpretability: Understanding which genes are most influential in the model's decision-making process for TLS+ and TLS- classification.
Overfitting Mitigation: Addressing challenges associated with overfitting due to class imbalance and high-dimensionality.

### Significance of FCNNs in this Study
Pattern Recognition in Complex Data:

FCNNs excel at detecting patterns in noisy, high-dimensional data, such as the expression of thousands of genes.
Exploration of Model Variants:

This study explores how different FCNN architectures (e.g., varying depths, dropout rates, and batch normalization) impact classification accuracy and generalization.
Importance of Interpretability:

By leveraging methods like saliency maps, the study identifies genes critical to TLS prediction, contributing to the biological understanding of TLS formation.
# Running the Notebook 
This Notebook contains 9 different sections. It was built on Google Colab using the A-100 GPU. I suggest using Google Colab to run this notebook, mapping your own google Drive. It contains 9 sections.

1. Getting Started: Downloading the necessary libraries to run the program
2. Mount Google Drive: Connecting to Google Drive to access data and save results
3. Download Data from Hugging Face and Save to Google Drive
4. Display Sample Data and Visualize It
5. Prepare Data for Training: Cleaning, and Visualizing
6. Build the Models: Preparing Architecture 
7. Prepare the Data to be Loaded into the Model
8. Train the Model
9. Analyze Results: Display key testing metrics
# Data Analysis 
To make predictions based on the genes expressed four different Fully Connected Neural Nets were created each with different architectures.
## Simple_FCNN
Description: A basic FCNN with three layers (input, hidden, and output).
Structure:
Input Layer: 17943 genes to 128 neurons.
Hidden Layer: 128 to 64 neurons, with Dropout to prevent overfitting.
Output Layer: 1 neuron (binary classification using Sigmoid).
Key Feature: Simple and lightweight, designed for basic classification tasks with minimal computation.
Use Case: Baseline performance model.
## Deep_FCNN
Description: A deeper and more complex FCNN with four layers and more neurons.
Structure:
Input Layer: 17943 genes to 256 neurons.
Hidden Layers: 256 → 128 → 64 neurons, each followed by Dropout for regularization.
Output Layer: 1 neuron (binary classification using Sigmoid).
Key Feature: Increased depth and number of neurons for learning more complex patterns.
Tradeoff: Higher computational cost and risk of overfitting compared to Simple_FCNN.
Use Case: Ideal for exploring more complex relationships in the data.
## BatchNorm_FCNN
Description: Similar to Deep_FCNN but with Batch Normalization layers added after each hidden layer.
Structure:
Batch Normalization: Normalizes activations during training to stabilize learning.
Hidden Layers: 256 → 128 → 64 neurons, each with Batch Normalization and Dropout.
Key Feature: Batch Normalization improves convergence speed and reduces internal covariate shift, enhancing generalization.
Tradeoff: Slightly more computational overhead.
Use Case: Suitable when training instability is observed, or faster convergence is desired.
## Residual_FCNN
Description: Incorporates Residual Connections, where outputs from earlier layers are added to later layers, enabling identity mappings.
Structure:
Residual Connection: Adds the output of the first hidden layer (with adjustments for dimensions) to the second hidden layer.
Hidden Layers: 256 → 128 → 64 neurons, with Dropout for regularization.
Key Feature: Helps mitigate vanishing gradients in deeper networks and retains information from earlier layers.
Tradeoff: Adds a bit of complexity to the architecture.
Use Case: Useful for deeper networks where gradient issues or loss of information is a concern.
# Results
Each model was separately used to analyze the data to get an idea of the differences each model picked up. Overall each model performed at around 96%. Which is decent, but mostly because almost all of the data set was TLS-. For each model the Loss, R-Score, and Accuracy for each Epoch was found. The best-performing epoch for each model was determined by finding the smallest Training loss. All results can be seen in the Result_Graphs folder within this GitHub. The key results are summarized in the table bellow.
<img width="737" alt="image" src="https://github.com/user-attachments/assets/089a63b1-fd5f-4696-8c4f-4b7ac864dfd5" />

Metrics were calculated based on the best model for each type of FCNN. They were:
Accuracy: The proportion of correctly classified samples out of the total number of samples.
Precision: The proportion of correctly predicted positive samples out of all samples predicted as positive.
Recall (Sensitivity or True Positive Rate): The proportion of actual positive samples that were correctly predicted as positive.
F1 Score: The harmonic mean of precision and recall, providing a single metric that balances both.
ROC-AUC (Receiver Operating Characteristic - Area Under the Curve): The area under the ROC curve, which plots the True Positive Rate (TPR or Recall) against the False Positive Rate (FPR).

Confusion matrices for all models were created to show the rates of correct classification. They showed a critical issue. True positives (TLS+) are underrepresented.

Saliency Maps were created to highlight the important genes for classifying TLS+ or TLS-. There were key genes that were used to help classify both TLS+ and TLS-, but the rate of expression could have a greater impact.
# Discussion

# Conclusion 

# Works Cited 
https://arxiv.org/pdf/2406.06393
https://huggingface.co/datasets/jiawennnn/STimage-1K4M/tree/main
Chat GPT
