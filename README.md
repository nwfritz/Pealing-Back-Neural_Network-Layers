# Pealing-Back-Neural_Network-Layers
Investigating Tertiary Lymphoid Structures (TLS) and Their Genetic Signatures Using Fully Connected Neural Networks (FCNNs)

# Project Overview 
Tertiary lymphoid structures (TLS) are organized aggregates of immune cells that form in non-lymphoid tissues during chronic inflammation, infection, or cancer. Unlike primary and secondary lymphoid organs, TLS develop in situ, often within tumor microenvironments, and are associated with improved outcomes in cancer immunotherapy. Their formation and function are driven by specific molecular signals and gene expression patterns, making them valuable biomarkers for disease progression and therapeutic response.

Gene expression profiling offers a powerful way to understand TLS by identifying key genes and pathways involved in their formation and anti-tumor immune interactions. However, the high dimensionality of these datasets presents challenges, necessitating advanced computational methods. Fully Connected Neural Networks (FCNNs) are well-suited for this task, as they excel at learning complex, non-linear relationships in high-dimensional data.

This project uses FCNNs to classify spatial transcriptomics regions as TLS+ or TLS- based on gene expression profiles. Key areas of focus include designing network architectures to improve accuracy, addressing overfitting due to class imbalance, and applying interpretability techniques to identify the most influential genes driving TLS prediction. By combining machine learning with biological insights, this study aims to advance our understanding of TLS and their role in cancer immunity.
# Data
![](https://github.com/user-attachments/assets/039425b3-6f2d-4303-9e5a-dcb43d07b55f)

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

# Data Analysis 

# Results

# Discussion

# Conclusion 

# Works Cited 
https://arxiv.org/pdf/2406.06393
https://huggingface.co/datasets/jiawennnn/STimage-1K4M/tree/main
Chat GPT
