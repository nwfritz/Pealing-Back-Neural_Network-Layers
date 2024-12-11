# Pealing-Back-Neural_Network-Layers
Investigating Tertiary Lymphoid Structures (TLS) and Their Genetic Signatures Using Fully Connected Neural Networks (FCNNs)

# Project Overview 
Tertiary lymphoid structures (TLS) are organized aggregates of immune cells that form in non-lymphoid tissues during chronic inflammation, infection, or cancer. Unlike primary and secondary lymphoid organs, TLS develop in situ, often within tumor microenvironments, and are associated with improved outcomes in cancer immunotherapy. Their formation and function are driven by specific molecular signals and gene expression patterns, making them valuable biomarkers for disease progression and therapeutic response.

Gene expression profiling offers a powerful way to understand TLS by identifying key genes and pathways involved in their formation and anti-tumor immune interactions. However, the high dimensionality of these datasets presents challenges, necessitating advanced computational methods. Fully Connected Neural Networks (FCNNs) are well-suited for this task, as they excel at learning complex, non-linear relationships in high-dimensional data.

This project uses FCNNs to classify spatial transcriptomics regions as TLS+ or TLS- based on gene expression profiles. Key areas of focus include designing network architectures to improve accuracy, addressing overfitting due to class imbalance, and applying interpretability techniques to identify the most influential genes driving TLS prediction. By combining machine learning with biological insights, this study aims to advance our understanding of TLS and their role in cancer immunity.
# Data
![](https://github.com/user-attachments/assets/039425b3-6f2d-4303-9e5a-dcb43d07b55f)
Data for this task was taken off the Hugging Face data set page. It is a small sub-section of the STimage-1K4M: A histopathology image-gene expression dataset for spatial transcriptomic (ST). Spatial transcriptomics is a cutting-edge technique that combines gene expression profiling with spatial localization, allowing researchers to study where specific genes are expressed within the architecture of a tissue. Although data was taken from an ST data set it was not utilized in this way. The data used came from Human Kidenies labeled GSE175540_GSM5924030, through GSE175540_GSM5924035. I downloaded the gene_expression and annotation CSV's for each of them. Each section of data represents a section of the same Human Kidney. The annotation CSV held the key to whether each section led to TLS+ or-. I merged these data sets to get one overarching data set which contained over 23,000 individual sections of data containing some variation of 17,943 genes that could have been expressed resulting in TLS+/-. 

![](<img width="747" alt="image" src="https://github.com/user-attachments/assets/4be88138-9474-4b92-9ff0-9060ed868f0a" />)

The ratio of TLS to NO_TLS is 1:27

![](<img width="399" alt="image" src="https://github.com/user-attachments/assets/5e1a60a6-d0d0-4b01-8039-4a72eb9a0682" />)

This data was cleaned for NaN and shuffled before being split into Training, Testing, and Validation sets. 

![](<img width="500" alt="image" src="https://github.com/user-attachments/assets/64b12dcc-718f-423f-a9e2-2a4ee46bfb6f" />)

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
