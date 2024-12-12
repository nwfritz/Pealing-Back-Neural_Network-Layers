# Pealing-Back-Neural_Network-Layers
Investigating Tertiary Lymphoid Structures (TLS) and Their Genetic Signatures Using Fully Connected Neural Networks (FCNNs)

# Project Overview 
Tertiary lymphoid structures (TLS) are organized aggregates of immune cells that form in non-lymphoid tissues during chronic inflammation, infection, or cancer. Unlike primary and secondary lymphoid organs, TLS develops in situ, often within tumor microenvironments, and is associated with improved outcomes in cancer immunotherapy. Their formation and function are driven by specific molecular signals and gene expression patterns, making them valuable biomarkers for disease progression and therapeutic response.

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
This Notebook contains 9 different sections. It was built on Google Colab using the A-100 GPU. I suggest using Google Colab to run this notebook, mapping your own Google Drive. It contains 9 sections.

1. Getting Started: Downloading the necessary libraries to run the program
2. Mount Google Drive: Connecting to Google Drive to access data and save results
3. Download Data from Hugging Face and Save to Google Drive
4. Display Sample Data and Visualize It
5. Prepare Data for Training: Cleaning, and Visualizing
6. Build the Models: Preparing Architecture 
7. Prepare the Data to be loaded into the Model
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
The four FCNN models all had an accuracy score of 97%. This sounds great but is extremely misleading due to the large disparities between TLS+ and TLS-. This ended up being detrimental to the effectiveness of the models, as there is clear signs of model bias towards the TLS- sections of the data, as well as overfitting in the TLS- direction. This is seen when looking at the number of correctly predicted TLS+/-. It was very effective at predicting TLS-, but was below 50% for all TLS+ predictions. Meaning it was almost always tailored to TLS- due to the greater amount of data in its favor with a couple of small indicators allowing essential coin flip to determine TLS+.

## Performance Metric Analysis 
#### Accuracy
All models achieve a high accuracy of 0.97, indicating they classify the majority class (TLS-) correctly.
#### Precision
Precision ranges from 0.51 to 0.62, showing how well the models handle TLS+ predictions with fewer false positives. Residual_FCNN performs best in this regard.
#### Recall
Recall ranges from 0.2 to 0.43, highlighting the models' struggles to correctly identify TLS+ samples. Deep_FCNN achieves the highest recall.
#### F1 Score
F1 score, balancing precision and recall, is highest for Deep_FCNN at 0.47, indicating its superior handling of TLS+ classification.
#### ROC-AUC
ROC-AUC is consistently high across models, with the Deep_FCNN achieving the highest at 0.97, indicating strong class separation overall.
#### Key Takeaways
While all models achieve similar accuracy, metrics like recall and F1 score reveal how Deep_FCNN handles TLS+ classification better than others.Residual_FCNN and BatchNorm_FCNN improve precision but do not significantly enhance recall, suggesting they focus more on reducing false positives for TLS+.
## TLS+ and TLS- Gene Contribution Analysis
#### TLS+ Contributions
SLC6A3 consistently appears as a top contributor in Simple_FCNN, Residual_FCNN, and Deep_FCNN, underscoring its importance in identifying TLS+ regions.
C12orf74 is a prominent contributor for both Simple_FCNN and Residual_FCNN, further emphasizing its relevance.
#### TLS- Contributions
SLC6A3 also contributes strongly to TLS-, suggesting that its role might be context-dependent, with differences in expression levels determining classification.
Deep_FCNN highlights genes like ADIPOQ and FCRL1 for TLS-, showcasing its ability to identify diverse patterns compared to simpler models.
#### Key Takeaways
The overlap in genes like SLC6A3 and C12orf74 for both TLS+ and TLS- suggests that expression intensity or dynamics, rather than binary presence, might drive classification.
Deep_FCNN identifies more unique genes for both TLS+ and TLS-, reflecting its capacity to learn nuanced relationships in the data.
## Correct Predictions
#### TLS+ Correct Predictions
Simple_FCNN correctly predicts 35 TLS+ samples, while Deep_FCNN predicts the most at 53.
#### TLS- Correct Predictions
Residual_FCNN predicts TLS- most accurately, with 3308 correct predictions, followed by BatchNorm_FCNN.
#### Key Takeaways and Call Backs
Deep_FCNN strikes a better balance by improving TLS+ predictions (highest recall and F1 score) while maintaining acceptable TLS- accuracy.
Models like BatchNorm_FCNN show strong TLS- performance but struggle with TLS+, leading to poor recall.
## Overall Model Comparison 
### Simple_FCNN
##### Strengths
High accuracy on this data set; easy to train and interpret.
##### Weaknesses
Poor recall (TLS+), indicating a strong bias toward TLS-.
### Residual_FCNN
##### Strengths
Improves precision for TLS+ while maintaining high TLS- accuracy.
##### Weaknesses
Struggles to improve recall or generalization significantly.
### BatchNorm_FCNN
##### Strengths
Stabilizes training with batch normalization; best TLS- predictions.
##### Weaknesses
Lowest recall and F1 score for TLS+, indicating limited ability to generalize across classes.
### Deep_FCNN
##### Strengths
Best overall performance with the highest recall, F1 score, and ROC-AUC. Captures nuanced features.
##### Weaknesses
Slightly lower TLS- accuracy, possibly due to a focus on learning complex TLS+ patterns.
## Past Work 
Based on past work done when exploring gene expression related to Tertiary Lymphoid Structures, the key has been to focus of genes that are expressed within the immune system. Especially infiltration with cells like B-Cells and CD8+ T-Cells. TLS are structures within tumors that promote ant-tumor immune responses so genes like CD19 (B-cell marker), CD8A (cytotoxic T-cell marker), CD79A (activated B-cell marker), and immunoglobulin genes (IGJ, IGKC) are often significantly upregulated in tumors with TLS present, indicating a strong immune response within the tumor microenvironment(Solis, 2023). What I found within my work did not greatly match up with these findings. I attribute that greatly to the data discrepancies between TLS+ and TLS-. However, my BatchNorm_FCNN model did highlight IGKC as the greatest indicator relating to TLS+ activation, but it was also a high indicator for TLS- activation.

# Conclusion 
In conclusion, this experiment provided a deeper understanding and learning opportunities for many things. This included PyTorch and FCNNs. I was able to write code for my own FCNN models and deepened my understanding of Neural Networks while learning about the intricacies and nuances that different types of FCNN models can bring. I learned about data sets and the importance of better split data, and how this will result in training biases in one direction over the other. Finally, I learned how to analyze key metrics to understand how each model did. Overall Deep_FCNN did the best as it was able to capture more complex relationships, making it the most effective model for TLS+ detection. 
## Future Work and Next Steps
##### Improving Recall 
The low recall across models indicates that TLS+ regions are often misclassified as TLS-. To hopefully address this I would like to try using Weighted Loss Functions during training. Additionally gathering more data off the Huggin Face website while only keeping enough to balance out the Number of TLS+ and TLS- expressions would be helpful in training and balancing the dataset. 
##### Gene Insights
Repeated identification of genes like SLC6A3 (dopamine transport protein) suggests potential biological relevance. Future research could validate these genes experimentally to better understand their roles in TLS formation. Through current work, I could not find any relevance or connection to TLS.
##### Future Model Enhancements
Consider integrating spatial transcriptomics data into a Graph Neural Network (GNN), which can incorporate spatial relationships between TLS regions for better prediction.
Explore ensemble learning by combining predictions from the best-performing models to balance TLS+ and TLS- predictions further.

# Works Cited 
Chatgpt. (n.d.). https://chatgpt.com/ 
Chen, J., Zhou, M., Wu, W., Zhang, J., Li, Y., & Li, D. (2024, June 20). Stimage-1K4M: A histopathology image-gene expression dataset for spatial transcriptomics. arXiv.org. https://arxiv.org/abs/2406.06393 
Hou, Y., Qiao, S., Li, M., Han, X., Wei, X., Pang, Y., & Mao, H. (2023, January 10). The gene signature of tertiary lymphoid structures within ovarian cancer predicts the prognosis and immunotherapy benefit. Frontiers in genetics. https://pmc.ncbi.nlm.nih.gov/articles/PMC9871364/ 
Solis, E. S. (n.d.). Https://www.annalsofoncology.org/action/showPdf?pi... Annals of Oncology. https://www.annalsofoncology.org/action/showPdf?pii=S0923-7534%2819%2945311-8 
