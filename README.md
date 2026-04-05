# CMPE188-MLProject: ToxScanAI
ToxScan AI accurately sorts through data gaming chats and identify whether there is any abusive language in the texts. A user can input text as a query and the model will evaluate its toxicity score based on context and wording. It looks through given datasets and picks out which ones have higher toxicity scores, and which texts likely do not have any animosity. Ideas for project extentions include: self-collected datasets from various games, real-time moderation/censoring, an additional explanation of why the AI thinks the sample is toxic, and tailoring it for one game specifically. 

### Current Implementation Progress: 
We are setting up the framework for the model and starting on the basic code. 

## Planned Model/System Approach:
The model is trained using labeled datasets. 
1. Data is inputted in batches
2. Forward pass where the model makes predictions
3. Compare preditions to the ground truth labels and do loss calculation 
4. Generate a toxicity score

## TEAM: 
Bryan Morales Sosa, 012060516

Janet Chiem, 017820933

Darren San, 017100720

## Datasets:
+ CONDA Dataset (CONtextual Dual-Annotated), University of Sydney, 2021 
+ CONDA Dataset (CONtextual Dual-Annotated), Kaggle, 2020
+ our own self acquired & self created datasets from various games (TBD)
