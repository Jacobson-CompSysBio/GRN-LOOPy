# network_processing_utils
This project contains code for network pre and post processing.

# iRF Pre / Post Processing Exploration: 

## Introduction
Depending uppon how the data are preprocessed and post processed, there remains a question as to how heavily the differences will impact the final network. The algorithm used in this project will be Iterative Random Forest (iRF). This program explores the network differences from: 
- Preprocessing: 
	- Raw Data
	- Low Variance Features Removed
	- Highly Correlated Features Removed
	- Low Variance and Highly Correlated Features Removed
- Post Processing: 
	- Base Feature Importance as Network Edges
	- R2 * Feature Importance Network Edge Weighting
	- Accuracy * Feature Importance Network Edge Weighting

## Implementation

 
