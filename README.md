# HP Network Generator: 
This project contains code for the high performance generation of networks from machine learning methods.

## Installation
To run this code, an system with MPI installed is required. When running this code on a slurm system, it is recommended that users refer to the system documentation and proper usage of MPI within that environment (i.e. specific lmod modules required to properly run MPI). 

Ex. 
Running the package code on OLCF's Frontier system requires 
```bash 
module load rocm
module load boost
module load ums/default
module load ums002/default
module load cray-mpich/8.1.23

# Load Environment w/ required dependencies: 
source activate /ccs/home/lanemj/environments/frontier/lightgbm
```

# Network Generation Processing: 


## Introduction
Depending uppon how the data are preprocessed and post processed, there remains a question as to how heavily the differences will impact the final network. The algorithm used in this project will be Iterative Random Forest (iRF). This program explores the network differences from: 
- Preprocessing: 
	- Raw Data
	- Low Variance Features Removed
	- Highly Correlated Features Removed
	- Low Variance and Highly Correlated Features Removed
- Processing: 
	
- Post Processing: 
	- Base Feature Importance as Network Edges
	- R2 * Feature Importance Network Edge Weighting
	- Accuracy * Feature Importance Network Edge Weighting

## Implementation

 
