# Master Thesis Repository of Reto Hendry

## Introduction
The objective of this Master Thesis was to explore the potential of applying machine learning techniques to medical resting-state functional MRI data. Specifically, the aim was to determine whether it is possible to classify subjects into control or sham groups. 

## Data (Not Public)
The data utilized for this analysis was provided by the University Hospital Zurich. It is available in NIfTI format, and the data used for the analysis is the MVPA data, which has already been processed and smoothed.

There are a total of 90 subjects, each with pre- and post-measurement data. The NIfTI files essentially represent volumetric (3D) data from the scans.

## Applied AutoML Tools:
- TPOT (http://epistasislab.github.io/tpot/)
- PyCaret (https://pycaret.gitbook.io/docs/)
- H2O (https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html)

## Explanation of scripts/code and folders
Each folder has its own readme.md file which briefly describes the usage of each file in the folder. Here is a quick overview what the content of each folder is:
- **parameter_sweep_[autoML tool].py**: They are used to find the optimal and best pipeline to classify the data/subject. 
- **functions**: The scripts in this folder are used in the parameter_sweep scripts. Each of the function has a description on the usage as well as a description of the input and return parameters.
- **results**: csv-files with results of the autoML tools / parameter sweeps and are used for the final analysis. 
- **[autoML tool]_best_model**: Model pipelines generated by the different autoML tools. 
- **neural_networks**: Scripts for the analysis of the data with CNNs. This was the approach at the beginning of this thesis. After some failed attempts to achieve a satisfying accuracy, this approach was dismissed. 
- **exploration**: Scripts which were used to build up the functions step by step or to test some scripts on simplified arrays for their correctness. Some of the scripts do not work anymore since some functions and arrays have changed over time.
- **analysis_and_plots**: Scripts for the final analysis and evaluation of the performed parameter sweeps. 
- **environments**: exported environments with pip (.txt) and conda (.yaml).