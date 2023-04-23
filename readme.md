# Master Thesis Repository of Reto Hendry

## Introduction
The objective of this Master Thesis was to explore the potential of applying machine learning techniques to medical resting-state functional MRI data. Specifically, the aim was to determine whether it is possible to classify subjects into control or sham groups. 

## Data (Not Public)
The data utilized for this analysis was provided by the University Hospital Zurich. It is available in NIfTI format, and the data used for the analysis is the MVPA data, which has already been processed and smoothed.

There are a total of 90 subjects, each with pre- and post-measurement data. The NIfTI files essentially represent volumetric (3D) data from the scans.

## Applied AutoML Tools:
- TPOT
- PyCaret
- H2O

## Explanation of scripts/code and folders
- **parameter_sweep_[autoML tool].py**: they are used to find the optimal and best pipeline to classify the data/subject. 
- **histogram_analysis_...**: 