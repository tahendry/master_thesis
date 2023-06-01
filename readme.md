# Master Thesis Repository of Reto Hendry

## Abstract
Chronic pain causes high healthcare costs and often cannot be cured with conventional pain medication. This master thesis is performed in collaboration with the Magnetic Resonance Imaging Center at the psychiatric University Hospital Zurich, where a study is conducted on a new non-drug-based intervention method against chronic pain involving exposure to a static electric field (EF) of 50 kV to the human body. Ninety subjects have undergone the intervention, of which 44 belong to the verum group (actual intervention with static EF) and 46 to the sham group (no static EF applied, placebo). This thesis aims to identify neurophysiological markers (brain regions) using different ML methods, indicating that a static EF may influence human neurophysiology by changing brain activity patterns. First, a histogram-based feature selection is applied to functional magnetic resonance imaging (fMRI) data to identify the most distinguishable brain regions between the verum and sham groups. The relevance of these regions is tested with classification models, autoML tools in particular, to classify the distinct regions of the brain into verum and sham groups. TPOT, PyCaret, and H2O are utilized for the classification. Classification test accuracies of up to 94% are achieved, indicating that a distinction based on distinct brain regions is possible and that there are changes in brain activity from exposure to static EF. In addition, the feature importance of the best models is evaluated, revealing the brain regions most important for the ML models to correctly classify the data into the two groups. The knowledge gained from this master thesis has the potential to further inspire research in this area by investigating the identified brain regions and improving the understanding of the effects of static EF exposure on the brain.

## Data (Not Public)
The data utilized for this analysis was provided by the University Hospital Zurich. It is available in NIfTI format, and the data used for the analysis is the MVPA data, which has already been processed and smoothed.

There are a total of 90 subjects, each with pre- and post-measurement data. The NIfTI files essentially represent volumetric (3D) data from the scans.

## Applied AutoML Tools:
- TPOT (http://epistasislab.github.io/tpot/)
- PyCaret (https://pycaret.gitbook.io/docs/)
- H2O (https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html)

## Hardware used
A Linux workstation is used for the thesis, running on 64-bit Ubuntu 20.04.5 LTS, equipped with the following hardware:
- Processor: Intel Xeon® Gold 5222 CPU @ 3.80GHz x 8
- RAM: 256 GB
- GPU: dual NVIDIA GeForce RTX 2080 Ti/PCIe/SSE2


## Explanation of scripts/code and folders
Each folder has its own readme.md file (work in progress) which briefly describes the usage of each file in the folder. Here is a quick overview what the content of each folder is:
### Files
- **parameter_sweep_[autoML tool].py**: This scripts allows to run a parameter sweep on the MVPA data using the autoML tool.
- **create_mask_of_features.py**: This script creates NIfTI files which show the positions of the most important features.
- **proof_of_correctness**: Aims to verify that the feature selection method and autoML tools work as expected.


### folders
- **analysis_and_plots**: Scripts for the final analysis and evaluation of the performed parameter sweeps. Also contains the analysis of the feature importance and its results.
- **environments**: exported environments with pip (.txt) and conda (.yaml).
- **example_data**: NIfTI files and headers of NIfTI files.
- **exploration**: Scripts which were used to build up the functions step by step or to test some scripts on simplified arrays for their correctness. Some of the scripts do not work anymore since some functions and arrays have changed over time.
- **figures**: all figures produced with Python that are used in my master thesis
- **functions**: all functions used throughout this thesis. The functions are used in the files above and in multiple jupyter notebooks. 
- **neural_networks**: jupyter notebook with many approaches. (none of them produced valuable results)
- **results**: csv-files with results of the autoML tools / parameter sweeps.


