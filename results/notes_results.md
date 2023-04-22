## Notes results folder

In here are the results from the autoML tool saved. 

The results ending with *...2xC1* are the ones where the same parameters were exactly run twice. This was to test the reproducibility of the autoML tools. \
The files without this ending contain the same information. But for the analysis, the results are filtered for duplicates of the parameters and only the first occurrence is taken.

The parameters for the reproducibility-check were the following:
- list_of_components = [1, 1]
- resample_cube_list = np.arange(1, 10, 1, dtype=int)
- number_of_feature_list = np.arange(10, 101, 10, dtype=int)

The *feature_list_df* contains information about each parameter configuration run with the list of the most important feature. It will be used to reverse the array flattening to find out what regions of the brain show the post prediction power.
