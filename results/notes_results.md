## Notes results folder

In here are the results from the autoML tool saved. 

The results ending with *...2xC1* are the ones where the same parameters were exactly run twice. This was to test the reproducibility of the autoML tools. NOTE: The results of the files *pycaret_results_df_2xC1.csv* and *tpot_results_df_2xC1.csv* might be wrong since I found an error in the code after running them. However, they still fulfill the purpose of testing the reproducibility. \

The parameters for the reproducibility-check were the following:
- list_of_components = [1, 1]
- resample_cube_list = np.arange(1, 10, 1, dtype=int)
- number_of_feature_list = np.arange(10, 101, 10, dtype=int)

The *feature_list_df* contains information about each parameter configuration run with the list of the most important feature. It will be used to reverse the array flattening to find out what regions of the brain show the post prediction power.
