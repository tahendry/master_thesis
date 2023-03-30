"""
Date: 10.03.2023
Author: Reto Hendry


"""

#%% import libraries
import os
import numpy as np
import pandas as pd
import time
import datetime
import nibabel as nib
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from function_reshape_array import reshape_array as ra

#%% constants

# deletes all the columns witch only contain zeros
del_zero_cols = False
# apply the reshape function (big 3d array to small 3d array)
reshape_mean_volume = True



#%% filepath
# path to data
#  - on Windows: C:/Users/tahendry/Desktop/Masterthesis_Reto/
#  - on Linux:   ../data/

try:
    print(os.uname())
    data_path = "../data/"
except:
    print("Windows")
    data_path = "C:/Users/tahendry/Desktop/Masterthesis_Reto/"

# read in the excel-file with the labels
label_file = "Conn_IDs_Matching.xlsx"

# read excel with only the first three columns
label_df = (pd.read_excel(os.path.join(data_path, label_file),
                            usecols=[0, 1, 2])
            .replace({"Cond": {1: 0}})
            .replace({"Cond": {2: 1}})
            )

label_df.head()

# read MVPA data
path_content = os.listdir(os.path.join(data_path, "Denoised_Data_6mm", "MVPA_data"))

# make two lists with pre (Condition002) and post (Condition003) data of first component
comp1_pre = sorted([x for x in path_content 
                    if "Component001" in x 
                    and "Condition002" in x])
comp1_post = sorted([x for x in path_content 
                    if "Component001" in x 
                    and "Condition003" in x])

print(comp1_pre[:5])

#%% prepare data
# create a dataset with the difference of pre and post data
comp1_diff = []
for pre, post in zip(comp1_pre, comp1_post):
    pre_vol = nib.load(os.path.join(data_path, "Denoised_Data_6mm", "MVPA_data", pre))
    post_vol = nib.load(os.path.join(data_path, "Denoised_Data_6mm", "MVPA_data", post))
    pre_vol_data = pre_vol.get_fdata()
    post_vol_data = post_vol.get_fdata()
    diff_vol_data = post_vol_data - pre_vol_data
    comp1_diff.append(diff_vol_data)

# check the shape of the data
print(comp1_diff[0].shape)

# check the type of the data
print(type(comp1_diff[0]))

# stack the data
# note: the first dimension is the number of samples
print(f"shape of one list element before stacking: {comp1_diff[0].shape=}")
inpt_comp1_diff = np.stack(comp1_diff, axis=0)

# normalize the input data (zero mean, unit variance)
inpt_comp1_diff = (inpt_comp1_diff - inpt_comp1_diff.mean()) / inpt_comp1_diff.std()

# apply the reshape function to every sample of the input data
if reshape_mean_volume:
    reshape_mean_volume = (5, 5, 5)
    inpt_comp1_diff = np.array([ra(x, reshape_mean_volume) for x in inpt_comp1_diff])
    print(f"{inpt_comp1_diff.shape=}")

# transform the data from a 3d array to a 2d array
x_dim = inpt_comp1_diff.shape[1]
y_dim = inpt_comp1_diff.shape[2] 
z_dim = inpt_comp1_diff.shape[3]
num_samples = inpt_comp1_diff.shape[0]
inpt_comp1_diff = inpt_comp1_diff.reshape(num_samples, x_dim*y_dim*z_dim)


print(f"{inpt_comp1_diff.shape=}",
      f"{inpt_comp1_diff.mean()=}",
      f"{inpt_comp1_diff.std()=}", sep="\n")

# delete the columns with only zeros
if del_zero_cols:
    df_inpt_comp1_diff = pd.DataFrame(inpt_comp1_diff)
    df_inpt_comp1_diff = df_inpt_comp1_diff.loc[:, (df_inpt_comp1_diff != 0).any(axis=0)]
    inpt_comp1_diff = np.array(df_inpt_comp1_diff)

x_train, x_test, y_train, y_test = train_test_split(inpt_comp1_diff, 
                                                    label_df["Cond"], 
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=label_df["Cond"]
                                                    )

# check the shape of the data
print(f"{x_train.shape=}",
        f"{x_test.shape=}",
        f"{y_train.shape=}",
        f"{y_test.shape=}", sep="\n")


#%% define the autoML class
# Define the cross-validation strategy
cv_stratified = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def tpot_class(nbr_generations, nbr_population_size, max_time=None, identifier=999):

    tpot = TPOTClassifier(generations=nbr_generations,  # number of iterations for optimization
                        population_size=nbr_population_size,  # default 100
                        max_time_mins=max_time,
                        early_stop=5,
                        scoring="accuracy",
                        cv=cv_stratified,  # cross validation fold (default)
                        n_jobs=-1, # nbr. of cores used (-1 = all)
                        max_eval_time_mins=10,  # default 5
                        # random_state=1,  # seed
                        memory=False,  # avoid fitting same model
                        periodic_checkpoint_folder=f"./tpot_folder_reduced_3",
                        verbosity=3,  # print minimal info
                        log_file=f"./tpot_folder_reduced_3/tpot_log_iteration_reduced_{identifier}",
                        )
    return tpot

# parameter list for the tpot class (number of generations and population size)
parameter_list = [10, 10, 10,
                  11, 11, 11,
                  12, 12, 12,
                  13, 13, 13,
                  14, 14, 14,
                  15, 15, 15]

for k, i in enumerate(parameter_list):
    # print current time
    print(f"current time: {datetime.datetime.now()}")
    # create the tpot class
    tpot = tpot_class(nbr_generations=i, nbr_population_size=i, max_time=180, identifier=k)
    start = time.time()
    tpot.fit(x_train, y_train)
    print(f"generation and pop_size: {i}",
          f"trained pipelines: {len(tpot.evaluated_individuals_)}",
          f"best model: {tpot.fitted_pipeline_}",
          f"best model score: {tpot.score(x_test, y_test)}",
          f"time needed: {(time.time() - start)/60:.2f} mins",
          sep="\n", end="\n\n")


#%% fit the model
tpot.fit(x_train, y_train)
print("done with fitting")

#%% evaluate the model
print("pipeline: /n", tpot.fitted_pipeline_)
print(tpot.score(x_test, y_test))
tpot.export('tpot_pipeline.py')
print("best pipeline exported")

print("done with evaluation, time: {datetime.datetime.now}")
