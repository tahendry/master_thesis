import nibabel as nib
import nilearn as nil
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import os
import pandas as pd

# print(os.listdir('./04_Data/test_data'))
# brain_vol = nib.load('../04_Data/test_data/fMRI_post.nii')

file_path = "../data/Denoised_Data_6mm"

# fMRI file
path_content = os.listdir(file_path)
brain_vol = nib.load(os.path.join(file_path, path_content[-1]))

# # ICA data
# path_content = os.listdir(os.path.join(file_path, "ICA_data"))
# brain_vol = nib.load(os.path.join(file_path, "ICA_data", path_content[0]))

# MVPA data
# path_content = os.listdir(os.path.join(file_path, "MVPA_data"))
# brain_vol = nib.load(os.path.join(file_path, "MVPA_data", path_content[0]))

# What is the type of this object?
print(type(brain_vol))

print(brain_vol.header)

brain_vol_data = brain_vol.get_fdata()
print(type(brain_vol_data))
print(brain_vol_data.shape)

brain_vol_data = brain_vol_data[:,:,:,300]  # 300th time point for fMRI data set
# brain_vol_data = brain_vol_data[:, :, :]

# make a histogram from a numpy array
plt.hist(brain_vol_data.flatten(), bins=100, log=True)

print(brain_vol_data.shape)

fig_rows = 4
fig_cols = 4
n_subplots = fig_rows * fig_cols
n_slice = brain_vol_data.shape[0]
step_size = n_slice // n_subplots
plot_range = n_subplots * step_size
start_stop = int((n_slice - plot_range) / 2)

fig, axs = plt.subplots(fig_rows, fig_cols, figsize=[10, 10])

for idx, img in enumerate(range(start_stop, plot_range, step_size)):
    axs.flat[idx].imshow(ndi.rotate(brain_vol_data[img, :, :], 90), cmap="gray")
    axs.flat[idx].axis("off")

plt.tight_layout()
plt.show()


