from skimage.morphology import remove_small_objects, remove_small_holes, opening, disk
from skimage.measure import label, regionprops
import spectral.io.envi as envi
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
import os

# extract and save the average reflectance of the cube

def avg_reflectance(hypercube_arr, mask, filename):
# def avg_reflectance(hypercube_arr, mask):

    # hypercube_arr = np.array(img.load())
    mask_bool = mask.astype(bool)

    # extract plant pixels from the boolean mask
    bean_pixels = hypercube_arr[mask_bool, :]

    # Calculate the average reflectance across all plant pixels for each band
    average_reflectance = np.mean(bean_pixels, axis=0)

    # return average reflectance for all the wavelengths (201) not just the sliced cube but whole
    # return average_reflectance

    # save the average reflectances into a 'csv' file

    data = {
        'wavelengths': list(range(350, 1006, 4)),
        'reflectance': average_reflectance
    }

    df = pd.DataFrame(data)

    df.to_csv(f'results/samples/{filename}.csv', index=False)
    return df

def avg_reflectance(hypercube_arr, mask):

    mask_bool = mask.astype(bool)
    bean_pixels = hypercube_arr[mask_bool, :]
    average_reflectance = np.mean(bean_pixels, axis=0)
    
    return average_reflectance  

def get_proprties(fresh_beans_mask):
    
    labeled_mask = label(fresh_beans_mask)
    num_obejcts = labeled_mask.max()
    print(f'There are {num_obejcts} beans in this genoype.')
    props = regionprops(labeled_mask)

    return props

def labeled_data(bean_ref, bean_mask, properties):

    minr, mincol, maxr, maxcol = properties.bbox
    bean_ref = bean_ref[minr:maxr, mincol:maxcol, :]
    bean_mask = bean_mask[minr:maxr, mincol:maxcol]

    return bean_ref, bean_mask


def viz_beans(beans, id=None, save=False):
    
    svm_maskk = beans[1]

    label_data_sm = label(svm_maskk)
    label_data_sm.max()

    label_data_sm.astype(bool)
    plt.close()
    plt.figure(figsize=(8, 5))
    plt.imshow(label_data_sm, cmap='Pastel1')

    if save==True:
        plt.savefig(f'results/images/gene_{id}.png', dpi=250)
    
    plt.show()


wavelength_list = list(range(350, 1006, 4))
# print(wavelength_list)
