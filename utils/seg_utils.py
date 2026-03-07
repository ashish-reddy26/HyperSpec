from skimage.morphology import remove_small_objects, remove_small_holes, opening, disk
from skimage.measure import label, regionprops
import spectral.io.envi as envi
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
import os


# train SVM classifier

def train_svm(hypercube, plant_coords, background_coords, C_reg=450):
    
    # hypercube = image_cube.load()

    # hypercube = slice_cube(image_cube)

    # --- Build the training data from your selected points ---
    X_train = []
    y_train = []
    bands = hypercube.shape[2]

    # Add plant pixels (Class 1)
    for r, c in plant_coords:
        X_train.append(hypercube[r, c, :].reshape(bands))
        y_train.append(1)

    # Add background pixels (Class 0)
    for r, c in background_coords:
        X_train.append(hypercube[r, c, :].reshape(bands))
        y_train.append(0)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    print(f"Training data created successfully!")
    print(f"Plant samples: {len(plant_coords)}")
    print(f"Background samples: {len(background_coords)}")
    print(f"Total training samples: {X_train.shape[0]}")

    # --- Step 2: Train the SVM Classifier ---
    # The 'C' and 'gamma' parameters can be tuned.
    from sklearn.svm import SVC

    classifier = SVC(C=C_reg, gamma='scale')
    classifier.fit(X_train, y_train)
    print("Classifier trained successfully!")

    # --- Step 3: Predict on the Entire Image ---
    # Reshape the hypercube to a long list of pixels (n_pixels, n_bands)
    rows, cols, bands = hypercube.shape
    img_reshaped = hypercube.reshape(rows * cols, bands)

    # Predict the class for every pixel
    predicted_labels = classifier.predict(img_reshaped)

    # Reshape the prediction back into a 2D image mask
    final_mask = predicted_labels.reshape(rows, cols)

    # --- Visualize the Result ---
    # plt.figure(figsize=(6, 6))
    # plt.imshow(final_mask, cmap='gray')
    # plt.title('Segmentation Mask from SVM Classifier')
    # plt.show()

    return final_mask, classifier

def predict_svm(svm_classifier, image_cube):

    # hypercube = image_cube.load()
    hypercube = image_cube
    
    rows, cols, bands = hypercube.shape
    img_reshaped = hypercube.reshape(rows * cols, bands)

    # Predict the class for every pixel
    predicted_labels = svm_classifier.predict(img_reshaped)

    # Reshape the prediction back into a 2D image mask
    mask1 = predicted_labels.reshape(rows, cols)

    return mask1

# --- Post-processing steps ---


def post_process(mask, min_size=80, area_threshold=0.1):

    # 1. Remove small white objects (noise in the background)
    # '150' is the minimum size in pixels for an object to be kept. You can adjust this.

    # converting the integer array with 0 and 1s into a boolean as skimage.morhology expects boolean array
    mask = mask.astype(bool)
    mask1 = remove_small_objects(mask, min_size=min_size)

    # 2. Fill small black holes inside the plant
    mask2 = remove_small_holes(mask1, area_threshold=area_threshold)

    return mask2

# --------------------------------------------------------------------------------------------------------#``

# def viz(final_mask, hypercube, filename, random_band=45):
def viz(final_mask, hypercube, random_band=2):

    fig, axes = plt.subplots(1, 2, figsize=(9, 5))

    single_band = hypercube[:, :, random_band]
    axes[0].imshow(single_band, cmap='gray')
    axes[0].set_title('Origninal')

    axes[1].imshow(final_mask, cmap='gray')
    axes[1].set_title('SVM Mask')
    plt.close()

# --- Visualize the huge improvement ---

def viz_all(initial_mask, final_mask, hypercube, random_band=2):

    fig, axes = plt.subplots(1, 3, figsize=(9, 5))

    axes[0].imshow(initial_mask, cmap='gray')
    axes[0].set_title('Initial Mask')

    single_band = hypercube[:, :, random_band]

    axes[1].imshow(single_band, cmap='gray')
    axes[1].set_title('Origninal')

    axes[2].imshow(final_mask, cmap='gray')
    axes[2].set_title('Final Mask')
    plt.close()
    
    plt.show()


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

def split_beans(reflectance, svm_mask, y_threshold):
    
    # split reflectance based on Y-axis
    upper_beans_ref = reflectance[:y_threshold, :, :]   # the 'shape' is height, width, wavelengths(bands)
    lower_beans_ref = reflectance[y_threshold:, :, :]

    # crop the mask based on the Y-axis
    upper_beans_mask = svm_mask[:y_threshold, :]
    lower_beans_mask = svm_mask[y_threshold:, :]

    upper_beans = [upper_beans_ref, upper_beans_mask]
    lower_beans = [lower_beans_ref, lower_beans_mask]

    return upper_beans, lower_beans # [ref, mask]


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



