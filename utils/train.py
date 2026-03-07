from skimage.morphology import remove_small_objects, remove_small_holes, opening, disk
from skimage.measure import label, regionprops
import spectral.io.envi as envi
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
import os


class SVMclassifier:

    def __init__(self):
        pass

    def train_svm(self, hypercube, X, y, C_reg=450):
        
        # hypercube = image_cube.load()
        # hypercube = slice_cube(image_cube)

        # --- Build the training data from your selected points ---
        bands = hypercube.shape[2]

        X_train = X
        y_train = y

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
        svm_mask = predicted_labels.reshape(rows, cols)

        # --- Visualize the Result ---
        # plt.figure(figsize=(6, 6))
        # plt.imshow(svm_mask, cmap='gray')
        # plt.title('Segmentation Mask from SVM Classifier')
        # plt.show()

        return svm_mask, classifier

    def predict_svm(self, svm_classifier, hypercube):

        # hypercube = image_cube.load()
        rows, cols, bands = hypercube.shape
        img_reshaped = hypercube.reshape(rows * cols, bands)

        # Predict the class for every pixel
        predicted_labels = svm_classifier.predict(img_reshaped)

        # Reshape the prediction back into a 2D image mask
        mask1 = predicted_labels.reshape(rows, cols)

        return mask1

    # --- Post-processing steps ---


    def post_process(self, mask, max_size=20, area_threshold=0.1):

        '''
        Args:
        mask: Binary Mask for removing noise.
        max_size (int): removes objects smaller than **or equal to** its value

        Returns:
        Refined binary mask
        '''

        # 1. Remove small white objects (noise in the background)

        # converting the integer array with 0 and 1s into a boolean as skimage.morhology expects boolean array
        mask = mask.astype(bool)
        mask_refined = remove_small_objects(mask, max_size=max_size)
        
        return mask_refined
    
    def display(self, hypercube, mask1, mask2=None):

        fig, axes = plt.subplots(1, 2, figsize=(9, 5))

        axes[0].imshow(mask1, cmap='gray')
        axes[0].set_title('SVM Mask')

        if mask2 is not None:
            axes[1].imshow(mask2, cmap='gray')
            axes[1].set_title('SVM Mask2')

        else: 
            random_band=70
            single_band = hypercube[:, :, random_band]
            axes[1].imshow(single_band, cmap='gray')
            axes[1].set_title(f'Random Band')