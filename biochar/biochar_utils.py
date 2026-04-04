import numpy as np
import spectral.io.envi as envi

def load_spectral_cube(header_file, bin_spectral_file):
    
    image_cube = envi.open(header_file, bin_spectral_file)
    
    # load cube as array
    hypercube_arr = np.array(image_cube.load())
    
    # replace NaN values with 0s
    final_cube_arr = np.nan_to_num(hypercube_arr, nan=0.0)

    return final_cube_arr, image_cube

import os
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
from PIL import Image

import os
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
from PIL import Image

def get_biochar_reflectance(hypercube, mask, output_dir="processed_objects"):
    """
    Saves the full mask as a PNG image, extracts average reflectance per object into a DataFrame 
    using wavelength columns (900 to 1700, step 4), 
    and saves isolated .npy cubes with np.nan backgrounds.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Save the overall 2D mask as a PNG image (NOT a numpy array)
    mask_uint8 = (mask * 255).astype(np.uint8)
    Image.fromarray(mask_uint8).save(os.path.join(output_dir, "mask.png"))
    print("Saved mask as PNG.")

    labeled_mask = label(mask)
    regions = regionprops(labeled_mask)
    
    print(f"Found {len(regions)} distinct objects. Processing...")

    # Generate the wavelength array: 900 to 1700 (inclusive) with a step of 4
    wavelengths = np.arange(900, 1704, 4)

    # List to hold the dictionary rows for our Pandas DataFrame
    all_object_averages = []

    for i, region in enumerate(regions):
        object_id = f"object_{i+1:02d}"
        
        # Get the bounding box and crop
        min_y, min_x, max_y, max_x = region.bbox
        cropped_cube = hypercube[min_y:max_y, min_x:max_x, :]
        
        # Create a boolean mask for JUST this object within its bounding box
        cropped_mask = labeled_mask[min_y:max_y, min_x:max_x] == (i + 1)
        
        # Extract ONLY the True pixels
        true_object_pixels = cropped_cube[cropped_mask] 
        
        # Calculate the average reflectance for each band
        mean_reflectance = np.mean(true_object_pixels, axis=0)
        
        # Build the row dictionary: ID, then Wavelengths
        row_data = {'Object_ID': object_id}
        for band_idx, refl_val in enumerate(mean_reflectance):
            # Map the calculated average to the specific wavelength column
            if band_idx < len(wavelengths):
                row_data[f'{wavelengths[band_idx]}'] = refl_val
            
        all_object_averages.append(row_data)
        
        # Eliminating Bounding Box Background
        # Cast to float so we can use np.nan for the background pixels
        cropped_cube_isolated = cropped_cube.astype(float) 
        cropped_cube_isolated[~cropped_mask] = np.nan 
        
        # 2. Save the individual reflectance numpy file
        filename = f"{object_id}.npy"
        np.save(os.path.join(output_dir, filename), cropped_cube_isolated)
        
        # (Optional) If you also want to save the individual cropped masks as PNGs, uncomment below:
        # cropped_mask_uint8 = (cropped_mask * 255).astype(np.uint8)
        # Image.fromarray(cropped_mask_uint8).save(os.path.join(output_dir, f"{object_id}_mask.png"))

    # Build the final CSV with rows = objects, columns = wavelengths
    df_averages = pd.DataFrame(all_object_averages)
    df_averages.to_csv(os.path.join(output_dir, "per_object_reflectance.csv"), index=False)
    
    print(f"Successfully saved {len(regions)} individual reflectance cubes, the full mask image, and the reflectance CSV.")


class TrainUtils:

    def __init__(self) -> None:
        pass

    def point_picker():
        pass

    def train_svm():
        pass

    def predict_svm():
        pass

