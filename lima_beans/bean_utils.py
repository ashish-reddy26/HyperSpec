import matplotlib.pyplot as plt
import numpy as np

class QtLinePicker:
    """
    Interactive Qt tool to set a horizontal dividing line with a single click.
    """
    def __init__(self, hypercube, display_band=70):
        self.display_band = hypercube[:, :, display_band]
        self.split_y = None
        
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        # Using a grayscale colormap often helps see boundaries better
        self.ax.imshow(self.display_band, cmap='gray') 
        self.ax.set_title("Click ONCE between the upper and lower objects.\nWindow will close automatically.")
        
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self._onclick)

    def _onclick(self, event):
        if event.inaxes != self.ax:
            return
            
        # Capture the Y-coordinate of the click
        self.split_y = int(event.ydata)
        
        # Automatically close the window after the click
        plt.close(self.fig)

    def get_y(self):
        """Opens the Qt window and blocks until clicked."""
        plt.show(block=True) 
        return self.split_y
    

import os
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops

def process_and_save_objects(hypercube, mask, split_y, output_dir="processed_objects"):
    """
    Saves the mask, extracts average reflectance per object into a DataFrame, 
    and saves isolated .npy cubes with np.nan backgrounds.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save the overall 2D mask
    np.save(os.path.join(output_dir, "full_mask.npy"), mask)
    print("Saved full mask.")

    labeled_mask = label(mask)
    regions = regionprops(labeled_mask)
    
    print(f"Found {len(regions)} distinct objects. Processing...")

    # List to hold the dictionary rows for our Pandas DataFrame
    all_object_averages = []

    for i, region in enumerate(regions):
        object_id = f"object_{i+1:02d}"
        
        # 1. Determine Upper vs Lower
        centroid_y, centroid_x = region.centroid
        position = "upper" if centroid_y < split_y else "lower"
        
        # 2. Get the bounding box and crop
        min_y, min_x, max_y, max_x = region.bbox
        cropped_cube = hypercube[min_y:max_y, min_x:max_x, :]
        
        # Create a boolean mask for JUST this object within its bounding box
        cropped_mask = labeled_mask[min_y:max_y, min_x:max_x] == (i + 1)
        
        # --- SOLVING POINT 1: Average Reflectance Row ---
        # Extract ONLY the True pixels (ignores the rectangular corners entirely)
        true_object_pixels = cropped_cube[cropped_mask] 
        mean_reflectance = np.mean(true_object_pixels, axis=0)
        
        # Build the row dictionary: ID, Position, then Band_0, Band_1, etc.
        row_data = {'Object_ID': object_id, 'Position': position}
        for band_idx, refl_val in enumerate(mean_reflectance):
            row_data[f'Band_{band_idx}'] = refl_val
            
        all_object_averages.append(row_data)
        
        # --- SOLVING POINT 2: Eliminating Bounding Box Background ---
        # Cast to float so we can use np.nan for the background pixels
        cropped_cube_isolated = cropped_cube.astype(float) 
        cropped_cube_isolated[~cropped_mask] = np.nan 
        
        # Save the individual numpy file
        filename = f"{object_id}_{position}.npy"
        np.save(os.path.join(output_dir, filename), cropped_cube_isolated)
        
    # Build the final CSV with rows = objects, columns = bands
    df_averages = pd.DataFrame(all_object_averages)
    df_averages.to_csv(os.path.join(output_dir, "per_object_reflectance.csv"), index=False)
    
    print(f"Successfully saved {len(regions)} individual cubes and the reflectance CSV.")

# --- How to use it ---
# process_and_save_objects(my_hypercube, final_mask_2d, custom_split_y)