import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Import your custom modules just like in the notebook
from utils.utils import run
from utils.train import SVMclassifier
from lima_beans.bean_utils import QtLinePicker, process_and_save_objects

def process_file(file_path):
    reflectance_np_file = Path(file_path)
    
    if not reflectance_np_file.exists():
        print(f"Error: Could not find file {file_path}")
        return

    print(f"\n--- Starting Pipeline for: {reflectance_np_file.stem} ---")
    
    # 1. Load Data
    reflectanceX = np.load(reflectance_np_file)
    
    # 2. Data Extraction (Window 1)
    print("\n STEP 1: Opening Pixel Picker...")
    print("   Select bean pixels (Left Click) and background (Right Click). Close window when done.")
    X, y = run(reflectanceX)
    
    # 3. Train SVM & Clean Mask
    print("\n STEP 2: Training SVM Classifier...")
    svm_classifier = SVMclassifier()
    svm_mask, trained_classifier = svm_classifier.train_svm(reflectanceX, X, y)
    
    print("   Cleaning mask noise...")
    final_mask = svm_classifier.post_process(mask=svm_mask, max_size=20)
    
    # 4. Interactive Split Line (Window 2)
    print("\n STEP 3: Opening Line Picker...")
    print("   Click exactly ONCE between the upper and lower beans.")
    line_picker = QtLinePicker(reflectanceX, display_band=70)
    custom_split_y = line_picker.get_y()
    
    if custom_split_y is None:
        print("No dividing line selected. Aborting save.")
        return
        
    print(f"Dividing line set at Y = {custom_split_y}")
    
    # 5. Save Results
    print("\n STEP 4: Extracting and Saving Data...")
    output_dir = f'results/{reflectance_np_file.stem}'
    process_and_save_objects(reflectanceX, final_mask, custom_split_y, output_dir=output_dir)
    
    print(f"---  Finished processing {reflectance_np_file.stem} ---\n")

if __name__ == "__main__":
    # Ensure matplotlib uses the correct backend for the CLI
    import matplotlib
    matplotlib.use('Qt5Agg') 
    
    # Set up the command line argument parser
    parser = argparse.ArgumentParser(description="Process Hyperspectral Bean Cubes")
    parser.add_argument("file", help="Path to the .npy reflectance file")
    
    args = parser.parse_args()
    process_file(args.file)