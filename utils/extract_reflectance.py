import numpy as np
import argparse
import cuvis
import os

def extract_reflectance_cube(session_filename):
    '''
    This function takes the session file ending with the extension '.cu3s' and processes RAW binary data into Reflectance data, 
    then scales the reflectance values to 0-1 as their actual values are unint16 ranging from 0 to >10000
    The processing  context is the key, which helps us set the processing mode which will be given as a processing argument to teh context.
    '''

    # 1. Path to session file
    # session_filename = "data/cuvis_files/sam_17.cu3s"
    if session_filename == '':
        raise FileNotFoundError

    # 2. LOAD THE RAW MEASUREMENT
    print("Loading session file...")
    session_file = cuvis.SessionFile(session_filename)
    raw_measurement = session_file.get_measurement(0)

    # 3. CREATE A PROCESSING CONTEXT
    print("Loading processing context from session file...")
    processing_context = cuvis.ProcessingContext(session_file)

    # 4. SET THE DESIRED PROCESSING MODE
    proc_args = cuvis.ProcessingArgs()
    proc_args.processing_mode = cuvis.ProcessingMode.Reflectance
    processing_context.set_processing_args(proc_args)

    # 5. APPLY THE PROCESSING
    print("Applying processing to create reflectance data...")
    processed_measurement = processing_context.apply(raw_measurement)

    # 6. EXTRACT THE INTEGER CUBE
    print("Extracting scaled integer cube to numpy array...")
    integer_cube = processed_measurement.cube.to_numpy()

    # 7. CONVERT TO FLOATING POINT REFLECTANCE
    print("Converting to float and scaling by 10000.0...")
    reflectance_cube = integer_cube.astype(np.float32) / 10000.0

    # clipping the reflectance values to 0-1, as there were a few outliers >1.0, which is technically not allowed wrt reflectance values
    reflectance_cube = np.clip(reflectance_cube, 0, 1.0)

    # 8. VERIFY AND VISUALIZE
    print(f"Final data type: {reflectance_cube.dtype}")

    return reflectance_cube

def run(cuvis_session_file, np_dest_location):
    
    reflectance = extract_reflectance_cube(cuvis_session_file)

    # 1. Extract ONLY the folder path (e.g., 'results/reflectance_data')
    parent_folder = os.path.dirname(np_dest_location)

    # 2. Check if there is a parent folder, and create it if it doesn't exist
    if parent_folder:
        # exist_ok=True means it won't crash if the folder is already there!
        os.makedirs(parent_folder, exist_ok=True)
        
    # os.makedirs(np_dest_location, exist_ok=True)
    np.save(np_dest_location, reflectance)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', 
                        help = 'Path to session file.', required=True)
    
    parser.add_argument('--output', 
                        help = 'Output NumPy file destination.', required=True)
    
    args = parser.parse_args()
    
    run(args.input, args.output)
    