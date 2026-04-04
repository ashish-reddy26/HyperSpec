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
    

import cuvis
import numpy as np

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

    # --- THIS IS THE NEW, FINAL STEP ---
    # 7. CONVERT TO FLOATING POINT REFLECTANCE
    print("Converting to float and scaling by 10000.0...")
    reflectance_cube = integer_cube.astype(np.float32) / 10000.0

    # clipping the reflectance values to 0-1, as there were a few outliers >1.0, which is technically not allowed wrt reflectance values
    reflectance_cube = np.clip(reflectance_cube, 0, 1.0)

    # 8. VERIFY AND VISUALIZE
    print(f"Final data type: {reflectance_cube.dtype}")

    return reflectance_cube

