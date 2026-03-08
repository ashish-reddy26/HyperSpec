import matplotlib.pyplot as plt
import numpy as np


class QtPixelPicker:
    """
    Interactive Qt tool to select object and background pixels in one window.
    """
    def __init__(self, hypercube, display_band=50):
        self.display_band = hypercube[:, :, display_band]
        
        self.object_coords = []
        self.background_coords = []
        
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.imshow(self.display_band, cmap='viridis')
        self.ax.set_title("Left Click: Object (Red) | Right Click: Background (Blue)\nCLOSE WINDOW to save coordinates.")
        
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self._onclick)

    def _onclick(self, event):
        if event.inaxes != self.ax:
            return
            
        x, y = int(event.xdata), int(event.ydata)
        
        if event.button == 1: # Left click
            self.object_coords.append([y, x])
            self.ax.plot(x, y, 'rx', markersize=8)
            
        elif event.button == 3: # Right click
            self.background_coords.append([y, x])
            self.ax.plot(x, y, 'bo', markersize=5, markerfacecolor='none')
            
        self.fig.canvas.draw()

    def get_data(self):
        """Opens the Qt window and blocks until it is closed."""
        plt.show(block=True) 
        return self.object_coords, self.background_coords
    
def run(my_hypercube, display_band=50):
    
    # 1. Open the Qt window and collect clicks
    picker = QtPixelPicker(my_hypercube, display_band)
    object_pixels, background_pixels = picker.get_data()

    print(f"Captured {len(object_pixels)} object pixels and {len(background_pixels)} background pixels.")

    # 2. Convert coordinate lists to numpy arrays for advanced indexing
    object_idx = np.array(object_pixels)
    bg_idx = np.array(background_pixels)

    # 3. Extract the full spectral vector across ALL bands
    # This grabs the pixel at (y, x) and pulls all data along the third axis
    object_spectra = my_hypercube[object_idx[:, 0], object_idx[:, 1], :]
    background_spectra = my_hypercube[bg_idx[:, 0], bg_idx[:, 1], :]

    # 4. Stack them into X (features) and y (labels) arrays for Scikit-Learn
    X = np.vstack((object_spectra, background_spectra))

    # Let's assign Object = 1, Background = 0
    y_object = np.ones(object_spectra.shape[0])
    y_bg = np.zeros(background_spectra.shape[0])
    y = np.concatenate((y_object, y_bg))

    print(f"Final feature matrix shape for SVM: {X.shape}")

    return X, y


