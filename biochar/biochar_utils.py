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