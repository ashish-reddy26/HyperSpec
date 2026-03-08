### Data Processing Workflow (`cube.ipynb`)

Follow the steps below to extract data, train the SVM, and process the bean reflectance cubes. Ensure you run the cells in order.

1. **`CELL1` (Setup):** Update the `reflectance_np_file` variable with the exact path to your reflectance NumPy file.
2. **`CELL2` (Load):** Run this cell to load the hyperspectral array into memory.
3. **`CELL3` (Interactive Data Extraction):** Running this cell will open a new pop-up window displaying the image.
   * **Left-click** to select **Bean** pixels (marks them with a red `X`).
   * **Right-click** to select **Background** pixels (marks them with a blue `O`).
   * *Note:* Ensure you select at least one pixel for every bean. Close the window when finished.
4. **`CELL4` (Train SVM):** Trains the SVM classifier based on your selections and displays the resulting mask. Inspect this visual output for any stray noise pixels in the background.
5. **`CELL5` (Post-Process):** Cleans up isolated noise from the mask. If you still see noise, adjust the `max_size` parameter and re-run.
6. **`CELL6` (Set Dividing Line):** A new window will appear showing Band 70. Click exactly once between the upper and lower beans to set the `Y` coordinate that divides them. The window will close automatically.
7. **`CELL7` (Save Results):** Run this final cell to process and save the following files to your `results/` directory:
   * The overall binary mask (`full_mask.npy`).
   * The mean reflectance of each bean across all bands (`per_object_reflectance.csv`).
   * Isolated reflectance cubes for each individual bean (`object_XX_upper/lower.npy`).