### HyperSpec

`utils/extract_reflectance.py`: Provide paths to CUVIS session file and the destination file for NumPy that stores the reflectance data.


* Example: `python utils/extract_reflectance.py --input path/to/session.cu3s --output dest/path/to/numpy/file.npy`


`seg_utils.py`: 

```
1. We should have the reflectance already as numpy files, extracted from previous step `extract_reflectance.py`.

2. Now, we need to load a numpy file - need to get the object pixels and background pixels.

3. Once it is done, train an SVM classifier with those selected pixels as they will automatically store the reflectance data at each selected pixel location (both the object and the background).

4. Once trainign is completed the Visualize the SVM mask, to validate the performance and/or to verify if there is any noise in predicted mask.

5. If there is no noise, we are good to use the mask, if not we need to post-process the mask to remove the noise.

6. Once we get the final mask, visualize to confirm.

------------------------------
custom adoption for lima beans
------------------------------

7. Now split the beans based on the y-axis, so that we get the upper and lower beans' mask separately.
    - the final output contains 2 items, first one being the cube and the second one is the correponding mask.

8. Now obtain the average reflectanc of each object and create a list of average reflectances for each object.

9. Finally save the average ref values of each object at available bands as rows of a dataframe.

10. Now we obtain the cube for each object, as we have the reflectance (numpy-file) and svm-masks for upper and lower beans, we can process them separately to obtain their beans.

11. Save the final extrcated cubes of each bean in the result folder.

12. The dataframe and these individual bean cubes are the final expected outputs.

```