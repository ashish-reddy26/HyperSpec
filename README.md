### HyperSpec

In the notebook `cube.ipynb` the required cells contain the cell number as `CELL{2}`

```
1. `CELL1` Update the location of  reflectance numpy file `reflectance_np_file and run`
2. `CELL2` Run it.
3. `CELL3` Data extraction - once we run this cell, a window pops up with a the image.
    - Here select the `bean` pixels with left-click, we should be able to see a `red color X mark` 
    - Select background pixels using the right-click, it shows `blue color circle`
    - Make sure there is atleast one pixel selected for each bean please refer to this image for reg=ference
4. `CELL4` Train an SVM classifier and displays the resultant mask.
    - look for any noise pixels in the background region
5. `CELL5` The post-process should clean the noise, if any, adjust the `max_size` accordingly
6. `CELL6` When you run this cell, a pop window appears, select a pixel where this its `y` coordinate should be able to separate the upper and lower beans 
7. `CELL7` Once we got the `y coordinate` run this to save the following results:
    - Overall binary mask as NumPy file
    - overall mean reflectance of each bean for all the available bands as CSV
    - Saves reflectance cubes for each bean as NumPy
    
```