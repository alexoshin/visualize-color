# visualize-color
Interactive visualization of color spaces and color thresholding

## Usage
Run using:

```python visualization.py <img_path>```

Two windows should appear. One allows you to edit the minimum and maximum values for each of the three channels of the color mask, using trackbar sliders. A visualization of the mask is shown below the sliders. When this window is focused, you can use the keys 1-9 on your keyboard to change the colorspace of the image. Press q to quit, which will also print out the final values to the console. The second window shows the image you are editing, as well as the effect of the mask on the image. Use this window to determine which values you want to use by hovering over the image and viewing the color channel values in the bottom right corner of the window.
