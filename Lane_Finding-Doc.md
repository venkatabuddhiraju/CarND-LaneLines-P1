Finding Lane Lines on the Road

The goals / steps of this project are the following:

1. Write simple code so that the lanes on the road are identified through the lines on the road on either side of the lane.
2. The lanes are extrapolated so that there are solid marked lines on either side of the current lane. 

**Pipeline**

1. Load 1 image 
2. Apply grayscale() : Use an open cv method to convert the image to grayscale. This is so that we have just the lines when we apply edge detector.
3. Apply canny() : Canny's edge detection algorith is one of the widely used algorithms. Used an open cv method to apply it on the grayscaled image.
4. Apply gaussian_blur() : Blur the edges so the lines are pronounced. 
5. Apply region_of_interest(): Find the specific region on the image that we want to find the lines in. 
6. Apply draw_lines() : A method that draws lines between a bunch of points. 

    1. Initially this method just draws a line between two points
    2. In the later versions, the lines are extrapolated. 
        1. Find the slope of a line
        2. Using the slope, determine if the line is left or right
        3. Save the line's co-ordinates and slopes.
        4. Calculate the averages of the co-ordinates and slopes
        5. Find the min and max of the co-ordinates using the averages
        6. Draw lines between the min and max co-ordinates.
    3. Smooth the lines
        SMOOTH LINES - not sure how to do this yet.
        
        One thing that you can do is to draw the lines based not only on the current frame but also on the past frames. You can create a list of parameters of the past frames and draw the lines based on the mean of these parameters.You can do it by creating a class or just using global variables.
        
7. Apply hough_lines() : Applies a hough transform on the image in place. 
8. Then you apply to every frame of the video (which is an image).

**Shortcomings**
1. At curves, the lines drawn are off.
2. Throws error when slope of line is 0. 

**Improvements**
1. Possible improvements would be to imagine lines on a road where there are no lines. 
2. During night, when lines could not be seen. 