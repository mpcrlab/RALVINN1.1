# RALVINN1.1 
WARNING: ONLY TESTED ON THE MPCR LAB LAPTOPS RUNNING UBUNTU!


RALVINN code in pygame, with separate image processing in opencv.


The RoverImageProcessor class has a method that takes a list of plain english colors e.g. ["orange, pink"] and returns a list of numbers (the rover's current "state").  Each number of the output corresponds to the color at the same index of the input list.  The number can indicate an object on the left (1), center (2), or right (3) third of the screen, or no object found (0).
