#                                 Rover Autonomous Land Vehicle in a Neural Network
-------------------------------------------------------

                        Machine Perception and Cognitive Robotics Laboratory
                            Center for Complex Systems and Brain Sciences
                                    Florida Atlantic University
-------------------------------------------------------
##Info

        This is the second version of the RALVINN code. It uses OpenCV to read the images from
        the BrookStone Rover rather than using Pygame. 

##Current Status

        This is a work in progress by the members of the FAU Machine Perception Club. You may 
        encounter bugs and other issues when running this code. 

##Getting Started with RALVINN.

        Before running the code you will need to make sure that the following python packages 
        have been installed

        1)OpenCV 3.0
        2)PyPNG
        3)PyQrCode and QRTools
        4)Pygame

##How to connect to BrookStone Rover.

        Open your wifi settings. Finds the Rover SSID, it usually starts with Rover_ROVERID. 
        Connecto the SSID ID. The green LED light on the Rover will stop flashig once 
        a connection has beeb established.

## How to control the BrookStone Rover 2.0:
        W - Forward
        S - Back
        D - Right
        A - LEFT
        U - Toggle Infrared Night Vision.
        J - Camera Up
        K - Camera Down
        I - Toggle Lights
        R - Neural Network
        Space bar - Take a pic


##To Do List
      1)Give the BrookStone Rover 2.0 the ability to read QR codes.
      2)Add more helper functions.
      3)Fix Bugs!
