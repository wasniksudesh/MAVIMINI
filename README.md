# MAVIMINI
A project based on Computer Vision using Raspberry pi
Instructions -
for using webcam or picamera with raspberry pi -
1)attach webcam/picamera and movidius stick.

2)assuming the graph folder is in the same directory - run the python command -

  python3 final-python-code-using-Picamera.py --graph graphs/mobilenetgraph
  
If you want to run a video and detect frame by frame - please input path in the python file.

The output of the files is in the form of audio via speakers and there is no display of the video feed to reduce processing time and power

There are going to be results on the terminal line - like the class and its confidence that was detected and also the fps recorded in the end.

There is no stopping of the process once started rather than through terminal or direct power off of the raspberry pi

Please note - the following graph file was generated using models given in NC-APPZOO (intel movidius software appzoo)
it classifies 20 different objects!
