# Obj_detection
Real-time object detection code ( Hackathon project- RGMCET)

This project aims to do real-time object detection through a laptop camera or webcam using OpenCV and MobileNetSSD. The idea is to loop over each frame of the video stream, detect objects like person, chair, dog, etc. and bound each detection in a box.


### How to run this code?
just create a clone / clone the repository and cd into the folder (new folder)

** one ** Install all the necessary libraries.These are some of the libraries I had to install:

```
pip install opencv-python
pip install opencv-contrib-python
pip install opencv-python-headless
pip install opencv-contrib-python-headless
pip install matplotlib
pip install imutils

** two** Make sure you have your video devices connected (e.g. Webcam, FaceTime HD Camera, etc.). You can list them by typing this in your terminal
```
system_profiler SPCameraDataType
system_profiler SPCameraDataType | grep "^    [^ ]" | sed "s/    //" | sed "s/://"
```
* three** To start your video stream and real-time object detection, run the following command:

```
python real_time.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel


**four ** If you need any help regarding the arguments you pass, try:

```
python real_time_.py --help
```
