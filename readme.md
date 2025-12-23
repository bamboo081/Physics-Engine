Math fair project - kinetic physics engine

# Change the Model
Around line 323, you see the function: _init_camera_and_model(self):
This function is where you define the computer vision model you will be using to help track your motion. YOLOv8 is what is used by default, but feel free
to use other models as well. 

# Changes
There are a few bugs, which I will be addressing later on. I will also add additional features.

# Installation
The list of libraries needed are in the requirements.txt folder, but the Torch version can be changed. This project uses the Pytorch CUDA 12.9 download version. The version can be changed with a different one if your device does not support CUDA.